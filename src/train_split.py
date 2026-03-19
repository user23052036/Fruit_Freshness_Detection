import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ensure_dirs

FEATURE_DIR = "Features"
MODEL_DIR   = "models"
ensure_dirs(MODEL_DIR)


def main(val_size=0.10, test_size=0.20, random_state=42):
    """
    Stratified 70 / 10 / 20 split.

    Train  → model fitting
    Val    → ALL threshold calibration (boundary, unstable_range,
              veg_conf, veg_gap, Mahalanobis thresholds)
    Test   → final reporting ONLY — never touched before evaluate_models.py
    """
    print("[INFO] Loading Features...")
    X       = np.load(os.path.join(FEATURE_DIR, "X.npy"))
    y_veg   = np.load(os.path.join(FEATURE_DIR, "y_veg.npy"))
    y_fresh = np.load(os.path.join(FEATURE_DIR, "y_fresh.npy"))

    # Load image paths if saved by extract_dataset_features.py
    paths_file = os.path.join(FEATURE_DIR, "image_paths.npy")
    image_paths = np.load(paths_file, allow_pickle=True) \
                  if os.path.exists(paths_file) else None

    if not (len(X) == len(y_veg) == len(y_fresh)):
        raise ValueError("Feature and label lengths do not match.")

    stratify_labels = np.array([f"{v}_{f}" for v, f in zip(y_veg, y_fresh)])
    idx = np.arange(len(X))

    # First cut: train vs (val + test)
    train_idx, valtest_idx = train_test_split(
        idx,
        test_size=val_size + test_size,
        stratify=stratify_labels,
        random_state=random_state,
    )

    # Second cut: val vs test (proportion relative to valtest pool)
    val_rel = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=1.0 - val_rel,
        stratify=stratify_labels[valtest_idx],
        random_state=random_state,
    )

    for name, split_idx in [("train", train_idx),
                             ("val",   val_idx),
                             ("test",  test_idx)]:
        np.save(os.path.join(MODEL_DIR, f"X_{name}.npy"),       X[split_idx])
        np.save(os.path.join(MODEL_DIR, f"y_veg_{name}.npy"),   y_veg[split_idx])
        np.save(os.path.join(MODEL_DIR, f"y_fresh_{name}.npy"), y_fresh[split_idx])
        if image_paths is not None:
            np.save(os.path.join(MODEL_DIR, f"val_image_paths.npy"
                                 if name == "val" else f"{name}_image_paths.npy"),
                    image_paths[split_idx])

    print(
        f"[SUCCESS] Split created — "
        f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}"
    )
    print("[INFO] Test set must remain untouched until evaluate_models.py.")


if __name__ == "__main__":
    main()