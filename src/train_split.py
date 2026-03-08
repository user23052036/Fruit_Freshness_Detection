import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ensure_dirs

FEATURE_DIR = "Features"
MODEL_DIR = "models"
ensure_dirs(MODEL_DIR)

def main(test_size=0.2, random_state=42):
    print("[INFO] Loading Features...")
    X = np.load(os.path.join(FEATURE_DIR, "X.npy"))
    y_veg = np.load(os.path.join(FEATURE_DIR, "y_veg.npy"))
    y_fresh = np.load(os.path.join(FEATURE_DIR, "y_fresh.npy"))

    if not (len(X) == len(y_veg) == len(y_fresh)):
        raise ValueError("Feature and label lengths do not match")

    # combined stratify label preserves veg+fresh distribution
    stratify_labels = np.array([f"{v}_{f}" for v, f in zip(y_veg, y_fresh)])
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=stratify_labels, random_state=random_state)

    np.save(os.path.join(MODEL_DIR, "X_train.npy"), X[train_idx])
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X[test_idx])
    np.save(os.path.join(MODEL_DIR, "y_veg_train.npy"), y_veg[train_idx])
    np.save(os.path.join(MODEL_DIR, "y_veg_test.npy"), y_veg[test_idx])
    np.save(os.path.join(MODEL_DIR, "y_fresh_train.npy"), y_fresh[train_idx])
    np.save(os.path.join(MODEL_DIR, "y_fresh_test.npy"), y_fresh[test_idx])

    print(f"[SUCCESS] Train/test split created. Train={len(train_idx)} Test={len(test_idx)}")

if __name__ == "__main__":
    main()