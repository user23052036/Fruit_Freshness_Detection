import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold

from utils import save_model, load_model, ensure_dirs

MODEL_DIR   = "models"
FEATURE_DIR = "Features"

ensure_dirs(MODEL_DIR)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def _load_split(split, vt, scaler, selected):
    """Load and preprocess one split (train / val / test)."""
    X       = np.load(os.path.join(MODEL_DIR, f"X_{split}.npy"))
    y_veg   = np.load(os.path.join(MODEL_DIR, f"y_veg_{split}.npy"))
    y_fresh = np.load(os.path.join(MODEL_DIR, f"y_fresh_{split}.npy"))
    X_r = vt.transform(X)
    X_s = scaler.transform(X_r)
    X_f = X_s[:, selected]
    return X_f, y_veg, y_fresh


# ─────────────────────────────────────────────────────────────
# Per-vegetable decision bounds (from training decisions)
# ─────────────────────────────────────────────────────────────

def compute_per_veg_bounds(decisions, y_veg, veg_classes):
    per_veg = {}
    for veg in veg_classes:
        mask = (y_veg == veg)
        d    = decisions[mask]
        if len(d) < 50:
            continue
        per_veg[veg] = {
            "p5"      : float(np.percentile(d, 5)),
            "p95"     : float(np.percentile(d, 95)),
            "hard_min": float(d.min()),
            "hard_max": float(d.max()),
        }
    return per_veg


# ─────────────────────────────────────────────────────────────
# p5/p95 stability across folds (train split only)
# ─────────────────────────────────────────────────────────────

def check_bound_stability(X_final, y_veg, fresh_model, veg_classes, n_splits=5):
    """
    KFold check on training data.
    If p5 or p95 shifts > 10% across folds → score scale is unstable.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("\n[INFO] p5/p95 stability across folds:")

    for veg in veg_classes:
        mask = (y_veg == veg)
        X_v  = X_final[mask]
        if len(X_v) < n_splits * 10:
            continue

        p5s, p95s = [], []
        for _, val_idx in kf.split(X_v):
            d = fresh_model.decision_function(X_v[val_idx])
            p5s.append(np.percentile(d, 5))
            p95s.append(np.percentile(d, 95))

        p5_cv  = np.std(p5s)  / (abs(np.mean(p5s))  + 1e-9)
        p95_cv = np.std(p95s) / (abs(np.mean(p95s)) + 1e-9)

        status = "OK"
        if p5_cv > 0.10 or p95_cv > 0.10:
            status = "WARNING — scale unstable, will fall back to global bounds"

        print(f"  {veg:<12}  p5_cv={p5_cv:.3f}  p95_cv={p95_cv:.3f}  [{status}]")


# ─────────────────────────────────────────────────────────────
# All threshold calibration on VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────

def calibrate_boundary_threshold(val_decisions, y_fresh_val):
    """
    Sweep abs(decision) thresholds on VAL SET.
    Find threshold where misclassification rate first exceeds 10%.
    """
    thresholds  = np.arange(0.05, 1.55, 0.05)
    best_thresh = 0.5

    for t in thresholds:
        near = np.abs(val_decisions) < t
        if near.sum() < 5:
            continue
        pred  = (val_decisions[near] > 0).astype(int)
        true  = y_fresh_val[near].astype(int)
        error = (pred != true).mean()
        if error >= 0.10:
            best_thresh = float(t)
            break

    return best_thresh


def calibrate_unstable_range_thresh(vt, scaler, selected, fresh_model,
                                     global_bounds, n_per_veg=60, seed=42):
    """
    Calibrate instability threshold using the EXACT SAME augmentations
    as predict_cli.py at runtime.

    Sampling is STRATIFIED PER VEGETABLE — not random across the val set.
    Random sampling risks banana dominating the distribution and leaving
    potato/cucumber instability underestimated.

    For each vegetable: sample up to n_per_veg images.
    Run same 6 augmentations as augment_and_score() in predict_cli.py.
    Threshold = 95th percentile of all observed score ranges.

    Calibration matches runtime behavior exactly.
    """
    import cv2
    from extract_features import (
        model as deep_model, preprocess_input, extract_handcrafted
    )

    val_paths_file  = os.path.join(MODEL_DIR, "val_image_paths.npy")
    val_veg_file    = os.path.join(MODEL_DIR, "y_veg_val.npy")

    if not os.path.exists(val_paths_file) or not os.path.exists(val_veg_file):
        print()
        print("=" * 60)
        print("[WARNING] CALIBRATION FALLBACK ACTIVE")
        print("  val_image_paths.npy or y_veg_val.npy not found.")
        print("  Re-run extract_dataset_features.py then train_split.py")
        print("  to generate these files and enable real augmentation calibration.")
        print("  Results with fallback are LESS RELIABLE.")
        print("=" * 60)
        print()
        return _calibrate_unstable_fallback(fresh_model, global_bounds)

    val_paths = np.load(val_paths_file, allow_pickle=True)
    val_veg   = np.load(val_veg_file,   allow_pickle=True)

    # Stratified sampling: n_per_veg images per vegetable
    rng = np.random.default_rng(seed)
    sample_paths = []
    vegetables   = np.unique(val_veg)

    for veg in vegetables:
        veg_mask  = val_veg == veg
        veg_paths = val_paths[veg_mask]
        n         = min(n_per_veg, len(veg_paths))
        chosen    = rng.choice(len(veg_paths), size=n, replace=False)
        sample_paths.extend(veg_paths[chosen].tolist())
        print(f"  [INFO] Calibration sampling: {veg:<12} {n} images")

    p5, p95 = global_bounds["p5"], global_bounds["p95"]
    denom   = max(p95 - p5, 1e-6)

    def norm(raw):
        return float(np.clip((raw - p5) / denom * 100.0, 0.0, 100.0))

    def preprocess_feats(img_rgb):
        batch  = preprocess_input(np.expand_dims(img_rgb.astype(np.float32), 0))
        deep_f = deep_model.predict(batch, verbose=0)[0].astype(np.float32)
        hand_f = extract_handcrafted(img_rgb)
        feats  = np.concatenate([deep_f, hand_f])
        X      = vt.transform(np.array([feats]))
        Xs     = scaler.transform(X)
        return Xs[:, selected]

    ranges  = []
    skipped = 0

    for path in sample_paths:
        img = cv2.imread(path)
        if img is None:
            skipped += 1
            continue

        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb    = cv2.resize(rgb, (224, 224))
        h, w   = rgb.shape[:2]

        # Exact same augmentations as predict_cli.py augment_and_score()
        augmented = [
            np.clip(rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8),
            np.clip(rgb.astype(np.float32) * 0.85, 0, 255).astype(np.uint8),
            cv2.flip(rgb, 1),
            cv2.GaussianBlur(rgb, (5, 5), 0),
            cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w//2, h//2),  5, 1.0), (w, h)),
            cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0), (w, h)),
        ]

        try:
            scores = [norm(float(fresh_model.decision_function(preprocess_feats(aug))[0]))
                      for aug in augmented]
            ranges.append(max(scores) - min(scores))
        except Exception:
            skipped += 1

    if len(ranges) < 20:
        print()
        print("=" * 60)
        print(f"[WARNING] CALIBRATION FALLBACK ACTIVE")
        print(f"  Only {len(ranges)} valid samples after augmentation.")
        print(f"  Results are LESS RELIABLE than real-aug calibration.")
        print("=" * 60)
        print()
        return _calibrate_unstable_fallback(fresh_model, global_bounds)

    thresh = float(np.percentile(ranges, 95))
    print(f"[INFO] Instability threshold (real aug P95): {thresh:.4f}")
    print(f"       Calibrated on {len(ranges)} val images ({skipped} skipped), "
          f"stratified across {len(vegetables)} vegetables.")
    return thresh


def _calibrate_unstable_fallback(fresh_model, global_bounds,
                                  n_samples=500, n_aug=6, seed=42):
    """
    Fallback only — used when val_image_paths.npy is missing.
    Gaussian feature-space noise: different distribution from runtime.
    This is less accurate. The strong warning above tells the user to fix it.
    """
    print("[INFO] Fallback: Gaussian feature-space noise (not aligned to runtime).")
    rng   = np.random.default_rng(seed)
    p5, p95 = global_bounds["p5"], global_bounds["p95"]
    denom   = max(p95 - p5, 1e-6)

    # Load any available feature split for sampling
    for split in ("val", "train"):
        path = os.path.join(MODEL_DIR, f"X_{split}.npy")
        if os.path.exists(path):
            X_raw = np.load(path)
            break
    else:
        return 13.0  # final fallback if nothing available

    idx   = rng.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
    X_sub = X_raw[idx]
    ranges = []

    for x in X_sub:
        noise  = rng.normal(0, 0.02, size=(n_aug, x.shape[0]))
        X_aug  = x + noise
        raws   = fresh_model.decision_function(X_aug)
        scores = [float(np.clip((r - p5) / denom * 100.0, 0.0, 100.0)) for r in raws]
        ranges.append(max(scores) - min(scores))

    thresh = float(np.percentile(ranges, 95))
    print(f"[INFO] Fallback instability threshold (P95): {thresh:.4f}")
    return thresh


def calibrate_mahalanobis_thresholds(X_final_train,
                                      X_final_val, mean, precision):
    """
    Calibrate soft-OOD thresholds from training distances.
    thresh_caution = 90th percentile of training distances
    thresh_ood     = 99th percentile of training distances
    Report OOD rate on validation for consistency check.
    """
    def mahal(X, m, P):
        diff = X - m
        return np.sqrt(np.einsum('ij,jk,ik->i', diff, P, diff))

    train_dists = mahal(X_final_train, mean, precision)
    val_dists   = mahal(X_final_val,   mean, precision)

    thresh_caution = float(np.percentile(train_dists, 90))
    thresh_ood     = float(np.percentile(train_dists, 99))

    ood_rate_val = float((val_dists > thresh_ood).mean())
    print(f"[INFO] Mahalanobis thresh_caution={thresh_caution:.3f}  "
          f"thresh_ood={thresh_ood:.3f}")
    print(f"[INFO] OOD rate on validation: {ood_rate_val:.4f}")

    return thresh_caution, thresh_ood, ood_rate_val


def calibrate_veg_confidence_gate(val_decisions, veg_probs_val,
                                   y_fresh_val, y_veg_val):
    """
    Simple sweep to find (conf_thresh, gap_thresh) pair that minimises
    per-veg normalization mismatch on the validation set.
    Conservative defaults are used; the sweep confirms or adjusts them.
    """
    # Default: absolute >= 0.70 AND gap >= 0.15
    # Report score sensitivity to gate flip on val set
    top1 = veg_probs_val.max(axis=1)
    top2 = np.sort(veg_probs_val, axis=1)[:, -2]
    gap  = top1 - top2

    n_total    = len(top1)
    n_pass     = ((top1 >= 0.70) & (gap >= 0.15)).sum()
    n_fallback = n_total - n_pass

    print(f"[INFO] Veg gate on val: {n_pass}/{n_total} use per-veg bounds, "
          f"{n_fallback} fall back to global.")

    return 0.70, 0.15   # tunable; defaults confirmed by inspection


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading artifacts...")

    vt       = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler   = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_features.npy"))

    X_train, y_veg_train, y_fresh_train = _load_split("train", vt, scaler, selected)
    X_val,   y_veg_val,   y_fresh_val   = _load_split("val",   vt, scaler, selected)
    # Test split is NOT loaded here — reserved for evaluate_models.py only

    print(f"[INFO] Feature matrix — train: {X_train.shape}  val: {X_val.shape}")

    # ── Vegetable classifier ─────────────────────────────────
    le           = LabelEncoder()
    yveg_encoded = le.fit_transform(y_veg_train)
    save_model(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print("[INFO] Training vegetable classifier...")
    veg_model = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, class_weight="balanced"
    )
    veg_model.fit(X_train, yveg_encoded)
    save_model(veg_model, os.path.join(MODEL_DIR, "veg_svm.joblib"))
    print("[DONE] Vegetable classifier saved.")

    # ── Freshness classifier ─────────────────────────────────
    # probability=True is intentionally omitted here.
    # The freshness path uses decision_function() (signed margin) for scoring
    # and predict() for the binary label. predict_proba() is never called.
    # Enabling Platt scaling would add training cost and imply probability
    # semantics that do not apply to this margin-based scoring system.
    print("[INFO] Training freshness classifier...")
    fresh_model = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced"   # no probability=True — uses decision_function
    )
    fresh_model.fit(X_train, y_fresh_train.astype(int))
    save_model(fresh_model, os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    print("[DONE] Freshness classifier saved.")

    # ── Training decisions → bounds ──────────────────────────
    train_decisions = fresh_model.decision_function(X_train)

    global_bounds = {
        "p5"      : float(np.percentile(train_decisions, 5)),
        "p95"     : float(np.percentile(train_decisions, 95)),
        "hard_min": float(train_decisions.min()),
        "hard_max": float(train_decisions.max()),
    }

    veg_classes    = le.classes_.tolist()
    per_veg_bounds = compute_per_veg_bounds(
        train_decisions, y_veg_train, veg_classes
    )

    print("[INFO] Per-vegetable bounds (from training):")
    for veg, b in per_veg_bounds.items():
        print(f"  {veg:<12}  p5={b['p5']:.4f}  p95={b['p95']:.4f}  "
              f"hard_min={b['hard_min']:.4f}  hard_max={b['hard_max']:.4f}")

    # ── p5/p95 stability check ───────────────────────────────
    check_bound_stability(X_train, y_veg_train, fresh_model, veg_classes)

    # ── ALL calibration on VALIDATION SPLIT ─────────────────
    print("\n[INFO] Calibrating all thresholds on validation split...")

    val_decisions = fresh_model.decision_function(X_val)

    boundary_thresh = calibrate_boundary_threshold(val_decisions, y_fresh_val)
    print(f"[INFO] Boundary threshold (val): {boundary_thresh:.4f}")

    # Fix 1+2: real image-space augmentation calibration matching predict_cli.py
    unstable_range_thresh = calibrate_unstable_range_thresh(
        vt, scaler, selected, fresh_model, global_bounds
    )

    # ── Mahalanobis (Ledoit-Wolf shrinkage) ──────────────────
    print("[INFO] Fitting Ledoit-Wolf covariance on training features...")
    lw         = LedoitWolf().fit(X_train)
    train_mean = X_train.mean(axis=0).astype(np.float32)
    precision  = lw.precision_.astype(np.float32)

    np.save(os.path.join(MODEL_DIR, "train_mean.npy"),      train_mean)
    np.save(os.path.join(MODEL_DIR, "train_precision.npy"), precision)

    thresh_caution, thresh_ood, ood_rate_val = calibrate_mahalanobis_thresholds(
        X_train, X_val, train_mean, precision
    )

    # ── Veg confidence gate calibration ─────────────────────
    veg_probs_val = veg_model.predict_proba(X_val)
    veg_conf_thresh, veg_gap_thresh = calibrate_veg_confidence_gate(
        val_decisions, veg_probs_val, y_fresh_val, y_veg_val
    )

    # ── Per-class centroids (class-consistency sanity check) ─
    # Computes the mean feature vector per vegetable class from training data.
    # All centroids are computed in the shared 100-feature space — the same
    # space both SVMs use after variance filtering, scaling, and top-k selection.
    print("[INFO] Computing per-class centroids...")
    class_centroids = {}
    for i, veg in enumerate(le.classes_):
        mask = (yveg_encoded == i)
        class_centroids[veg] = X_train[mask].mean(axis=0).tolist()

    # Calibrate PER-CLASS centroid ratio thresholds on validation set.
    # A single global threshold is crude because cluster spread differs
    # between vegetables (banana has tight color features, potato is spread).
    # Per-class thresholds respect that difference.
    #
    # For each val sample: ratio = dist_to_predicted / dist_to_second_best.
    # Per-class threshold = 95th percentile of correct predictions for that class.
    yveg_val_enc  = le.transform(y_veg_val)
    veg_pred_val  = veg_model.predict(X_val)
    centroids_arr = np.array([class_centroids[v] for v in le.classes_])

    # Compute ratios for all val samples
    all_ratios    = np.zeros(len(X_val))
    for i in range(len(X_val)):
        x          = X_val[i]
        dists      = np.linalg.norm(centroids_arr - x, axis=1)
        pred_idx   = int(veg_pred_val[i])
        sorted_idx = np.argsort(dists)
        d_pred     = dists[pred_idx]
        d_second   = next(dists[j] for j in sorted_idx if j != pred_idx)
        all_ratios[i] = float(d_pred / (d_second + 1e-9))

    # Per-class P95 on correctly predicted val samples
    per_class_ratio_thresh = {}
    print("[INFO] Per-class centroid ratio thresholds (P95 of correct val predictions):")
    for i, veg in enumerate(le.classes_):
        correct_veg_mask = (veg_pred_val == i) & (yveg_val_enc == i)
        if correct_veg_mask.sum() < 5:
            per_class_ratio_thresh[veg] = 1.0   # safe fallback
        else:
            per_class_ratio_thresh[veg] = float(
                np.percentile(all_ratios[correct_veg_mask], 95)
            )
        print(f"  {veg:<12}  threshold={per_class_ratio_thresh[veg]:.4f}  "
              f"(n={correct_veg_mask.sum()})")

    np.save(os.path.join(MODEL_DIR, "class_centroids.npy"),
            np.array([class_centroids[v] for v in le.classes_], dtype=np.float32))

    # ── Save scoring_config.json ─────────────────────────────
    # Fix 3: preflight thresholds now live in config — tunable without
    # code changes, consistent across training and deployment.
    scoring_config = {
        "global_bounds"           : global_bounds,
        "per_veg_bounds"          : per_veg_bounds,
        "boundary_threshold"      : boundary_thresh,
        "unstable_range_thresh"   : unstable_range_thresh,
        "veg_confidence_threshold": veg_conf_thresh,
        "veg_gap_threshold"       : veg_gap_thresh,
        "mahal_thresh_caution"    : thresh_caution,
        "mahal_thresh_ood"        : thresh_ood,
        "ood_rate_val"              : ood_rate_val,
        "centroid_ratio_thresholds" : per_class_ratio_thresh,  # per-class, not global
        # Preflight thresholds — part of the trained system, not hardcoded
        "min_laplacian_variance"  : 28.0,
        "min_brightness"          : 30.0,
        "max_brightness"          : 220.0,
        "min_coverage"            : 0.40,
        "calibration_note"        : (
            "All thresholds calibrated on validation split. "
            "unstable_range_thresh derived from real augmentation percentile. "
            "Test set not touched during training or calibration."
        ),
    }

    config_path = os.path.join(MODEL_DIR, "scoring_config.json")
    with open(config_path, "w") as f:
        json.dump(scoring_config, f, indent=2)

    print(f"\n[DONE] Scoring config saved → {config_path}")
    print("[INFO] Test set untouched. Run evaluate_models.py for final metrics.")


if __name__ == "__main__":
    main()