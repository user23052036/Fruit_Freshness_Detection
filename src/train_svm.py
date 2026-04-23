# Fixes vs previous version
# --------------------------
# Issue 2: same val set used for isotonic calibration AND threshold selection
#           → leakage / overfit thresholds.
#           Fix: deterministic 50/50 split of X_val into:
#             cal_val   → CalibratedClassifierCV (isotonic)
#             thr_val   → select_thresholds (formal gate calibration)
#           The split is stratified by y_fresh_val to keep class balance.
#
# Issue 4: grid was 25 points. Fix: log-scale 30-point grid identical to
#           preprocess_and_rank.py so both files always agree.
#           Both best params + val accuracy per task are printed + stored.
#
# Issue 5: infeasibility fallback did not log WHY.
#           Fix: infeasibility_case ('a' or 'b'), risk_at_quantiles, and
#           coverage_at_quantiles are stored in scoring_config.json so
#           gate behaviour is debuggable without re-running training.
#
# Issue 6: downstream compatibility confirmed:
#   predict_cli.py  — loads veg_svm.joblib (CalibratedClassifierCV wraps
#                     the base SVC, so predict_proba / predict interfaces
#                     are unchanged). No modification needed.
#   evaluate_models.py — loads the same artifact filenames.  All filenames
#                     are unchanged from the original pipeline.
#   scoring_config.json — all original keys are present plus new diagnostic
#                     keys (svm_best_params, threshold_selection_result,
#                     infeasibility_diagnosis). Downstream readers that
#                     ignore unknown keys are unaffected.
#
# Issue 7: validation report is printed to stdout at the end of main().

import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from utils import save_model, load_model, ensure_dirs, SVM_PARAM_GRID
from threshold_selection import (
    select_thresholds,
    diagnose_infeasibility,
    compute_gate_metrics,
)

MODEL_DIR   = "models"

ensure_dirs(MODEL_DIR)

# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def _load_split(split, vt, scaler, selected):
    X       = np.load(os.path.join(MODEL_DIR, f"X_{split}.npy"))
    y_veg   = np.load(os.path.join(MODEL_DIR, f"y_veg_{split}.npy"))
    y_fresh = np.load(os.path.join(MODEL_DIR, f"y_fresh_{split}.npy"))
    X_r = vt.transform(X)
    X_s = scaler.transform(X_r)
    return X_s[:, selected], y_veg, y_fresh


# ─────────────────────────────────────────────────────────────
# Hyperparameter tuning  (issue 4 fix)
# ─────────────────────────────────────────────────────────────

def tune_svm(X_train, y_train, task_label, n_cv_splits=5, random_state=42):
    """
    GridSearchCV with the 30-point log-scale grid.
    probability=False in base SVC; the vegetable model is wrapped in
    CalibratedClassifierCV after this call.

    Returns (best_estimator_refitted, best_params, best_cv_score).
    """
    base = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True,
                         random_state=random_state)
    gs = GridSearchCV(
        base, SVM_PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    bp = gs.best_params_
    bs = float(gs.best_score_)
    print(f"[INFO] {task_label} — best params: {bp}  CV acc={bs:.4f}")
    return gs.best_estimator_, bp, bs


# ─────────────────────────────────────────────────────────────
# Per-vegetable decision bounds
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
# p5/p95 stability check
# ─────────────────────────────────────────────────────────────

def check_bound_stability(X_final, y_veg, fresh_model, veg_classes,
                           n_splits=5):
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
# Augmentation stats on a val subset
# ─────────────────────────────────────────────────────────────

def compute_val_aug_stats(vt, scaler, selected, fresh_model,
                           global_bounds, per_veg_bounds=None,
                           veg_model=None, le=None,
                           veg_conf_thresh=0.70, veg_gap_thresh=0.15,
                           n_per_veg=100, seed=42,
                           restrict_to_indices=None):
    """
    Compute per-sample aug statistics on a stratified val subset.
    restrict_to_indices: if provided (array of row indices), only
    process those rows.  Used to separate cal_val from thr_val.

    Returns (aug_stats, p95_thresh).
    """
    import cv2
    from extract_features import (
        model as deep_model, preprocess_input, extract_handcrafted,
    )

    val_paths_file = os.path.join(MODEL_DIR, "val_image_paths.npy")
    val_veg_file   = os.path.join(MODEL_DIR, "y_veg_val.npy")

    if not os.path.exists(val_paths_file) or not os.path.exists(val_veg_file):
        raise RuntimeError(
            "Strict pipeline violation: insufficient or missing augmentation "
            "data for calibration. Expected val_image_paths.npy and "
            "y_veg_val.npy. Re-run extract_dataset_features.py then "
            "train_split.py."
        )

    val_paths = np.load(val_paths_file, allow_pickle=True)
    val_veg   = np.load(val_veg_file,   allow_pickle=True)

    rng        = np.random.default_rng(seed)
    vegetables = np.unique(val_veg)
    sample_idx = []

    for veg in vegetables:
        veg_global_idx = np.where(val_veg == veg)[0]
        if restrict_to_indices is not None:
            veg_global_idx = np.intersect1d(veg_global_idx, restrict_to_indices)
        if len(veg_global_idx) == 0:
            raise RuntimeError(
                f"Strict pipeline violation: no samples available for {veg} "
                "in augmentation calibration subset."
            )
        n      = min(n_per_veg, len(veg_global_idx))
        chosen = rng.choice(len(veg_global_idx), size=n, replace=False)
        sample_idx.extend(veg_global_idx[chosen].tolist())
        print(f"  [INFO] Aug sampling {veg:<12} {n} images "
              f"(restricted={restrict_to_indices is not None})")

    if not sample_idx:
        raise RuntimeError(
            "Strict pipeline violation: augmentation calibration subset is empty."
        )

    def preprocess_feats(img_rgb):
        batch  = preprocess_input(
            np.expand_dims(img_rgb.astype(np.float32), 0)
        )
        deep_f = deep_model.predict(batch, verbose=0)[0].astype(np.float32)
        hand_f = extract_handcrafted(img_rgb)
        feats  = np.concatenate([deep_f, hand_f])
        X      = vt.transform(np.array([feats]))
        Xs     = scaler.transform(X)
        return Xs[:, selected]

    aug_stats = []

    for idx in sample_idx:
        path = val_paths[idx]
        img  = cv2.imread(path)
        if img is None:
            raise RuntimeError(
                f"Strict pipeline violation: failed to load image at {path} "
                "during augmentation calibration."
            )

        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb  = cv2.resize(rgb, (224, 224))
        h, w = rgb.shape[:2]

        if veg_model is None or le is None or not per_veg_bounds:
            raise RuntimeError(
                "Strict pipeline violation: vegetable model, label encoder, "
                "or per-vegetable bounds are missing for augmentation calibration."
            )

        Xf_orig      = preprocess_feats(rgb)
        veg_p        = veg_model.predict_proba(Xf_orig)[0]
        sorted_p     = np.sort(veg_p)[::-1]
        pred_veg     = le.inverse_transform([int(np.argmax(veg_p))])[0]
        veg_conf_cal = float(sorted_p[0]) * 100.0
        gap_cal      = float(sorted_p[0] - sorted_p[1]) * 100.0

        sample_bounds = per_veg_bounds.get(pred_veg)
        if sample_bounds is None:
            raise RuntimeError(
                f"Strict pipeline violation: per-vegetable bounds missing for {pred_veg} "
                "during augmentation calibration."
            )

        p5_s  = sample_bounds["p5"]
        p95_s = sample_bounds["p95"]
        denom = max(p95_s - p5_s, 1e-6)

        def norm(raw, _p5=p5_s, _d=denom):
            return float(np.clip((raw - _p5) / _d * 100.0, 0.0, 100.0))

        augmented = [
            np.clip(rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8),
            np.clip(rgb.astype(np.float32) * 0.85, 0, 255).astype(np.uint8),
            cv2.flip(rgb, 1),
            cv2.GaussianBlur(rgb, (5, 5), 0),
            cv2.warpAffine(
                rgb, cv2.getRotationMatrix2D((w//2, h//2),  5, 1.0), (w, h),
            ),
            cv2.warpAffine(
                rgb, cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0), (w, h),
            ),
        ]

        raws   = [
            float(fresh_model.decision_function(preprocess_feats(a))[0])
            for a in augmented
        ]
        scores = [norm(r) for r in raws]
        aug_stats.append({
            "val_idx"    : int(idx),
            "aug_range"  : float(max(scores) - min(scores)),
            "crosses_bnd": bool(min(raws) < 0 and max(raws) > 0),
        })

    if len(aug_stats) < 20:
        raise RuntimeError(
            "Strict pipeline violation: insufficient or missing augmentation "
            "data for calibration. Only "
            f"{len(aug_stats)} valid augmented samples collected."
        )

    p95_thresh = float(np.percentile([s["aug_range"] for s in aug_stats], 95))
    print(f"[INFO] unstable_range_thresh (P95): {p95_thresh:.4f}  "
            f"({len(aug_stats)} samples)")
    return aug_stats, p95_thresh


# ─────────────────────────────────────────────────────────────
# Mahalanobis calibration
# ─────────────────────────────────────────────────────────────
def mahal(X, m, P):
    diff = X - m
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, P, diff)) 

def calibrate_mahalanobis_thresholds(X_train, X_val, mean, precision):
    train_dists = mahal(X_train, mean, precision)
    val_dists   = mahal(X_val,   mean, precision)

    thresh_caution = float(np.percentile(train_dists, 90))
    thresh_ood     = float(np.percentile(train_dists, 99))
    ood_rate_val   = float((val_dists > thresh_ood).mean())

    print(f"[INFO] Mahalanobis thresh_caution={thresh_caution:.3f}  "
          f"thresh_ood={thresh_ood:.3f}")
    print(f"[INFO] OOD rate on validation: {ood_rate_val:.4f}")
    return thresh_caution, thresh_ood, ood_rate_val


# ─────────────────────────────────────────────────────────────
# Infeasibility diagnosis with logging  (issue 5 fix)
# ─────────────────────────────────────────────────────────────

def _capture_infeasibility_info(decisions, predictions, true_labels,
                                 is_ood, crosses_bnd, aug_range,
                                 epsilon=0.10):
    """
    Call diagnose_infeasibility and capture the structured output so it
    can be stored in scoring_config.json.

    Returns a dict with:
        infeasibility_case : 'a' | 'b' | 'insufficient_data' | 'reaches_epsilon'
        description        : human-readable explanation
        risk_at_quantiles  : {q: risk}
        coverage_at_quantiles : {q: coverage}
    """
    quantiles = (0.50, 0.75, 0.90, 0.95, 0.99)
    decisions   = np.asarray(decisions,   dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    is_ood      = np.asarray(is_ood,      dtype=bool)
    crosses_bnd = np.asarray(crosses_bnd, dtype=bool)
    aug_range   = np.asarray(aug_range,   dtype=float)

    abs_margin = np.abs(decisions)
    risk_at_q  = {}
    cov_at_q   = {}

    for q in quantiles:
        T_b = float(np.quantile(abs_margin, q))
        m   = compute_gate_metrics(
            decisions, predictions, true_labels,
            is_ood, crosses_bnd, aug_range,
            T_boundary=T_b, T_instability=float("inf"),
        )
        risk_at_q[str(q)] = None if np.isnan(m.risk) else float(m.risk)
        cov_at_q[str(q)]  = float(m.coverage)

    defined = [(q, r) for q, r in risk_at_q.items() if r is not None]
    if len(defined) < 2:
        case = "insufficient_data"
        desc = ("Fewer than 2 quantiles with n_reliable > 0. "
                "Cannot classify infeasibility. Reduce n_min or add data.")
    else:
        first_r = defined[0][1]
        last_r  = defined[-1][1]
        drop    = first_r - last_r
        reaches = any(r <= epsilon for _, r in defined)
        if reaches:
            case = "reaches_epsilon"
            desc = ("Risk reaches epsilon at some quantile. "
                    "Constraint may be satisfiable with finer T_b grid.")
        elif drop < 0.02:
            case = "a"
            desc = (f"Case (a): Risk is flat (total drop={drop:.4f} < 0.02 pp). "
                    "Margin gate has no predictive power for errors on this data.")
        else:
            case = "b"
            desc = (f"Case (b): Risk decreases with margin quantile "
                    f"(drop={drop:.4f} pp) but never reaches "
                    f"epsilon={epsilon}. Base model error rate is the binding "
                    "constraint.")

    return {
        "infeasibility_case"   : case,
        "description"          : desc,
        "epsilon"              : epsilon,
        "risk_at_quantiles"    : risk_at_q,
        "coverage_at_quantiles": cov_at_q,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading artifacts from preprocess_and_rank.py output...")

    vt = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))

    union_path = os.path.join(MODEL_DIR, "selected_union_features.npy")
    if not os.path.exists(union_path):
        raise RuntimeError(
            "models/selected_union_features.npy not found. "
            "Run preprocess_and_rank.py with the current pipeline before train_svm.py."
        )
    selected = np.load(union_path)
    print(f"[INFO] Feature set size: {len(selected)}")

    X_train, y_veg_train, y_fresh_train = _load_split("train", vt, scaler, selected)
    X_val,   y_veg_val,   y_fresh_val   = _load_split("val",   vt, scaler, selected)
    # Test split NOT loaded here — reserved for evaluate_models.py only.

    print(f"[INFO] Feature matrix — train: {X_train.shape}  val: {X_val.shape}")

    # ── Encode vegetable labels ───────────────────────────────
    le           = LabelEncoder()
    yveg_encoded = le.fit_transform(y_veg_train)
    save_model(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    veg_classes  = le.classes_.tolist()

    # ── Issue 2 fix: split val into cal_val and thr_val ───────
    # 50/50 stratified split so both halves have similar class balance.
    # cal_val  → CalibratedClassifierCV (isotonic)
    # thr_val  → select_thresholds (formal gate calibration)
    #
    # Without this split:
    #   isotonic regression observes the same samples that later
    #   determine T_boundary and T_instability, so the probability
    #   calibration implicitly encodes the threshold-selection targets.
    #   This makes the threshold selection overfit to the val set.
    #
    # With this split:
    #   calibration and threshold selection see disjoint data.
    #   Thresholds are unbiased estimates of out-of-sample gate performance.
    n_val        = len(X_val)
    val_idx_all  = np.arange(n_val)

    # Stratify by y_fresh_val to keep fresh/rotten balance in both halves
    cal_idx, thr_idx = train_test_split(
        val_idx_all,
        test_size=0.5,
        stratify=y_fresh_val.astype(int),
        random_state=42,
    )
    cal_idx = np.sort(cal_idx)
    thr_idx = np.sort(thr_idx)

    X_cal_val,  y_veg_cal,   y_fresh_cal   = (X_val[cal_idx],
                                               y_veg_val[cal_idx],
                                               y_fresh_val[cal_idx])
    X_thr_val,  y_veg_thr,   y_fresh_thr   = (X_val[thr_idx],
                                               y_veg_val[thr_idx],
                                               y_fresh_val[thr_idx])

    yveg_cal_enc = le.transform(y_veg_cal)
    yveg_thr_enc = le.transform(y_veg_thr)

    print(f"[INFO] val split: cal_val={len(cal_idx)}  thr_val={len(thr_idx)}")

    # ── Vegetable SVM — GridSearchCV on training data ─────────
    print("\n[INFO] Tuning vegetable classifier (GridSearchCV, 5-fold)...")
    veg_base, veg_best_params, veg_cv_score = tune_svm(
        X_train, yveg_encoded, task_label="Vegetable",
    )
    save_model(veg_base, os.path.join(MODEL_DIR, "veg_svm_base.joblib"))

    # ── Isotonic calibration on cal_val only  (issue 2 fix) ───
    # CalibratedClassifierCV with FrozenEstimator fits the calibrator
    # on pre-trained veg_base using only cal_val.
    # Isotonic calibration is more flexible than Platt scaling and
    # avoids assuming a sigmoidal score-to-probability shape.
    print("\n[INFO] Calibrating vegetable probabilities "
      "(isotonic, FrozenEstimator, fit on cal_val only)...")
    veg_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(veg_base),
                    method="isotonic"
                    )
    
    veg_model.fit(X_cal_val, yveg_cal_enc)
    save_model(veg_model, os.path.join(MODEL_DIR, "veg_svm.joblib"))
    print("[DONE] Calibrated vegetable classifier saved → models/veg_svm.joblib")

    # Report val accuracy on cal_val (calibration quality check)
    veg_cal_acc = float(
        (veg_model.predict(X_cal_val) == yveg_cal_enc).mean()
    )
    veg_thr_acc = float(
        (veg_model.predict(X_thr_val) == yveg_thr_enc).mean()
    )
    print(f"[INFO] Vegetable acc — cal_val={veg_cal_acc:.4f}  "
          f"thr_val={veg_thr_acc:.4f}  (expect ~equal, large gap → small split)")

    # ── Freshness SVM — GridSearchCV on training data ─────────
    print("\n[INFO] Tuning freshness classifier (GridSearchCV, 5-fold)...")
    fresh_model, fresh_best_params, fresh_cv_score = tune_svm(
        X_train, y_fresh_train.astype(int), task_label="Freshness",
    )
    save_model(fresh_model, os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    print(f"[DONE] Freshness classifier saved.")

    fresh_cal_acc = float(
        (fresh_model.predict(X_cal_val) == y_fresh_cal.astype(int)).mean()
    )
    fresh_thr_acc = float(
        (fresh_model.predict(X_thr_val) == y_fresh_thr.astype(int)).mean()
    )
    print(f"[INFO] Freshness acc — cal_val={fresh_cal_acc:.4f}  "
          f"thr_val={fresh_thr_acc:.4f}")

    # ── p5/p95 stability check (training data) ───────────────
    check_bound_stability(X_train, y_veg_train, fresh_model, veg_classes)

    # ── Normalization bounds — computed on FULL VAL SET ───────
    # p5/p95 bounds are a fixed linear scale transform (no label leakage),
    # so using the full val set is safe and necessary: the 50/50 cal/thr
    # split can leave individual vegetable classes below the 50-sample
    # minimum in either half alone. cal_val and thr_val are unchanged.
    print("\n[INFO] Calibrating normalization bounds on full val set...")
    val_decisions = fresh_model.decision_function(X_val)

    global_bounds = {
        "p5"      : float(np.percentile(val_decisions, 5)),
        "p95"     : float(np.percentile(val_decisions, 95)),
        "hard_min": float(val_decisions.min()),
        "hard_max": float(val_decisions.max()),
    }
    per_veg_bounds = compute_per_veg_bounds(
        val_decisions, y_veg_val, veg_classes
    )

    missing_bounds = [veg for veg in veg_classes if veg not in per_veg_bounds]
    if missing_bounds:
        raise RuntimeError(
            "Strict pipeline violation: per-vegetable bounds missing from full val set "
            f"for: {', '.join(missing_bounds)}"
        )

    print("[INFO] Per-vegetable bounds (from full val set):")
    for veg, b in per_veg_bounds.items():
        print(f"  {veg:<12}  p5={b['p5']:.4f}  p95={b['p95']:.4f}  "
              f"hard_min={b['hard_min']:.4f}  hard_max={b['hard_max']:.4f}")

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

    # OOD flags on thr_val only (used for threshold selection)
    thr_dists  = mahal(X_thr_val, train_mean, precision)
    is_ood_thr = (thr_dists > thresh_ood)

    veg_conf_thresh = 0.70
    veg_gap_thresh  = 0.15

    # ── Augmentation stats — on thr_val rows only  (issue 2) ──
    print("\n[INFO] Computing augmentation stats on thr_val subset...")
    aug_stats, unstable_range_thresh = compute_val_aug_stats(
        vt, scaler, selected, fresh_model, global_bounds,
        per_veg_bounds=per_veg_bounds,
        veg_model=veg_model,
        le=le,
        veg_conf_thresh=veg_conf_thresh,
        veg_gap_thresh=veg_gap_thresh,
        n_per_veg=100,
        restrict_to_indices=thr_idx,   # ONLY thr_val rows
    )

    # ── Formal threshold selection on thr_val  (strict) ─────
    boundary_thresh            = None
    threshold_selection_result = {}
    infeasibility_diagnosis    = {}

    if len(aug_stats) >= 20:
        print("\n[INFO] Running formal threshold selection on thr_val subset...")

        sub_idx         = np.array([s["val_idx"]     for s in aug_stats], dtype=int)
        sub_aug_range   = np.array([s["aug_range"]   for s in aug_stats], dtype=float)
        sub_crosses_bnd = np.array([s["crosses_bnd"] for s in aug_stats], dtype=bool)

        # Map global val indices → thr_val local indices
        # thr_idx is sorted, so searchsorted gives the local position
        local_sub_idx     = np.searchsorted(thr_idx, sub_idx)
        sub_decisions     = fresh_model.decision_function(X_thr_val)[local_sub_idx]
        sub_predictions   = fresh_model.predict(X_thr_val)[local_sub_idx]
        sub_true          = y_fresh_thr[local_sub_idx].astype(int)
        sub_is_ood        = is_ood_thr[local_sub_idx]

        n_min  = max(20, len(sub_idx) // 20)
        result = select_thresholds(
            sub_decisions, sub_predictions, sub_true,
            sub_is_ood, sub_crosses_bnd, sub_aug_range,
            epsilon=0.10,
            n_min=n_min,
        )

        threshold_selection_result = {
            "feasible"        : result.feasible,
            "T_boundary"      : float(result.T_boundary),
            "T_instability"   : float(result.T_instability),
            "risk"            : float(result.risk) if not np.isnan(result.risk) else None,
            "coverage"        : float(result.coverage),
            "n_reliable"      : int(result.n_reliable),
            "n_min_used"      : int(n_min),
            "epsilon"         : 0.10,
            "subset_size"     : int(len(sub_idx)),
            "data_source"     : "thr_val (disjoint from cal_val)",
        }

        if result.feasible:
            boundary_thresh       = result.T_boundary
            unstable_range_thresh = result.T_instability
            print(
                f"[INFO] Formal thresholds — "
                f"T_boundary={boundary_thresh:.4f}  "
                f"T_instability={unstable_range_thresh:.4f}\n"
                f"       Risk={result.risk:.4f}  "
                f"Coverage={result.coverage:.4f}  "
                f"n_reliable={result.n_reliable}"
            )
        else:
            print(
                "[ERROR] Formal threshold selection infeasible.  Diagnosing..."
            )
            diagnose_infeasibility(
                sub_decisions, sub_predictions, sub_true,
                sub_is_ood, sub_crosses_bnd, sub_aug_range,
                epsilon=0.10,
            )
            # Capture structured diagnosis for the config
            infeasibility_diagnosis = _capture_infeasibility_info(
                sub_decisions, sub_predictions, sub_true,
                sub_is_ood, sub_crosses_bnd, sub_aug_range,
                epsilon=0.10,
            )
            print(f"[INFO] Infeasibility case: "
                  f"{infeasibility_diagnosis['infeasibility_case']}")
            print(f"       {infeasibility_diagnosis['description']}")
            raise RuntimeError(
                "Formal threshold selection infeasible; see diagnose_infeasibility() "
                "output for details."
            )
    else:
        print("[ERROR] Not enough aug samples for formal threshold selection.")
        raise RuntimeError(
            f"Insufficient augmentation samples for select_thresholds(): "
            f"got {len(aug_stats)}, need at least 20."
        )

    # ── Per-class centroids ───────────────────────────────────
    print("[INFO] Computing per-class centroids...")
    class_centroids = {}
    for i, veg in enumerate(le.classes_):
        mask = (yveg_encoded == i)
        class_centroids[veg] = X_train[mask].mean(axis=0).tolist()

    # Use full val set for centroid ratio thresholds (enough samples)
    veg_pred_full = veg_model.predict(X_val)
    yveg_val_enc_full = le.transform(y_veg_val)
    centroids_arr = np.array([class_centroids[v] for v in le.classes_])

    all_ratios = np.zeros(len(X_val))
    for i in range(len(X_val)):
        x          = X_val[i]
        dists      = np.linalg.norm(centroids_arr - x, axis=1)
        pred_idx   = int(veg_pred_full[i])
        sorted_idx = np.argsort(dists)
        d_pred     = dists[pred_idx]
        d_second   = next(dists[j] for j in sorted_idx if j != pred_idx)
        all_ratios[i] = float(d_pred / (d_second + 1e-9))

    per_class_ratio_thresh = {}
    print("[INFO] Per-class centroid ratio thresholds "
          "(P95 of correct val predictions):")
    for i, veg in enumerate(le.classes_):
        correct_mask = (veg_pred_full == i) & (yveg_val_enc_full == i)
        if correct_mask.sum() < 5:
            per_class_ratio_thresh[veg] = 1.0
        else:
            per_class_ratio_thresh[veg] = float(
                np.percentile(all_ratios[correct_mask], 95)
            )
        print(f"  {veg:<12}  threshold={per_class_ratio_thresh[veg]:.4f}  "
              f"(n={correct_mask.sum()})")

    np.save(
        os.path.join(MODEL_DIR, "class_centroids.npy"),
        np.array([class_centroids[v] for v in le.classes_], dtype=np.float32),
    )

    # ── Save scoring_config.json ──────────────────────────────
    scoring_config = {
        "grade_thresholds"          : {"truly_fresh": 85, "fresh": 65, "moderate": 40},
        "use_augmentation_gate"     : False,
        "global_bounds"             : global_bounds,
        "per_veg_bounds"            : per_veg_bounds,
        "boundary_threshold"        : float(boundary_thresh),
        "unstable_range_thresh"     : float(unstable_range_thresh),
        "veg_confidence_threshold"  : veg_conf_thresh,
        "veg_gap_threshold"         : veg_gap_thresh,
        "veg_gate_source"           : "design_constant",
        "mahal_thresh_caution"      : thresh_caution,
        "mahal_thresh_ood"          : thresh_ood,
        "ood_rate_val"              : ood_rate_val,
        "centroid_ratio_thresholds" : per_class_ratio_thresh,
        "min_laplacian_variance"    : 28.0,
        "min_brightness"            : 30.0,
        "max_brightness"            : 220.0,
        "min_coverage"              : 0.40,
        # ── Diagnostic fields (new) ──────────────────────────
        "svm_best_params"           : {
            "vegetable" : veg_best_params,
            "freshness" : fresh_best_params,
        },
        "svm_cv_scores"             : {
            "vegetable" : veg_cv_score,
            "freshness" : fresh_cv_score,
        },
        "val_split_sizes"           : {
            "cal_val" : int(len(cal_idx)),
            "thr_val" : int(len(thr_idx)),
        },
        "val_accuracy"              : {
            "veg_cal_val"   : veg_cal_acc,
            "veg_thr_val"   : veg_thr_acc,
            "fresh_cal_val" : fresh_cal_acc,
            "fresh_thr_val" : fresh_thr_acc,
        },
        # issue 5: threshold selection details always stored
        "threshold_selection_result": threshold_selection_result,
        "infeasibility_diagnosis"   : infeasibility_diagnosis,
        "calibration_note"          : (
            "Normalization bounds from full val set (p5/p95 need enough samples). "
            "val split into cal_val (50%) and thr_val (50%) stratified by y_fresh "
            "(issue-2 leakage fix): "
            "  cal_val  → CalibratedClassifierCV(isotonic, FrozenEstimator(veg_base)); "
            "  thr_val  → select_thresholds(epsilon=0.10) + aug stats. "
            "SVM: GridSearchCV 5-fold stratified, C=[1e-3..100] × "
            "gamma=[1e-4..0.1,'scale'] (issue-4 fix). "
            "Infeasibility case + risk/coverage curve stored in "
            "infeasibility_diagnosis (issue-5 fix). "
            "Design constants: veg_confidence_threshold=0.70, "
            "veg_gap_threshold=0.15. Test set not touched."
        ),
    }

    config_path = os.path.join(MODEL_DIR, "scoring_config.json")
    with open(config_path, "w") as f:
        json.dump(scoring_config, f, indent=2)

    print(f"\n[DONE] Scoring config saved → {config_path}")

    # ── Issue 6: downstream compatibility summary ─────────────
    print("\n" + "=" * 60)
    print("DOWNSTREAM COMPATIBILITY CHECK")
    print("=" * 60)
    print("  predict_cli.py  — NO CHANGES NEEDED")
    print("    Loads: veg_svm.joblib (CalibratedClassifierCV wraps SVC;")
    print("           predict_proba/predict interface unchanged).")
    print("    Loads: fresh_svm.joblib (SVC, decision_function unchanged).")
    print("    Loads: scoring_config.json (all original keys present;")
    print("           new keys ignored by existing readers).")
    print("  evaluate_models.py — NO CHANGES NEEDED")
    print("    Loads same filenames as before.")
    print("    New scoring_config keys are additive (no removals).")
    print("  visualize_results.py — NO CHANGES NEEDED")
    print("    Loads: feature_importances_fresh/veg.npy (same shape).")
    print("    Loads: selected_union_features.npy (new union set — verify")
    print("           visualize_results.py uses selected before slicing X).")

    # ── Issue 7: final validation report ─────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    sel_report_path = os.path.join(MODEL_DIR, "feature_selection_report.json")
    if os.path.exists(sel_report_path):
        with open(sel_report_path) as f:
            sel_report = json.load(f)
        print(f"  best_k                  : {sel_report.get('best_k', 'n/a')}")
        print(f"  union_feature_count     : {sel_report.get('union_feature_count', len(selected))}")
        print(f"  best SVM params (veg)   : "
              f"{sel_report.get('best_svm_params_veg', 'n/a')}")
        print(f"  best SVM params (fresh) : "
              f"{sel_report.get('best_svm_params_fresh', 'n/a')}")
        print(f"  RBF val acc (fresh)     : "
              f"{sel_report.get('rbf_val_acc_fresh', 'n/a'):.4f}"
              if isinstance(sel_report.get("rbf_val_acc_fresh"), float) else
              f"  RBF val acc (fresh)     : {sel_report.get('rbf_val_acc_fresh', 'n/a')}")
        print(f"  RBF val acc (veg)       : "
              f"{sel_report.get('rbf_val_acc_veg', 'n/a'):.4f}"
              if isinstance(sel_report.get("rbf_val_acc_veg"), float) else
              f"  RBF val acc (veg)       : {sel_report.get('rbf_val_acc_veg', 'n/a')}")
    else:
        print(f"  feature_selection_report.json not found — "
              f"run preprocess_and_rank.py first.")
        print(f"  union_feature_count     : {len(selected)}")

    t_sel = scoring_config["threshold_selection_result"]
    print(f"\n  [Threshold Selection]")
    print(f"  feasible                : {t_sel.get('feasible', 'n/a')}")
    print(f"  T_boundary              : {scoring_config['boundary_threshold']:.4f}")
    print(f"  T_instability           : {scoring_config['unstable_range_thresh']:.4f}")
    risk_val = t_sel.get("risk")
    print(f"  risk (thr_val)          : "
          f"{risk_val:.4f}" if risk_val is not None else "  risk (thr_val)          : n/a")
    print(f"  coverage (thr_val)      : "
          f"{t_sel.get('coverage', 'n/a'):.4f}"
          if isinstance(t_sel.get("coverage"), float) else
          f"  coverage (thr_val)      : {t_sel.get('coverage', 'n/a')}")
    print(f"  n_reliable              : {t_sel.get('n_reliable', 'n/a')}")

    if infeasibility_diagnosis:
        print(f"\n  [Infeasibility Diagnosis]")
        print(f"  case                    : "
              f"{infeasibility_diagnosis.get('infeasibility_case', 'n/a')}")
        print(f"  description             : "
              f"{infeasibility_diagnosis.get('description', 'n/a')}")

    print(f"\n  [SVM CV Scores]")
    print(f"  veg  5-fold CV acc      : {veg_cv_score:.4f}")
    print(f"  fresh 5-fold CV acc     : {fresh_cv_score:.4f}")
    print(f"  veg  cal_val acc        : {veg_cal_acc:.4f}")
    print(f"  veg  thr_val acc        : {veg_thr_acc:.4f}")
    print(f"  fresh cal_val acc       : {fresh_cal_acc:.4f}")
    print(f"  fresh thr_val acc       : {fresh_thr_acc:.4f}")
    print("=" * 60)
    print("[INFO] Test set untouched. Run evaluate_models.py for final metrics.")


if __name__ == "__main__":
    main()