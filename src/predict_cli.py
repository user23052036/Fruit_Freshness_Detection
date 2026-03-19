import argparse
import os
import json
import numpy as np
import cv2

from extract_features import (
    extract_features, extract_handcrafted,
    model as deep_model, preprocess_input,
)
from utils import load_model, grade_from_score

MODEL_DIR = "models"


# ─────────────────────────────────────────────────────────────
# Pre-flight checks — enforcement, not documentation
# ─────────────────────────────────────────────────────────────

def compute_object_coverage(gray):
    """
    Crude but sufficient reject signal.
    Returns fraction of frame covered by the largest contour.
    """
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    h, w = gray.shape
    return cv2.contourArea(max(contours, key=cv2.contourArea)) / (h * w)


def preflight_checks(image_path, config):
    """
    Run before any feature extraction.
    Returns ("OK", None) or ("UNRELIABLE", reason_string).

    Coverage is a WARNING only — Otsu thresholding is too unreliable
    on varied real-world backgrounds to justify a hard rejection.
    Blur and brightness are enforced as hard rejects.
    All thresholds read from scoring_config.json (written by train_svm.py).
    """
    img = cv2.imread(image_path)
    if img is None:
        return "UNRELIABLE", "Image unreadable"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur — hard reject
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    min_lap = config.get("min_laplacian_variance", 28.0)
    if lap_var < min_lap:
        return "UNRELIABLE", f"Image out of focus (lap_var={lap_var:.1f} < {min_lap})"

    # Brightness — hard reject
    mean_b = float(gray.mean())
    min_b  = config.get("min_brightness", 30.0)
    max_b  = config.get("max_brightness", 220.0)
    if not (min_b <= mean_b <= max_b):
        return "UNRELIABLE", f"Brightness out of range ({mean_b:.1f} not in [{min_b}, {max_b}])"

    # Coverage — WARNING only, not a reject.
    # Otsu thresholding is unreliable on varied backgrounds (potato on wood,
    # capsicum on dark surface, etc.). A low coverage number may mean the
    # background was not separable, not that the produce is absent.
    coverage     = compute_object_coverage(gray)
    min_coverage = config.get("min_coverage", 0.40)
    if coverage < min_coverage:
        # Return OK but embed coverage warning in reason field for caller to log
        return "OK_LOW_COVERAGE", (
            f"Low object coverage ({coverage:.2f} < {min_coverage}). "
            f"Score may be affected by background. Not rejected."
        )

    return "OK", None


# ─────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────

def load_pipeline_artifacts():
    vt       = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler   = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_features.npy"))
    veg_svm  = load_model(os.path.join(MODEL_DIR, "veg_svm.joblib"))
    fresh_svm= load_model(os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    le       = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    train_mean      = np.load(os.path.join(MODEL_DIR, "train_mean.npy"))
    train_precision = np.load(os.path.join(MODEL_DIR, "train_precision.npy"))
    class_centroids = np.load(os.path.join(MODEL_DIR, "class_centroids.npy"))

    with open(os.path.join(MODEL_DIR, "scoring_config.json")) as f:
        cfg = json.load(f)

    return vt, scaler, selected, veg_svm, fresh_svm, le, \
           train_mean, train_precision, class_centroids, cfg


# ─────────────────────────────────────────────────────────────
# Feature preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_features(feats, vt, scaler, selected):
    X      = vt.transform(np.array([feats]))
    Xs     = scaler.transform(X)
    return Xs[:, selected]


# ─────────────────────────────────────────────────────────────
# Score normalization  (grade_from_score stays pure in utils.py)
# ─────────────────────────────────────────────────────────────

def normalize_score(raw, bounds):
    p5, p95 = bounds["p5"], bounds["p95"]
    denom   = p95 - p5
    if abs(denom) < 1e-6:
        return 50.0
    return float(np.clip((raw - p5) / denom * 100.0, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────
# Mahalanobis distance (OOD flag only — never used for scoring)
# ─────────────────────────────────────────────────────────────

def mahalanobis_dist(x, mean, precision):
    diff = x.flatten() - mean
    return float(np.sqrt(diff @ precision @ diff))


def mahal_zone(dist, thresh_caution, thresh_ood):
    if dist >= thresh_ood:
        return "ood"
    if dist >= thresh_caution:
        return "caution"
    return "trusted"


# ─────────────────────────────────────────────────────────────
# Augmentation — score range (NOT std) drives instability flag
# ─────────────────────────────────────────────────────────────

def augment_and_score(image_path, vt, scaler, selected,
                      fresh_svm, veg_name, cfg):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0, [], []   # (score_range, score_std, scores, aug_raws)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    h, w = rgb.shape[:2]

    augmented = [
        np.clip(rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8),
        np.clip(rgb.astype(np.float32) * 0.85, 0, 255).astype(np.uint8),
        cv2.flip(rgb, 1),
        cv2.GaussianBlur(rgb, (5, 5), 0),
        cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w//2, h//2),  5, 1.0), (w, h)),
        cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0), (w, h)),
    ]

    per_veg = cfg["per_veg_bounds"]
    globl   = cfg["global_bounds"]
    bounds  = per_veg.get(veg_name, globl)

    scores    = []
    aug_raws  = []   # raw decision values per augmentation — used for sign-flip check
    for aug in augmented:
        batch  = preprocess_input(np.expand_dims(aug.astype(np.float32), 0))
        deep_f = deep_model.predict(batch, verbose=0)[0].astype(np.float32)
        hand_f = extract_handcrafted(aug)
        feats  = np.concatenate([deep_f, hand_f])
        Xf     = preprocess_features(feats, vt, scaler, selected)
        aug_raw = float(fresh_svm.decision_function(Xf)[0])
        aug_raws.append(aug_raw)
        scores.append(normalize_score(aug_raw, bounds))

    score_range = float(max(scores) - min(scores))
    score_std   = float(np.std(scores))
    return score_range, score_std, scores, aug_raws


# ─────────────────────────────────────────────────────────────
# Core predict
# ─────────────────────────────────────────────────────────────

def predict(image_path: str, compute_uncertainty: bool = True):

    vt, scaler, selected, veg_svm, fresh_svm, le, \
        train_mean, train_precision, class_centroids, cfg = load_pipeline_artifacts()

    # ── Pre-flight ──────────────────────────────────────────
    status, reason = preflight_checks(image_path, cfg)
    if status == "UNRELIABLE":
        result = {
            "state"      : "UNRELIABLE",
            "reason"     : reason,
            "score"      : None,
            "raw"        : None,
            "fresh_label": None,
            "grade"      : None,
        }
        print(f"[UNRELIABLE] Pre-flight failed: {reason}")
        return result

    preflight_warnings = []
    if status == "OK_LOW_COVERAGE":
        preflight_warnings.append(f"[!] {reason}")

    # ── Feature extraction ──────────────────────────────────
    feats  = extract_features(image_path)
    Xfinal = preprocess_features(feats, vt, scaler, selected)

    # ── Vegetable prediction ────────────────────────────────
    veg_probs     = veg_svm.predict_proba(Xfinal)[0]
    sorted_probs  = np.sort(veg_probs)[::-1]
    veg_idx       = int(np.argmax(veg_probs))
    veg_name      = le.inverse_transform([veg_idx])[0]
    veg_conf      = float(sorted_probs[0]) * 100.0
    conf_gap      = float(sorted_probs[0] - sorted_probs[1]) * 100.0

    veg_conf_thresh = cfg.get("veg_confidence_threshold", 0.70) * 100.0
    veg_gap_thresh  = cfg.get("veg_gap_threshold",        0.15) * 100.0
    veg_confident   = (veg_conf >= veg_conf_thresh) and (conf_gap >= veg_gap_thresh)

    per_veg = cfg["per_veg_bounds"]
    globl   = cfg["global_bounds"]
    bounds  = per_veg.get(veg_name, globl) if veg_confident else globl
    norm_source = "per-veg" if (veg_confident and veg_name in per_veg) else "global"

    # ── Class-consistency centroid check ────────────────────
    # Detects confident but semantically wrong classifications.
    # Computes ratio: dist_to_predicted_centroid / dist_to_second_best_centroid.
    # If ratio > threshold, the sample is not clearly in the predicted class
    # cluster — flag as class-inconsistent even if confidence is high.
    # This is a sanity check only — never used for scoring.
    x_flat = Xfinal.flatten()
    dists_to_centroids  = np.linalg.norm(class_centroids - x_flat, axis=1)
    sorted_centroid_idx = np.argsort(dists_to_centroids)
    d_pred   = dists_to_centroids[veg_idx]
    d_second = next(dists_to_centroids[j]
                    for j in sorted_centroid_idx if j != veg_idx)
    centroid_ratio = float(d_pred / (d_second + 1e-9))

    # Per-class threshold — each vegetable has its own calibrated bound
    # reflecting its natural cluster spread in feature space.
    per_class_thresholds  = cfg.get("centroid_ratio_thresholds", {})
    centroid_ratio_thresh = float(per_class_thresholds.get(veg_name, 1.0))
    class_inconsistent    = centroid_ratio > centroid_ratio_thresh

    # ── Freshness signal ────────────────────────────────────
    raw         = float(fresh_svm.decision_function(Xfinal)[0])
    score       = normalize_score(raw, bounds)
    fresh_class = int(fresh_svm.predict(Xfinal)[0])
    fresh_label = "Fresh" if fresh_class == 1 else "Rotten"

    # ── Mahalanobis (OOD flag only) ─────────────────────────
    dist  = mahalanobis_dist(Xfinal, train_mean, train_precision)
    zone  = mahal_zone(dist,
                       cfg["mahal_thresh_caution"],
                       cfg["mahal_thresh_ood"])
    is_ood = (zone == "ood")

    # ── Augmentation instability ────────────────────────────
    use_aug_gate = cfg.get("use_augmentation_gate", False)
    score_range, score_std, aug_scores, aug_raws = 0.0, 0.0, [], []
    if use_aug_gate and compute_uncertainty:
        score_range, score_std, aug_scores, aug_raws = augment_and_score(
            image_path, vt, scaler, selected,
            fresh_svm, veg_name if veg_confident else "__global__", cfg
        )

    unstable_range_thresh = cfg.get("unstable_range_thresh", 13.0)
    if use_aug_gate:
        high_range = score_range >= unstable_range_thresh
        crosses_boundary = (
            len(aug_raws) > 0
            and min(aug_raws) < 0
            and max(aug_raws) > 0
        )
        unstable = high_range and crosses_boundary
        sensitive_only = (
            high_range
            and not crosses_boundary
            and score_range > unstable_range_thresh * 1.5
        )
    else:
        high_range       = False
        crosses_boundary = False
        unstable         = False
        sensitive_only   = False

    # ── Boundary proximity ──────────────────────────────────
    boundary_thresh = cfg["boundary_threshold"]
    near_boundary   = abs(raw) < boundary_thresh

    # ── Fix 4 — High-confidence override ───────────────────
    # When veg confidence > 95%, raw is far from boundary, and
    # no actual class flip occurs under augmentation, the system
    # has strong multi-signal evidence that the prediction is
    # correct. A moderate score range alone should not suppress
    # the result. Force RELIABLE in this case.
    high_conf_override = (
        veg_conf > 95.0
        and not near_boundary
        and not crosses_boundary
        and not is_ood
        and not class_inconsistent   # do not override when class cluster is wrong
    )

    # ── Two-level uncertainty gate ──────────────────────────
    #
    # Level 1 — score validity
    #   unstable (high range + raw crosses 0) OR is_ood
    #   → score itself unreliable, invalidate entirely
    #   Exception: high_conf_override skips this
    #
    # Level 2 — decision validity
    #   near_boundary OR severe sensitive_only OR low veg confidence
    #   OR class_inconsistent (centroid sanity check)
    #   → score valid, suppress class and grade only
    #   Exception: high_conf_override skips this too
    #
    if high_conf_override:
        score_unreliable    = False
        decision_unreliable = False
    else:
        score_unreliable    = unstable or is_ood
        decision_unreliable = (near_boundary or sensitive_only
                               or (not veg_confident) or class_inconsistent)

    warnings = []

    if high_conf_override:
        warnings.append(
            f"HIGH-CONFIDENCE OVERRIDE — veg_conf={veg_conf:.1f}%, "
            f"raw far from boundary, no class flip. "
            f"Score range={score_range:.2f} ignored. Forced RELIABLE."
        )
    if class_inconsistent:
        warnings.append(
            f"CLASS INCONSISTENCY — centroid ratio={centroid_ratio:.3f} "
            f"(threshold={centroid_ratio_thresh:.3f}). "
            f"Sample is not clearly in the {veg_name} cluster. "
            f"Vegetable prediction may be wrong despite high confidence."
        )
    if not veg_confident:
        warnings.append(
            f"Low veg confidence ({veg_conf:.1f}%, gap={conf_gap:.1f}%) "
            f"— using global normalization."
        )
    if near_boundary:
        warnings.append(
            f"MODEL UNCERTAINTY — near decision boundary "
            f"(|raw|={abs(raw):.4f} < {boundary_thresh:.4f}). "
            f"Classifier is unsure."
        )
    if unstable:
        warnings.append(
            f"TRUE INSTABILITY — score range={score_range:.2f} pts AND "
            f"raw margin crosses zero (min={min(aug_raws):.3f}, max={max(aug_raws):.3f}). "
            f"Prediction flips under augmentation."
        )
    if sensitive_only:
        warnings.append(
            f"INPUT SENSITIVITY — score range={score_range:.2f} pts "
            f"(threshold={unstable_range_thresh:.2f}, severe cutoff={unstable_range_thresh*1.5:.2f}). "
            f"All augmentations stay {'fresh' if raw > 0 else 'rotten'} side. "
            f"Prediction consistent but score sensitive to imaging conditions."
        )
    if is_ood:
        warnings.append(
            f"OOD — Mahalanobis dist={dist:.3f} > threshold={cfg['mahal_thresh_ood']:.3f}. "
            f"Outside training distribution."
        )
    if zone == "caution":
        warnings.append(
            f"CAUTION — Mahalanobis dist={dist:.3f} in caution zone "
            f"[{cfg['mahal_thresh_caution']:.3f}, {cfg['mahal_thresh_ood']:.3f}]."
        )

    # ── Build result under strict contract ──────────────────
    if score_unreliable:
        # Invalidate score entirely — no hidden values
        result = {
            "state"      : "UNRELIABLE",
            "veg"        : veg_name,
            "veg_conf"   : veg_conf,
            "score"      : None,
            "raw"        : None,
            "fresh_label": None,
            "grade"      : None,
            "mahal_dist" : dist,
            "mahal_zone" : zone,
            "warnings"   : warnings,
        }

    elif decision_unreliable:
        # Score is valid; decision (class + grade) is not
        result = {
            "state"      : "TENTATIVE",
            "veg"        : veg_name,
            "veg_conf"   : veg_conf,
            "score"      : score,
            "score_range": score_range,
            "raw"        : raw,
            "fresh_label": None,   # tentative — never shown as hard label
            "grade"      : None,
            "norm_source": norm_source,
            "mahal_dist" : dist,
            "mahal_zone" : zone,
            "warnings"   : warnings,
        }

    else:
        # Fully reliable
        grade = grade_from_score(score, cfg)   # thresholds sourced from scoring_config
        result = {
            "state"      : "RELIABLE",
            "veg"        : veg_name,
            "veg_conf"   : veg_conf,
            "score"      : score,
            "score_range": score_range,
            "raw"        : raw,
            "fresh_label": fresh_label,
            "grade"      : grade,
            "norm_source": norm_source,
            "mahal_dist" : dist,
            "mahal_zone" : zone,
            "warnings"   : warnings,
        }

    # ── Print ───────────────────────────────────────────────
    print(f"Vegetable : {veg_name} ({veg_conf:.2f}%,  gap={conf_gap:.2f}%)")
    print(f"State     : {result['state']}")

    if result["score"] is not None:
        if compute_uncertainty:
            print(f"Score     : {result['score']:.2f}  range=±{score_range:.2f} / 100")
        else:
            print(f"Score     : {result['score']:.2f} / 100")
        print(f"Norm      : {norm_source}")

    if result["fresh_label"] is not None:
        print(f"Freshness : {result['fresh_label']}")

    if result["grade"] is not None:
        print(f"Grade     : {result['grade']}")

    print(f"Mahal     : {dist:.3f}  [{zone}]")

    for w in preflight_warnings:
        print(w)
    for w in warnings:
        print(f"[!] {w}")

    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--no-uncertainty", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    predict(args.image, compute_uncertainty=not args.no_uncertainty)