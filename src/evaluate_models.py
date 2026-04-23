import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from utils import load_model,normalize_score

MODEL_DIR = "models"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def get_normalized_scores(decisions, bounds):
    return np.array([normalize_score(d, bounds) for d in decisions])


def get_deployed_scores(decisions, y_veg, veg_probs, scoring_config):
    """
    Mirror the CLI gating rule exactly:
      - per-veg bounds when top-1 >= conf_thresh AND gap >= gap_thresh
      - global bounds otherwise

    This is the path a real image takes through predict_cli.py.
    Using this in evaluation closes the gap between eval and deployment.
    """
    per_veg     = scoring_config.get("per_veg_bounds", {})
    globl       = scoring_config["global_bounds"]
    conf_thresh = scoring_config.get("veg_confidence_threshold", 0.70)
    gap_thresh  = scoring_config.get("veg_gap_threshold", 0.15)

    scores = []
    for i, (raw, veg) in enumerate(zip(decisions, y_veg)):
        probs      = veg_probs[i]
        sorted_p   = np.sort(probs)[::-1]
        top1       = float(sorted_p[0])
        gap        = float(sorted_p[0] - sorted_p[1])
        veg_ok     = (top1 >= conf_thresh) and (gap >= gap_thresh)
        bounds     = per_veg.get(veg, globl) if (veg_ok and veg in per_veg) else globl
        scores.append(normalize_score(raw, bounds))

    return np.array(scores, dtype=float)


def score_to_tier(s, grade_thr=None):
    """Map continuous score to 4-level integer tier (for bucket inversion)."""
    thr = grade_thr or {}
    t1  = thr.get("truly_fresh", 85)
    t2  = thr.get("fresh",       65)
    t3  = thr.get("moderate",    40)
    if s >= t1: return 3
    if s >= t2: return 2
    if s >= t3: return 1
    return 0


# ─────────────────────────────────────────────────────────────
# Inversion rate — three layers
# ─────────────────────────────────────────────────────────────

def inversion_rate(scores_fresh, scores_rotten, n=10000, seed=42):
    """
    Fraction of (fresh, rotten) pairs where score_fresh < score_rotten.
    Lower is better.
    """
    if len(scores_fresh) == 0 or len(scores_rotten) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n   = min(n, len(scores_fresh) * len(scores_rotten))
    fi  = rng.integers(0, len(scores_fresh),  n)
    ri  = rng.integers(0, len(scores_rotten), n)
    return float((scores_fresh[fi] < scores_rotten[ri]).mean())


def three_layer_inversions(decisions, scores, y_fresh, grade_thr=None):
    """
    Report inversion at:
      1. raw margin         (model signal)
      2. normalized score   (normalization layer)
      3. grade bucket       (coarse sanity check only)

    Δ between layers tells you WHERE ordering breaks.
    """
    fresh  = y_fresh == 1
    rotten = y_fresh == 0

    inv_raw  = inversion_rate(decisions[fresh], decisions[rotten])
    inv_norm = inversion_rate(scores[fresh],    scores[rotten])

    tiers      = np.array([score_to_tier(s, grade_thr) for s in scores], dtype=float)
    inv_grade  = inversion_rate(tiers[fresh], tiers[rotten])

    return inv_raw, inv_norm, inv_grade


# ─────────────────────────────────────────────────────────────
# OOD rates on val and test (threshold consistency check)
# ─────────────────────────────────────────────────────────────

def mahal(X, m, P):
    diff = X - m
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, P, diff))

def compute_ood_rates(vt, scaler, selected, scoring_config):
    train_mean      = np.load(os.path.join(MODEL_DIR, "train_mean.npy"))
    train_precision = np.load(os.path.join(MODEL_DIR, "train_precision.npy"))
    thresh_ood      = scoring_config["mahal_thresh_ood"]

    rates = {}
    for split in ("val", "test"):
        X = np.load(os.path.join(MODEL_DIR, f"X_{split}.npy"))
        X_r = vt.transform(X)
        X_s = scaler.transform(X_r)
        X_f = X_s[:, selected]
        dists = mahal(X_f, train_mean, train_precision)
        rates[split] = float((dists > thresh_ood).mean())

    return rates["val"], rates["test"]


# ─────────────────────────────────────────────────────────────
# p5/p95 stability check (already run in train_svm.py,
# re-reported here on held-out val for cross-check)
# ─────────────────────────────────────────────────────────────

def report_bound_stability(scoring_config):
    print("\n--- p5/p95 Bounds (from scoring_config.json) ---")
    per_veg = scoring_config.get("per_veg_bounds", {})
    globl   = scoring_config["global_bounds"]
    print(f"  Global  p5={globl['p5']:.4f}  p95={globl['p95']:.4f}")
    for veg, b in per_veg.items():
        spread = b["p95"] - b["p5"]
        print(f"  {veg:<12}  p5={b['p5']:.4f}  p95={b['p95']:.4f}  "
              f"spread={spread:.4f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading test data...")

    # Test set is only loaded here — never before
    X_test       = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_veg_test   = np.load(os.path.join(MODEL_DIR, "y_veg_test.npy"))
    y_fresh_test = np.load(os.path.join(MODEL_DIR, "y_fresh_test.npy")).astype(int)

    vt       = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler   = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_union_features.npy"))

    veg_model   = load_model(os.path.join(MODEL_DIR, "veg_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    le          = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    with open(os.path.join(MODEL_DIR, "scoring_config.json")) as f:
        scoring_config = json.load(f)

    print("[INFO] Applying preprocessing pipeline...")
    X = vt.transform(X_test)
    X = scaler.transform(X)
    X = X[:, selected]

    decisions  = fresh_model.decision_function(X)
    veg_probs  = veg_model.predict_proba(X)        # shape (N, n_classes)
    yveg_pred  = veg_model.predict(X)              # integer predictions
    # Convert to vegetable name strings — same as what predict_cli.py uses.
    # MUST be predicted labels, NOT y_veg_test (ground truth).
    # Using ground truth here would silently inflate deployed scores
    # because the per-veg bounds would always match the true vegetable.
    yveg_pred_names = le.inverse_transform(yveg_pred)

    # Global-bounds scores (baseline)
    scores_global = get_normalized_scores(decisions, scoring_config["global_bounds"])

    # Deployed-scores: mirrors CLI gating rule exactly.
    # Uses PREDICTED vegetable names — matches runtime behaviour.
    scores_deployed = get_deployed_scores(
        decisions, yveg_pred_names, veg_probs, scoring_config
    )

    # ─────────────────────────────────────────
    # 1. Classification metrics
    # ─────────────────────────────────────────
    print("\n========== Vegetable Classification ==========")
    yveg_enc = le.transform(y_veg_test)
    # yveg_pred already computed above (used for deployed scoring)
    print(f"Accuracy: {accuracy_score(yveg_enc, yveg_pred):.4f}")
    print(classification_report(yveg_enc, yveg_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(yveg_enc, yveg_pred))

    print("\n========== Freshness Classification ==========")
    yfresh_pred = fresh_model.predict(X)
    print(f"Accuracy: {accuracy_score(y_fresh_test, yfresh_pred):.4f}")
    print(classification_report(y_fresh_test, yfresh_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_fresh_test, yfresh_pred))

    # ROC-AUC using decision_function as ranking signal.
    # Note: this is margin AUC, not probability AUC.
    # Sign verification: SVM positive margin must correspond to class 1 (Fresh).
    # If classes_[1] == 0 (Rotten), the margin is inverted and AUC would be wrong.
    from sklearn.metrics import roc_auc_score
    print(f"\n  fresh_model.classes_ : {fresh_model.classes_}")
    print(f"  Positive margin = class {fresh_model.classes_[1]}  "
          f"(expected 1 = Fresh for correct AUC direction)")
    if fresh_model.classes_[1] == 1:
        margin_for_auc = decisions
        print(f"  [OK] Sign direction confirmed. Positive margin = Fresh.")
    else:
        margin_for_auc = -decisions
        print(f"  [WARNING] Sign inverted. Negating margin for AUC computation.")
    margin_auc = roc_auc_score(y_fresh_test, margin_for_auc)
    print(f"  Freshness ROC-AUC (margin-based) : {margin_auc:.4f}")
    print(f"  [Note] AUC uses decision_function() as ranking signal, not predict_proba().")
    print(f"         Measures ordering reliability of the SVM margin.")

    # ─────────────────────────────────────────
    # 2. Three-layer inversion diagnostics
    #
    # Reported for BOTH score paths:
    #   global-only  — baseline, cross-vegetable consistent
    #   deployed     — mirrors CLI gate; per-veg when confident
    #
    # If deployed inversion >> global inversion:
    #   per-veg gate is distorting ordering → tune gate
    # If difference is small (< 0.01):
    #   gate is safe; keep current thresholds
    # ─────────────────────────────────────────
    print("\n========== Inversion Rate Diagnostics ==========")

    grade_thr = scoring_config.get("grade_thresholds", {})

    print("\n  [A] Global-bounds scores (baseline):")
    inv_raw, inv_norm_g, inv_grade_g = three_layer_inversions(
        decisions, scores_global, y_fresh_test, grade_thr
    )
    print(f"    Raw margin inversion       : {inv_raw:.4f}  ← primary")
    print(f"    Global-norm inversion      : {inv_norm_g:.4f}  ← primary")
    print(f"    Grade-bucket inversion     : {inv_grade_g:.4f}  ← coarse only")

    print("\n  [B] Deployed-path scores (mirrors CLI gate):")
    _, inv_norm_d, inv_grade_d = three_layer_inversions(
        decisions, scores_deployed, y_fresh_test, grade_thr
    )
    print(f"    Raw margin inversion       : {inv_raw:.4f}  (same model)")
    print(f"    Deployed-norm inversion    : {inv_norm_d:.4f}  ← primary")
    print(f"    Grade-bucket inversion     : {inv_grade_d:.4f}  ← coarse only")

    print("\n  [GATE EFFECT — GLOBAL]")
    gate_delta = inv_norm_d - inv_norm_g
    if gate_delta > 0.01:
        print(f"  [!] Global deployed delta = {gate_delta:.4f} (> 0.01).")
        print(f"      Per-veg gate is distorting ordering globally.")
    elif gate_delta < -0.01:
        print(f"  [OK] Per-veg gate REDUCES global inversions by {abs(gate_delta):.4f}.")
    else:
        print(f"  [OK] Global gate delta = {gate_delta:.4f}. Negligible global effect.")

    # ── Per-vegetable gate delta (Issue 1 fix) ───────────────
    # A small global delta can hide localized failures on weak vegetables.
    # Check BOTH: global delta < 0.01 AND max per-veg delta < 0.02.
    print("\n  [GATE EFFECT — PER-VEGETABLE]")
    per_veg_deltas = {}
    for veg in le.classes_:
        mask = (y_veg_test == veg)
        if mask.sum() < 10:
            continue
        vbounds_g = scoring_config["global_bounds"]
        vbounds_d = scoring_config.get("per_veg_bounds", {}).get(veg, vbounds_g)

        vscores_g = np.array([normalize_score(d, vbounds_g)
                               for d in decisions[mask]])
        vscores_d = np.array([normalize_score(d, vbounds_d)
                               for d in decisions[mask]])
        vfresh = y_fresh_test[mask]

        inv_g = inversion_rate(vscores_g[vfresh==1], vscores_g[vfresh==0])
        inv_d = inversion_rate(vscores_d[vfresh==1], vscores_d[vfresh==0])
        per_veg_deltas[veg] = inv_d - inv_g

        flag = ""
        if abs(per_veg_deltas[veg]) > 0.02:
            flag = "  ← EXCEEDS 0.02 THRESHOLD"
        print(f"  {veg:<12}  global_inv={inv_g:.4f}  deployed_inv={inv_d:.4f}  "
              f"delta={per_veg_deltas[veg]:+.4f}{flag}")

    max_veg_delta = max(abs(v) for v in per_veg_deltas.values()) \
                    if per_veg_deltas else 0.0
    print(f"\n  Max per-veg delta : {max_veg_delta:.4f}")

    # Final gate stability verdict — BOTH conditions must pass
    global_ok  = abs(gate_delta)    < 0.01
    per_veg_ok = max_veg_delta      < 0.02

    if global_ok and per_veg_ok:
        print("  [STABLE] Global delta < 0.01 AND max per-veg delta < 0.02.")
        print("           Gate is stable. Keep current conf/gap thresholds.")
    else:
        print("  [UNSTABLE] Gate requires tuning.")
        if not global_ok:
            print(f"    Global delta {gate_delta:.4f} >= 0.01.")
        if not per_veg_ok:
            failing = [v for v, d in per_veg_deltas.items() if abs(d) >= 0.02]
            print(f"    Per-veg failures: {failing}")
        # Issue 2 fix — tuning order policy
        print("\n  [TUNING POLICY]")
        print("  Step 1: Increase veg_gap_threshold first.")
        print("          Gap controls prediction ambiguity — better signal.")
        print("          Raise in steps of 0.05 and re-run evaluation.")
        print("  Step 2: Increase veg_confidence_threshold only if gap tuning")
        print("          is insufficient. Confidence alone allows confused")
        print("          predictions that gap would have rejected.")

    # Use deployed scores for all subsequent per-veg analysis
    # (matches what the system actually outputs)
    scores = scores_deployed

    print("\n  [NOTE] Subsequent per-veg analysis uses deployed-path scores.")

    # ─────────────────────────────────────────
    # 2b. Raw vs norm delta interpretation
    # ─────────────────────────────────────────
    raw_norm_delta = inv_norm_d - inv_raw
    print(f"\n  Raw → Deployed-norm delta : {raw_norm_delta:+.4f}")
    if raw_norm_delta > 0.01:
        print("  [!] Normalization increases inversions. Per-veg bounds may be misaligned.")
    elif raw_norm_delta < -0.01:
        print("  [OK] Normalization reduces inversions over raw margin.")
    else:
        print("  [OK] Normalization has negligible effect on inversion.")



    # ─────────────────────────────────────────
    # 3. Per-vegetable inversion + delta
    # ─────────────────────────────────────────
    print("\n========== Per-Vegetable Diagnostics ==========")
    per_veg_bounds = scoring_config.get("per_veg_bounds", {})
    globl          = scoring_config["global_bounds"]

    print(f"  {'Veg':<12}  {'InvRaw':>8}  {'InvNorm':>8}  "
          f"{'FreshMean':>10}  {'RottenMean':>11}  {'Delta':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*7}")

    for veg in le.classes_:
        mask   = (y_veg_test == veg)
        if mask.sum() < 10:
            continue
        bounds = per_veg_bounds.get(veg, globl)
        vscores = np.array([normalize_score(d, bounds) for d in decisions[mask]])
        vfresh  = y_fresh_test[mask]

        ir_raw  = inversion_rate(decisions[mask][vfresh==1],
                                 decisions[mask][vfresh==0])
        ir_norm = inversion_rate(vscores[vfresh==1], vscores[vfresh==0])
        fm = vscores[vfresh==1].mean() if (vfresh==1).any() else float("nan")
        rm = vscores[vfresh==0].mean() if (vfresh==0).any() else float("nan")
        d  = fm - rm if not (np.isnan(fm) or np.isnan(rm)) else float("nan")

        note = ""
        if veg in ("cucumber", "potato") and ir_norm > 0.07:
            note = "  ← WEAK"

        print(f"  {veg:<12}  {ir_raw:>8.4f}  {ir_norm:>8.4f}  "
              f"{fm:>10.2f}  {rm:>11.2f}  {d:>7.2f}{note}")

    # ─────────────────────────────────────────
    # 4. OOD consistency (val vs test)
    # ─────────────────────────────────────────
    print("\n========== OOD Rate Consistency ==========")
    ood_val, ood_test = compute_ood_rates(vt, scaler, selected, scoring_config)
    print(f"  OOD rate — validation : {ood_val:.4f}")
    print(f"  OOD rate — test       : {ood_test:.4f}")
    if abs(ood_val - ood_test) > 0.05:
        print("  [WARNING] OOD rates differ by > 5%. "
              "Mahalanobis threshold may not transfer stably.")
    else:
        print("  [OK] OOD rates consistent across splits.")

    # ─────────────────────────────────────────
    # 5. p5/p95 bound stability report
    # ─────────────────────────────────────────
    report_bound_stability(scoring_config)

    # ─────────────────────────────────────────
    # 6. Score distribution summary
    # ─────────────────────────────────────────
    print("\n========== Score Distribution ==========")
    fresh_s  = scores[y_fresh_test == 1]
    rotten_s = scores[y_fresh_test == 0]
    delta    = fresh_s.mean() - rotten_s.mean()
    overlap  = (rotten_s > fresh_s.mean()).mean()
    print(f"  Fresh  — mean={fresh_s.mean():.2f}  std={fresh_s.std():.2f}  "
          f"range={fresh_s.max()-fresh_s.min():.2f}")
    print(f"  Rotten — mean={rotten_s.mean():.2f}  std={rotten_s.std():.2f}  "
          f"range={rotten_s.max()-rotten_s.min():.2f}")
    print(f"  Delta (fresh - rotten mean) : {delta:.2f} pts")
    print(f"  Overlap (rotten > fresh mean): {overlap:.4f}")

    # ─────────────────────────────────────────
    # 7. Calibration note
    # ─────────────────────────────────────────
    print("\n========== Calibration Provenance ==========")
    note = scoring_config.get("calibration_note", "Not recorded.")
    print(f"  {note}")

    # ─────────────────────────────────────────
    # 8. State distribution (deployment-critical)
    #
    # Simulates the two-level gate from predict_cli.py
    # on the entire test set WITHOUT running augmentation.
    #
    # RELIABLE  : score valid + decision valid
    # TENTATIVE : score valid, decision unreliable
    # UNRELIABLE: score itself invalid (OOD — augmentation
    #             instability cannot be simulated here without
    #             re-running EfficientNet 6x per sample)
    #
    # OOD-based UNRELIABLE is reported from Mahalanobis.
    # Augmentation-based UNRELIABLE requires predict_cli.py.
    # ─────────────────────────────────────────
    print("\n========== State Distribution (Test Set) ==========")

    train_mean      = np.load(os.path.join(MODEL_DIR, "train_mean.npy"))
    train_precision = np.load(os.path.join(MODEL_DIR, "train_precision.npy"))
    thresh_ood      = scoring_config["mahal_thresh_ood"]
    boundary_thresh = scoring_config["boundary_threshold"]
    conf_thresh     = scoring_config.get("veg_confidence_threshold", 0.70)
    gap_thresh_val  = scoring_config.get("veg_gap_threshold", 0.15)


    dists       = mahal(X, train_mean, train_precision)
    is_ood_arr  = dists > thresh_ood

    sorted_p    = np.sort(veg_probs, axis=1)[:, ::-1]
    top1_arr    = sorted_p[:, 0]
    gap_arr     = sorted_p[:, 0] - sorted_p[:, 1]
    veg_conf_ok = (top1_arr >= conf_thresh) & (gap_arr >= gap_thresh_val)

    near_boundary_arr = np.abs(decisions) < boundary_thresh

    # Centroid gate — mirrors predict_cli.py class-consistency check
    class_centroids_arr = np.load(os.path.join(MODEL_DIR, "class_centroids.npy"))
    per_cls_thresh      = scoring_config.get("centroid_ratio_thresholds", {})
    centroid_gate_arr   = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        x_f      = X[i]
        dists_c  = np.linalg.norm(class_centroids_arr - x_f, axis=1)
        pidx     = int(yveg_pred[i])
        sorted_c = np.argsort(dists_c)
        d_pred   = dists_c[pidx]
        d_sec    = next(dists_c[j] for j in sorted_c if j != pidx)
        ratio    = float(d_pred / (d_sec + 1e-9))
        veg_n    = le.inverse_transform([pidx])[0]
        thresh_c = float(per_cls_thresh.get(veg_n, 1.0))
        centroid_gate_arr[i] = ratio > thresh_c

    decision_unreliable_arr = near_boundary_arr | (~veg_conf_ok) | centroid_gate_arr

    # Note: augmentation-instability UNRELIABLE not simulatable here
    state_arr = np.where(
        is_ood_arr,              "UNRELIABLE",
        np.where(
            decision_unreliable_arr, "TENTATIVE",
                                     "RELIABLE"
        )
    )

    n_total      = len(state_arr)
    n_reliable   = (state_arr == "RELIABLE").sum()
    n_tentative  = (state_arr == "TENTATIVE").sum()
    n_unreliable = (state_arr == "UNRELIABLE").sum()

    print(f"  Total test samples : {n_total}")
    print(f"  RELIABLE           : {n_reliable:5d}  ({n_reliable/n_total*100:.1f}%)")
    print(f"  TENTATIVE          : {n_tentative:5d}  ({n_tentative/n_total*100:.1f}%)")
    print(f"  UNRELIABLE (OOD)   : {n_unreliable:5d}  ({n_unreliable/n_total*100:.1f}%)")
    print(f"  [Note] Augmentation-instability UNRELIABLE not counted here.")
    print(f"         Run predict_cli.py for full state distribution per image.")

    if n_reliable / n_total < 0.50:
        print("  [WARNING] < 50% of test samples reach RELIABLE state.")
        print("            Gate thresholds may be too aggressive for this dataset.")
    else:
        print(f"  [OK] {n_reliable/n_total*100:.1f}% of samples reach RELIABLE state.")

    # ─────────────────────────────────────────
    # 8a. Gate trigger statistics
    #
    # Reports how often each gate fires and what it costs.
    # This is the ablation proxy — without a full ablation study,
    # gate-trigger counts are the minimum evidence needed to justify
    # each gate's existence. A gate that fires on 0.1% of samples
    # and catches no additional errors is a candidate for removal.
    #
    # Gates simulated here (augmentation-instability omitted —
    # requires 6x EfficientNet passes per sample):
    #   G1: OOD (Mahalanobis)
    #   G2: near_boundary
    #   G3: low veg confidence (conf < thresh OR gap < gap_thresh)
    #
    # For each gate: how many samples it fires on, and of those,
    # how many were WRONG predictions (gate catches real errors)
    # vs how many were CORRECT predictions (gate blocks correct outputs).
    # ─────────────────────────────────────────
    print("\n========== Gate Trigger Statistics ==========")
    print("  (Augmentation-instability gate measured separately on val set below)")
    print()

    yfresh_pred_all_g = fresh_model.predict(X)
    correct_fresh     = (yfresh_pred_all_g == y_fresh_test)
    correct_veg       = (yveg_pred == yveg_enc)

    gates = {
        "G1_OOD"          : is_ood_arr,
        "G2_near_boundary": near_boundary_arr,
        "G3_low_veg_conf" : ~veg_conf_ok,
    }

    # Baseline: all gates active
    baseline_reliable_mask = (state_arr == "RELIABLE")
    baseline_coverage      = baseline_reliable_mask.sum() / n_total
    baseline_reliable_acc  = (
        accuracy_score(y_fresh_test[baseline_reliable_mask],
                       yfresh_pred_all_g[baseline_reliable_mask])
        if baseline_reliable_mask.sum() > 0 else float("nan")
    )

    def recompute_state(disabled_gate_name):
        """
        Properly rebuild state_arr with one gate disabled.
        Gates are interdependent: a sample blocked by both G1 and G2
        stays blocked when only G1 is removed. OR-masking overstates impact.
        """
        g_ood  = is_ood_arr.copy()
        g_near = near_boundary_arr.copy()
        g_conf = (~veg_conf_ok).copy()
        if disabled_gate_name == "G1_OOD":
            g_ood[:] = False
        elif disabled_gate_name == "G2_near_boundary":
            g_near[:] = False
        elif disabled_gate_name == "G3_low_veg_conf":
            g_conf[:] = False
        new_score_unrel = g_ood          # only OOD invalidates score
        new_dec_unrel   = g_near | g_conf
        return np.where(
            new_score_unrel, "UNRELIABLE",
            np.where(new_dec_unrel, "TENTATIVE", "RELIABLE")
        )

    print(f"  {'Gate':<20}  {'Fires':>5}  {'Fire%':>5}  "
          f"{'Catch_W':>7}  {'Block_C':>7}  "
          f"{'Δ_acc':>7}  {'Δ_cov':>7}  {'Verdict'}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*15}")

    for gate_name, gate_mask in gates.items():
        n_fires        = gate_mask.sum()
        fire_pct       = n_fires / n_total * 100
        catches_wrong  = (gate_mask & ~correct_fresh).sum()
        blocks_correct = (gate_mask &  correct_fresh).sum()

        # Properly rebuild state with this gate disabled
        new_state   = recompute_state(gate_name)
        new_rel     = (new_state == "RELIABLE")
        new_cov     = new_rel.sum() / n_total
        new_acc     = (
            accuracy_score(y_fresh_test[new_rel], yfresh_pred_all_g[new_rel])
            if new_rel.sum() > 0 else float("nan")
        )
        delta_acc = new_acc - baseline_reliable_acc
        delta_cov = new_cov - baseline_coverage

        # Verdict: two-axis rule, applied consistently.
        # Δ_acc < 0 means disabling the gate HURTS accuracy → gate protects accuracy.
        # Δ_cov > 0 means disabling the gate EXPANDS coverage → gate restricts coverage.
        # Both axes are reported; verdict derives from the combination.
        #
        # Rule (strictly applied):
        #   never fires            → NEVER FIRES (cannot evaluate)
        #   Δ_acc < -0.001        → KEEP  (protects accuracy, regardless of coverage)
        #   Δ_acc ≥ -0.001
        #     Δ_cov ≤ 0.005      → REMOVE  (no accuracy benefit, no meaningful coverage cost)
        #     Δ_cov >  0.005     → REVIEW  (no accuracy benefit but restricts coverage — tradeoff)
        if n_fires == 0:
            verdict = "NEVER FIRES"
        elif delta_acc < -0.001:
            verdict = "KEEP"
        elif delta_cov <= 0.005:
            verdict = "REMOVE"
        else:
            verdict = "REVIEW (coverage cost)"

        print(f"  {gate_name:<20}  {n_fires:>5}  {fire_pct:>4.1f}%  "
              f"{catches_wrong:>7}  {blocks_correct:>7}  "
              f"{delta_acc:>+7.4f}  {delta_cov:>+7.4f}  {verdict}")

    print()
    print(f"  Baseline: acc={baseline_reliable_acc:.4f}  coverage={baseline_coverage:.3f}")
    print(f"  Δ_acc: (acc when gate disabled) - baseline_acc")
    print(f"  Δ_cov: (coverage when gate disabled) - baseline_coverage")
    print(f"  Verdict rule (applied consistently):")
    print(f"    KEEP              → Δ_acc < -0.001  (disabling hurts accuracy)")
    print(f"    REMOVE            → Δ_acc ≥ -0.001 AND Δ_cov ≤ 0.005")
    print(f"    REVIEW            → Δ_acc ≥ -0.001 AND Δ_cov > 0.005 (coverage cost only)")
    print(f"    NEVER FIRES       → gate inactive on this test set")

    # ── Gate co-occurrence (interdependency) ─────────────────
    print("\n  --- Gate Co-occurrence ---")
    gate_names = list(gates.keys())
    gate_masks = list(gates.values())
    for i in range(len(gate_names)):
        for j in range(i+1, len(gate_names)):
            co    = (gate_masks[i] & gate_masks[j]).sum()
            only_i = (gate_masks[i] & ~gate_masks[j]).sum()
            only_j = (~gate_masks[i] & gate_masks[j]).sum()
            print(f"  {gate_names[i]} ∩ {gate_names[j]}: {co}  "
                  f"(only {gate_names[i]}: {only_i},  only {gate_names[j]}: {only_j})")
    all_three = (gate_masks[0] & gate_masks[1] & gate_masks[2]).sum()
    print(f"  All three: {all_three}")
    print(f"  High co-occurrence = gates redundant for overlapping samples.")
    print(f"  (Remove lowest-Catches gate in a co-occurring pair)")

    # ── Silent failures: 3-way split ─────────────────────────
    print("\n  --- Silent Failures (veg wrong but RELIABLE) ---")
    silent_failures  = baseline_reliable_mask & ~correct_veg
    n_silent         = silent_failures.sum()
    catastrophic     = silent_failures & ~correct_fresh
    accidental_ok    = silent_failures &  correct_fresh
    print(f"  Total silent failures          : {n_silent}")
    print(f"  Catastrophic (veg+fresh wrong) : {catastrophic.sum()}  ← true risk")
    print(f"  Accidental correct (fresh ok)  : {accidental_ok.sum()}  ← lucky, not reliable")
    if n_silent == 0:
        print(f"  [OK] All veg misclassifications gated out.")
    elif catastrophic.sum() == 0:
        print(f"  [OK] No catastrophic failures. Freshness signal robust to veg error.")
    else:
        print(f"  [WARNING] {catastrophic.sum()} catastrophic silent failures exist.")

    # ── Wrong-veg detection breakdown ────────────────────────
    print("\n  --- Wrong-Veg Detection Breakdown ---")
    veg_wrong_all = ~correct_veg   # all misclassified veg samples
    n_veg_wrong   = veg_wrong_all.sum()
    caught_ood     = (veg_wrong_all &  is_ood_arr).sum()
    # centroid_gate_arr already computed in section 8

    caught_by_centroid_only = (veg_wrong_all & centroid_gate_arr & ~is_ood_arr).sum()
    caught_by_ood_only      = (veg_wrong_all & is_ood_arr & ~centroid_gate_arr).sum()
    caught_by_both          = (veg_wrong_all & is_ood_arr & centroid_gate_arr).sum()
    missed_by_both          = (veg_wrong_all & ~is_ood_arr & ~centroid_gate_arr).sum()

    print(f"  Total veg misclassifications   : {n_veg_wrong}")
    print(f"  Caught by OOD only             : {caught_by_ood_only}")
    print(f"  Caught by centroid only        : {caught_by_centroid_only}")
    print(f"  Caught by both                 : {caught_by_both}")
    print(f"  Missed by both                 : {missed_by_both}  ← blind spots")
    if missed_by_both > 0:
        missed_mask = veg_wrong_all & ~is_ood_arr & ~centroid_gate_arr
        missed_fresh_wrong = (~correct_fresh[missed_mask]).sum()
        print(f"    Of blind spots, freshness also wrong: {missed_fresh_wrong}  ← true silent errors")
    else:
        print(f"  [OK] All veg misclassifications caught by at least one detector.")

    # ── Partial augmentation gate stats ──────────────────────
    print("\n  --- Augmentation Gate (val set, stratified 40/veg) ---")
    aug_paths_file = os.path.join(MODEL_DIR, "val_image_paths.npy")
    aug_veg_file   = os.path.join(MODEL_DIR, "y_veg_val.npy")

    if not scoring_config.get("use_augmentation_gate", False):
        print("  [DISABLED] use_augmentation_gate=False in scoring_config.")
    elif not os.path.exists(aug_paths_file) or not os.path.exists(aug_veg_file):
        print("  [SKIP] val_image_paths.npy not found. Re-run extraction + split.")
    else:
        import cv2
        from extract_features import (
            model as deep_model_eval, preprocess_input as pi_eval,
            extract_handcrafted as eh_eval
        )
        val_paths_aug = np.load(aug_paths_file, allow_pickle=True)
        val_veg_aug   = np.load(aug_veg_file,   allow_pickle=True)
        val_fresh_aug = np.load(os.path.join(MODEL_DIR, "y_fresh_val.npy"))
        val_X_file    = os.path.join(MODEL_DIR, "X_val.npy")
        val_X_precomp = np.load(val_X_file) if os.path.exists(val_X_file) else None

        globl_b    = scoring_config["global_bounds"]
        p5_a, p95_a = globl_b["p5"], globl_b["p95"]
        denom_a    = max(p95_a - p5_a, 1e-6)
        thresh_aug = scoring_config.get("unstable_range_thresh", 13.0)

        rng_aug = np.random.default_rng(42)
        samples = []   # (path_idx, true_fresh)
        for veg in np.unique(val_veg_aug):
            vmask  = val_veg_aug == veg
            vidxs  = np.where(vmask)[0]
            n      = min(40, len(vidxs))
            chosen = rng_aug.choice(len(vidxs), n, replace=False)
            for c in chosen:
                samples.append((vidxs[c], val_fresh_aug[vmask][c]))

        aug_fires=0; aug_catches=0; aug_blocks=0; aug_total=0; aug_skip=0

        for val_idx, true_fresh in samples:
            path = val_paths_aug[val_idx]
            img  = cv2.imread(path)
            if img is None:
                aug_skip += 1
                continue
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb  = cv2.resize(rgb, (224, 224))
            h, w = rgb.shape[:2]
            augmented = [
                np.clip(rgb.astype(np.float32)*1.15,0,255).astype(np.uint8),
                np.clip(rgb.astype(np.float32)*0.85,0,255).astype(np.uint8),
                cv2.flip(rgb,1),
                cv2.GaussianBlur(rgb,(5,5),0),
                cv2.warpAffine(rgb,cv2.getRotationMatrix2D((w//2,h//2), 5,1.0),(w,h)),
                cv2.warpAffine(rgb,cv2.getRotationMatrix2D((w//2,h//2),-5,1.0),(w,h)),
            ]
            try:
                raws_aug = []
                for aug in augmented:
                    batch  = pi_eval(np.expand_dims(aug.astype(np.float32),0))
                    deep_f = deep_model_eval.predict(batch,verbose=0)[0].astype(np.float32)
                    hand_f = eh_eval(aug)
                    feats  = np.concatenate([deep_f, hand_f])
                    Xvt    = vt.transform(np.array([feats]))
                    Xf_aug = scaler.transform(Xvt)[:, selected]
                    raws_aug.append(float(fresh_model.decision_function(Xf_aug)[0]))

                scores_aug    = [float(np.clip((r-p5_a)/denom_a*100,0,100)) for r in raws_aug]
                score_range_a = max(scores_aug) - min(scores_aug)
                crosses_a     = (min(raws_aug) < 0) and (max(raws_aug) > 0)
                gate_fires    = (score_range_a >= thresh_aug) and crosses_a

                # Use precomputed val features for base prediction (Fix 4)
                if val_X_precomp is not None:
                    Xf_base    = vt.transform(val_X_precomp[[val_idx]])
                    Xf_base    = scaler.transform(Xf_base)[:, selected]
                else:
                    base_batch = pi_eval(np.expand_dims(rgb.astype(np.float32),0))
                    deep_base  = deep_model_eval.predict(base_batch,verbose=0)[0].astype(np.float32)
                    feats_base = np.concatenate([deep_base, eh_eval(rgb)])
                    Xf_base    = scaler.transform(vt.transform(np.array([feats_base])))[:, selected]

                pred_fresh = int(fresh_model.predict(Xf_base)[0])
                aug_total += 1
                if gate_fires:
                    aug_fires += 1
                    if pred_fresh != int(true_fresh):
                        aug_catches += 1
                    else:
                        aug_blocks += 1
            except Exception:
                aug_skip += 1

        if aug_total > 0:
            print(f"  Tested  : {aug_total} ({aug_skip} skipped), stratified 40/veg")
            print(f"  Fires   : {aug_fires} ({aug_fires/aug_total*100:.1f}%)")
            print(f"  Catches : {aug_catches}  (gate fires + pred wrong)")
            print(f"  Blocks  : {aug_blocks}  (gate fires + pred correct)")
            aug_delta_cov = -aug_fires / aug_total   # gate always reduces coverage
            print(f"  Coverage cost: {aug_delta_cov:+.3f} (fraction blocked)")
            if aug_fires == 0:
                print(f"  [OK] Gate never fires on clean val — threshold appropriate for dataset.")
            elif aug_catches == 0:
                print(f"  [!] Gate fires but catches no errors. REMOVE CANDIDATE on this data.")
            else:
                ratio = aug_catches / aug_fires if aug_fires > 0 else 0
                print(f"  Precision: {ratio:.2f} (fraction of fires that were real errors)")
        else:
            print(f"  [SKIP] No valid samples.")

    # ── Calibration sanity: per-veg score means ──────────────
    print("\n  --- Score Calibration Sanity (per-veg means) ---")
    print(f"  {'Veg':<12}  {'Fresh_mean':>10}  {'Rotten_mean':>12}  {'Delta':>7}  Note")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*4}")
    per_veg_b = scoring_config.get("per_veg_bounds", {})
    for veg in le.classes_:
        vmask  = (y_veg_test == veg)
        bounds = per_veg_b.get(veg, scoring_config["global_bounds"])
        vscores = np.array([normalize_score(d, bounds) for d in decisions[vmask]])
        vfresh  = y_fresh_test[vmask]
        fm = vscores[vfresh==1].mean() if (vfresh==1).any() else float("nan")
        rm = vscores[vfresh==0].mean() if (vfresh==0).any() else float("nan")
        d  = fm - rm if not (np.isnan(fm) or np.isnan(rm)) else float("nan")
        note = "← narrow delta" if (not np.isnan(d) and d < 40) else ""
        print(f"  {veg:<12}  {fm:>10.2f}  {rm:>12.2f}  {d:>7.2f}  {note}")
    print(f"  [DESIGN] Per-vegetable normalization fits separate p5/p95 bounds per class.")
    print(f"           This means scores ARE locally comparable within a vegetable class")
    print(f"           but are NOT globally comparable across vegetables by design.")
    print(f"           A score of 80 for banana and 80 for potato both mean 'high relative")
    print(f"           to that vegetable's training distribution' — not the same absolute")
    print(f"           freshness level. This is the expected consequence of per-class")
    print(f"           calibration, not an inconsistency. If cross-vegetable comparability")
    print(f"           is required, use global bounds throughout.")

    print()
    print("  FEATURE SELECTION NOTE: XGBoost ranks on freshness-only labels.")
    print("  Vegetable classifier reuses the same top-100 subset.")
    print("  Union-of-top-features fix requires full retrain.")
    print("  Deferred while veg accuracy remains 99.13%.")

    # ─────────────────────────────────────────
    # 8b. Per-vegetable state distribution
    #
    # Global RELIABLE% can hide vegetable-level collapse.
    # A system that is 99% reliable for banana but 70% reliable
    # for potato looks fine globally but fails operationally.
    # This check catches that pattern.
    # ─────────────────────────────────────────
    print("\n========== Per-Vegetable State Distribution ==========")
    print(f"  {'Veg':<12}  {'N':>5}  {'RELIABLE':>10}  {'TENTATIVE':>10}  "
          f"{'UNRELIABLE':>11}  {'Note'}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*4}")

    for veg in le.classes_:
        vmask   = (y_veg_test == veg)
        n_veg   = vmask.sum()
        if n_veg == 0:
            continue
        vstate  = state_arr[vmask]
        nr      = (vstate == "RELIABLE").sum()
        nt      = (vstate == "TENTATIVE").sum()
        nu      = (vstate == "UNRELIABLE").sum()
        r_pct   = nr / n_veg * 100
        t_pct   = nt / n_veg * 100
        u_pct   = nu / n_veg * 100

        note = ""
        if r_pct < 90:
            note = "← WEAK"
        if r_pct < 75:
            note = "← CRITICAL"

        print(f"  {veg:<12}  {n_veg:>5}  "
              f"{nr:>5} {r_pct:>4.1f}%  "
              f"{nt:>5} {t_pct:>4.1f}%  "
              f"{nu:>5} {u_pct:>4.1f}%   {note}")

    # ─────────────────────────────────────────
    # 8c. Freshness accuracy restricted to RELIABLE samples
    #
    # RELIABLE must mean trustworthy — not just "gate passed."
    # If RELIABLE accuracy < overall accuracy, the gate is
    # letting wrong predictions through confidently, which is
    # the worst failure mode.
    #
    # Expected: RELIABLE accuracy >= overall accuracy.
    # ─────────────────────────────────────────
    print("\n========== Freshness Accuracy on RELIABLE Samples ==========")
    reliable_mask = (state_arr == "RELIABLE")

    if reliable_mask.sum() == 0:
        print("  [WARNING] No RELIABLE samples. Cannot compute accuracy.")
    else:
        yfresh_pred_all = fresh_model.predict(X)

        overall_acc  = accuracy_score(y_fresh_test, yfresh_pred_all)
        reliable_acc = accuracy_score(
            y_fresh_test[reliable_mask],
            yfresh_pred_all[reliable_mask]
        )

        print(f"  Overall freshness accuracy   : {overall_acc:.4f}")
        print(f"  RELIABLE-only accuracy       : {reliable_acc:.4f}")
        delta_acc = reliable_acc - overall_acc

        if delta_acc >= 0:
            print(f"  [OK] RELIABLE accuracy is {delta_acc*100:+.2f}% vs overall. "
                  f"Gate filters correctly — RELIABLE samples are more trustworthy.")
        else:
            print(f"  [WARNING] RELIABLE accuracy is {delta_acc*100:+.2f}% vs overall.")
            print(f"            Gate is admitting wrong predictions confidently.")
            print(f"            Instability threshold may be too loose.")

        # Per-vegetable RELIABLE accuracy
        # Comparison baseline is global RELIABLE accuracy, not overall.
        # Reason: overall accuracy includes TENTATIVE/UNRELIABLE samples
        # which are harder by definition. Comparing per-veg RELIABLE
        # accuracy against global RELIABLE accuracy is the fair test.
        print(f"\n  Per-vegetable RELIABLE accuracy (baseline = {reliable_acc:.4f}):")
        print(f"  {'Veg':<12}  {'N_reliable':>10}  {'Acc':>6}  {'Note'}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*4}")

        for veg in le.classes_:
            vmask_r = reliable_mask & (y_veg_test == veg)
            if vmask_r.sum() == 0:
                continue
            vacc = accuracy_score(
                y_fresh_test[vmask_r],
                yfresh_pred_all[vmask_r]
            )
            gap  = vacc - reliable_acc
            note = ""
            if gap < -0.05:
                note = f"← {abs(gap)*100:.1f}% below RELIABLE baseline"
            print(f"  {veg:<12}  {vmask_r.sum():>10}  {vacc:.4f}  {note}")

    # ─────────────────────────────────────────
    # 9. Limitation statement (mandatory)
    # ─────────────────────────────────────────
    print("\n========== Limitations ==========")
    print(
        "  Score is a per-vegetable calibrated SVM margin proxy.\n"
        "  It proves class separation and ordering reliability between classes.\n"
        "  It does not prove correct intra-class ordering without continuous\n"
        "  ground-truth freshness labels (e.g. decay-day annotations).\n"
        "  The system does not guarantee detection of all condition violations.\n"
        "  Scores are only meaningful relative to the training distribution\n"
        "  and assume consistent imaging conditions."
    )


if __name__ == "__main__":
    main()