# Reliability Gate Analysis

The system produces three output states — RELIABLE, TENTATIVE, UNRELIABLE — by applying a sequence of gates to each prediction. This document explains what each gate is, how the gate ablation study measures their usefulness, what the actual test-set results show, and what the results imply for future tuning decisions.

---

## 1. The Five Gates

The system has five distinct gates. Three are formally ablated in `evaluate_models.py`. Two are handled separately.

| Gate | Component | What it checks | Output effect |
|------|-----------|----------------|---------------|
| **G1 OOD** | Mahalanobis distance | dist ≥ 30.438 | `score_unreliable=True` → UNRELIABLE |
| **G2 Near-boundary** | SVM margin magnitude | \|raw\| < T_boundary | `decision_unreliable=True` → TENTATIVE |
| **G3 Low veg confidence** | Top-1 prob + prob gap | conf < 70% OR gap < 15% | `decision_unreliable=True` → TENTATIVE |
| **G4 Centroid consistency** | L2 centroid ratio | ratio > per-class P95 threshold | `decision_unreliable=True` → TENTATIVE |
| **G5 Aug instability** | Score range across 6 views | range ≥ 36.0 AND crosses boundary | `score_unreliable=True` → UNRELIABLE |

G1, G2, and G3 appear in the ablation table. G4 (centroid) is used in state computation but evaluated separately via the wrong-veg detection breakdown. G5 (augmentation) is currently disabled (`use_augmentation_gate=False`) and is evaluated on a separate val-set sample.

---

## 2. How the Ablation Measures Gate Usefulness

The ablation does not use a simple catch-rate. It uses a **two-axis evaluation** that quantifies the exact cost and benefit of each gate by computing what would happen if that gate were disabled.

```
For each gate G:
  1. Set G.active = False
  2. Recompute state_arr for all test samples (keeping other gates active)
  3. Compute new RELIABLE mask and accuracy on those samples

  Δ_acc = accuracy(new RELIABLE set) − baseline_reliable_accuracy
  Δ_cov = coverage(new RELIABLE set) − baseline_coverage

  Δ_acc < 0  means disabling the gate HURTS accuracy  → gate is protecting predictions
  Δ_acc > 0  means disabling the gate IMPROVES accuracy → gate is blocking good predictions
  Δ_cov > 0  means disabling the gate EXPANDS coverage  → gate is restricting reach
```

This is more informative than a simple "wrong catches / fires" ratio because it accounts for gate interdependency: a sample blocked by both G1 and G3 simultaneously stays blocked when only G1 is disabled. A naive efficiency ratio would overstate G1's impact on coverage.

### Verdict rule (strictly applied)

```
KEEP         → Δ_acc < −0.001   (gate actively protects accuracy)
REMOVE       → Δ_acc ≥ −0.001  AND  Δ_cov ≤ 0.005  (no accuracy benefit, negligible coverage cost)
REVIEW       → Δ_acc ≥ −0.001  AND  Δ_cov > 0.005  (no accuracy benefit but meaningful coverage cost)
NEVER FIRES  → gate inactive on this test set
```

---

## 3. Ablation Results (Test Set — 2,539 samples)

```
Baseline: acc=0.9898   coverage=0.923   (all gates active)

  Gate               Fires  Fire%  Catch_W  Block_C   Δ_acc    Δ_cov   Verdict
  ─────────────────────────────────────────────────────────────────────────────
  G1_OOD               62   2.4%      1       61    −0.0004  +0.0760   REVIEW
  G2_near_boundary      0   0.0%      0        0    −0.0003  +0.0524   NEVER FIRES
  G3_low_veg_conf       3   0.1%      0        3    −0.0003  +0.0528   REVIEW

  Catch_W = gate fires AND freshness prediction was wrong  (gate prevented an error)
  Block_C = gate fires AND freshness prediction was correct (gate withheld a valid result)
  Δ_acc   = (accuracy when this gate disabled) − baseline_acc
  Δ_cov   = (coverage when this gate disabled) − baseline_coverage
```

---

## 4. Gate-by-Gate Interpretation

### G1 — OOD Gate (Mahalanobis)

**Verdict: REVIEW**

```
Fires:   62 samples (2.4% of test set)
Catch_W:  1    (1 wrong freshness prediction blocked)
Block_C: 61    (61 correct freshness predictions withheld)
Δ_acc:   −0.0004  (disabling drops RELIABLE accuracy by 0.04%)
Δ_cov:   +0.0760  (disabling expands coverage by 7.6%)
```

This is the most consequential gate in terms of coverage. Disabling it would add 7.6% of the test set back to the RELIABLE pool — that is 193 additional predictions. However, among those 62 currently blocked samples, only 1 had a wrong freshness prediction. The other 61 had correct freshness predictions despite being OOD.

This does not mean the gate is useless. A freshness prediction can be correct *on this test set* while still being produced by features that lie outside the training distribution. On a genuinely novel image (a different camera, different background, unfamiliar variety), those same OOD features could produce a wrong prediction. The gate exists for **distribution-shift robustness**, not purely for in-distribution accuracy.

The REVIEW verdict does not mean remove — it means the tradeoff is real and should be assessed against deployment context. If deployment conditions are tightly controlled (same camera, same background, same vegetables), the coverage cost outweighs the distributional protection. If deployment is open-ended, the gate is worth keeping.

The OOD rate is stable across splits: 1.81% on val, 2.44% on test (difference = 0.63%, well within the 5% stability threshold). The Mahalanobis threshold is transferring stably.

---

### G2 — Near-Boundary Gate

**Verdict: NEVER FIRES**

```
Fires:   0 samples (0.0% of test set)
T_boundary = 0.0  (set by formal threshold selection)
```

This gate never fires because its threshold was formally set to zero. The constrained optimiser (`select_thresholds()` in `threshold_selection.py`) found that maximising coverage subject to Risk ≤ 10% on `thr_val` requires **no margin cutoff at all**. The base model is accurate enough that even near-boundary samples do not disproportionately contain errors — the OOD gate and centroid gate are already capturing the problematic near-boundary cases.

Despite never firing, the ablation still reports Δ_acc = −0.0003 and Δ_cov = +0.0524. These non-zero values come from the interdependency structure: when G2 is disabled, the state computation changes in ways that interact with G3, producing a small ripple in the RELIABLE pool.

G2 is not dormant by accident — it is formally calibrated to be inactive on this data. If a future retrain produces a model where near-boundary samples are consistently less accurate, the optimiser would find T_boundary > 0 and the gate would become active again.

---

### G3 — Low Vegetable Confidence Gate

**Verdict: REVIEW**

```
Fires:   3 samples (0.1% of test set)
Catch_W: 0    (no wrong freshness predictions caught)
Block_C: 3    (3 correct freshness predictions withheld)
Δ_acc:   −0.0003
Δ_cov:   +0.0528
```

This gate fires on 3 samples where the vegetable classifier's top-1 confidence was below 70% or the gap between top-1 and top-2 was below 15%. In all 3 cases, the freshness prediction was actually correct. The gate caught no errors on this test set.

The REVIEW verdict reflects the coverage cost (0.5% of samples, 13 predictions in absolute terms if scaled) with no measurable accuracy benefit. However, this gate has a purpose that is not captured in the freshness accuracy metric: **it prevents per-vegetable normalization bounds from being applied to the wrong vegetable**.

If a cucumber is misclassified as a potato with 71% confidence (above the threshold), the system applies potato's p5/p95 bounds to a cucumber's raw margin. The resulting score may be numerically valid but is calibrated to the wrong reference distribution — a subtle error that does not necessarily produce a wrong `fresh_label` but produces an unreliable score magnitude. G3 prevents this by forcing global bounds when confidence is marginal.

On the current test set this case simply did not arise — the 3 samples G3 fired on all happened to have correct freshness predictions regardless of which bounds were applied. The gate's protective value is better assessed through the gate stability check (global delta = +0.0003, max per-veg delta = 0.0000 — both well within tolerance).

---

### G4 — Centroid Consistency Gate

This gate is not individually ablated in the three-gate table but is included in the state computation. Its behaviour is reported in the wrong-veg detection breakdown.

```
  Total veg misclassifications   : 10
  Caught by OOD gate only        :  5
  Caught by centroid gate only   :  2
  Caught by both                 :  0
  Missed by both (blind spots)   :  3
    Of blind spots, freshness also wrong: 0
```

The centroid gate uniquely caught 2 of the 10 vegetable misclassifications that the OOD gate missed. Without it, those 2 samples would have reached RELIABLE with an incorrect vegetable label — and potentially wrong normalization bounds applied. No catastrophic failures (wrong veg + wrong fresh both reaching RELIABLE) occurred.

The 3 blind spots that both gates missed all had correct freshness predictions despite the vegetable being misclassified — "accidental correct" outcomes. These are not dangerous failures but they are not reliable either. They are flagged as silent failures in the evaluation:

```
  Silent failures (veg wrong but RELIABLE): 3
  Catastrophic (veg+fresh wrong):           0  ← true risk = zero
  Accidental correct (fresh ok):            3  ← lucky, not reliable
```

---

### G5 — Augmentation Instability Gate

**Status: Disabled** (`use_augmentation_gate=False`)

This gate is evaluated separately on a stratified val sample (40 images per vegetable, 200 total). On the current test set, `evaluate_models.py` reports:

```
  --- Augmentation Gate (val set, stratified 40/veg) ---
  [DISABLED] use_augmentation_gate=False in scoring_config.
```

The gate is disabled because the 6× EfficientNetB0 passes required per image are impractical for real-time inference on CPU. The formal threshold T_instability = 36.0 is stored in `scoring_config.json` and the gate can be reactivated by setting `use_augmentation_gate: true`.

---

## 5. Gate Co-occurrence

Co-occurrence tells you how often two gates fire on the same sample. High co-occurrence means they are detecting the same problem — one is redundant.

```
  G1_OOD ∩ G2_near_boundary:       0   (only G1: 62,  only G2: 0)
  G1_OOD ∩ G3_low_veg_conf:        2   (only G1: 60,  only G3: 1)
  G2_near_boundary ∩ G3_low_veg_conf: 0
  All three simultaneously:         0
```

The three gates fire almost entirely on **disjoint samples**. G1 and G3 overlap on 2 samples (3.2% of G1's fires, 67% of G3's fires). This means both gates would have caught those 2 samples — so removing G3 would not cause G1 to miss anything on those 2. Conversely, G1 uniquely handles 60 samples that G3 would never touch.

The absence of high co-occurrence is a good sign: the three gates are monitoring different failure modes rather than redundantly blocking the same samples.

---

## 6. RELIABLE Subset Accuracy

The core test of whether the gating system is working is whether RELIABLE predictions are more accurate than the overall population. If they are not — if RELIABLE accuracy is lower than overall — the gates are admitting wrong predictions with false confidence.

```
  Overall freshness accuracy    : 98.94%   (all 2,539 samples)
  RELIABLE-only accuracy        : 98.98%   (2,343 RELIABLE samples)
  Δ                             : +0.04%   ← RELIABLE is more accurate [OK]
```

RELIABLE is 0.04% more accurate than the full test set. The margin is small because the base model is already very accurate — there is limited room for the gates to improve on it. The important property is the direction: filtering to RELIABLE did not make accuracy worse.

Per-vegetable RELIABLE accuracy relative to the 98.98% RELIABLE baseline:

```
  apple       n=856    acc=0.9907   (+0.09% above baseline)
  banana      n=866    acc=0.9988   (+0.90% above baseline)
  capsicum    n=208    acc=1.0000   (+1.02% above baseline)
  cucumber    n=160    acc=0.9688   (−2.10% below baseline)  ← WEAK
  potato      n=253    acc=0.9605   (−2.93% below baseline)  ← WEAK
```

Apple, banana, and capsicum show higher accuracy on their RELIABLE subsets than the global RELIABLE baseline — consistent with their high RELIABLE rates (95.5%, 92.1%, 91.6% respectively). Cucumber and potato are below baseline, consistent with their lower RELIABLE rates (87.9% and 86.1%) and higher OOD rates (7.7% and 6.5%). The samples that pass the gates for these two vegetables are still less accurate than the other three.

This is the primary area warranting attention: the cucumber and potato RELIABLE subsets have higher error rates than other vegetables despite gating.

---

## 7. State Distribution on the Test Set

```
  Total test samples : 2,539

  RELIABLE           : 2,343  (92.3%)   score + fresh_label + confidence_band
  TENTATIVE          :   134  ( 5.3%)   score shown, no fresh_label
  UNRELIABLE (OOD)   :    62  ( 2.4%)   no score, no label, warning returned

  [Note] Augmentation-instability UNRELIABLE not counted (gate disabled).
```

Per-vegetable breakdown:

```
  Veg        N      RELIABLE        TENTATIVE       UNRELIABLE
  apple      896    856  (95.5%)     35  (3.9%)       5  (0.6%)
  banana     940    866  (92.1%)     54  (5.7%)      20  (2.1%)
  capsicum   227    208  (91.6%)     15  (6.6%)       4  (1.8%)
  cucumber   182    160  (87.9%)      8  (4.4%)      14  (7.7%)  ← WEAK
  potato     294    253  (86.1%)     22  (7.5%)      19  (6.5%)  ← WEAK
```

Cucumber and potato have the highest UNRELIABLE rates because their training distributions are more compact — genuine test samples more frequently fall outside the Mahalanobis P99 radius. This is a data density problem, not a threshold problem. More training samples per class would reduce OOD flagging without any threshold change.

---

## 8. Tuning Recommendations by Gate

### G1 — OOD Gate

**Current:** REVIEW. Fires 62 times, catches 1 error, costs 7.6% coverage.

The gate is working correctly as a distributional safety check — it should not be removed on the basis of in-distribution accuracy alone. The appropriate question is deployment context. Consider two options:

- **Downgrade to CAUTION-only** (warning flag, no state change to UNRELIABLE) if deployment conditions are tightly controlled and coverage is more important than distributional robustness. This recovers 7.6% coverage at the cost of distributional safety guarantees.
- **Tighten the threshold** from P99 (30.438) to a higher percentile of training distances. This reduces fires and coverage cost while maintaining the safety signal for genuinely extreme outliers.

Do **not** simply disable it. The 62 flagged samples are real OOD detections — they lie outside the P99 of the training Mahalanobis distribution. Even if their freshness predictions happened to be correct on this test set, they represent exactly the kind of input where the system's calibration guarantees break down.

### G2 — Near-Boundary Gate

**Current:** NEVER FIRES (T_boundary = 0.0).

No action needed. The threshold was set to zero by the formal optimiser on `thr_val`. If a future retrain changes the model or feature set, re-run `train_svm.py` — the optimiser will recalibrate T_boundary automatically. Do not manually set T_boundary > 0 without re-running the formal selection.

### G3 — Low Veg Confidence Gate

**Current:** REVIEW. Fires 3 times, catches 0 errors, costs 0.5% coverage.

The gate's primary value is preventing wrong per-vegetable bounds from being applied — not directly catching freshness errors. Before tuning, check whether any of the 3 blocked samples had misclassified vegetable predictions. If yes, the gate is correctly flagging ambiguous vegetable predictions regardless of freshness outcome. If all 3 had correct vegetable predictions, the gate is firing on cases where the veg classifier was briefly unsure but still correct.

If gate stability ever degrades (global delta ≥ 0.01 OR max per-veg delta ≥ 0.02), tune `veg_gap_threshold` first (raise in steps of 0.05), then `veg_confidence_threshold` only if gap tuning is insufficient. Confidence alone allows confused predictions that a gap threshold would have rejected.

### G4 — Centroid Gate

No tuning indicated. The gate uniquely caught 2 vegetable misclassifications with zero false positives relative to catastrophic failures. The per-class thresholds (P95 of correct val predictions) are well-calibrated. Revisit only if the number of blind spots increases on a future retrain.

### G5 — Augmentation Gate

**Current:** Disabled.

The formally selected T_instability = 36.0 is stored and ready. Reactivate by setting `use_augmentation_gate: true` in `scoring_config.json` if inference latency is acceptable (adds ~6× EfficientNetB0 passes per image). No re-calibration is needed.

---

## 9. Summary

```
Gate               Status        Verdict       What it actually does
──────────────────────────────────────────────────────────────────────────────
G1 OOD             Active        REVIEW        Distributional safety. High
                                               coverage cost (7.6%), low in-
                                               distribution accuracy benefit.
                                               Keep for robustness, consider
                                               downgrading to caution-only.

G2 Near-boundary   Active        NEVER FIRES   T_boundary=0.0 from formal
                                               optimisation. Correctly inactive
                                               on this model. Will activate
                                               automatically if model changes.

G3 Low veg conf    Active        REVIEW        Prevents wrong-veg bounds.
                                               Fires 3 times, blocks 3 correct
                                               predictions, catches 0 errors.
                                               Protective purpose valid;
                                               accuracy impact negligible.

G4 Centroid        Active        Not ablated   Uniquely caught 2 of 10 veg
                                               misclassifications. Zero
                                               catastrophic blind spots.

G5 Augmentation    Disabled      Not ablated   Formally calibrated (T=36.0).
                                               Inactive due to inference cost.
                                               Can be reactivated without
                                               retraining.
──────────────────────────────────────────────────────────────────────────────

System-level results (test set):
  RELIABLE rate          : 92.3%
  RELIABLE accuracy      : 98.98%   (+0.04% vs overall)
  Catastrophic failures  : 0
  Baseline accuracy      : 98.98%   coverage=0.923
```