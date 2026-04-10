# Augmentation Instability Gate

**Current status:** configured and formally calibrated, but **disabled at inference time** (`use_augmentation_gate = False` in `scoring_config.json`). The thresholds are stored and the gate can be reactivated without retraining. This document describes the gate in full, including its design, calibration, limitations, and the reasoning behind its current disabled status.

---

## 1. What Problem This Gate Solves

The freshness SVM produces a single score for a single image. That score is computed from one particular lighting condition, one camera angle, one blur level. It says nothing about whether the prediction would hold up under slightly different conditions.

A sample can sit near the decision boundary in a region where a small real-world perturbation — a phone camera slightly overexposing, a hand-held shake, a slight rotation — would push it across the fresh/rotten boundary. The model has no way to detect this on its own; it just returns a score.

The augmentation gate addresses this directly: **it synthetically perturbs the image in six ways and checks whether the model's decision is consistent across all perturbations**. Inconsistency indicates a fragile prediction that should not be called RELIABLE.

---

## 2. The Six Augmentations

Each augmentation is applied to the resized 224×224 RGB image. The original image is not included in the set — only the six perturbed versions are scored.

```
Augmentation 1:   Pixel values × 1.15, clipped to [0, 255]   (brightness +15%)
Augmentation 2:   Pixel values × 0.85, clipped to [0, 255]   (brightness −15%)
Augmentation 3:   cv2.flip(rgb, 1)                           (horizontal mirror)
Augmentation 4:   cv2.GaussianBlur(rgb, (5, 5), 0)           (mild blur, σ auto)
Augmentation 5:   cv2.warpAffine(..., rotation=+5°)          (clockwise rotation)
Augmentation 6:   cv2.warpAffine(..., rotation=−5°)          (counter-clockwise)
```

These simulate the kind of variation that occurs naturally in deployment: inconsistent phone camera exposure, a photographer who is slightly off-axis, a blurry low-quality image. They are **not random** — the same six augmentations are applied to every image in the same order, making the gate fully deterministic.

Each augmented view goes through the full inference sub-pipeline independently:

```
augmented image
       │
       ▼
EfficientNetB0 + handcrafted → [1312]
       │
VarianceThreshold → StandardScaler → union_349 slice → [349]
       │
fresh_svm.decision_function() → aug_raw   (raw SVM margin)
       │
normalize(aug_raw, per_veg_bounds)  → aug_score  (0–100)
```

This means EfficientNetB0 is run **six separate times** per image when the gate is active. This is the primary reason the gate is currently disabled — the inference cost is 6× that of a standard prediction.

---

## 3. The Two Computed Statistics

After scoring all six augmentations, two statistics are derived:

### score_range

```
score_range = max(aug_scores) − min(aug_scores)
```

This is the spread of the normalised 0–100 freshness scores across the six augmented views. A small range means the model's output is insensitive to the applied perturbations. A large range means the output shifts substantially depending on lighting or angle.

**Example — stable prediction:**
```
aug_scores  = [82.1, 79.4, 83.6, 80.2, 81.7, 80.9]
score_range = 83.6 − 79.4 = 4.2
→ Consistent fresh signal across all perturbations. Gate does not fire.
```

**Example — unstable prediction:**
```
aug_scores  = [88.3, 22.1, 85.7, 18.4, 84.9, 25.6]
score_range = 88.3 − 18.4 = 69.9
→ Large spread. Some augmentations score in the fresh range; others score
  in the rotten range. Gate evaluates the second condition.
```

### crosses_boundary

```
crosses_boundary = (min(aug_raws) < 0) AND (max(aug_raws) > 0)
```

This operates on the **raw SVM margins** (not the normalised scores). The raw margin is positive on the fresh side of the hyperplane and negative on the rotten side. `crosses_boundary = True` means at least one augmented view falls on the fresh side and at least one falls on the rotten side — the model is literally contradicting itself across views.

**Example:**
```
aug_raws = [1.21, 0.84, −0.31, 1.03, −0.48, 0.91]
min = −0.48  (< 0)
max =  1.21  (> 0)
→ crosses_boundary = True
```

---

## 4. The Three-Level Gate Logic

The gate produces three possible outcomes — not two. This is important because the appropriate response to high range without a boundary flip is different from the response to a flip.

```
high_range     = (score_range ≥ T_instability)  →  T_instability = 36.0
crosses_bnd    = (min(aug_raws) < 0 AND max(aug_raws) > 0)

unstable       = high_range AND crosses_bnd
sensitive_only = high_range AND NOT crosses_bnd AND score_range > T_instability × 1.5
               = high_range AND NOT crosses_bnd AND score_range > 54.0
```

| Condition | Name | Effect | Output State |
|---|---|---|---|
| `unstable = True` | TRUE INSTABILITY | `score_unreliable = True` | **UNRELIABLE** — score and label both withheld |
| `sensitive_only = True` | INPUT SENSITIVITY | `decision_unreliable = True` | **TENTATIVE** — score shown, label withheld |
| neither | Stable | No effect | Gate does not contribute to state |

### Why three levels, not two

**Scenario A — high range, no flip:**
```
aug_scores = [88, 62, 85, 70, 87, 68]    score_range = 26
aug_raws   = [1.8, 0.6, 1.6, 0.8, 1.7, 0.7]
→ All margins positive. All views say "fresh."
→ crosses_boundary = False
→ Score is sensitive to perturbations but the decision direction is consistent.
→ Appropriate response: TENTATIVE (magnitude unreliable, direction reliable)
```

**Scenario B — high range + boundary flip:**
```
aug_scores = [90, 18, 87, 24, 88, 21]    score_range = 72
aug_raws   = [1.9, −0.8, 1.7, −0.6, 1.8, −0.7]
→ Three views: positive margin (fresh). Three views: negative margin (rotten).
→ crosses_boundary = True
→ The model flatly contradicts itself. The prediction has no stable direction.
→ Appropriate response: UNRELIABLE (score and label both meaningless)
```

The severity threshold for `sensitive_only` is set at `1.5 × T_instability = 54.0` — a tighter cutoff than the standard `T_instability` — because a sample that reaches `sensitive_only` is already at `high_range` without a flip, so only an extreme range (large enough to be clearly pathological) triggers a TENTATIVE downgrade.

---

## 5. Where T_instability = 36.0 Comes From

This value is **not** the P95 of augmentation ranges on the validation set. It is the result of a formal constrained optimisation.

### Step 1 — Compute augmentation statistics on thr_val

Augmentation statistics are computed on **thr_val only** (the 50% of val disjoint from the probability calibration set). A stratified sample is drawn:

```
apple      100 images  (from thr_val rows)
banana     100 images
capsicum    59 images
cucumber    52 images
potato      69 images
──────────────────────
Total:     380 images
```

For each sampled image, all six augmentations are run and `aug_range` and `crosses_boundary` are recorded. The raw P95 of `aug_range` across all 380 samples:

```
P95 of aug_range (380 samples) = 29.4715
```

This P95 is an input to the threshold optimiser — it is not used directly as the threshold.

### Step 2 — Formal threshold selection

`T_instability` is selected jointly with `T_boundary` by solving the same constrained optimisation used for the boundary gate:

```
RELIABLE_i = (
    NOT is_ood_i
    AND NOT (crosses_bnd_i AND aug_range_i > T_instability)
    AND abs(decision_i) > T_boundary
)

Optimise: Maximise Coverage = P(RELIABLE)
Subject to: Risk = P(error | RELIABLE) ≤ ε = 0.10
            n_reliable ≥ n_min
```

The grid sweep over `T_instability` uses `np.arange(0.0, max_aug_range + 1.0, 0.5)`, where `max_aug_range` is derived from the 380-sample aug stats. The result from the actual training run:

```
T_boundary    = 0.0000
T_instability = 36.0000    ← formally selected, not the raw P95
Risk          = 0.0188     (1.88% error rate on RELIABLE samples)
Coverage      = 97.89%
n_reliable    = 372
feasible      = True
```

`T_instability = 36.0` is higher than the raw P95 of 29.4715 because the optimiser found that setting `T_instability = 36.0` (rather than 29.47) allows more samples into RELIABLE while still satisfying Risk ≤ 10%. The formal selection accounts for the full joint gate behaviour — not just the aug range in isolation.

---

## 6. Where the Gate Lives in the Inference Pipeline

```
fresh_svm.decision_function() → raw margin
       │
normalize → score
       │
Mahalanobis OOD check         ← runs before aug gate
       │
AUGMENTATION GATE  (currently skipped: use_augmentation_gate=False)
  │
  ├── Run 6 augmented views through full sub-pipeline
  │     (EfficientNetB0 × 6 passes)
  │
  ├── score_range = max(aug_scores) − min(aug_scores)
  │   crosses_boundary = min(aug_raws) < 0 AND max(aug_raws) > 0
  │
  ├── unstable = (score_range ≥ 36.0) AND crosses_boundary
  │       → score_unreliable = True → UNRELIABLE
  │
  └── sensitive_only = (score_range ≥ 36.0) AND NOT crosses_boundary
                       AND score_range > 54.0
            → decision_unreliable = True → TENTATIVE
       │
Boundary gate: abs(raw) < T_boundary
       │
Reliability decision: RELIABLE / TENTATIVE / UNRELIABLE
```

When the gate is disabled, `score_range`, `score_std`, `aug_scores`, and `aug_raws` are all set to `0.0 / []` and `unstable = sensitive_only = crosses_boundary = False`. The rest of the pipeline proceeds identically.

---

## 7. Current Disabled Status

The gate is disabled because:

1. **Inference cost.** EfficientNetB0 must run six additional forward passes per image. On CPU (the deployment environment — Intel i7 12th-gen, no GPU), this adds several seconds per prediction, making real-time use impractical.

2. **Gate ablation result.** The evaluation in `evaluate_models.py` runs a partial augmentation gate test on the validation set (40 images per vegetable, 200 total). With the gate enabled:
   - The gate fires on a small fraction of samples
   - `T_instability = 36.0` is a relatively loose threshold — most real images have augmentation ranges well below 36
   - The base model's accuracy on near-boundary samples is already controlled by the OOD gate and centroid gate, which fire without the inference cost

3. **T_boundary = 0.0.** The formal optimiser found no benefit to a margin cutoff. This implies that samples near the decision boundary are not, in practice, less accurate — the OOD and centroid gates are doing the filtering work. The aug gate would be most valuable when those two gates are insufficient, which does not appear to be the case on the current dataset.

To reactivate:

```python
# In scoring_config.json:
"use_augmentation_gate": true

# At inference:
predict(image_path, compute_uncertainty=True)   # default — runs aug gate
predict(image_path, compute_uncertainty=False)  # skips aug gate even when enabled
```

---

## 8. Interaction With Other Gates

The aug gate and the OOD gate are independent:

```
score_unreliable = unstable OR is_ood
```

Both can fire simultaneously. A sample can be OOD (Mahalanobis distance > 30.438) AND unstable (score_range > 36.0 with a boundary flip). Either condition alone produces UNRELIABLE. The gate trigger statistics from `evaluate_models.py` capture this:

```
Gate co-occurrence (test set, aug gate disabled):
  G1_OOD ∩ G2_near_boundary:  0   (T_boundary=0.0, G2 never fires)
  G1_OOD ∩ G3_low_veg_conf:   2
  All three:                   0
```

When the aug gate is enabled, a fourth gate (G4_aug_instability) would appear in this table. Based on the P95 calibration range of 29.47, the gate would fire on approximately 5% of samples. Of those, only the subset with `crosses_boundary = True` would produce UNRELIABLE — the remainder would be `sensitive_only` → TENTATIVE.

---

## 9. Limitations and Assumptions

### Augmentation coverage

The six augmentations were chosen to represent common real-world variation. They are not a comprehensive sample of deployment conditions. Conditions not covered include:

- Brightness changes > 15% (harsh overhead lighting, direct sunlight)
- Camera noise / grain (low-end phone sensors)
- Partial occlusion (vegetable partially out of frame or behind other objects)
- Background clutter (multiple objects in scene)
- JPEG compression artefacts
- Non-uniform lighting (shadows across the vegetable)

If real deployment conditions produce variation outside this range, two failure modes are possible:

| Failure mode | Condition | Result |
|---|---|---|
| False confidence | Real variation exceeds aug severity; aug range looks small | Gate does not fire; system calls RELIABLE on a truly unstable sample |
| Over-triggering | Deployment images are noisier than validation images | Gate fires frequently; RELIABLE rate drops; system becomes over-cautious |

Both failure modes are detectable: monitor the RELIABLE rate and the fraction of samples exceeding T_instability in production logs. A sudden shift in either indicates a domain gap between validation and deployment conditions.

### Normalization dependency

`score_range` is computed on normalised scores (0–100), not raw margins. The normalisation uses per-vegetable p5/p95 bounds. This means score_range reflects **two sources of variation** — the model's actual sensitivity to augmentation, and any variation in which vegetable the model predicts across augmentations (which would change which bounds are applied).

In practice, the vegetable prediction is stable across mild augmentations for well-classified samples. For samples where the vegetable prediction itself is ambiguous, the aug gate and the low-veg-confidence gate would both fire.

`crosses_boundary` is computed on raw margins, not normalised scores, so it is free of this normalisation dependency. The boundary flip is the more reliable signal of the two.

### Sample size for calibration

The aug statistics were computed on 380 samples (up to 100 per vegetable from thr_val). Capsicum contributed 59 and cucumber 52 — both below the 100 target because thr_val had fewer samples for these classes. The P95 estimate of 29.4715 is therefore less stable for these two vegetables. A larger thr_val or more samples per vegetable would produce a more reliable T_instability estimate.

### Gate is deterministic

The six augmentations are fixed transforms with no randomness. Two predictions of the same image will produce identical aug statistics. This is intentional — it makes debugging reproducible — but it means the gate cannot detect instability that only manifests under a different perturbation from the six fixed ones.

---

## 10. Summary

| Property | Value |
|---|---|
| Number of augmented views | 6 |
| Augmentations | ±15% brightness, horizontal flip, Gaussian blur (5×5), ±5° rotation |
| Primary instability metric | `score_range = max(aug_scores) − min(aug_scores)` |
| Secondary metric | `crosses_boundary = min(aug_raws) < 0 AND max(aug_raws) > 0` |
| T_instability | **36.0** (formally selected; not the raw P95) |
| Raw P95 of aug ranges (380 samples) | 29.4715 |
| Calibration data source | thr_val (disjoint from cal_val) |
| `unstable = True` → | UNRELIABLE (score and label withheld) |
| `sensitive_only = True` → | TENTATIVE (score shown, label withheld) |
| Inference cost when active | 6× EfficientNetB0 forward passes per image |
| Current status | **Disabled** (`use_augmentation_gate = False`) |
| Reactivation | Set `use_augmentation_gate: true` in `scoring_config.json` |