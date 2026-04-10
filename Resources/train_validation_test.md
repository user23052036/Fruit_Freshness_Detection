# Train, Validation, and Test Sets — Why Each Split Exists

This document explains the three-way data split used in this pipeline, with particular focus on one question that commonly causes confusion: why normalization bounds are computed on the validation set rather than the training set. The answer matters because choosing the wrong data source for bounds directly corrupts the freshness score for every prediction the system makes.

---

## 1. The Three Jobs That Cannot Share Data

The pipeline has three distinct jobs, each requiring a different slice of the dataset:

```
Job 1: Train the SVM          →  needs labelled examples to fit from
Job 2: Calibrate everything   →  needs unseen data to set honest thresholds
Job 3: Evaluate performance   →  needs untouched data to report trustworthy accuracy
```

These three jobs must use **disjoint data**. If any two jobs share data, the result of the third is contaminated. The three-way 70 / 10 / 20 split enforces this:

```
Full dataset (12,691 images)
        │
        ├── TRAIN (70% = 8,883 images)
        │   → Fit VarianceThreshold, StandardScaler
        │   → Train vegetable SVM and freshness SVM
        │   → Compute per-class centroids for centroid gate
        │   → Fit LedoitWolf covariance for Mahalanobis OOD detector
        │
        ├── VALIDATION (10% = 1,269 images)
        │   → Compute p5/p95 normalization bounds (per-vegetable)
        │   → Select T_boundary and T_instability (formal threshold selection)
        │   → Calibrate vegetable probabilities (isotonic, on cal_val half)
        │   → Compute per-class centroid ratio thresholds
        │   → Measure Mahalanobis OOD rate
        │
        └── TEST (20% = 2,539 images)
            → Final accuracy measurement
            → Gate ablation
            → State distribution report
            → NEVER opened before evaluate_models.py
```

The rest of this document focuses on one specific calibration job: **computing p5/p95 normalization bounds**, and why the validation set must be used rather than the training set.

---

## 2. What the Raw Margin Is and Why It Needs Normalisation

The freshness SVM outputs a signed real number for every image — the decision function value, or raw margin:

```python
raw = float(fresh_svm.decision_function(Xfinal)[0])
```

This value is the geometric distance from the sample to the SVM's decision hyperplane. Positive means fresh side; negative means rotten side. Larger magnitude means more confident.

```
Rotten           Boundary          Fresh
  ←──────────────────│──────────────────→
raw = −2.5       raw = 0        raw = +1.8

|raw| = 2.5  →  far from boundary  →  confident rotten
|raw| = 0.3  →  near boundary      →  uncertain
|raw| = 1.8  →  far from boundary  →  confident fresh
```

The raw margin is a real number with no natural [0, 100] scale. Its range depends on the SVM hyperparameters (C, gamma) and the data. To produce a meaningful freshness score between 0 and 100, the raw margin is linearly rescaled using p5 and p95 bounds:

```python
# predict_cli.py

def normalize_score(raw, bounds):
    p5, p95 = bounds["p5"], bounds["p95"]
    denom   = p95 - p5
    if abs(denom) < 1e-6:
        return 50.0
    return float(np.clip((raw - p5) / denom * 100.0, 0.0, 100.0))
```

```
score = clip( (raw − p5) / (p95 − p5) × 100, 0, 100 )

A raw margin equal to p5  → score = 0
A raw margin equal to p95 → score = 100
A raw margin between p5 and p95 → score interpolated linearly
```

The bounds `(p5, p95)` are the 5th and 95th percentiles of the raw margin distribution across a reference dataset. The choice of which dataset to use for computing these percentiles is what this document is about.

---

## 3. Why Training Margins Are Wider Than Validation Margins

An SVM is trained to maximise the margin — the gap between the decision boundary and the nearest training samples (the support vectors). This optimisation process has a direct consequence: training samples that the SVM has seen and fitted on tend to sit farther from the boundary than genuinely new, unseen samples.

This is not a flaw or a coincidence. It is a structural property of how the SVM learns:

- The SVM adjusts the hyperplane specifically to push training samples as far from the boundary as possible (subject to the regularisation parameter C).
- The resulting hyperplane is calibrated to the training samples. New samples that were not part of this optimisation land wherever the hyperplane happens to place them — which is generally closer to the boundary because the hyperplane was not tuned for them.

The result is that raw margins on the training set span a **wider range** than raw margins on unseen data.

---

## 4. A Numerical Example Using This Project's Actual Bounds

The actual per-vegetable bounds computed on the **validation set** from the training run are:

```
  Vegetable    p5        p95       spread
  apple       -2.5635    2.1198    4.6833
  banana      -2.0173    1.8217    3.8390
  capsicum    -1.2853    1.8389    3.1241
  cucumber    -1.6697    1.6762    3.3460
  potato      -1.8869    1.6565    3.5434
```

Now consider what would happen if bounds were computed from training margins instead. Training margins for a well-fitted SVM are systematically wider — a reasonable estimate is that the training spread would be 30–50% wider than validation spread (the exact amount depends on C and gamma). Suppose for banana the training bounds were approximately:

```
  Training bounds (hypothetical):  p5 = −3.1,  p95 = +2.9   spread = 6.0
  Validation bounds (actual):      p5 = −2.017, p95 = +1.822  spread = 3.84
```

Now take a real fresh banana at inference with raw margin = +1.82 (which is at the 95th percentile of the validation distribution — a strongly fresh banana):

```
Normalising with TRAINING bounds:
  score = (+1.82 − (−3.1)) / (2.9 − (−3.1)) × 100
        = (4.92) / (6.0) × 100
        = 82.0

Normalising with VALIDATION bounds:
  score = (+1.82 − (−2.017)) / (1.822 − (−2.017)) × 100
        = (3.837) / (3.839) × 100
        ≈ 100.0
```

The same sample, the same raw margin, the same model — but completely different scores:

| Bounds source | Score | Interpretation |
|---------------|-------|----------------|
| Training | 82 (Medium confidence band) | Appears moderately fresh |
| Validation | 100 (High confidence band) | Correctly registers as maximally fresh |

Now take a rotten banana at inference with raw margin = −1.82 (symmetrically, at the 5th percentile of the validation distribution — strongly rotten):

```
Normalising with TRAINING bounds:
  score = (−1.82 − (−3.1)) / (6.0) × 100
        = (1.28) / (6.0) × 100
        = 21.3

Normalising with VALIDATION bounds:
  score = (−1.82 − (−2.017)) / (3.839) × 100
        = (0.197) / (3.839) × 100
        ≈ 5.1
```

| Bounds source | Score | Interpretation |
|---------------|-------|----------------|
| Training | 21 (Low — barely registers as rotten) | Appears only slightly rotten |
| Validation | 5 (Very Low — correctly very rotten) | Correctly registers as strongly rotten |

The practical consequence: **training bounds compress all scores toward the middle**. Fresh produce looks less fresh than it is. Rotten produce looks less rotten than it is. The score loses its dynamic range and its ability to distinguish between degrees of freshness.

---

## 5. Visualising the Compression Effect

```
RAW MARGIN DISTRIBUTION:

  Training data (SVM fitted on these):
  ← rotten ─────────────────────── fresh →
  ████████████████████████████████████████
  −4.5                  0               +4.5
  │← ─────────── spread ≈ 9.0 ─────────── →│

  Validation data (model has NOT seen these):
  ← rotten ──────────────── fresh →
      ████████████████████████
      −2.0                  +2.0
          │← spread ≈ 4.0 →│


WHAT NORMALISATION DOES:

  Using TRAINING bounds (p5=−4.5, p95=+4.5):
    Scores the validation samples against a reference window of 9.0 units.
    A validation margin of +2.0 sits at position 2/4.5 = 44% from centre.
    → Score ≈ 72 for a strongly fresh sample (should be 100)

  Using VALIDATION bounds (p5=−2.0, p95=+2.0):
    Scores the validation samples against their own natural range of 4.0 units.
    A validation margin of +2.0 sits at the top of this range.
    → Score = 100 for a strongly fresh sample (correct)
```

The validation distribution is the honest reference. It represents the margin range the model produces on data it has not seen — which is exactly the condition under which the model will be used at inference time.

---

## 6. Why the Full Validation Set Is Used (Not Just Half)

The validation set is split 50/50 into `cal_val` (634 samples) and `thr_val` (635 samples) for other calibration steps. However, normalization bounds are computed on the **full** 1,269-sample validation set:

```python
# train_svm.py

print("\n[INFO] Calibrating normalization bounds on full val set...")
val_decisions = fresh_model.decision_function(X_val)   # all 1,269 val samples

global_bounds = {
    "p5"      : float(np.percentile(val_decisions, 5)),
    "p95"     : float(np.percentile(val_decisions, 95)),
    ...
}
per_veg_bounds = compute_per_veg_bounds(
    val_decisions, y_veg_val, veg_classes
)
```

The reason: per-vegetable bounds require enough samples per class to make the 5th and 95th percentiles statistically stable. The five vegetable classes are not equally sized. Cucumber has 182 test-set samples, which corresponds to roughly 91 val samples before the 70/10/20 split. If only `cal_val` (50% of val) were used, cucumber might have only 45 val samples — not enough for a reliable percentile estimate.

Per-vegetable stability was confirmed with a 5-fold cross-validation check on training data:

```
p5/p95 stability check (5-fold CV on X_train):
  apple    p5_cv=0.016  p95_cv=0.013  [OK]
  banana   p5_cv=0.021  p95_cv=0.016  [OK]
  capsicum p5_cv=0.061  p95_cv=0.016  [OK]
  cucumber p5_cv=0.036  p95_cv=0.050  [OK]
  potato   p5_cv=0.057  p95_cv=0.020  [OK]
```

All coefficients of variation are below 0.10, meaning the bounds are stable across different subsets of training data. The full validation set gives even more stable estimates.

Using the full validation set for bounds is safe because a linear percentile transform cannot encode label information. It is not "fitting a model" on validation data — it is measuring a statistical property of the model's output distribution.

---

## 7. What the Bounds Actually Are in This Project

```
Global bounds (all vegetables, full val set):
  p5  = −2.2678
  p95 = +1.9306

Per-vegetable bounds (full val set):
  apple     p5=−2.5635   p95=+2.1198   spread=4.6833
  banana    p5=−2.0173   p95=+1.8217   spread=3.8390
  capsicum  p5=−1.2853   p95=+1.8389   spread=3.1241
  cucumber  p5=−1.6697   p95=+1.6762   spread=3.3460
  potato    p5=−1.8869   p95=+1.6565   spread=3.5434
```

At inference, per-vegetable bounds are applied when the vegetable classifier is confident (top-1 prob ≥ 70%, gap ≥ 15%) AND the centroid gate passes. Global bounds are the fallback. This matters because capsicum's spread (3.124) and banana's spread (3.839) are different — applying capsicum's bounds to a banana or vice versa would produce miscalibrated scores.

The resulting test-set score distribution confirms the bounds are working correctly:

```
Fresh  — mean=86.84   std=10.39   range=69.93
Rotten — mean=16.10   std=12.38   range=85.96

Delta (fresh mean − rotten mean) : 70.73 points
Overlap (rotten scoring > fresh mean): 0.0000
```

Zero rotten samples scored above the average fresh score. The 70.73-point separation between class means is large precisely because the validation bounds correctly represent the model's real-world margin distribution — preserving the score's dynamic range rather than compressing it.

---

## 8. The Three Splits — One Sentence Each

**Train set (70%, 8,883 samples):** The only data the SVM ever sees during fitting; everything trained here is biased toward this data and therefore not suitable for honest calibration or evaluation.

**Validation set (10%, 1,269 samples):** Never used for fitting; used exclusively for calibration of all thresholds, bounds, and gate parameters; represents the model's genuine generalisation performance on unseen data.

**Test set (20%, 2,539 samples):** Never opened until `evaluate_models.py`; provides a completely fresh measurement of accuracy, score distribution, state distribution, and gate behaviour that is free from any calibration decisions made on validation data.

---

## 9. The Key Principle to Remember

The p5/p95 normalization bounds answer one question:

> *"What range of raw margins should I expect from images the model has never seen?"*

Only the validation set can answer this honestly. The training set answers a different question: *"What range of raw margins does the model produce on the data it was optimised for?"* — which is systematically wider and therefore the wrong reference for real-world score calibration.