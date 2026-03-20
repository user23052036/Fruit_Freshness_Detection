# 1. Feature Extraction

```text
vegetable_Dataset/
        │
        ▼
Scan folders
(parse "Fresh"/"Rotten" prefix + vegetable name from folder name)
        │
        ▼
Filter only target vegetables
{apple, banana, capsicum, cucumber, potato}
        │
        ▼
Load images (threaded, parallel with ThreadPoolExecutor)
        │
        ▼
Resize → 224×224
Convert BGR → RGB
        │
        ▼
────────────────────────────────────────
Feature Extraction (per image)
────────────────────────────────────────
        │
        ├── EfficientNetB0 (imagenet weights, no top layer)
        │       │
        │       ▼
        │   GlobalAveragePooling
        │       │
        │       ▼
        │   1280 deep features
        │   (texture, shape, colour patterns learned from ImageNet)
        │
        └── Handcrafted features (extract_handcrafted_from_array)
                │
                ├ RGB mean / std          → 6 values
                ├ HSV mean / std          → 6 values
                ├ Grayscale mean / std    → 2 values
                ├ Edge density (Canny)    → 1 value
                ├ Laplacian variance      → 1 value
                └ Luminance histogram     → 8 bins (normalized)
                        │
                        ▼
                    32 features (padded to exactly 32)
        │
        ▼
Concatenate
[deep(1280) | handcrafted(32)]
        │
        ▼
1312 feature vector per image
        │
        ▼
Feature Matrix saved to Features/
        │
        ├ X.npy             # shape (12642, 1312)
        ├ y_veg.npy         # vegetable name strings
        ├ y_fresh.npy       # 1=fresh, 0=rotten
        └ image_paths.npy   # absolute path per row
                            # (needed for real augmentation calibration in Phase 5)
```

---

# 2. Dataset Split

```text
X.npy  (12642, 1312)
y_veg.npy
y_fresh.npy
image_paths.npy
        │
        ▼
train_split.py
        │
        ▼
Stratified Split
combined label = vegetable + freshness
e.g. "banana_1", "potato_0", "capsicum_1"
Ensures balanced distribution across ALL three sets
        │
        ├──────────────────┬──────────────────┬──────────────────
        │                  │                  │
        ▼                  ▼                  ▼
   TRAIN (70%)         VAL (10%)          TEST (20%)
   8883 samples        1269 samples       2539 samples
        │                  │                  │
        │   Model fitting   │  Threshold        │  Final evaluation
        │   (SVMs learn)    │  calibration      │  ONLY — never
        │                   │  (all thresholds  │  touched before
        │                   │  set here)        │  Phase 6
        │
        ├ X_train.npy              ├ X_val.npy              ├ X_test.npy
        ├ y_veg_train.npy          ├ y_veg_val.npy          ├ y_veg_test.npy
        ├ y_fresh_train.npy        ├ y_fresh_val.npy        ├ y_fresh_test.npy
        └ train_image_paths.npy    └ val_image_paths.npy    └ test_image_paths.npy

⚠ Three-way split is required:
  - Using test for calibration would overfit the thresholds to test data.
  - Validation set acts as a separate "threshold fitting set."
  - Test set is only touched in Phase 6.
```

---

# 3. Feature Optimization

```text
X_train.npy  (8883 × 1312)
        │
        ▼
VarianceThreshold
(remove features with zero variance across all training samples)
        │
        ▼
X_train_vt  (8883 × ~1304)
        │
        ▼
StandardScaler
(fit on TRAIN ONLY — normalize to mean=0, std=1)
⚠ Scaler is never fit on val or test — that would be data leakage
        │
        ▼
X_train_scaled  (8883 × ~1304)
        │
        ▼
XGBoost Classifier
(label = FRESHNESS ONLY — y_fresh)
⚠ NOT combined veg+freshness — freshness-only ranking prevents
  vegetable identity from polluting the freshness feature selection
        │
        ▼
Feature Importance Array (gain scores)
[0.033, 0.004, 0.021, 0.0001, ...]
        │
        ▼
Stability check: run ranking across 3 seeds → min overlap ≥ 0.80?
(warns if selection is seed-sensitive and fragile)
        │
        ▼
Diagnostic sweep: k = 50, 100, 150, 200 features
Cross-val accuracy plateaus near k=100 → 100 is justified
        │
        ▼
Pick top 100 indices (canonical seed=42)
        │
        ▼
X_train_final  (8883 × 100)
        │
        ▼
Save preprocessing artifacts to models/
        │
        ├ variance.joblib           # fitted VarianceThreshold
        ├ scaler.joblib             # fitted StandardScaler
        ├ selected_features.npy    # top 100 feature indices
        └ feature_importances.npy  # used for visualization only

Note: Vegetable classifier reuses the same top-100 features
      even though XGBoost ranked on freshness labels.
      Works in practice (98.94% veg accuracy) but is a known
      limitation — a union of separate feature sets would be ideal.
```

---

# 4. Model Training

```text
X_train_final  (8883 × 100)
        │
        ▼
Two Separate Models
```

### Vegetable Classifier (SVM 1)

```text
X_train_final  (8883 × 100)
        │
        ▼
LabelEncoder
(vegetable names → integers)
apple=0, banana=1, capsicum=2, cucumber=3, potato=4
        │
        ▼
RBF SVM
  kernel='rbf'
  C=1.0, gamma='scale'
  probability=True          ← enables predict_proba() for confidence %
  class_weight='balanced'   ← compensates for class imbalance
        │
        ▼
veg_svm.joblib
label_encoder.joblib
```

### Freshness Classifier (SVM 2)

```text
X_train_final  (8883 × 100)
        │
        ▼
RBF SVM
  kernel='rbf'
  C=1.0, gamma='scale'
  probability=False         ← intentionally omitted
  class_weight='balanced'   ← binary: 0=rotten, 1=fresh
        │
  ⚠ probability=True is NOT used here.
    Platt scaling would distort the decision_function margin
    which is the core scoring signal. Enabling it adds cost
    with no benefit since predict_proba() is never called
    on the freshness SVM.
        │
        ▼
fresh_svm.joblib
```

### Post-Training: Calibration on Validation Set (ALL thresholds)

```text
─────────────────────────────────────────────────────────────
STEP A — Compute normalization bounds from VALIDATION decisions
─────────────────────────────────────────────────────────────

fresh_svm.decision_function(X_val_final)
→ raw signed distances from hyperplane for all 1269 val samples
  positive = fresh side
  negative = rotten side

⚠ Bounds anchored to VALIDATION, not training (C2 fix).
  Training decisions are inflated — the SVM was optimised on
  training data so training margins are artificially wider.
  Validation decisions reflect real deployment behaviour.

        │
        ▼
Per-vegetable bounds (from val decisions, split by val veg label)
        │
        ├ apple    → p5=-2.5742  p95=2.1143  hard_min=-3.2166  hard_max=2.9179
        ├ banana   → p5=-2.4324  p95=2.1062  hard_min=-2.9441  hard_max=3.0366
        ├ capsicum → p5=-1.4356  p95=2.1542  hard_min=-2.2829  hard_max=2.3113
        ├ cucumber → p5=-1.7849  p95=1.6414  hard_min=-2.0149  hard_max=1.7207
        └ potato   → p5=-1.8389  p95=1.7821  hard_min=-2.2838  hard_max=2.0282
        │
Global bounds (from all val decisions, fallback)
        └ p5=-2.4208  p95=2.0882

─────────────────────────────────────────────────────────────
STEP B — Boundary threshold calibration (val set)
─────────────────────────────────────────────────────────────

Sweep t from 0.05 → 1.50 (step 0.05)
For each t: find val samples where abs(decision) < t
Check misclassification rate on those near-boundary samples
Find first t where error rate ≥ 10%
→ boundary_threshold = 0.05

⚠ This is a heuristic. It does not formally guarantee
  < 10% error for all predictions above 0.05.
  It is calibrated empirically from the validation set.

─────────────────────────────────────────────────────────────
STEP C — Augmentation instability threshold (real image augmentation)
─────────────────────────────────────────────────────────────

Sample 60 images per vegetable from val_image_paths (stratified)
→ 300 validation images total

For each sampled image:
  Apply 6 augmentations:
    ├ Brightness +15%
    ├ Brightness −15%
    ├ Horizontal flip
    ├ Gaussian blur (5×5)
    ├ Rotation +5°
    └ Rotation −5°
  For each augmentation:
    → run full EfficientNetB0 + handcrafted pipeline
    → apply vt / scaler / selector
    → decision_function → normalize to score

  score_range = max(aug_scores) − min(aug_scores)

Collect score_range across all 300 images
→ instability_threshold = 95th percentile of all ranges
→ instability_threshold = 32.72

─────────────────────────────────────────────────────────────
STEP D — Mahalanobis OOD thresholds (Ledoit-Wolf covariance)
─────────────────────────────────────────────────────────────

Fit Ledoit-Wolf covariance estimator on X_train_final
→ train_mean  (100-element vector)
→ precision   (100×100 precision matrix with shrinkage)

Compute Mahalanobis distance for every training sample
→ thresh_caution = 90th percentile of train distances = 13.102
→ thresh_ood     = 99th percentile of train distances = 16.852

Validation consistency check:
  OOD rate on val  = 1.02%  ← calibration reference
  OOD rate on test = 0.91%  ← must be consistent (no leakage)

Save: train_mean.npy, train_precision.npy

─────────────────────────────────────────────────────────────
STEP E — Per-class centroids and centroid ratio thresholds
─────────────────────────────────────────────────────────────

Compute mean feature vector per vegetable class from X_train_final
→ class_centroids  (5 × 100 array)

For each validation sample:
  Compute distances to all 5 class centroids
  ratio = dist_to_predicted / dist_to_second_closest
  (higher ratio = sample is farther from its predicted class cluster)

Per-class threshold = 95th percentile of ratios for
                      CORRECTLY predicted validation samples only
→ apple    : ratio_threshold = 1.1013  (n=444 correct val preds)
→ banana   : ratio_threshold = 1.0954  (n=469 correct val preds)
→ capsicum : ratio_threshold = 1.0406  (n=113 correct val preds)
→ cucumber : ratio_threshold = 1.0124  (n=90  correct val preds)
→ potato   : ratio_threshold = 1.0085  (n=147 correct val preds)

Save: class_centroids.npy

─────────────────────────────────────────────────────────────
Save scoring_config.json (all calibration values unified)
─────────────────────────────────────────────────────────────
        │
        ├ global_bounds              { p5, p95, hard_min, hard_max }
        ├ per_veg_bounds             { apple, banana, capsicum, cucumber, potato }
        │                            each with p5, p95, hard_min, hard_max
        │                            ⚠ from VALIDATION decisions (not training)
        ├ boundary_threshold         = 0.05      (heuristic, val-calibrated)
        ├ unstable_range_thresh      = 32.72     (real aug P95, val-calibrated)
        ├ veg_confidence_threshold   = 0.70      (design constant)
        ├ veg_gap_threshold          = 0.15      (design constant)
        ├ mahal_thresh_caution       = 13.102    (val-calibrated)
        ├ mahal_thresh_ood           = 16.852    (val-calibrated)
        ├ ood_rate_val               = 0.0102
        ├ centroid_ratio_thresholds  { per-class val-calibrated thresholds }
        ├ grade_thresholds           { truly_fresh:85, fresh:65, moderate:40 }
        │                            (band cutoffs — design constants)
        ├ min_laplacian_variance     = 28.0      (preflight, design constant)
        ├ min_brightness             = 30.0      (preflight, design constant)
        ├ max_brightness             = 220.0     (preflight, design constant)
        ├ min_coverage               = 0.40      (preflight, design constant)
        └ use_augmentation_gate      = False     (aug gate currently disabled)
```

---

# 5. Model Evaluation

```text
X_test.npy  (2539 × 1312)  ← first time test set is used
        │
        ▼
Apply SAVED pipeline (same fitted objects from Phase 3):
variance.transform()
scaler.transform()
X[:, selected_features]
        │
        ▼
X_test_final  (2539 × 100)
        │
        ├────────────────────────────────────────────────────────
        │                                                        │
        ▼                                                        ▼
veg_svm.predict(X_test_final)               fresh_svm.predict(X_test_final)
        │                                                        │
        ▼                                                        ▼
Predicted vegetable labels                  Predicted fresh/rotten labels
        │                                                        │
        ▼                                                        ▼
Accuracy: 0.9894                            Accuracy: 0.9799
Confusion Matrix:                           Confusion Matrix:
[[890   0   0   0   6]                      [[1294   29]
 [  1 939   0   0   0]                        [  22 1194]]
 [  0   1 223   1   2]
 [  0   3   2 171   6]
 [  1   1   1   2 289]]

                                            ROC-AUC: 0.9979
                                            (margin-based, not predict_proba)

────────────────────────────────────────────────────────────────
Score Validation
────────────────────────────────────────────────────────────────

fresh_svm.decision_function(X_test_final)
→ raw distances for all 2539 test samples
        │
        ▼
Normalize using DEPLOYED-PATH scoring (mirrors predict_cli.py exactly)
Per-veg bounds if (veg_confident AND NOT class_inconsistent), else global
        │
        ├── A. Inversion rate (primary ordering metric)
        │       Global-norm inversion  = 0.0022  (0.22% of fresh/rotten pairs wrong)
        │       Deployed-norm inversion = 0.0033  (per-veg gate adds 0.0011 delta)
        │       [STABLE] gate delta < 0.01 — per-veg normalization not distorting order
        │
        ├── B. Score distribution + Delta
        │       Fresh  mean=85.95  std=12.22  range=91.00 points
        │       Rotten mean=17.76  std=13.82  range=82.71 points
        │       Δ = 85.95 − 17.76 = 68.20 points  ← key metric
        │       Overlap = 0.0000 ← zero rotten samples above fresh mean
        │
        ├── C. Intra-class spread (non-collapse check)
        │       Wide range and std confirm score is continuous
        │       (not collapsed to just two values)
        │
        └── D. Per-vegetable summary (deployed-path scores)
                Vegetable    InvNorm    FreshMean  RottenMean    Delta
                apple         0.0003       86.04       19.56    66.48
                banana        0.0000       89.34       14.43    74.90
                capsicum      0.0000       86.89       10.82    76.08
                cucumber      0.0194       82.13       20.63    61.50  ← weaker
                potato        0.0299       76.10       21.86    54.24  ← weaker

────────────────────────────────────────────────────────────────
State Distribution
────────────────────────────────────────────────────────────────

RELIABLE           :  2353  (92.7%)  ← score + label shown
TENTATIVE          :   163  ( 6.4%)  ← score only, label withheld
UNRELIABLE (OOD)   :    23  ( 0.9%)  ← nothing shown

Per-vegetable RELIABLE coverage:
  apple      95.3%  ✓
  banana     94.9%  ✓
  capsicum   93.8%  ✓
  cucumber   82.4%  ← WEAK (15.4% TENTATIVE)
  potato     83.0%  ← WEAK (13.9% TENTATIVE)

RELIABLE-only freshness accuracy : 0.9843
Overall freshness accuracy       : 0.9799
Gate is working correctly — RELIABLE samples are measurably more accurate.

────────────────────────────────────────────────────────────────
Gate Trigger Statistics
────────────────────────────────────────────────────────────────

Gate               Fires   Fire%   Catches Wrong   Verdict
-----------------  -----   -----   -------------   -------
G1_OOD                23    0.9%       1           REVIEW (cost > benefit)
G2_near_boundary      17    0.7%       5           KEEP
G3_low_veg_conf       28    1.1%       5           KEEP

Baseline (all gates active): acc=0.9843  coverage=0.927

────────────────────────────────────────────────────────────────
Silent Failure Analysis
────────────────────────────────────────────────────────────────

Total veg misclassifications       : 27
  Caught by OOD gate               :  4
  Caught by centroid gate          :  4
  Caught by both                   :  0
  Missed by both (blind spots)     : 19
    Of blind spots, fresh also wrong:  2  ← true silent errors
Catastrophic failures (veg+fresh both wrong, RELIABLE state): 0
```

---

# 6. Prediction Pipeline (Single Image)

```text
Input: photo.jpg
        │
        ▼
─────────────────────────────────────
PRE-FLIGHT CHECKS  (before any ML)
─────────────────────────────────────
        │
        ├── Blur check
        │     Laplacian variance < 28.0 → REJECT as unfocused
        │
        ├── Brightness check
        │     mean pixel brightness < 30  → REJECT as too dark
        │     mean pixel brightness > 220 → REJECT as overexposed
        │
        └── Coverage check (WARNING only, not reject)
              largest contour < 40% of frame → warn "low coverage"
              (Otsu thresholding is unreliable on varied backgrounds;
               low coverage often means background wasn't separable,
               not that the vegetable is absent)
        │
        │ any HARD fail → return UNRELIABLE immediately
        │
        ▼
─────────────────────────────────────
FEATURE EXTRACTION
─────────────────────────────────────
Resize → 224×224, BGR→RGB
EfficientNetB0  → 1280 deep features
Handcrafted     →   32 features
Concatenate     → 1312 feature vector
        │
        ▼
─────────────────────────────────────
PREPROCESSING  (must match training exactly)
─────────────────────────────────────
variance.transform()       # remove same constant features
scaler.transform()         # normalize with saved mean/std
X[:, selected_features]    # keep same top-100 indices
        │
        ▼
X_final  (1 × 100)
        │
        ├────────────────────────────────────────
        │                                        │
        ▼                                        ▼
veg_svm.predict_proba(X_final)        fresh_svm.decision_function(X_final)
        │                                        │
        ▼                                        ▼
Class probabilities                    raw signed distance from hyperplane
e.g. banana=96.3% apple=2.1%          e.g. +0.85
                                       positive = fresh side
top1_conf  = 96.3%                     negative = rotten side
top1_name  = "banana"
conf_gap   = 96.3% − 2.1% = 94.2%
```

---

# 7. Centroid Consistency Check  ← NEW (C3 fix — runs BEFORE bounds selection)

```text
X_final  (1 × 100)
class_centroids  (5 × 100)
        │
        ▼
Compute L2 distance from X_final to each of 5 class centroids

distances = [dist_apple, dist_banana, dist_capsicum, dist_cucumber, dist_potato]
        │
        ▼
pred_idx = argmax(veg_probs)   e.g. banana = index 1
d_pred   = distances[pred_idx]  e.g. 2.14  (dist to banana centroid)
d_second = min distance to any OTHER centroid  e.g. 3.81

centroid_ratio = d_pred / d_second
              = 2.14 / 3.81 = 0.56

Compare against per-class threshold (banana: 1.0954)
centroid_ratio = 0.56  < threshold 1.0954
→ class_inconsistent = False  ← sample is clearly in the banana cluster

If centroid_ratio > threshold:
→ class_inconsistent = True
→ override: use global bounds regardless of veg confidence
→ attach CLASS INCONSISTENCY warning

⚠ Why this check runs BEFORE bounds selection:
  Without it, a confident but wrong vegetable prediction would
  apply the wrong vegetable's p5/p95 bounds to the score,
  producing a misleading number with no warning.
  e.g. capsicum misidentified as banana → banana bounds applied
  → capsicum score is completely wrong but appears to be precise.
```

---

# 8. Scoring System

```text
─────────────────────────────────────
Bounds selection (uses centroid result from above)
─────────────────────────────────────

veg_confident = (top1_conf ≥ 70%) AND (conf_gap ≥ 15%)
use_per_veg   = veg_confident AND NOT class_inconsistent

        │
        ├── use_per_veg = True
        │     → per-vegetable bounds
        │     e.g. banana: p5=-2.4324  p95=2.1062
        │     norm_source = "per-veg"
        │
        └── use_per_veg = False
              → global bounds
              → p5=-2.4208  p95=2.0882
              norm_source = "global"

⚠ Per-veg scores are NOT comparable across vegetables.
  banana score 80 ≠ potato score 80 in absolute freshness.
  Both mean "high relative to that vegetable's val distribution."

─────────────────────────────────────
Percentile normalization
─────────────────────────────────────

score = (raw − p5) / (p95 − p5) × 100
score = clip(score, 0, 100)

epsilon guard: if abs(p95 − p5) < 1e-6 → return 50.0

Example (banana, use_per_veg=True):
  raw = +0.85
  p5  = −2.4324
  p95 = +2.1062

  score = (0.85 − (−2.4324)) / (2.1062 − (−2.4324)) × 100
        = 3.2824 / 4.5386 × 100
        = 72.3

─────────────────────────────────────
Confidence Band Assignment
─────────────────────────────────────
```

| Score   | Confidence Band | Meaning |
| ------- | --------------- | ------- |
| ≥ 85    | **High**        | Model strongly in fresh region |
| 65 – 84 | **Medium**      | Model comfortably in fresh region |
| 40 – 64 | **Low**         | Model in ambiguous zone |
| < 40    | **Very Low**    | Model in rotten region |

```text
⚠ These are model confidence bands, not quality grades.
  They describe where the sample sits relative to the
  fresh/rotten decision boundary — not biological freshness
  or shelf life. "High" does not mean the vegetable will
  stay fresh for X days.
```

---

# 9. Mahalanobis OOD Check

```text
X_final  (1 × 100)
train_mean       (100 elements)
train_precision  (100 × 100 Ledoit-Wolf precision matrix)
        │
        ▼
diff = X_final.flatten() − train_mean
dist = sqrt( diff @ precision @ diff )
        │
        ▼
Compare to calibrated thresholds:

dist < 13.102            → zone = "trusted"
13.102 ≤ dist < 16.852   → zone = "caution"  (attach CAUTION warning)
dist ≥ 16.852            → zone = "ood"       (set is_ood = True)

─────────────────────────────────────
OOD rate on test: 0.91% (23 of 2539 samples)
Consistent with val rate of 1.02% — thresholds transfer stably.
─────────────────────────────────────
```

---

# 10. Augmentation Instability (currently disabled)

```text
use_augmentation_gate = False in scoring_config.json
(Gate infrastructure exists and was calibrated; disabled for deployment
 because the 6× EfficientNetB0 passes per image is expensive
 and the P95 threshold of 32.72 means it rarely fires in production)

─────────────────────────────────────
When enabled, the check works as follows:
─────────────────────────────────────

Apply 6 augmentations to the image:
  ├ Brightness +15%
  ├ Brightness −15%
  ├ Horizontal flip
  ├ Gaussian blur (5×5)
  ├ Rotation +5°
  └ Rotation −5°

For each augmented image:
  → full pipeline → decision_function → normalize → aug_score

aug_scores = [s1, s2, s3, s4, s5, s6]
aug_raws   = [r1, r2, r3, r4, r5, r6]  (raw decision values before normalization)

score_range = max(aug_scores) − min(aug_scores)   ← metric is RANGE, not std

─────────────────────────────────────
Two conditions for TRUE INSTABILITY:
─────────────────────────────────────

crosses_boundary = (min(aug_raws) < 0 AND max(aug_raws) > 0)
  → some augmentations say "rotten", others say "fresh"

high_range = score_range ≥ 32.72

TRUE INSTABILITY = high_range AND crosses_boundary
  → set score_unreliable = True  → state becomes UNRELIABLE

sensitive_only = high_range AND NOT crosses_boundary
                 AND score_range > 49.08  (1.5 × threshold)
  → does NOT invalidate score
  → attaches INPUT SENSITIVITY warning
  → prediction direction is consistent even if magnitude varies

⚠ score_std is NOT used — only score_range (max−min).
  Range is more sensitive to extreme augmentation outcomes.
```

---

# 11. Two-Level Reliability Gate

```text
All checks combined into two independent levels:

─────────────────────────────────────
HIGH-CONFIDENCE OVERRIDE (skips both gates)
─────────────────────────────────────
If ALL of:
  veg_conf > 95%
  NOT near_boundary
  NOT crosses_boundary  (aug gate — if enabled)
  NOT is_ood
  NOT class_inconsistent

→ force score_unreliable = False, decision_unreliable = False
→ state = RELIABLE regardless of score_range
→ attach "HIGH-CONFIDENCE OVERRIDE" notice

─────────────────────────────────────
LEVEL 1 — Score validity (UNRELIABLE if fails)
─────────────────────────────────────
score_unreliable = is_ood
                   OR (use_aug_gate AND unstable)
                         where unstable = high_range AND crosses_boundary

If score_unreliable → state = UNRELIABLE
  score = None, fresh_label = None, band = None

─────────────────────────────────────
LEVEL 2 — Decision validity (TENTATIVE if fails)
─────────────────────────────────────
decision_unreliable = near_boundary       (|raw| < 0.05)
                      OR sensitive_only   (high range, consistent direction)
                      OR NOT veg_confident
                      OR class_inconsistent

If decision_unreliable → state = TENTATIVE
  score = valid (shown)
  fresh_label = None (withheld — too uncertain)
  band = None (withheld)

─────────────────────────────────────
Otherwise → state = RELIABLE
  score = valid
  fresh_label = "Fresh" or "Rotten"
  band = confidence band
─────────────────────────────────────
```

---

# 12. Warning Flags

```text
All warnings are independent — multiple can appear simultaneously.

        ├── HIGH-CONFIDENCE OVERRIDE
        │     Condition: veg_conf > 95% AND not near boundary AND not OOD
        │     AND not class_inconsistent AND not crossing boundary
        │     Message: "HIGH-CONFIDENCE OVERRIDE — forced RELIABLE"
        │     Effect: overrides both gates
        │
        ├── CLASS INCONSISTENCY  (C3 fix)
        │     Condition: centroid_ratio > per-class threshold
        │     Message: "CLASS INCONSISTENCY — sample not clearly in
        │               predicted vegetable cluster. Bounds: global."
        │     Effect: forces global bounds, sets decision_unreliable=True
        │
        ├── Low veg confidence
        │     Condition: veg_conf < 70% OR conf_gap < 15%
        │     Message: "Low veg confidence (XX%, gap=XX%) — using global"
        │     Effect: forces global bounds, sets decision_unreliable=True
        │
        ├── MODEL UNCERTAINTY (near boundary)
        │     Condition: abs(raw) < boundary_threshold (0.05)
        │     Message: "MODEL UNCERTAINTY — near decision boundary
        │               (|raw|=0.03 < 0.05)"
        │     Effect: sets decision_unreliable=True (TENTATIVE)
        │     Meaning: a small feature change could flip fresh/rotten
        │
        ├── TRUE INSTABILITY  (aug gate, when enabled)
        │     Condition: score_range ≥ 32.72 AND raw crosses zero
        │     Message: "TRUE INSTABILITY — prediction flips under augmentation"
        │     Effect: sets score_unreliable=True (UNRELIABLE)
        │     Meaning: the model disagrees with itself on different views
        │
        ├── INPUT SENSITIVITY  (aug gate, when enabled)
        │     Condition: score_range ≥ 32.72 AND NOT crossing boundary
        │                AND range > 49.08 (severe)
        │     Message: "INPUT SENSITIVITY — score sensitive to conditions,
        │               prediction direction is consistent"
        │     Effect: sets decision_unreliable=True (TENTATIVE only)
        │     Meaning: direction stable, but magnitude varies with lighting
        │
        ├── OOD  (Mahalanobis)
        │     Condition: mahal_dist ≥ 16.852
        │     Message: "OOD — Mahalanobis dist=XX > threshold=16.852"
        │     Effect: sets score_unreliable=True (UNRELIABLE)
        │     Meaning: image outside training distribution
        │
        └── CAUTION  (soft OOD zone)
              Condition: 13.102 ≤ mahal_dist < 16.852
              Message: "CAUTION — Mahalanobis dist=XX in caution zone"
              Effect: warning only, no state change
              Meaning: unusual but not flagged as OOD
```

---

# 13. Final Output

```text
Image
        │
        ▼
Vegetable type + confidence % + gap %
        │
        ▼
State: RELIABLE / TENTATIVE / UNRELIABLE
        │
        ▼
(if not UNRELIABLE) Score (0–100)
                    Norm source: per-veg or global
(if RELIABLE)       Freshness label: Fresh / Rotten
                    Freshness confidence band: High / Medium / Low / Very Low
(if aug enabled)    Score range: ±XX / 100
        │
        ▼
Mahalanobis distance + zone
        │
        ▼
Warning flags (if any)
```

**Example — normal RELIABLE output:**

```text
Vegetable : banana (96.30%,  gap=94.20%)
State     : RELIABLE
Score     : 72.30 / 100
Norm      : per-veg
Freshness : Fresh
Confidence: Medium
Mahal     : 8.23  [trusted]
```

**Example — TENTATIVE output (near boundary):**

```text
Vegetable : potato (71.20%,  gap=18.40%)
State     : TENTATIVE
Score     : 49.10 / 100
Norm      : per-veg
Mahal     : 11.40  [trusted]

[!] MODEL UNCERTAINTY — near decision boundary
    (|raw|=0.03 < threshold=0.05)
    Classifier is unsure. Fresh_label withheld.
```

**Example — UNRELIABLE output (OOD):**

```text
Vegetable : apple (88.10%,  gap=62.30%)
State     : UNRELIABLE
Mahal     : 18.32  [ood]

[!] OOD — Mahalanobis dist=18.32 > threshold=16.852.
    Outside training distribution.
```

---

# 14. Full System (Ultra-Compact View)

```text
Image
 ↓
[PRE-FLIGHT: blur / brightness / coverage]
 ↓
EfficientNetB0 (1280) + Handcrafted (32)
 ↓
1312 Features
 ↓
VarianceThreshold → ~1304
 ↓
StandardScaler
 ↓
Top 100 Features (XGBoost freshness-ranked)
 ↓
 ┌────────────────────────────────────────────────────────┐
 │                         │                              │
 ▼                         ▼                              │
Vegetable SVM          Freshness SVM                      │
(RBF, probability=T)   (RBF, probability=F)               │
 │                         │                              │
 ▼                         ▼                              │
name, conf%, gap%    decision_function()                  │
                     (raw signed distance)                │
 │                         │                              │
 └──────────────┬───────────┘                             │
                │                                         │
  CENTROID CONSISTENCY CHECK ← (C3 fix — runs first)      │
  dist_pred / dist_second_best vs per-class threshold     │
                │                                         │
  veg_confident AND NOT class_inconsistent?               │
                │                                         │
        ┌───YES─┴──NO────┐                                │
        │                │                                │
  per-veg bounds    global bounds                         │
  (val-anchored)    (val-anchored) ← (C2 fix)             │
        └────────┬────────┘                               │
                 │                                        │
  score = (raw − p5) / (p95 − p5) × 100                  │
                 │                                        │
  MAHALANOBIS OOD CHECK                                   │
  dist > 16.852 → is_ood = True                           │
                 │                                        │
  TWO-LEVEL GATE                                          │
  Level 1: is_ood → UNRELIABLE                            │
  Level 2: near_boundary OR low_conf OR inconsistent      │
           → TENTATIVE                                    │
  Otherwise → RELIABLE                                    │
                 │                                        │
  CONFIDENCE BAND (C1 fix — not a grade)                  │
  score ≥ 85 → High                                       │
  score ≥ 65 → Medium                                     │
  score ≥ 40 → Low                                        │
  score  < 40 → Very Low                                  │
                 │                                        │
       Warning flags ─────────────────────────────────────┘
       (centroid / boundary / sensitivity / OOD / caution)
```

---

# 15. What Changed From Previous Version (Audit Fixes C1–C4)

```text
PREVIOUS VERSION                   →   CURRENT VERSION (post-audit)
───────────────────────────────────────────────────────────────────────

C1 — Output semantics
  Grade names implied quality       →   Confidence band names reflect model certainty
  "Truly Fresh / Fresh / Moderate   →   "High / Medium / Low / Very Low"
   / Rotten"                             (SVM distance, not biological quality)
  Result key: "grade"               →   Result key: "freshness_confidence_band"

C2 — Normalization bounds source
  p5/p95 from TRAINING decisions    →   p5/p95 from VALIDATION decisions
  (inflated — model saw train data) →   (reflects real deployment behaviour)
  Affected fresh/rotten to look     →   Anchors represent what model produces
  artificially moderate in deploy        on unseen data

C3 — Centroid check ordering
  Centroid check ran AFTER bounds   →   Centroid check runs BEFORE bounds
  selection                              selection
  Wrong-veg prediction could apply  →   class_inconsistent flag now gates
  wrong per-veg bounds silently          use_per_veg — wrong veg → global bounds

C4 — Threshold calibration labelling
  Boundary threshold described as   →   Code and config explicitly note:
  calibrated (implied rigorous)          "heuristic (not formally guaranteed)"
                                         threshold_selection.py documents
                                         formal alternative using GateMetrics

───────────────────────────────────────────────────────────────────────

STRUCTURAL CHANGES (pipeline flow)

  2-way split: train/test           →   3-way split: train / val / test
  All calibration on test set       →   All calibration on validation set
                                         (test only touched in evaluation)

  XGBoost label: combined           →   XGBoost label: freshness only
  veg+freshness e.g. "banana_1"         y_fresh (0/1) only — vegetable
                                         identity not mixed into ranking

  OOD: hard min/max bounds          →   OOD: Mahalanobis distance (Ledoit-Wolf)
  (only catches extreme outliers)        (accounts for feature correlations,
                                         detects off-distribution points
                                         within the value range too)

  Instability metric: score_std     →   Instability metric: score_range
  threshold: 9.0                         threshold: 32.72 (P95 of val ranges)
                                         True instability = high range AND
                                         raw decision crosses zero

  No centroid consistency check     →   Per-class centroid ratio thresholds
                                         calibrated on val correct predictions

  Single confidence threshold       →   Dual gate: confidence (70%)
  for per-veg normalization              AND confidence gap (15%)
                                         AND centroid consistency
───────────────────────────────────────────────────────────────────────
```