# Pipeline Flow — Vegetable Freshness Grading with Dual SVM

All numbers are from the actual training run. Outdated values (pre-audit architecture, old thresholds, single-task feature ranking) have been removed.

---

# 1. Feature Extraction

```text
vegetable_Dataset/
        │
        ▼
Scan all subfolders
Parse folder name → (vegetable, freshness)
  "fresh"  prefix + remainder = vegetable name, label = 1
  "rotten" prefix + remainder = vegetable name, label = 0
  e.g. "freshbanana" → ("banana", 1)
       "rottenpotato" → ("potato", 0)
        │
        ▼
Filter to TARGET_VEGETABLES only
  {apple, banana, capsicum, cucumber, potato}
        │
        ▼
Load images (ThreadPoolExecutor, parallel)
  cv2.imread → BGR → RGB
  cv2.resize → 224 × 224
  Failed loads → silently filtered, paths not included
        │
        ▼
────────────────────────────────────────────────────────────────
Feature Extraction per image
────────────────────────────────────────────────────────────────
        │
        ├── EfficientNetB0
        │     weights = "imagenet"
        │     include_top = False
        │     pooling = "avg"     ← GlobalAveragePooling output
        │           │
        │           ▼
        │     preprocess_input(image)   ← EfficientNet normalisation
        │           │
        │           ▼
        │     1280 deep features
        │     (learned texture / colour / shape patterns)
        │
        └── Handcrafted features  (extract_handcrafted_from_array)
                  │
                  ├ RGB mean per channel          [3]  r, g, b average
                  ├ RGB std per channel           [3]  colour variance
                  ├ HSV mean per channel          [3]  hue, saturation, value
                  ├ HSV std per channel           [3]  colour tone spread
                  ├ Grayscale mean                [1]  overall brightness
                  ├ Grayscale std                 [1]  contrast
                  ├ Edge density (Canny 100/200)  [1]  fraction of edge pixels
                  ├ Laplacian variance            [1]  whole-image sharpness
                  └ Luminance histogram (8 bins)  [8]  brightness distribution
                            │
                            ▼  zero-pad to exactly 32
                        32 features
        │
        ▼
Concatenate:  [deep(1280) | handcrafted(32)]
        │
        ▼
1312-dimensional feature vector per image
        │
        ▼
Saved to Features/

  X.npy              shape (12691, 1312)   float32
  y_veg.npy          ["banana", "apple", ...]  string labels
  y_fresh.npy        [1, 0, 1, ...]  int  (1=fresh, 0=rotten)
  image_paths.npy    absolute path aligned to each row
                     ↑ required for real augmentation in Phase 5
```

---

# 2. Dataset Split

```text
Features/
  X.npy  (12691 × 1312)
  y_veg.npy
  y_fresh.npy
  image_paths.npy
        │
        ▼
train_split.py
        │
        ▼
Composite stratification label = "{vegetable}_{freshness}"
  e.g.  "banana_1", "potato_0", "capsicum_1"
  Every (veg, fresh) combination proportionally represented
  in ALL three splits — no class can be missing from any split
        │
        ├─────────────────────┬──────────────────────┬─────────────────────
        │                     │                      │
        ▼                     ▼                      ▼
   TRAIN  70%            VAL  10%              TEST  20%
   8,883 samples         1,269 samples         2,539 samples
        │                     │                      │
  Model fitting          All calibration        Final evaluation
  Both SVMs learn        All thresholds,        ONLY — locked until
  only from here         bounds, OOD            evaluate_models.py
                         detectors set here
        │                     │                      │
  X_train.npy           X_val.npy              X_test.npy
  y_veg_train.npy       y_veg_val.npy          y_veg_test.npy
  y_fresh_train.npy     y_fresh_val.npy        y_fresh_test.npy
  train_image_paths.npy val_image_paths.npy    (test paths not used)

⚠  The val set is split AGAIN inside Phase 5 into:
     cal_val (50%  = 634 samples) — isotonic probability calibration
     thr_val (50%  = 635 samples) — formal threshold selection
   This second split prevents calibration leakage (see Phase 5).

⚠  Test set is NEVER opened before evaluate_models.py.
   Using test data for calibration would overfit thresholds to test — 
   the evaluation would no longer be an honest estimate of deployment accuracy.
```

---

# 3. Feature Preprocessing and Selection

```text
X_train.npy  (8883 × 1312)
        │
        ▼
─────────────────────────────────────────────────────────
STEP A — VarianceThreshold
─────────────────────────────────────────────────────────
Fit on TRAIN only.
Remove any feature with zero variance across all training samples.
A constant feature carries no information — knowing it tells the
model nothing about freshness or vegetable identity.

  Before: 1312 features
  After:  1304 features  (8 zero-variance features removed)

        │
        ▼
─────────────────────────────────────────────────────────
STEP B — StandardScaler
─────────────────────────────────────────────────────────
Fit on TRAIN only. Transform → mean=0, std=1 per feature.

  Why: Laplacian variance spans ~50–800.
       HSV mean spans 0.0–1.0.
  Without scaling, large-range features dominate SVM distance
  calculations and low-range features become irrelevant.

  ⚠  Never fit on val or test — that is data leakage.
     The saved scaler is applied identically to val and test.

  X_train_scaled:  (8883 × 1304), zero-mean, unit-variance

        │
        ▼
─────────────────────────────────────────────────────────
STEP C — Dual XGBoost feature ranking (2 tasks, 5 seeds each)
─────────────────────────────────────────────────────────
Feature importance is computed SEPARATELY for each task.

  FRESHNESS ranking (binary: y_fresh  0/1)
  ┌──────────────────────────────────────────────────────┐
  │  Seed  42 → XGBoost(100 trees) → gain[1304]         │
  │  Seed   7 → XGBoost(100 trees) → gain[1304]         │
  │  Seed 123 → XGBoost(100 trees) → gain[1304]         │
  │  Seed  17 → XGBoost(100 trees) → gain[1304]         │
  │  Seed  99 → XGBoost(100 trees) → gain[1304]         │
  │                  Average → avg_imp_fresh[1304]        │
  └──────────────────────────────────────────────────────┘

  VEGETABLE ranking (5-class: y_veg)
  ┌──────────────────────────────────────────────────────┐
  │  Same 5-seed procedure → avg_imp_veg[1304]           │
  └──────────────────────────────────────────────────────┘

  Why separate rankings?
    A feature discriminating fresh vs rotten (edge density, HSV hue)
    may carry zero signal for identifying which vegetable it is.
    A feature discriminating banana vs potato (EfficientNet shape
    channels) may carry zero freshness signal.
    Using a freshness-only ranking for the vegetable classifier
    is scientifically invalid — this is the dual-ranking fix.

  Why 5 seeds?
    XGBoost with randomness can vary which features appear in
    the top-k between runs. Averaging over 5 seeds produces a
    stable ranking that does not change between training runs.

  Stability check results:
    [Stability 'freshness'] min pairwise seed overlap = 1.000  [OK]
    [Stability 'vegetable'] min pairwise seed overlap = 1.000  [OK]
    → All 5 seeds agree perfectly on the top features for both tasks.

  Speed optimisation: rankings are computed ONCE at max_k (250),
  then sliced cheaply per k candidate. No re-running XGBoost per k.
  Total XGBoost fits: 5 seeds × 2 tasks = 10  (not 50 as before)

        │
        ▼
─────────────────────────────────────────────────────────
STEP D — Two-phase k selection
─────────────────────────────────────────────────────────
k_candidates = {50, 100, 150, 200, 250}

For each k:
  sel_fresh = top-k indices from avg_imp_fresh
  sel_veg   = top-k indices from avg_imp_veg
  union_k   = sel_fresh ∪ sel_veg

PHASE 1 — Proxy sweep (LinearSVC, fast)

  k     union  fresh_val   veg_val  combined
  ----  -----  ---------  --------  --------
    50     98     0.9464    0.9850    0.9657
   100    187     0.9598    0.9913    0.9756
   150    270     0.9661    0.9929    0.9795
   200    349     0.9653    0.9945    0.9799
   250    432     0.9661    0.9953    0.9807

  Proxy winner: k=250  (combined=0.9807)

  ⚠ LinearSVC uses a linear boundary. The actual model uses RBF.
    LinearSVC can favour larger feature sets where RBF does not.
    Phase 2 corrects this.

PHASE 2 — RBF SVM confirmation (3-fold sweep, 5-fold refit)

  k     union  rbf_fresh   rbf_veg  combined
  ----  -----  ---------  --------  --------
    50     98     0.9835    0.9945    0.9890
   100    187     0.9866    0.9968    0.9917
   150    270     0.9858    0.9976    0.9917
   200    349     0.9858    0.9992    0.9925   ← winner
   250    432     0.9842    0.9976    0.9909

  RBF winner: k=200  (combined=0.9925)
  Note: proxy said k=250; RBF corrects to k=200.
        k=250 adds 83 more features that help LinearSVC but add
        noise for the RBF kernel.

  5-fold refit of winner k=200:
    [veg   5-fold] C=10.0, gamma=0.001   CV acc=0.9958
    [fresh 5-fold] C=10.0, gamma='scale' CV acc=0.9865

        │
        ▼
─────────────────────────────────────────────────────────
STEP E — Construct union feature set
─────────────────────────────────────────────────────────

  selected_fresh = top-200 from avg_imp_fresh  → 200 indices
  selected_veg   = top-200 from avg_imp_veg    → 200 indices

  ┌──────────────┬─────────────┬──────────────┐
  │ Fresh-only   │   Shared    │   Veg-only   │
  │  149 feats   │  51 feats   │  149 feats   │
  └──────────────┴─────────────┴──────────────┘
         ←────────── 349 features ──────────→

  union_set = 349 indices into the 1304-column scaled matrix
  Both SVMs are trained on this same 349-feature union.
  Task-specific rankings determined which features to include —
  the shared space preserves full signal for both classifiers.

Saved to models/
  variance.joblib                   fitted VarianceThreshold
  scaler.joblib                     fitted StandardScaler
  selected_union_features.npy       349 column indices
  selected_fresh_features.npy       200 freshness-task indices
  selected_veg_features.npy         200 vegetable-task indices
  feature_importances_fresh.npy     avg_imp_fresh[1304]
  feature_importances_veg.npy       avg_imp_veg[1304]
  feature_selection_report.json     full sweep tables + params
```

---

# 4. Model Training

```text
X_train[:, union_349]  (8883 × 349)
        │
        ▼
─────────────────────────────────────────────────────────────
STEP A — Val set disjoint split  (calibration leakage fix)
─────────────────────────────────────────────────────────────

X_val  (1269 samples)  stratified by y_fresh
        │
        ├── 50%  →  cal_val  (634 samples)
        │           Used ONLY for isotonic probability calibration
        │
        └── 50%  →  thr_val  (635 samples)
                    Used ONLY for threshold selection + aug stats

  Why split the val set?
    If you calibrate probabilities on the same data used for
    threshold selection, the isotonic layer learns to predict
    probabilities that align with the threshold-selection targets.
    Thresholds look tight in validation but fail on new data.
    Disjoint halves break this feedback loop entirely.

─────────────────────────────────────────────────────────────
STEP B — Vegetable classifier (GridSearchCV on X_train)
─────────────────────────────────────────────────────────────

LabelEncoder: apple=0  banana=1  capsicum=2  cucumber=3  potato=4

Base SVC:
  kernel        = "rbf"
  class_weight  = "balanced"     ← handles class-size imbalance
  probability   = False          ← GridSearchCV stage only
  random_state  = 42

GridSearchCV:
  CV            = StratifiedKFold(5 folds, shuffled)
  param grid    = C     ∈ {0.001, 0.01, 0.1, 1, 10, 100}
                  gamma ∈ {0.0001, 0.001, 0.01, 0.1, "scale"}
                  → 30 combinations × 5 folds = 150 fits
  scoring       = "accuracy"
  refit         = True

  Best params:  C=10.0, gamma=0.001   CV acc=0.9958

→ veg_base (best refitted SVC, probability=False)

─────────────────────────────────────────────────────────────
STEP C — Isotonic probability calibration  (on cal_val only)
─────────────────────────────────────────────────────────────

  veg_model = CalibratedClassifierCV(
                  estimator = FrozenEstimator(veg_base),
                  method    = "isotonic"
              )
  veg_model.fit(X_cal_val, y_veg_cal)

  FrozenEstimator: veg_base weights are FROZEN — never updated.
    Only the isotonic calibration layer is fit on cal_val.
    This maps the SVC decision scores to calibrated probabilities
    without changing how the SVC partitions feature space.

  Isotonic vs Platt scaling:
    Platt scaling assumes the score-to-probability curve is
    sigmoidal. Isotonic regression is non-parametric — it learns
    the actual shape of the curve from data. More flexible and
    more accurate on class-imbalanced datasets.

  Calibration quality check:
    Vegetable acc — cal_val = 1.0000
    Vegetable acc — thr_val = 1.0000
    → No gap between halves; split sizes are large enough.

→ veg_svm.joblib   (CalibratedClassifierCV wrapping frozen veg_base)
  Provides: predict_proba() → 5-class probability vector
            predict()       → integer class index

─────────────────────────────────────────────────────────────
STEP D — Freshness classifier  (GridSearchCV on X_train)
─────────────────────────────────────────────────────────────

Same RBF SVC, same GridSearchCV procedure.
  Binary target: y_fresh  (0=rotten, 1=fresh)

  Best params:  C=10.0, gamma='scale'   CV acc=0.9865

  ⚠  probability=False — intentional.
     Platt scaling would internally perturb the decision_function
     values that are the core scoring signal. The raw margin is
     what gets normalised to the 0–100 score. Enabling probability
     would add cost (internal 5-fold refit) with no benefit since
     predict_proba() is never called on the freshness SVM.

  Fresh acc — cal_val = 0.9858
  Fresh acc — thr_val = 0.9858

→ fresh_svm.joblib
  Provides: decision_function() → raw signed distance (real number)
            predict()           → 0 (rotten) or 1 (fresh)

─────────────────────────────────────────────────────────────
STEP E — p5/p95 stability check  (on X_train, 5-fold)
─────────────────────────────────────────────────────────────

Estimates how stable the per-vegetable normalization bounds
would be if training data were slightly different.

  For each vegetable, run 5-fold split on X_train samples of that veg.
  Per fold: compute p5 and p95 of fresh_svm.decision_function().
  Stability = coefficient of variation of p5/p95 across folds.
  Threshold for warning: CV > 0.10

  Results:
    apple    p5_cv=0.016  p95_cv=0.013  [OK]
    banana   p5_cv=0.021  p95_cv=0.016  [OK]
    capsicum p5_cv=0.061  p95_cv=0.016  [OK]
    cucumber p5_cv=0.036  p95_cv=0.050  [OK]
    potato   p5_cv=0.057  p95_cv=0.020  [OK]
  All well below 0.10 → bounds are stable across training splits.

Saved to models/
  veg_svm_base.joblib           raw GridSearchCV winner (pre-calibration)
  veg_svm.joblib                CalibratedClassifierCV (final)
  fresh_svm.joblib              freshness SVC
  label_encoder.joblib          LabelEncoder for vegetable names
```

---

# 5. Per-Vegetable Normalization Bounds

```text
─────────────────────────────────────────────────────────────
Computed on the FULL val set (1,269 samples)
─────────────────────────────────────────────────────────────

Why FULL val set (not cal_val or thr_val)?
  p5/p95 percentiles need enough samples per vegetable class
  for stable estimates. The 50/50 split can leave thin classes
  (cucumber = ~105 val samples) below the 50-sample minimum
  in either half alone. The full val set provides stable bounds.

  p5/p95 is a fixed linear transform — it cannot encode label
  information or leak targets into the model. Using the full
  val set here is safe.

  ⚠  If any vegetable class has < 50 val samples, a hard
     RuntimeError is raised. There is no silent fallback.

val_decisions = fresh_svm.decision_function(X_val[:, union_349])

Global bounds (fallback):
  p5 = -2.2678   p95 = 1.9306

Per-vegetable bounds:

  Vegetable    p5        p95      spread    hard_min   hard_max
  ---------  -------   ------   --------  ---------  ---------
  apple      -2.5635   2.1198    4.6833    -3.2613     2.6114
  banana     -2.0173   1.8217    3.8390    -2.6101     2.5962
  capsicum   -1.2853   1.8389    3.1241    -1.9306     2.0868
  cucumber   -1.6697   1.6762    3.3460    -1.9225     2.0566
  potato     -1.8869   1.6565    3.5434    -2.7631     2.0914

Score normalization formula:
  score = clip( (raw - p5_veg) / (p95_veg - p5_veg) × 100, 0, 100 )

  A score of 0   means raw ≤ p5  (bottom 5% of val distribution)
  A score of 100 means raw ≥ p95 (top 5% of val distribution)
  A score of 50  means raw is exactly at the distribution midpoint

  ⚠  Scores are locally comparable within a vegetable class.
     A banana score of 80 and a potato score of 80 both mean
     "high relative to that vegetable's training distribution" —
     NOT the same absolute freshness level. Cross-vegetable
     score comparison requires global bounds throughout.

Per-veg bounds are ONLY applied when:
  veg_confident = True   (top-1 prob ≥ 70% AND gap ≥ 15%)
  AND class_inconsistent = False  (centroid ratio within threshold)

Otherwise: global bounds are used.

Saved to models/scoring_config.json
  "per_veg_bounds": {vegetable → {p5, p95, hard_min, hard_max}}
  "global_bounds":  {p5, p95, hard_min, hard_max}
```

---

# 6. Mahalanobis OOD Detector

```text
─────────────────────────────────────────────────────────────
Fit on X_train[:, union_349]
─────────────────────────────────────────────────────────────

LedoitWolf covariance estimation:
  Shrinks the sample covariance matrix toward a structured
  estimate. Critical for high-dimensional data (349 features)
  where the sample covariance is noisy or singular.

  train_mean      = X_train.mean(axis=0)        shape [349]
  precision       = LedoitWolf().precision_     shape [349×349]
  (precision = inverse of regularised covariance)

Mahalanobis distance:
  mahal_dist(x) = sqrt( (x − mean)ᵀ · precision · (x − mean) )

  Unlike Euclidean distance, this accounts for feature
  correlations. A sample can be within the value range of all
  features but still be OOD if its feature correlations are
  wrong. Mahalanobis catches this; hard min/max bounds do not.

Thresholds (from training distribution):
  thresh_caution = P90 of mahal distances on X_train = 24.167
  thresh_ood     = P99 of mahal distances on X_train = 30.438

  zone(x) = "trusted"  if dist < 24.167
           = "caution"  if 24.167 ≤ dist < 30.438
           = "ood"      if dist ≥ 30.438

OOD rate measured after calibration:
  Validation: 0.0181  (1.81% of 1,269 val samples flagged OOD)
  Test:       0.0244  (2.44% of 2,539 test samples flagged OOD)
  Difference: 0.63%   → within 5% stability threshold  [OK]

is_ood = (zone == "ood")   → sets score_unreliable = True

Saved to models/
  train_mean.npy          centroid of training distribution
  train_precision.npy     LedoitWolf precision matrix
  scoring_config.json → "mahal_thresh_caution": 24.167
                       → "mahal_thresh_ood":     30.438
                       → "ood_rate_val":          0.0181
```

---

# 7. Augmentation Stability Statistics

```text
─────────────────────────────────────────────────────────────
Computed on thr_val rows ONLY (disjoint from cal_val)
─────────────────────────────────────────────────────────────

Stratified sample from thr_val:
  apple     100 images  (restricted to thr_val indices)
  banana    100 images
  capsicum   59 images  (< 100 available in thr_val)
  cucumber   52 images
  potato     69 images
  ─────────────────────
  Total:    380 images

For each sampled image:
  Load original image from val_image_paths.npy
  Run 6 augmentations:
    1. Brightness × 1.15    (brighter)
    2. Brightness × 0.85    (darker)
    3. cv2.flip(rgb, 1)     (horizontal flip)
    4. GaussianBlur(5×5)    (blur)
    5. Rotation  +5°        (slight clockwise)
    6. Rotation  −5°        (slight counter-clockwise)

  For each augmented view:
    Extract EfficientNetB0 + handcrafted features → [1312]
    Apply same VarianceThreshold → StandardScaler → union_349 slice
    aug_raw = fresh_svm.decision_function(Xf)[0]
    aug_score = normalize(aug_raw, per_veg_bounds)

  Per sample statistics:
    aug_range   = max(aug_scores) − min(aug_scores)
    crosses_bnd = (min(aug_raws) < 0) AND (max(aug_raws) > 0)
    (crosses_bnd = True means augmentations flip fresh ↔ rotten)

  P95 of aug_range across 380 samples:
    unstable_range_thresh = 29.4715  (rounded input; T_instability=36.0)

  ⚠  These stats are computed on thr_val only (not cal_val).
     If aug stats were computed on cal_val, the threshold selection
     would see augmentation statistics that were already "seen"
     by the isotonic calibration — a form of leakage.

Saved to models/scoring_config.json
  "unstable_range_thresh":  36.0   (T_instability from formal selection)
  "use_augmentation_gate":  False  (gate stored but inactive at inference)
```

---

# 8. Formal Threshold Selection

```text
─────────────────────────────────────────────────────────────
Runs on thr_val subset (380 augmented samples from thr_val)
─────────────────────────────────────────────────────────────

RELIABILITY FORMULA (fixed contract — never changed between runs):

  RELIABLE_i = (
    NOT is_ood_i
    AND NOT (crosses_bnd_i AND aug_range_i > T_instability)
    AND abs(decision_i) > T_boundary
  )

OPTIMISATION PROBLEM:

  Find (T_boundary*, T_instability*) that:
    Maximise:   Coverage = P(RELIABLE)
    Subject to: Risk = P(error | RELIABLE) ≤ ε = 0.10
                n_reliable ≥ n_min

  n_min = max(20, len(sub_idx) // 20) = max(20, 380//20) = 20

  Grid:
    T_boundary    ∈ [0.0, 3.0]              step 0.05  → 61 values
    T_instability ∈ [0.0, max_aug_range]    step 0.5   → variable

  For each (T_b, T_i) pair:
    Compute RELIABLE mask using the formula above
    risk     = P(error | RELIABLE)
    coverage = n_reliable / N
    If risk ≤ 0.10 AND n_reliable ≥ n_min:
      update best_feasible if coverage is higher
    If n_reliable ≥ 1 (risk defined):
      update best_min_risk  (used as fallback if no feasible pair)

RESULT (actual run):

  feasible       = True
  T_boundary     = 0.0000
  T_instability  = 36.0000
  Risk           = 0.0188   (1.88% — well below ε=10%)
  Coverage       = 0.9789   (97.89% of thr_val samples reach RELIABLE)
  n_reliable     = 372

  T_boundary = 0.0 interpretation:
    The optimiser found that maximising coverage while keeping
    Risk ≤ 10% requires NO margin cutoff. The base model is
    sufficiently accurate that all samples (including near-boundary
    ones) can be included in RELIABLE without exceeding the risk
    budget. The OOD and centroid gates handle the cases where a
    margin cutoff would have mattered.

  T_instability = 36.0 interpretation:
    The aug instability gate is currently disabled
    (use_augmentation_gate=False). This value is formally selected
    and stored for future activation — it will require T_instability
    to be exceeded AND the raw margin to cross zero for a sample
    to be flagged as unstable.

Infeasibility handling:
  If no (T_b, T_i) pair satisfies Risk ≤ ε AND n_reliable ≥ n_min,
  the infeasible fallback returns the pair with minimum risk.
  diagnose_infeasibility() then prints a sweep table classifying
  the failure as:
    Case (a): Risk flat across margin quantiles (margin has no
              predictive power for errors on this data)
    Case (b): Risk decreases with margin but never reaches ε
              (base model error is the binding constraint)
  For this run: feasible = True, so diagnosis was not triggered.

Saved to models/scoring_config.json
  "boundary_threshold":         0.0
  "unstable_range_thresh":      36.0
  "threshold_selection_result": {feasible, T_boundary, T_instability,
                                 risk, coverage, n_reliable, epsilon,
                                 subset_size, data_source}
```

---

# 9. Per-Class Centroid Consistency Gate

```text
─────────────────────────────────────────────────────────────
Computed on X_train (centroids) and X_val (thresholds)
─────────────────────────────────────────────────────────────

Class centroids (from X_train):
  For each vegetable class i:
    centroids[i] = mean of X_train rows where label == i
    shape: [5 × 349]

Centroid ratio at inference:
  x_flat = Xfinal.flatten()  (shape [349])
  dists  = L2_norm(centroids − x_flat, axis=1)  → 5 distances

  d_pred   = dists[predicted_class_index]
  d_second = min distance to any OTHER class centroid
  ratio    = d_pred / (d_second + 1e-9)

  ratio < 1.0 → sample is closer to its predicted class than
                to any other class (expected for correct predictions)
  ratio > 1.0 → sample is closer to another class's centroid
                than to its own — likely a misclassification

Per-class threshold = P95 of ratio on CORRECTLY classified val samples:

  apple    threshold=1.0220  (n=448 correct val predictions)
  banana   threshold=0.9552  (n=469)
  capsicum threshold=0.9973  (n=114)
  cucumber threshold=1.0257  (n=91)
  potato   threshold=0.9740  (n=147)

  class_inconsistent = (ratio > per_class_threshold)

At inference:
  If class_inconsistent:
    • Per-veg normalization bounds are NOT applied
      (using wrong-veg bounds on a misclassified sample would
       silently produce a miscalibrated score)
    • Global bounds are used instead
    • decision_unreliable = True   →  output is TENTATIVE

Test-set detection breakdown:
  Total veg misclassifications: 10
    Caught by OOD gate only:      5
    Caught by centroid gate only: 2
    Caught by both:               0
    Missed by both (blind spots): 3
      Of blind spots, freshness also wrong: 0  ← zero catastrophic failures

Saved to models/
  class_centroids.npy              shape [5 × 349]
  scoring_config.json → "centroid_ratio_thresholds": {veg → threshold}
```

---

# 10. Inference Pipeline  (predict_cli.py)

```text
New image: image.jpg
        │
        ▼
────────────────────────────────────────────────────────────────────
STAGE 1 — Preflight image quality checks
────────────────────────────────────────────────────────────────────
  Load with cv2.imread → convert to grayscale

  Laplacian variance  (sharpness):
    cv2.Laplacian(gray, CV_64F).var()
    < 28.0  →  UNRELIABLE  "Image out of focus"

  Mean brightness:
    gray.mean()
    < 30.0  →  UNRELIABLE  "Too dark"
    > 220.0 →  UNRELIABLE  "Too bright"

  Object coverage:
    Otsu threshold → largest contour area / (h × w)
    < 0.40  →  WARNING only (not rejected, continues)
               "Low object coverage. Score may be affected."

  ⚠  If UNRELIABLE: return immediately, no further processing.
        │
        ▼  (pass)

────────────────────────────────────────────────────────────────────
STAGE 2 — Feature extraction + preprocessing
────────────────────────────────────────────────────────────────────
  extract_features(image_path)
    → EfficientNetB0(224×224) + handcrafted
    → concatenate → [1312]

  vt.transform([feats])        → [1304]  (VarianceThreshold)
  scaler.transform(X)          → [1304]  (StandardScaler)
  X[:, union_349_indices]      → [349]   (feature selection)

  Xfinal: shape (1, 349)
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 3 — Vegetable classification
────────────────────────────────────────────────────────────────────
  veg_probs   = veg_svm.predict_proba(Xfinal)[0]   → 5 probs
  sorted_probs = veg_probs sorted descending

  veg_idx  = argmax(veg_probs)
  veg_name = le.inverse_transform([veg_idx])[0]
  veg_conf = sorted_probs[0] × 100     (top-1 probability %)
  conf_gap = (sorted_probs[0] − sorted_probs[1]) × 100

  veg_confident = (veg_conf ≥ 70%) AND (conf_gap ≥ 15%)

  If NOT veg_confident:
    → global bounds will be used for normalization
    → decision_unreliable = True  (TENTATIVE)
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 4 — Centroid consistency check  (runs BEFORE bound selection)
────────────────────────────────────────────────────────────────────
  dists_to_centroids = L2_norm(class_centroids − Xfinal, axis=1)
  d_pred   = dists_to_centroids[veg_idx]
  d_second = min of dists_to_centroids excluding veg_idx
  centroid_ratio = d_pred / (d_second + 1e-9)

  centroid_ratio_thresh = per_class_thresholds[veg_name]
  class_inconsistent = (centroid_ratio > centroid_ratio_thresh)

  If class_inconsistent:
    → global bounds are forced regardless of veg_confident
    → decision_unreliable = True  (TENTATIVE)
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 5 — Freshness scoring + bound selection
────────────────────────────────────────────────────────────────────
  raw         = fresh_svm.decision_function(Xfinal)[0]
  fresh_class = fresh_svm.predict(Xfinal)[0]
  fresh_label = "Fresh" if fresh_class == 1 else "Rotten"

  Bound selection:
    use_per_veg = veg_confident AND NOT class_inconsistent
    bounds = per_veg_bounds[veg_name] if use_per_veg else global_bounds
    norm_source = "per-veg" or "global"

  score = clip( (raw − bounds["p5"]) / (bounds["p95"] − bounds["p5"]) × 100, 0, 100 )
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 6 — Mahalanobis OOD gate
────────────────────────────────────────────────────────────────────
  diff = Xfinal.flatten() − train_mean
  dist = sqrt( diffᵀ · train_precision · diff )

  zone:
    dist < 24.167              → "trusted"
    24.167 ≤ dist < 30.438    → "caution"   (warning, no state change)
    dist ≥ 30.438              → "ood"

  is_ood = (zone == "ood")
  If is_ood: score_unreliable = True  → output will be UNRELIABLE
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 7 — Augmentation instability gate  (currently DISABLED)
────────────────────────────────────────────────────────────────────
  use_augmentation_gate = False in scoring_config.json
  → This stage is skipped at inference.
  → T_instability = 36.0 is stored for future activation.

  When enabled, this stage would:
    Run 6 augmented views (±brightness, flip, blur, ±rotation)
    Compute score for each → aug_range = max − min
    crosses_boundary = (min(raws) < 0 AND max(raws) > 0)
    unstable = (aug_range ≥ 36.0) AND crosses_boundary
    If unstable: score_unreliable = True → UNRELIABLE
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 8 — Reliability decision
────────────────────────────────────────────────────────────────────
  near_boundary = abs(raw) < T_boundary  (= abs(raw) < 0.0)
  Currently: T_boundary=0.0 → near_boundary is always False

  decision_unreliable =  near_boundary
                      OR (NOT veg_confident)
                      OR class_inconsistent
                      OR (conf_gap < 10%)

  score_unreliable = is_ood OR unstable (aug gate disabled → always False)

  HIGH-CONFIDENCE OVERRIDE:
    If ALL of the following:
      veg_conf > 95%
      NOT near_boundary
      NOT crosses_boundary
      NOT is_ood
      NOT class_inconsistent
    Then: force RELIABLE regardless of decision_unreliable

  Final state assignment:
    score_unreliable = True      →  UNRELIABLE   (no score, no label)
    decision_unreliable = True   →  TENTATIVE    (score, no label)
    neither = True               →  RELIABLE     (score + label + band)
        │
        ▼

────────────────────────────────────────────────────────────────────
STAGE 9 — Confidence band  (RELIABLE only)
────────────────────────────────────────────────────────────────────
  score ≥ 85  →  "High"
  score ≥ 65  →  "Medium"
  score ≥ 40  →  "Low"
  score  < 40 →  "Very Low"

  ⚠  Band reflects MODEL CERTAINTY in the fresh/rotten classification,
     not biological freshness quality. It is derived from the SVM
     margin distance, not from any nutritional or sensory measurement.
     The fresh_label field is the primary actionable decision.
```

---

# 11. Warning Flags

```text
Warnings are accumulated and returned in the result dict.
Multiple warnings can fire simultaneously.

  CLASS INCONSISTENCY
    Condition:  centroid_ratio > per_class_threshold[veg_name]
    Effect:     forces global bounds, decision_unreliable = True
    Message:    "CLASS INCONSISTENCY — centroid ratio=X.XXX
                 (threshold=X.XXX). Sample not clearly in {veg} cluster.
                 Global normalization bounds applied."

  Low veg confidence
    Condition:  veg_conf < 70%  OR  conf_gap < 15%
    Effect:     forces global bounds, decision_unreliable = True
    Message:    "Low veg confidence (XX.X%, gap=XX.X%) —
                 using global normalization."

  MODEL UNCERTAINTY (near boundary)
    Condition:  abs(raw) < T_boundary  (currently inactive, T_boundary=0.0)
    Effect:     decision_unreliable = True  (TENTATIVE)
    Message:    "MODEL UNCERTAINTY — near decision boundary
                 (|raw|=X.XXXX < X.XXXX). Classifier is unsure."

  TRUE INSTABILITY  (aug gate — currently disabled)
    Condition:  aug_range ≥ 36.0  AND  crosses_boundary = True
    Effect:     score_unreliable = True  (UNRELIABLE)
    Message:    "TRUE INSTABILITY — score range=XX.XX pts AND
                 raw margin crosses zero (min=X.XXX, max=X.XXX).
                 Prediction flips under augmentation."

  INPUT SENSITIVITY  (aug gate — currently disabled)
    Condition:  aug_range ≥ 36.0  AND  NOT crosses_boundary
                AND  aug_range > 54.0  (36.0 × 1.5)
    Effect:     decision_unreliable = True  (TENTATIVE only)
    Message:    "INPUT SENSITIVITY — score range=XX.XX pts.
                 All augmentations stay on same side of boundary."
    Meaning:    Direction is stable (consistently fresh or rotten)
                but score magnitude varies with small image changes.

  OOD
    Condition:  mahal_dist ≥ 30.438
    Effect:     score_unreliable = True  (UNRELIABLE)
    Message:    "OOD — Mahalanobis dist=XX.XXX > threshold=30.438.
                 Outside training distribution."

  CAUTION  (soft OOD zone)
    Condition:  24.167 ≤ mahal_dist < 30.438
    Effect:     warning only, state unchanged
    Message:    "CAUTION — Mahalanobis dist=XX.XXX in caution zone
                 [24.167, 30.438]."

  HIGH-CONFIDENCE OVERRIDE  (suppresses all gates above when fired)
    Condition:  veg_conf > 95% AND not near_boundary AND not
                crosses_boundary AND not is_ood AND not class_inconsistent
    Effect:     decision_unreliable = False, score_unreliable = False
                → forces RELIABLE
    Message:    "HIGH-CONFIDENCE OVERRIDE — veg_conf=XX.X%,
                 raw far from boundary, no class flip. Forced RELIABLE."
```

---

# 12. Final Output

```text
result dict returned by predict()

  Always present:
    "state"                     "RELIABLE" / "TENTATIVE" / "UNRELIABLE"

  If state == "UNRELIABLE" (preflight):
    "reason"                    human-readable failure cause
    "score"                     None
    "raw"                       None
    "fresh_label"               None
    "freshness_confidence_band" None

  If state == "UNRELIABLE" (OOD/instability):
    "veg"                       predicted vegetable name
    "veg_conf"                  top-1 probability %
    "score"                     None
    "raw"                       None
    "fresh_label"               None
    "freshness_confidence_band" None
    "mahal_dist"                float
    "mahal_zone"                "trusted" / "caution" / "ood"
    "warnings"                  [list of warning strings]

  If state == "TENTATIVE":
    "veg"                       predicted vegetable name
    "veg_conf"                  top-1 probability %
    "score"                     float 0–100  (shown but uncertain)
    "score_range"               float  (aug range, 0.0 if gate disabled)
    "raw"                       float  (raw SVM margin)
    "fresh_label"               None   (withheld)
    "freshness_confidence_band" None   (withheld)
    "norm_source"               "per-veg" or "global"
    "mahal_dist"                float
    "mahal_zone"                string
    "warnings"                  [list]

  If state == "RELIABLE":
    "veg"                       predicted vegetable name
    "veg_conf"                  top-1 probability %
    "score"                     float 0–100
    "score_range"               float  (0.0 if aug gate disabled)
    "raw"                       float
    "fresh_label"               "Fresh" or "Rotten"
    "freshness_confidence_band" "High" / "Medium" / "Low" / "Very Low"
    "norm_source"               "per-veg" or "global"
    "mahal_dist"                float
    "mahal_zone"                string
    "warnings"                  [] (empty unless override triggered)
```

**Example — RELIABLE, fresh banana:**
```text
Vegetable : banana (98.40%,  gap=96.80%)
State     : RELIABLE
Score     : 77.20  range=±0.00 / 100
Norm      : per-veg
Freshness : Fresh
Confidence: Medium
Mahal     : 14.20  [trusted]
```

**Example — TENTATIVE (low vegetable confidence):**
```text
Vegetable : potato (61.30%,  gap=8.20%)
State     : TENTATIVE
Score     : 42.10  range=±0.00 / 100
Norm      : global
Mahal     : 18.70  [caution]
[!] Low veg confidence (61.3%, gap=8.2%) — using global normalization.
[!] CAUTION — Mahalanobis dist=18.7 in caution zone [24.167, 30.438].
```

**Example — UNRELIABLE (OOD):**
```text
Vegetable : apple (88.10%,  gap=62.30%)
State     : UNRELIABLE
Mahal     : 32.15  [ood]
[!] OOD — Mahalanobis dist=32.15 > threshold=30.438.
    Outside training distribution.
```

**Example — UNRELIABLE (image quality failure):**
```text
[UNRELIABLE] Pre-flight failed: Image out of focus (lap_var=12.3 < 28.0)
```

---

# 13. Full System — Compact View

```text
Image (any size, JPEG/PNG)
  │
  ▼  resize → 224×224, BGR→RGB
[PREFLIGHT: Laplacian variance / brightness / coverage]
  │ fail → UNRELIABLE immediately
  │ pass ↓
EfficientNetB0(1280) + Handcrafted(32)  =  1312 features
  │
VarianceThreshold  →  1304
StandardScaler     →  1304  (zero-mean, unit-variance)
union_349 slice    →   349  (top-200 fresh ∪ top-200 veg)
  │
  ├──────────────────────────────────────┐
  │                                      │
  ▼                                      ▼
VEGETABLE SVM                       FRESHNESS SVM
CalibratedClassifierCV              SVC(RBF, prob=False)
(isotonic, FrozenEstimator)         C=10.0, gamma='scale'
C=10.0, gamma=0.001                 decision_function()
predict_proba()                     → raw margin (real number)
→ veg_name, veg_conf%, conf_gap%    → fresh_class (0/1)
  │                                      │
  └──────────────┬───────────────────────┘
                 │
CENTROID CONSISTENCY CHECK  (runs before bound selection)
  d_pred / d_second > per_class_threshold?
  class_inconsistent = True  → global bounds, decision_unreliable
                 │
  veg_confident AND NOT class_inconsistent?
     YES               NO
      │                 │
  per-veg bounds   global bounds
  (p5/p95 per veg) (p5=-2.268, p95=1.931)
      └─────────┬──────┘
                │
  score = (raw − p5) / (p95 − p5) × 100   clipped to [0, 100]
                │
MAHALANOBIS OOD
  dist ≥ 30.438 → is_ood=True → score_unreliable=True
  dist ≥ 24.167 → caution warning
                │
AUGMENTATION GATE  (disabled: use_augmentation_gate=False)
  [stored: T_instability=36.0 for future activation]
                │
BOUNDARY GATE
  abs(raw) < T_boundary=0.0  →  near_boundary=True  (currently inactive)
                │
RELIABILITY DECISION
  score_unreliable    →  UNRELIABLE  (no score, no label)
  decision_unreliable →  TENTATIVE   (score, no fresh_label)
  neither             →  RELIABLE    (score + label + band)
                │
CONFIDENCE BAND (RELIABLE only)
  ≥ 85 → High  │  ≥ 65 → Medium  │  ≥ 40 → Low  │  < 40 → Very Low
                │
              OUTPUT
```

---

# 14. What Changed From the Previous Version

```text
PREVIOUS VERSION                      CURRENT VERSION
──────────────────────────────────────────────────────────────────────

Feature selection
  Single freshness-only XGBoost    →  Dual task ranking (freshness +
  ranking, top-100 features            vegetable independently)
  Vegetable SVM reused freshness   →  Union of top-200 per task = 349
  features (scientific flaw)           features (149 fresh-only +
                                        149 veg-only + 51 shared)

k selection
  Fixed at k=100 with a simple     →  Two-phase sweep:
  LinearSVC proxy sweep                Phase 1: LinearSVC proxy
                                        Phase 2: RBF confirmation
                                        Proxy best_k=250 corrected
                                        to RBF best_k=200

XGBoost
  3 seeds, one ranking per run     →  5 seeds per task; rankings
  potentially unstable                 computed once at max_k,
                                        sliced per k (10 fits total)
  Stability: 3 seeds               →  Stability: 5 seeds
                                        min pairwise overlap = 1.000

SVM training
  SVC(probability=True) for veg   →  GridSearchCV(RBF) → veg_base
  C=1.0, gamma='scale'  (default) →  Best: C=10.0, gamma=0.001 (veg)
                                             C=10.0, gamma='scale' (fresh)
  No probability calibration       →  CalibratedClassifierCV(
                                        FrozenEstimator(veg_base),
                                        method="isotonic")
                                        Fit on cal_val only

Val set usage
  Entire val set used for both     →  50/50 disjoint split:
  calibration and thresholds           cal_val → isotonic calibration
  (calibration leakage)                thr_val → threshold selection
                                        Prevents leakage between stages

Normalization bounds
  Computed on training decisions   →  Computed on FULL val set
  (inflated — model saw train)         (reflects unseen-data behaviour)
  Single global bounds only        →  Per-vegetable p5/p95 + global fallback
  No thin-class protection         →  Hard RuntimeError if any veg < 50 samples

OOD detection
  Hard min/max bounds only         →  Mahalanobis (LedoitWolf precision)
  (catches only extreme outliers)      Accounts for feature correlations
  Thresholds: none documented      →  thresh_caution=24.167 (P90 train)
                                        thresh_ood=30.438 (P99 train)

Threshold selection
  Heuristic P95 of aug ranges      →  Formal constrained optimisation:
                                        Maximise Coverage
                                        subject to Risk ≤ ε=0.10
                                        n_reliable ≥ n_min
  T_boundary = 0.05  (arbitrary)   →  T_boundary = 0.0000 (formal)
  T_instability = 32.72            →  T_instability = 36.0000 (formal)

Augmentation gate
  Enabled (score_std ≥ 9.0)       →  Formally calibrated but currently
                                        disabled (use_augmentation_gate=False)
                                        T_instability stored for reactivation

Centroid gate
  None                             →  Per-class P95 thresholds from
                                        correctly classified val samples
  Ran after bound selection        →  Runs BEFORE bound selection
                                        (prevents wrong-veg bounds being
                                        applied to misclassified samples)

Evaluation metrics (test set)
  Veg accuracy: not reported       →  99.61%
  Fresh accuracy: 97.99%          →  98.94%
  ROC-AUC: 0.9979                 →  0.9994
  RELIABLE rate: 92.7%            →  92.3%
  Catastrophic silent failures    →  0  (formally verified)

Output naming
  "grade" key                      →  "freshness_confidence_band"
  "Truly Fresh / Fresh / Moderate  →  "High / Medium / Low / Very Low"
   / Rotten" grade names               (reflects SVM certainty, not
                                        biological quality)
──────────────────────────────────────────────────────────────────────
```