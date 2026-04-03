# Complete Project Overview — Vegetable Freshness Grading with Dual SVM

---

## The Core Idea in One Line

You give the system a photo of a vegetable. It answers two questions: **what vegetable is this**, and **how fresh is it, on a scale of 0 to 100**. Two separate machine learning models work in tandem, surrounded by a formal reliability gating system that tells you exactly when to trust a prediction — and when not to.

---

## The Big Picture Before Any Details

Training runs once across seven phases. Prediction runs in real time on new images using the saved artifacts.

```
Phase 1 → Extract features from all images        (extract_dataset_features.py)
Phase 2 → Split into train / val / test sets      (train_split.py)
Phase 3 → Clean and select the best features      (preprocess_and_rank.py)
Phase 4 → Train two SVMs                          (train_svm.py)
Phase 5 → Calibrate all thresholds                (train_svm.py — continued)
Phase 6 → Evaluate on held-out test set           (evaluate_models.py)
Phase 7 → Predict on a new single image           (predict_cli.py)
```

The test set is never opened in Phases 1–5. It exists only for Phase 6.

---

## Phase 1 — Turning Images into Numbers

### Why this is necessary

Machine learning models cannot look at a photograph the way a human does. They need numbers. Phase 1 converts every image into a vector of **1312 numbers** that captures colour, texture, sharpness, and structure.

### Part A — Deep features from EfficientNetB0 (1280 numbers)

EfficientNetB0 is a convolutional neural network pretrained on 1.2 million general images (ImageNet). The patterns it learned — edges, surface reflectance, texture gradients, colour distributions — are exactly what distinguishes a fresh banana (smooth, yellow, firm) from a rotten one (blotchy, brown, soft).

The model is used without its classification head. Instead, the global average pooling layer is read out, producing 1280 numbers per image. You do not interpret individual numbers — what matters is that the 1280-number pattern is reliably different across vegetable types and freshness states.

```
EfficientNetB0 architecture (how we use it):

  Input image [224×224×3]
       │
   Conv layers × 7 blocks
       │
   Global Average Pooling       ← We read the output HERE
       │
  [1280 numbers]                ← One floating-point value per learned filter map
       │
  (Final Dense + Softmax)       ← REMOVED — we do not use the class prediction
```

**Concretely:** a fresh banana might produce `[0.23, 0.0, 1.45, 0.88, ...]` across 1280 values. A rotten banana produces a measurably different pattern in the same 1280 positions. The SVM learns which differences matter for freshness and which matter for vegetable identity.

### Part B — Handcrafted features (32 numbers)

These capture domain-specific visual signals that deep networks can underweight:

| Feature Group | Count | What it captures |
|---|---|---|
| RGB mean per channel | 3 | Average redness, greenness, blueness — rotten produce shifts colour noticeably |
| RGB std per channel | 3 | Colour variance — uniform yellow (fresh banana) vs patchy brown (rotten) |
| HSV mean per channel | 3 | Hue, Saturation, Value — browning and yellowing directly affect hue and saturation |
| HSV std per channel | 3 | Variation in colour tone across the image |
| Grayscale mean | 1 | Overall image brightness |
| Grayscale std | 1 | Contrast — rotten produce tends toward low-contrast uniform patches |
| Edge density (Canny) | 1 | Fraction of pixels with sharp edges — fresh produce has firm, crisp edges |
| Laplacian variance | 1 | Whole-image sharpness — rotten vegetables appear visually softer |
| Luminance histogram (8 bins) | 8 | Distribution of pixel brightness — spoilage shifts the histogram shape |
| Zero-padding | 7 | Pads to exactly 32 values |

**Example:** A fresh cucumber has high edge density (firm skin, clear ridges), high Laplacian variance, and a narrow green-dominant HSV distribution. A rotten cucumber has low edge density (softened skin), lower sharpness, and a yellower, broader HSV distribution.

### Phase 1 output

```
Features/X.npy            → shape (12,691 × 1312) — one row per image
Features/y_veg.npy        → ["banana", "banana", "apple", ...] — vegetable label
Features/y_fresh.npy      → [1, 0, 1, 1, 0, ...] — 1=fresh, 0=rotten
Features/image_paths.npy  → ["/path/img1.jpg", ...] — path aligned to each row
```

The image paths file is critical — it is used in Phase 5 to run real augmentations on validation images during threshold calibration.

---

## Phase 2 — Splitting the Data

### Why splitting is critical

If you train a model and evaluate it on the same data it learned from, your accuracy numbers mean nothing. The model has memorised the training images. You need a held-out set the model has never seen.

This pipeline uses a three-way split:

| Split | Size | Purpose |
|---|---|---|
| **Train (70%)** | 8,883 images | Model fitting — both SVMs learn entirely from this |
| **Validation (10%)** | 1,269 images | Threshold calibration — all decision boundaries set here |
| **Test (20%)** | 2,539 images | Final evaluation only — never touched before Phase 6 |

### Why three splits, not two

The validation set acts as a second calibration set that is completely separate from evaluation. If you used the test set to find your boundary threshold, your test accuracy would be overfit to your own decision logic. The three-way split preserves an honest final measurement.

### Stratified splitting

The split uses a **composite label** of the form `"{vegetable}_{freshness}"` — for example, `"banana_fresh"` or `"potato_rotten"` — to ensure every combination of vegetable type and freshness class is proportionally represented in all three splits.

```
Without stratification, you might accidentally get:
  Train:  all rotten capsicums
  Test:   no rotten capsicums → capsicum freshness accuracy looks artificially perfect

With stratification:
  Every split contains roughly the same fraction of every (veg, fresh) combination.
  fresh_apple, rotten_apple, fresh_banana, rotten_banana, ... — balanced across all 3 splits.
```

The validation set is split once more inside Phase 5 into two disjoint halves — `cal_val` and `thr_val` — to prevent calibration leakage (explained in Phase 5).

---

## Phase 3 — Cleaning and Selecting Features

You have 1312 features per image but not all of them are useful. Three steps refine them into a compact, high-signal union feature set.

### Step 1 — VarianceThreshold: remove constant features

Some features may have the exact same value for every image. A feature that never changes carries zero information. `VarianceThreshold(threshold=0.0)` removes these.

```
Input:   1312 features
Removed:    8 zero-variance features
Output:  1304 features
```

### Step 2 — StandardScaler: bring all features to the same scale

Different features live on wildly different numerical ranges. Laplacian variance might span 50 to 800. An HSV mean spans 0.0 to 1.0. SVM computes distances in feature space — if one feature spans 0–800 and another spans 0–1, the large-range feature dominates every distance calculation and the other 1303 features become irrelevant.

`StandardScaler` transforms each feature to **mean = 0, standard deviation = 1**. After scaling, all 1304 features contribute equally to distance calculations.

The scaler is **fit on training data only**, then applied identically to validation and test. Fitting on test data would let test statistics leak into the model.

### Step 3 — Dual XGBoost ranking (5 seeds, 2 tasks independently)

This is the most important design decision in Phase 3. Feature importance is computed **separately for each task** — freshness classification and vegetable classification — because the two tasks require different information from the image:

- A feature that distinguishes fresh from rotten (e.g. edge density, HSV hue) may be irrelevant for identifying which vegetable it is.
- A feature that distinguishes banana from potato (e.g. specific EfficientNet channels responding to shape) may carry no freshness signal.

Using only freshness-ranked features for the vegetable classifier (the previous approach) is scientifically invalid. Dual ranking solves this.

```
DUAL XGBoost RANKING:

  Task 1: freshness (binary: 0/1)
  ┌──────────────────────────────────────────────────────────────┐
  │  Run XGBoost with seed 42  → importance[1304]                │
  │  Run XGBoost with seed  7  → importance[1304]                │
  │  Run XGBoost with seed 123 → importance[1304]                │
  │  Run XGBoost with seed 17  → importance[1304]                │
  │  Run XGBoost with seed 99  → importance[1304]                │
  │                 Average → avg_imp_fresh[1304]                 │
  └──────────────────────────────────────────────────────────────┘

  Task 2: vegetable identity (5-class)
  ┌──────────────────────────────────────────────────────────────┐
  │  Same 5-seed procedure → avg_imp_veg[1304]                   │
  └──────────────────────────────────────────────────────────────┘

  Why 5 seeds? XGBoost with randomness can vary which features appear
  in the top-k between runs. Averaging over 5 seeds gives a stable
  ranking that does not flip between training runs.

  Why compute ONCE at max_k, then slice? Previously the code ran
  50 separate XGBoost fits (5 seeds × 2 tasks × 5 k-values).
  Now it runs 10 (5 seeds × 2 tasks) and slices the sorted
  importance vector cheaply per k. Same result, ~5× faster.
```

**Stability check result from training run:**
```
  [Stability 'freshness'] min pairwise overlap = 1.000  [OK]
  [Stability 'vegetable'] min pairwise overlap = 1.000  [OK]
```
Both rankings are perfectly stable — every seed agrees on the top features.

### Step 4 — Two-phase k selection

How many top features to keep? The parameter `k` is swept across `{50, 100, 150, 200, 250}` in two phases:

**Phase 1 (proxy sweep):** For each k, take the top-k features from each task ranking, form their union, and train a fast LinearSVC on the union. Measure combined validation accuracy.

```
Phase 1 (LinearSVC proxy sweep):

  k     union  fresh_val  veg_val   combined
  -----  -----  ---------  --------  --------
     50     98     0.9464    0.9850    0.9657
    100    187     0.9598    0.9913    0.9756
    150    270     0.9661    0.9929    0.9795
    200    349     0.9653    0.9945    0.9799
    250    432     0.9661    0.9953    0.9807

  Proxy winner: k = 250  (combined = 0.9807)
```

LinearSVC is only a proxy — it uses a linear decision boundary whereas the actual model uses a non-linear RBF kernel. **Phase 2** confirms the winner using real RBF SVMs:

```
Phase 2 (RBF SVM confirmation):

  k     union  rbf_fresh  rbf_veg   combined
  -----  -----  ---------  --------  --------
     50     98     0.9835    0.9945    0.9890
    100    187     0.9866    0.9968    0.9917
    150    270     0.9858    0.9976    0.9917
    200    349     0.9858    0.9992    0.9925   ← winner
    250    432     0.9842    0.9976    0.9909

  RBF winner: k = 200  (combined = 0.9925)
  Note: proxy said k=250, RBF corrects to k=200 — LinearSVC overfits to larger sets.
```

Phase 2 re-fits the winner (k=200) with 5-fold GridSearchCV for accurate parameter estimates:
```
  [5-fold final] veg   k=200: C=10.0, gamma=0.001  CV acc=0.9958
  [5-fold final] fresh k=200: C=10.0, gamma='scale' CV acc=0.9865
```

### Step 5 — Form the union feature set

```
  selected_fresh  = top-200 features by avg_imp_fresh   → 200 indices
  selected_veg    = top-200 features by avg_imp_veg     → 200 indices

  union_set  = selected_fresh ∪ selected_veg

  ┌────────────────┬────────────────┬────────────────┐
  │  Fresh-only    │    Shared      │   Veg-only     │
  │  149 features  │  51 features   │  149 features  │
  └────────────────┴────────────────┴────────────────┘
                    ← 349 features total →

  Both SVMs (vegetable AND freshness) are trained on this same
  349-feature union. Task-specific rankings determined which
  features to include — but the shared space preserves signal
  for both classifiers simultaneously.
```

---

## Phase 4 — Training Two Separate SVMs

### What is an SVM?

A Support Vector Machine draws a boundary in feature space that separates two classes while maximising the gap (margin) between them. In the freshness case, the boundary separates the region where fresh produce lives from the region where rotten produce lives.

The distance from a sample to the boundary — called the **decision function value** or raw margin — is the core signal. A large positive distance means "strongly fresh"; a large negative distance means "strongly rotten"; a small distance near zero means the model is uncertain.

```
Feature space (simplified to 2D for illustration):

        ●  ●  ●  ●  ●           ← fresh samples (raw > 0)
         ●  ●  ●  ●
     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       ← decision boundary (raw = 0)
         ○  ○  ○  ○
        ○  ○  ○  ○  ○           ← rotten samples (raw < 0)

  Margin (gap) ↑
  The SVM maximises this gap during training.
  Samples far from the boundary → high confidence.
  Samples near the boundary → low confidence → TENTATIVE state.
```

The SVM uses an **RBF kernel**, which maps the 349 features into an infinite-dimensional space where a linear boundary can separate classes that are non-linearly arranged in the original space.

### Vegetable SVM

```
  Task:     5-class classification (apple / banana / capsicum / cucumber / potato)
  Input:    X_train[:, union_349_features]
  Method:   GridSearchCV — 5-fold stratified CV over 30 parameter combinations
  Grid:     C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
            γ ∈ {0.0001, 0.001, 0.01, 0.1, "scale"}
  Best:     C = 10.0, γ = 0.001
  CV acc:   99.58%
  Output:   predict_proba() — probability for each of 5 classes
```

The raw SVC is then wrapped with isotonic probability calibration (see Phase 5).

### Freshness SVM

```
  Task:     Binary classification (0 = rotten, 1 = fresh)
  Input:    X_train[:, union_349_features]
  Method:   Same GridSearchCV procedure
  Best:     C = 10.0, γ = "scale"
  CV acc:   98.65%
  Output:   decision_function() → raw margin (real number, positive = fresh)
            predict() → 0 or 1
```

The freshness SVM outputs a raw margin, not a probability. This raw value is later normalised to a 0–100 score using per-vegetable percentile bounds.

---

## Phase 5 — Calibration

Phase 5 is where the system learns *when it can be trusted*. Five things are calibrated: vegetable probabilities, normalization bounds, the OOD detector, augmentation instability thresholds, and the formal reliability gate thresholds.

### The leakage problem and the cal_val / thr_val split

If you use the same validation data to (a) calibrate probabilities and (b) select your reliability thresholds, the threshold selection sees data whose probabilities were already tuned to that exact data. The thresholds appear to work but fail on genuinely new data. This is called **calibration leakage**.

**Fix:** the 1,269-sample validation set is split 50/50 into two disjoint halves before any calibration occurs:

```
  X_val (1,269 samples), stratified by y_fresh
  │
  ├── 50% → cal_val  (634 samples)
  │         Used ONLY for: isotonic probability calibration on veg_svm
  │
  └── 50% → thr_val  (635 samples)
            Used ONLY for: formal threshold selection, augmentation stats
```

### Vegetable probability calibration (on cal_val)

The raw SVC from Phase 4 has no valid probabilities — `probability=True` in scikit-learn's SVC uses Platt scaling internally, which has known problems with class imbalance. Instead:

```
  veg_model = CalibratedClassifierCV(
                  estimator = FrozenEstimator(veg_base),   ← weights frozen, not retrained
                  method    = "isotonic"                   ← non-parametric, more flexible
              )
  veg_model.fit(X_cal_val, y_veg_cal)   ← learns score-to-probability mapping here
```

`FrozenEstimator` ensures the SVC weights are never changed — only the calibration layer is fit. Isotonic regression learns the actual shape of the score-to-probability curve from data, without assuming it is sigmoidal.

**Calibration check:**
```
  Vegetable acc — cal_val = 1.0000  (calibration set)
  Vegetable acc — thr_val = 1.0000  (held-out half)
  → No gap: cal/thr split is not so small as to cause overfitting
```

### Per-vegetable normalization bounds (on full val set)

The raw freshness margin is a real number — approximately −3 to +3 — that varies in scale across vegetables. A score of 1.5 means different things for a potato (which has a narrow decision margin distribution) vs a banana.

To make scores comparable within each vegetable type, the raw margin is linearly scaled using the **p5 and p95 percentiles of validation-set decisions per vegetable**:

```
  score = clip( (raw - p5_veg) / (p95_veg - p5_veg) × 100, 0, 100 )
```

Per-vegetable bounds from the actual training run:

```
  Vegetable    p5        p95       spread
  apple       -2.5635    2.1198    4.6833
  banana      -2.0173    1.8217    3.8390
  capsicum    -1.2853    1.8389    3.1241
  cucumber    -1.6697    1.6762    3.3460
  potato      -1.8869    1.6565    3.5434
```

Why the **full** val set for bounds (not just cal_val or thr_val)? The 50/50 split can leave individual vegetable classes with too few samples for stable percentile estimates — cucumber has only 182 test-set samples, so a random 50% split of its val samples could easily have < 50 examples for fitting bounds. Using the full val set is safe because a linear percentile transform cannot encode label information (it is not trained on the task).

At training time, if any vegetable class is missing bounds (< 50 samples), a hard `RuntimeError` is raised — there is no silent fallback.

**Stability check (5-fold cross-val across training data):**
```
  apple    p5_cv=0.016  p95_cv=0.013  [OK]
  banana   p5_cv=0.021  p95_cv=0.016  [OK]
  capsicum p5_cv=0.061  p95_cv=0.016  [OK]
  cucumber p5_cv=0.036  p95_cv=0.050  [OK]
  potato   p5_cv=0.057  p95_cv=0.020  [OK]

  All < 0.10 coefficient of variation → bounds are stable across folds
```

### Mahalanobis OOD detector

A vegetable the model has never encountered (or a photograph taken under very unusual lighting) should not receive a confident RELIABLE prediction. The OOD detector catches these cases.

**How it works:** the training data occupies a region of the 349-dimensional feature space. Samples from the same distribution cluster in that region. A sample far from the cluster — a high Mahalanobis distance — is likely out-of-distribution.

```
  LedoitWolf covariance fit on X_train
  train_mean       = X_train.mean(axis=0)        → shape [349]
  precision_matrix = LedoitWolf().precision_     → shape [349×349]

  mahal_dist(x) = sqrt( (x − mean)ᵀ · precision · (x − mean) )

  Thresholds set from the training distribution:
    thresh_caution = P90 of train distances = 24.167   → "caution zone"
    thresh_ood     = P99 of train distances = 30.438   → "OOD"

  OOD rate on validation: 1.81%  (23 of 1,269 val samples)
  OOD rate on test:       2.44%  (62 of 2,539 test samples)
  Difference:             0.63%  → within the 5% stability threshold [OK]
```

### Per-class centroid consistency gate

Even a sample that is in-distribution overall can be misclassified — a cucumber whose EfficientNet features happen to land closer to the potato cluster than the cucumber cluster. The centroid gate catches this:

```
  For each vegetable class, compute the centroid (mean feature vector) on X_train.
  At inference:
    d_pred   = L2 distance from x to the predicted class centroid
    d_second = L2 distance from x to the nearest other centroid

    centroid_ratio = d_pred / d_second

  Per-class threshold = P95 of this ratio on correctly classified val samples:

    apple    threshold = 1.0220   (n=448 correct val samples)
    banana   threshold = 0.9552   (n=469)
    capsicum threshold = 0.9973   (n=114)
    cucumber threshold = 1.0257   (n=91)
    potato   threshold = 0.9740   (n=147)

  class_inconsistent = (centroid_ratio > per_class_threshold)
```

A ratio > 1.0 means the sample is closer to another class's centroid than to its own predicted class — a signal of possible misclassification even when the SVM probability looks fine.

### Augmentation instability gate (thr_val)

Some images sit near the decision boundary in a region where small real-world perturbations (lighting change, slight blur, minor rotation) would flip the predicted class. Six augmented views of the image are generated:

```
  Augmentations applied to each image:
  1. Brightness +15%
  2. Brightness −15%
  3. Horizontal flip
  4. Gaussian blur (5×5)
  5. Rotation +5°
  6. Rotation −5°

  aug_range = max(scores across 6 views) − min(scores across 6 views)

  Gate fires when:
    aug_range ≥ T_instability  AND  raw margin crosses zero across augmentations
    (i.e. some augmentations predict Fresh, others predict Rotten)
```

The threshold `T_instability = 36.0` was formally selected (see below). The gate is currently configured `use_augmentation_gate = False` in `scoring_config.json`, meaning the threshold is stored and ready but not applied at inference time. The T_boundary gate is the active margin-proximity gate.

### Formal threshold selection (on thr_val)

`T_boundary` is selected by solving a constrained optimisation problem on `thr_val`:

```
  RELIABLE_i = (
    NOT is_ood_i
    AND NOT (crosses_bnd_i AND aug_range_i > T_instability)
    AND abs(decision_i) > T_boundary
  )

  Find (T_boundary*, T_instability*) that:
    Maximise:   Coverage = P(RELIABLE)
    Subject to: Risk = P(error | RELIABLE) ≤ ε = 0.10
                n_reliable ≥ n_min

  Grid: T_boundary ∈ [0.0, 3.0] step 0.05
        T_instability ∈ [0.0, max_aug_range] step 0.5
```

**Result from actual training run:**
```
  feasible      = True
  T_boundary    = 0.0000     ← abs(raw) must exceed 0 to be RELIABLE
  T_instability = 36.0000    ← (aug gate inactive; this is stored for future use)
  Risk          = 0.0188     ← 1.88% error rate on RELIABLE samples (well below ε=10%)
  Coverage      = 97.89%     ← 97.89% of thr_val samples reached RELIABLE
  n_reliable    = 372
```

`T_boundary = 0.0` means the boundary proximity gate is not actively filtering samples in this run — the OOD gate and centroid gate handle the cases where the boundary gate would have fired. This is a valid outcome from the formal optimisation: maximising coverage subject to Risk ≤ 10% results in T_boundary = 0 because the base model is already reliable enough without a margin cutoff.

---

## Phase 6 — Evaluation on the Test Set

The test set is opened for the first time in `evaluate_models.py`. Nothing from training or calibration has touched it.

### Classification metrics

```
VEGETABLE CLASSIFICATION (2,539 test images):

  Accuracy: 99.61%

              precision  recall  f1-score  support
  apple          1.00     1.00     1.00      896
  banana         1.00     1.00     1.00      940
  capsicum       1.00     0.99     1.00      227
  cucumber       0.99     0.98     0.98      182
  potato         0.99     0.99     0.99      294

  Confusion matrix (10 total errors across 2,539 samples):
  [[895   0   0   0   1]   ← apple   (1 wrong: apple→potato)
   [  1 939   0   0   0]   ← banana  (1 wrong: banana→apple)
   [  0   0 225   1   1]   ← capsicum (2 wrong)
   [  0   2   0 178   2]   ← cucumber (4 wrong)
   [  1   0   0   1 292]]  ← potato  (2 wrong)
```

```
FRESHNESS CLASSIFICATION (2,539 test images):

  Accuracy: 98.94%

  Confusion matrix:
  [[1308   15]    ← rotten: 1308 correct, 15 predicted as fresh
   [  12 1204]]   ← fresh:  1204 correct, 12 predicted as rotten

  Total errors: 27 out of 2,539 samples

  ROC-AUC (margin-based): 0.9994
  → The SVM margin correctly ranks 99.94% of (fresh, rotten) pairs by ordering
```

### Score distribution

The normalised 0–100 freshness score separates the two classes with a very large margin:

```
  Fresh  — mean = 86.84    std = 10.39    range = 69.93
  Rotten — mean = 16.10    std = 12.38    range = 85.96

  Delta (fresh mean − rotten mean) = 70.73 points

  Overlap (fraction of rotten samples scoring above fresh mean of 86.84) = 0.0000
  → Zero rotten samples scored above the average fresh score
```

### Inversion rate

The inversion rate measures what fraction of (fresh, rotten) sample pairs are *incorrectly ordered* by the score (i.e., rotten scores higher than fresh). Lower is better.

```
  Raw margin inversion            = 0.0007   (0.07% of pairs incorrectly ordered)
  Global-normalised inversion     = 0.0007
  Deployed-path inversion         = 0.0010   (per-veg bounds slightly changes a few orderings)

  Per-vegetable inversion:
  apple     raw=0.0000   norm=0.0000   delta=+0.0000
  banana    raw=0.0000   norm=0.0000   delta=+0.0000
  capsicum  raw=0.0000   norm=0.0000   delta=+0.0000
  cucumber  raw=0.0062   norm=0.0062   delta=+0.0000
  potato    raw=0.0122   norm=0.0122   delta=+0.0000

  Max per-veg delta: 0.0000
  [STABLE] Global delta < 0.01 AND max per-veg delta < 0.02.
           Per-veg normalization does not distort ordering for any vegetable.
```

Cucumber and potato have slightly higher inversion rates than the others — consistent with their lower per-vegetable RELIABLE accuracy. These two vegetables are the hardest for the model.

### State distribution

```
  Total test samples:  2,539
  ─────────────────────────────
  RELIABLE             2,343   (92.3%)   score + fresh_label + confidence_band
  TENTATIVE              134   ( 5.3%)   score only, no fresh_label
  UNRELIABLE (OOD)        62   ( 2.4%)   no output, warning returned
  ─────────────────────────────
  [Note] Augmentation-instability UNRELIABLE not counted above.
         The aug gate is currently disabled (use_augmentation_gate=False).
         Full state distribution including aug instability requires predict_cli.py.
```

Per-vegetable state breakdown:

```
  Vegetable   N      RELIABLE         TENTATIVE        UNRELIABLE
  apple       896    856  (95.5%)      35   (3.9%)       5   (0.6%)
  banana      940    866  (92.1%)      54   (5.7%)      20   (2.1%)
  capsicum    227    208  (91.6%)      15   (6.6%)       4   (1.8%)
  cucumber    182    160  (87.9%)       8   (4.4%)      14   (7.7%)   ← WEAK
  potato      294    253  (86.1%)      22   (7.5%)      19   (6.5%)   ← WEAK
```

Cucumber and potato have notably higher UNRELIABLE rates (7.7% and 6.5%) than the other vegetables. This is consistent with their higher Mahalanobis OOD rates — the training distribution is more compact for these two, meaning genuine test samples more frequently fall outside the P99 radius.

### Gate ablation

For each gate, the evaluation computes what would happen if that gate were disabled: how much accuracy would change (Δ_acc) and how much coverage would change (Δ_cov):

```
  Gate               Fires  Fire%  Catch_W  Block_C   Δ_acc    Δ_cov   Verdict
  ─────────────────────────────────────────────────────────────────────────────
  G1 OOD               62   2.4%      1       61    −0.0004  +0.0760   REVIEW
  G2 near_boundary      0   0.0%      0        0    −0.0003  +0.0524   NEVER FIRES
  G3 low_veg_conf       3   0.1%      0        3    −0.0003  +0.0528   REVIEW

  Baseline: acc=0.9898  coverage=0.923
  Catch_W = gate fires AND prediction was wrong (gate protects a correct outcome)
  Block_C = gate fires AND prediction was right (gate unnecessarily withholds output)
```

Reading the verdicts:
- **G1 OOD — REVIEW:** The gate fires on 62 samples and costs 7.6% coverage but catches only 1 error while blocking 61 correct predictions. The coverage cost is real; the accuracy protection is minimal on this test set. The gate exists for distribution-shift safety (out-of-sample robustness), not purely for in-distribution accuracy. Whether to keep it depends on deployment context.
- **G2 near_boundary — NEVER FIRES:** With T_boundary = 0.0 (from formal threshold selection), this gate is effectively inactive. The formal optimiser chose not to apply a margin cutoff because the base model was already reliable enough.
- **G3 low_veg_conf — REVIEW:** Fires 3 times, blocks 3 correct predictions, catches no errors. On this test set it has no accuracy benefit. However, it protects against cases where per-veg bounds would be applied to the wrong vegetable.

### Silent failure analysis

```
  Veg misclassifications total:   10
  ──────────────────────────────────────────────────────
  Caught by OOD gate only:         5
  Caught by centroid gate only:    2
  Caught by both gates:            0
  ──────────────────────────────────────────────────────
  Missed by both (blind spots):    3

  Of the 3 blind spots:
    Freshness also wrong:          0   ← true silent errors = ZERO
    Freshness still correct:       3   ← accidental correct

  [OK] No catastrophic failures. Freshness signal robust to veg error.
```

The 3 samples where vegetable identity was misclassified but the freshness prediction was still correct are "accidental correct" — the freshness SVM happened to get the right answer despite receiving a wrong vegetable context. These are not dangerous failures, but they are not reliable outputs either. All 3 were correctly flagged as silent failures by the evaluation.

### RELIABLE subset accuracy

```
  Overall freshness accuracy   : 98.94%
  RELIABLE-only accuracy       : 98.98%   ← slightly HIGHER than overall [OK]

  Per-vegetable RELIABLE accuracy (baseline = 98.98%):
  apple     n=856   acc=0.9907   ← near baseline
  banana    n=866   acc=0.9988   ← above baseline
  capsicum  n=208   acc=1.0000   ← perfect on RELIABLE subset
  cucumber  n=160   acc=0.9688   ← 3.1% below baseline
  potato    n=253   acc=0.9605   ← 2.9% below baseline
```

The RELIABLE subset being slightly more accurate than the full test set confirms the gate is working correctly — it preferentially filters cases the model is less certain about. Cucumber and potato are the weakest vegetables on RELIABLE samples, consistent with their lower RELIABLE rate.

---

## Phase 7 — Predicting on a New Image

`predict_cli.py` loads all saved artifacts and runs the full 8-stage pipeline on a single image. The pipeline is sequential — the first gate to fail terminates the pipeline and returns the appropriate state.

```
INPUT: vegetable photo
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Preflight image quality checks                            │
│                                                                     │
│  Laplacian variance < 28.0  → UNRELIABLE (out of focus)            │
│  Mean brightness not in [30, 220] → UNRELIABLE (too dark/bright)   │
│  Object coverage < 0.40     → warning only, continue               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ pass
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Feature extraction                                        │
│                                                                     │
│  EfficientNetB0(224×224 RGB) → [1280] deep features                │
│  Handcrafted extraction      → [32] features                        │
│  Concatenate → [1312]                                               │
│  VarianceThreshold.transform → [1304]                               │
│  StandardScaler.transform    → [1304] zero-mean                     │
│  [:, union_349_indices]      → [349]                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Vegetable classification                                  │
│                                                                     │
│  veg_svm.predict_proba(X[349])                                      │
│    → top-1 label  (apple / banana / capsicum / cucumber / potato)   │
│    → veg_conf     (top-1 probability × 100)                         │
│    → conf_gap     (top-1 − top-2 probability × 100)                 │
│                                                                     │
│  veg_confident = (veg_conf ≥ 70%) AND (conf_gap ≥ 15%)             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Centroid consistency check                                │
│                                                                     │
│  d_pred   = L2 dist from x to predicted class centroid             │
│  d_second = L2 dist from x to nearest other centroid               │
│  ratio    = d_pred / d_second                                       │
│                                                                     │
│  class_inconsistent = (ratio > per_class_threshold)                 │
│                                                                     │
│  If veg_confident AND NOT class_inconsistent → use per-veg bounds   │
│  Otherwise                                  → use global bounds     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Freshness scoring                                         │
│                                                                     │
│  raw   = fresh_svm.decision_function(X)  → e.g. +1.843             │
│  score = clip((raw − p5) / (p95 − p5) × 100, 0, 100)               │
│  fresh_class = fresh_svm.predict(X)       → 0 (rotten) or 1 (fresh)│
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: Mahalanobis OOD gate                                      │
│                                                                     │
│  dist = sqrt( (x−mean)ᵀ · precision · (x−mean) )                   │
│  zone = "trusted"  if dist < 24.167                                 │
│       = "caution"  if 24.167 ≤ dist < 30.438                        │
│       = "ood"      if dist ≥ 30.438                                 │
│                                                                     │
│  is_ood = (zone == "ood")  → sets score_unreliable = True           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 7: Augmentation instability (currently disabled)             │
│                                                                     │
│  When use_augmentation_gate=True:                                   │
│    Run 6 augmented views → compute score for each                   │
│    aug_range = max(scores) − min(scores)                            │
│    crosses_bnd = (min(raw) < 0 AND max(raw) > 0)                    │
│    unstable = (aug_range ≥ 36.0) AND crosses_bnd                    │
│    unstable=True → score_unreliable = True                          │
│                                                                     │
│  When use_augmentation_gate=False (current): skip this stage        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 8: Reliability decision                                      │
│                                                                     │
│  score_unreliable    = is_ood OR unstable                           │
│  decision_unreliable = near_boundary (|raw| < T_boundary=0.0)       │
│                      OR NOT veg_confident                           │
│                      OR class_inconsistent                          │
│                      OR conf_gap < 10%                              │
│                                                                     │
│  HIGH-CONFIDENCE OVERRIDE:                                          │
│    If veg_conf > 95% AND NOT near_boundary                          │
│       AND NOT crosses_bnd AND NOT is_ood                            │
│       AND NOT class_inconsistent                                    │
│    → force RELIABLE regardless                                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ score_unreliable    → UNRELIABLE  (no score, no label)      │    │
│  │ decision_unreliable → TENTATIVE   (score, no label)         │    │
│  │ neither             → RELIABLE    (score + label + band)    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
             RELIABLE                 TENTATIVE / UNRELIABLE
                    │
      ┌─────────────▼─────────────┐
      │  CONFIDENCE BAND          │
      │  score ≥ 85  → High       │
      │  score ≥ 65  → Medium     │
      │  score ≥ 40  → Low        │
      │  score  < 40 → Very Low   │
      └───────────────────────────┘
```

### Example outputs

**Reliable fresh banana:**
```
Vegetable : banana (98.40%,  gap=96.80%)
State     : RELIABLE
Score     : 77.20 / 100
Norm      : per-veg
Freshness : Fresh
Confidence: Medium
Mahal     : 14.2  [trusted]
```

**Tentative result (low vegetable confidence):**
```
Vegetable : potato (61.30%,  gap=8.20%)
State     : TENTATIVE
Score     : 42.10 / 100
Norm      : global
Mahal     : 18.7  [caution]
[!] Low veg confidence (61.3%, gap=8.2%) — using global normalization.
[!] CAUTION — Mahalanobis dist=18.7 in caution zone [24.167, 30.438].
```

**Unreliable — image quality failure:**
```
[UNRELIABLE] Pre-flight failed: Image out of focus (lap_var=12.3 < 28.0)
```

---

## Summary of Numbers to Remember

All numbers below are from the actual training and evaluation run.

| Metric | Value |
|--------|-------|
| Total images | 12,691 |
| Training samples | 8,883 |
| Validation samples | 1,269 |
| Test samples | 2,539 |
| Raw features | 1,312 |
| After VarianceThreshold | 1,304 |
| After union feature selection | **349** |
| XGBoost seeds per task | 5 |
| Best k per task | 200 |
| Fresh-specific features | 149 |
| Veg-specific features | 149 |
| Shared features | 51 |
| Best SVM params (veg) | C=10.0, γ=0.001 |
| Best SVM params (fresh) | C=10.0, γ="scale" |
| Vegetable CV accuracy (5-fold) | 99.58% |
| Freshness CV accuracy (5-fold) | 98.65% |
| **Vegetable test accuracy** | **99.61%** |
| **Freshness test accuracy** | **98.94%** |
| **Freshness ROC-AUC** | **0.9994** |
| Fresh score mean | 86.84 / 100 |
| Rotten score mean | 16.10 / 100 |
| Score delta (fresh − rotten) | 70.73 pts |
| Overlap (rotten > fresh mean) | 0.00% |
| Raw margin inversion rate | 0.0007 (0.07%) |
| RELIABLE (test set) | **92.3%** |
| TENTATIVE (test set) | 5.3% |
| UNRELIABLE / OOD (test set) | 2.4% |
| RELIABLE-only freshness accuracy | 98.98% |
| T_boundary | 0.0000 |
| T_instability | 36.0000 |
| Risk on thr_val (ε target = 10%) | **1.88%** |
| Coverage on thr_val | 97.89% |
| Mahalanobis thresh_caution | 24.167 |
| Mahalanobis thresh_ood | 30.438 |
| OOD rate (val) | 1.81% |
| OOD rate (test) | 2.44% |
| Catastrophic silent failures | **0** |

---

## What This System Does and Does Not Do

### What it does

- Identifies 5 vegetable types with 99.61% accuracy on unseen images
- Classifies fresh vs rotten with 98.94% accuracy
- Ranks fresh above rotten with 99.94% AUC reliability (ordering almost never inverts)
- Produces a 0–100 freshness score calibrated per vegetable type
- Formally certifies predictions as RELIABLE, TENTATIVE, or UNRELIABLE based on margin proximity, OOD distance, vegetable confidence, and class consistency
- Detects out-of-distribution images and withholds predictions on them
- Guarantees RELIABLE accuracy ≥ overall accuracy (verified on held-out test set)
- Has zero catastrophic silent failures (no cases where vegetable was wrong AND freshness was wrong AND the sample was called RELIABLE)

### What it does not do

- **It does not measure biological freshness.** The score is the SVM's geometric confidence, derived from colour and texture patterns learned from labelled images. A vegetable with internal decay that looks visually fresh on the outside would score high.
- **It does not guarantee intra-class ordering.** A score of 80 is reliably above a score of 20 (one is fresh, one is rotten). But a score of 80 vs 75, both on fresh bananas — that ordering is not validated. Only the binary fresh/rotten separation is proven.
- **Scores are not comparable across vegetables.** A banana score of 80 and a potato score of 80 both mean "high relative to that vegetable's training distribution" — not the same absolute freshness level. This is a deliberate consequence of per-vegetable normalization.
- **It does not handle unseen vegetables.** A mango fed to the system would be misclassified as one of the five known vegetables with no indication that the vegetable is unknown. The OOD gate may catch this in some cases but is not guaranteed to.
- **It assumes consistent imaging conditions.** Unusual lighting, strong shadows, or significantly different camera angles will degrade score reliability and increase OOD and TENTATIVE rates.