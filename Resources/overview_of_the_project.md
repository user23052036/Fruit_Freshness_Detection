# Complete Project Overview — Vegetable Freshness Grading with Dual SVM

---

## The Core Idea in One Line

You give the system a photo of a vegetable. It answers two questions: **what vegetable is this**, and **how confident is the model that it is fresh, on a scale of 0 to 100**. It uses two separate machine learning models working together, surrounded by safety checks to tell you when it is uncertain.

---

## The Big Picture Before Any Details

The system is built in seven phases that run in order. You only run each phase once (or again if you retrain). After training is complete, the predict script runs in real time on new images.

```
Phase 1 → Extract features from all images        (extract_dataset_features.py)
Phase 2 → Split into train / val / test sets      (train_split.py)
Phase 3 → Clean and select the best features      (preprocess_and_rank.py)
Phase 4 → Train two SVMs                          (train_svm.py)
Phase 5 → Calibrate all thresholds                (train_svm.py — continued)
Phase 6 → Evaluate on held-out test set           (evaluate_models.py)
Phase 7 → Predict on a new single image           (predict_cli.py)
```

The test set is never touched in Phases 1–5. It exists only for Phase 6.

---

## Phase 1 — Turning Images into Numbers

### Why this is necessary

Machine learning models cannot look at a photograph the way a human does. They need numbers. Phase 1 converts every image into a list of 1312 numbers that captures what the image looks like — its colours, textures, sharpness, and structure.

### Part A — Deep features from EfficientNetB0 (1280 numbers)

EfficientNetB0 is a deep neural network originally trained on 1.2 million general images (ImageNet). The key insight is that the patterns it learned — edges, textures, colour gradients, surface reflectance — are exactly the patterns that distinguish a fresh banana (smooth, yellow, firm) from a rotten one (blotchy, brown, soft).

When you pass your vegetable photo through EfficientNetB0, you remove the final classification layer and instead read out the 1280 numbers just before it. Each number loosely represents something the network "noticed" — but you do not control or interpret individual numbers. What matters is that the 1280-number pattern is reliably different between fresh and rotten vegetables, and between different vegetable types.

**Concretely:** a fresh banana fed through EfficientNetB0 might produce values like `[0.23, 0.0, 1.45, 0.0, 0.88, ...]` (1280 values). A rotten banana produces a different pattern. The model learns which differences matter.

### Part B — Handcrafted features you compute manually (32 numbers)

These capture domain-specific signals that deep networks sometimes miss:

| Feature | Numbers | What it captures |
|---------|---------|-----------------|
| RGB mean per channel | 3 | Average redness, greenness, blueness — rotten produce shifts colour noticeably |
| RGB std per channel | 3 | Colour variance — uniform yellow (fresh banana) vs patchy brown (rotten) |
| HSV mean per channel | 3 | Hue, Saturation, Value — browning and yellowing directly affect hue and saturation |
| HSV std per channel | 3 | Variation in colour tone across the image |
| Grayscale mean | 1 | Overall brightness |
| Grayscale std | 1 | Contrast |
| Edge density | 1 | Fraction of pixels with sharp edges — fresh produce has crisp, firm edges; rotten produce looks soft and blurry |
| Laplacian variance | 1 | Sharpness of the whole image — rotten vegetables appear visually softer |
| Luminance histogram (8 bins) | 8 | Distribution of pixel brightness — discolouration and spoilage shift the histogram shape |
| Padding to exact 32 | 8 | Zeros |

**Example:** A fresh cucumber has high edge density (firm skin, clear ridges), high Laplacian variance (sharp), and a narrow green-dominant HSV distribution. A rotten cucumber has low edge density (softened), lower sharpness, and a yellower, wider HSV distribution.

### The output of Phase 1

For every image you produce one row of 1312 numbers. Three files are saved:

```
Features/X.npy          → shape (12642, 1312) — one row per image
Features/y_veg.npy      → ["banana", "banana", "apple", ...] — vegetable label
Features/y_fresh.npy    → [1, 0, 1, 1, 0, ...] — 1=fresh, 0=rotten
Features/image_paths.npy → ["/path/img1.jpg", ...] — path per row
```

The image paths file is important — it gets used in Phase 5 to run real augmentations during calibration.

---

## Phase 2 — Splitting the Data

### Why splitting is critical

If you train a model and evaluate it on the same data it learned from, your accuracy numbers are meaningless — the model has memorised the training images. You need a held-out set the model has never seen.

This pipeline uses a **three-way split**:

| Split | Size | Purpose |
|-------|------|---------|
| Train (70%) | 8,883 images | Model fitting — both SVMs learn from this |
| Validation (10%) | 1,269 images | Threshold calibration — used after training to set all decision boundaries |
| Test (20%) | 2,539 images | Final evaluation only — never touched before Phase 6 |

### Why three splits, not two

The validation split exists to calibrate thresholds without contaminating the test set. If you used the test set to find your boundary threshold, your test evaluation would be overfit. The validation set acts as a "second training set for the decision logic" — distinct from both training and testing.

### Stratified splitting

The split uses **stratified sampling**: every combination of vegetable type and freshness class is proportionally represented in all three splits. Without this, you might accidentally put all rotten capsicums into training with none in test, making evaluation unreliable.

```
Every split contains roughly the same fraction of:
  fresh apples, rotten apples,
  fresh bananas, rotten bananas,
  fresh capsicums, rotten capsicums,
  ... and so on for cucumber and potato.
```

---

## Phase 3 — Cleaning and Selecting Features

You now have 1312 features per image but not all are useful. Three steps reduce and refine them.

### Step 1 — VarianceThreshold: remove constant features

Some features may have the same value for every single image. A feature that never changes carries zero information — knowing it is always 0.0 tells you nothing about whether the produce is fresh or rotten.

```
Before: 1312 features
After:  1304 features  (8 constant features removed)
```

### Step 2 — StandardScaler: bring features to the same scale

Different features have wildly different numerical ranges. Laplacian variance might range from 50 to 800. An HSV mean might range from 0.0 to 1.0. SVM uses distances between points in feature space — if one feature spans 0–800 and another spans 0–1, the large-range feature completely dominates all distance calculations. The other features become irrelevant.

StandardScaler transforms every feature to **mean = 0, standard deviation = 1**. After scaling, all 1304 features contribute equally to distance calculations.

**Important:** the scaler is **fit on training data only**, then applied to validation and test. You must not fit the scaler on test data — that would let test statistics leak into your model.

### Step 3 — XGBoost ranking: keep the top 100

You train an XGBoost gradient boosted classifier on all 1304 features and ask it: "which features contribute most to correctly classifying freshness?" XGBoost internally measures how much each feature improves split quality (called "gain"). You take the 100 highest-gain features.

Why 100? A sweep across k = 50, 100, 150, 200 shows accuracy stabilises near 100 — adding more features beyond this point adds noise rather than signal.

**From now on, every image is represented by exactly 100 numbers** — the 100 most predictive ones from the original 1312.

One known limitation: the top-100 are selected specifically for freshness prediction. The vegetable classifier reuses the same 100 features even though it has a different task. In practice this works well (98.94% vegetable accuracy) but a theoretically correct approach would select separate feature sets for each task.

---

## Phase 4 — Training Two Separate SVMs

Two completely independent models are trained on the same 100-feature vectors.

### Why two models, not one?

A single model that predicts both vegetable type and freshness simultaneously would mix the two tasks. Errors in vegetable identification would corrupt freshness prediction. Two separate models keep the tasks cleanly separated and allow independent diagnosis of each.

### Model 1 — Vegetable Classifier

- **Input:** 8,883 rows × 100 features (training set)
- **Target:** which of the 5 vegetable types (apple, banana, capsicum, cucumber, potato)
- **Algorithm:** RBF kernel SVM with `probability=True` and `class_weight=balanced`

The SVM finds a decision boundary in 100-dimensional feature space that separates the 5 classes. RBF kernel allows the boundary to be curved and complex — necessary because fresh apples and fresh capsicums, for example, occupy very different regions.

`probability=True` enables the SVM to return a confidence probability for each class using Platt scaling. This is needed to compute:
- **Top-1 confidence** — how certain the model is about its prediction (e.g. 96.3% banana)
- **Confidence gap** — difference between first and second choice (e.g. 96.3% banana, 2.1% apple → gap = 94.2%)

Both metrics gate whether per-vegetable normalisation is used in Phase 5.

**From the terminal:**
```
Accuracy: 0.9894  (98.94% on 2,539 test images)
```

### Model 2 — Freshness Classifier

- **Input:** same 8,883 rows × 100 features
- **Target:** binary — 0 = rotten, 1 = fresh
- **Algorithm:** RBF kernel SVM, **no** `probability=True`

`probability=True` is deliberately omitted. The freshness model does not use predicted probabilities — it uses `decision_function`, which is fundamentally different. Enabling Platt scaling would add training cost while implying probability semantics that do not apply here.

### Understanding decision_function

This is the most important concept in the whole system.

When the SVM is trained, it creates a **hyperplane** (a flat surface in 100-dimensional space) that separates fresh samples on one side from rotten samples on the other. Every point in the feature space has a signed distance from this boundary:

```
decision_function output → meaning

  +3.1   Very clearly fresh. Far from boundary on the fresh side.
  +0.6   Fresh, but not very far from the boundary. More uncertain.
  +0.05  Fresh, but right near the boundary. Very uncertain.
   0.0   Exactly on the boundary. No opinion.
  -0.1   Rotten, but right near the boundary. Very uncertain.
  -1.8   Rotten. Clear prediction.
  -3.2   Very clearly rotten. Far from boundary on the rotten side.
```

This signed distance is what gets converted into your 0–100 freshness score. A large positive value means "the model is very confident this is fresh." A large negative value means "the model is very confident this is rotten." Values near zero mean "the model is unsure."

**From the terminal:**
```
Accuracy: 0.9799  (97.99% on 2,539 test images)
ROC-AUC:  0.9979  (ordering reliability — fresh scores above rotten 99.79% of the time)
```

---

## Phase 5 — Calibration on the Validation Set

After training, everything is calibrated using the validation set. The test set is still untouched.

### Normalization bounds: converting raw distances to 0–100

A raw decision_function value of +1.8 is meaningless without context. Does that mean "very fresh" or "just barely fresh"? You need to know the typical range for that vegetable.

The pipeline computes **p5 and p95** (5th and 95th percentile) of the validation-set decision function values for each vegetable separately. These become the normalization anchors:

```
score = clip( (raw - p5) / (p95 - p5) × 100, 0, 100 )
```

**Example with banana:**
```
Banana validation bounds: p5 = -2.43,  p95 = +2.11
If raw = +0.85:
  score = (+0.85 - (-2.43)) / (2.11 - (-2.43)) × 100
        = 3.28 / 4.54 × 100
        = 72.2
```

A score of 72 means "this banana is in the upper portion of the fresh distribution from the validation set."

**Why validation bounds, not training bounds?**

This is a critical design decision (fix C2 in the audit). Training-set decisions are inflated — the model was optimised on training data, so its margins on training images are artificially larger than on new images. Using training bounds as anchors would compress all deployment scores toward the middle, making fresh things look less fresh and rotten things look less rotten. Validation bounds reflect what the model actually produces on unseen data.

**From the terminal — validation bounds used:**
```
apple:    p5=-2.5742  p95=2.1143
banana:   p5=-2.4324  p95=2.1062
capsicum: p5=-1.4356  p95=2.1542
cucumber: p5=-1.7849  p95=1.6414
potato:   p5=-1.8389  p95=1.7821
Global:   p5=-2.4208  p95=2.0882
```

Notice capsicum has a narrower spread than apple or banana. This is because capsicum's fresh/rotten decision function values cluster in a smaller range — its features are already highly separable without needing a wide margin. Cucumber and potato have tighter bounds too, which is why their scores tend to be harder to interpret.

### When per-vegetable bounds are used vs global bounds

Per-vegetable bounds are only used when the system is confident it has identified the vegetable correctly. Two conditions must **both** hold:

1. **Vegetable confidence ≥ 70%** — the top-1 probability from the vegetable SVM is at least 70%
2. **Confidence gap ≥ 15%** — the gap between first and second prediction is at least 15 percentage points
3. **Centroid consistency** (new in C3 fix) — the sample must lie close to the predicted vegetable's cluster in feature space

If any condition fails, global bounds are used. This prevents a confidently wrong vegetable prediction from applying the wrong vegetable's normalization (e.g. capsicum bounds applied to a cucumber, which would wildly distort the score).

### Boundary threshold: detecting uncertain predictions

You sweep the threshold `t` from 0.05 to 1.5. For each value, you find all validation samples where `|decision_function| < t` (close to the boundary) and measure how often the model is wrong on those samples. The threshold is set to the smallest `t` where the error rate first hits 10%.

**From the terminal:**
```
Boundary threshold: 0.05
```

This means any prediction with `|raw decision| < 0.05` is flagged as near-boundary — the classifier is genuinely unsure.

Note: this threshold is a heuristic. It does not formally guarantee that all predictions above 0.05 have less than 10% error rate — it only calibrates the decision boundary empirically from the validation set.

### Instability threshold: detecting sensitivity to imaging conditions

The system runs 6 augmented versions of each image through the full pipeline:

1. Brightness +15%
2. Brightness −15%
3. Horizontal flip
4. Gaussian blur (5×5)
5. Rotation +5°
6. Rotation −5°

For each augmentation, a score is computed. The **range** (max score minus min score) across all 6 augmentations measures how sensitive the prediction is to minor imaging changes.

Using 300 stratified validation images (60 per vegetable), the system finds the 95th percentile of these ranges:

```
Instability threshold (P95): 32.72
```

A score range above 32.72 combined with the raw decision crossing zero (the fresh/rotten boundary) indicates "true instability" — the model is predicting opposite things under minor variations. This triggers an UNRELIABLE flag.

### Mahalanobis OOD detection: catching out-of-distribution images

Some images may look completely unlike anything in the training set — an unusual background, a vegetable variety never seen, extreme lighting. The system flags these as out-of-distribution (OOD) using Mahalanobis distance: how far is this 100-feature vector from the centroid of the training distribution, accounting for feature correlations (via Ledoit-Wolf covariance shrinkage)?

**From the terminal:**
```
Mahalanobis thresh_caution = 13.102  (yellow warning zone)
Mahalanobis thresh_ood     = 16.852  (hard OOD flag)
OOD rate on validation:       1.02%
OOD rate on test:             0.91%  (consistent — no leakage)
```

About 1% of images are flagged as OOD. These receive the UNRELIABLE state regardless of the freshness prediction.

### Centroid consistency: catching vegetable misidentification

Even with 99% vegetable accuracy, confident wrong predictions happen. A misidentified vegetable would apply the wrong vegetable's normalization bounds, producing a misleading score.

The fix (C3): the centroid consistency check now runs **before** bounds selection. For each prediction, the ratio of the sample's distance to its predicted vegetable's centroid vs its distance to the next-closest centroid is computed. If this ratio is too high, the sample is not clearly inside the predicted class cluster — even if the SVM said "99% banana."

**From the terminal — per-class thresholds:**
```
apple:    ratio threshold = 1.1013
banana:   ratio threshold = 1.0954
capsicum: ratio threshold = 1.0406
cucumber: ratio threshold = 1.0124
potato:   ratio threshold = 1.0085
```

When this check fails, global bounds are used instead of per-vegetable bounds, and a warning is attached to the result.

---

## Phase 6 — Evaluation on the Held-Out Test Set

This is the only phase where the test set is used. It gives the honest final numbers.

### Classification accuracy

```
Vegetable classification:  98.94%  (2,513 / 2,539 correct)
Freshness classification:  97.99%  (2,488 / 2,539 correct)
Freshness ROC-AUC:         0.9979
```

The confusion matrix for freshness:
```
               Predicted Rotten   Predicted Fresh
Actual Rotten:     1294               29   (29 rotten items called fresh)
Actual Fresh:        22             1194   (22 fresh items called rotten)
```

51 errors out of 2,539 images. The errors are roughly balanced — about as many rotten items are called fresh as fresh items are called rotten.

### Score distribution

```
Fresh samples:  mean score = 85.95 / 100,  std = 12.22,  range = 91 points
Rotten samples: mean score = 17.76 / 100,  std = 13.82,  range = 83 points
Delta (gap):    68.20 points
Overlap:        0.0000  (zero rotten samples score above the fresh mean)
```

A delta of 68 points and zero overlap means the score is an excellent proxy for the fresh/rotten binary classification. The wide standard deviations (12–14 points) confirm the score is producing genuinely continuous output across its range — not just clustering at two values.

### Per-vegetable score separation

```
Vegetable   Fresh Mean   Rotten Mean    Delta
---------   ----------   -----------   ------
banana          89.34         14.43    74.90   ← strongest separation
capsicum        86.89         10.82    76.08   ← strongest separation
apple           86.04         19.56    66.48
cucumber        82.13         20.63    61.50
potato          76.10         21.86    54.24   ← weakest separation
```

Banana and capsicum show near-perfect separation. Cucumber and potato are harder — their visual fresh/rotten cues are subtler. A rotten potato can still look quite firm and green from a top-down photo; the decay may be internal.

### Reliability gate: what fraction of predictions are trusted?

```
RELIABLE:           2,353  (92.7%)  — score and fresh_label both shown
TENTATIVE:            163  ( 6.4%)  — score shown, fresh_label withheld
UNRELIABLE (OOD):      23  ( 0.9%)  — nothing shown, image flagged
```

92.7% of test images receive a full RELIABLE prediction. The 6.4% TENTATIVE samples have valid scores but the binary classification is withheld because the model is too close to the decision boundary. The 0.9% OOD samples are rejected entirely.

**Gate effectiveness:**
```
Gate              Fires  Catches wrong predictions  Verdict
-----------       -----  -------------------------  -------
G1_OOD              23   1 wrong prediction caught  REVIEW (has coverage cost)
G2_near_boundary    17   5 wrong predictions caught KEEP
G3_low_veg_conf     28   5 wrong predictions caught KEEP
```

The near-boundary and low-confidence gates are actively protecting accuracy. Disabling either one would lower the RELIABLE-only accuracy. The OOD gate is under review — it catches 1 error but blocks 22 correct predictions.

**RELIABLE-only accuracy vs overall:**
```
Overall freshness accuracy:    97.99%
RELIABLE-only accuracy:        98.43%
```

The gate is working correctly — RELIABLE samples are measurably more accurate than average.

### Per-vegetable RELIABLE coverage

```
Vegetable   RELIABLE%   TENTATIVE%   UNRELIABLE%
---------   ---------   ----------   -----------
apple          95.3%        4.4%          0.3%
banana         94.9%        4.5%          0.6%
capsicum       93.8%        5.7%          0.4%
cucumber       82.4%       15.4%          2.2%   ← WEAK
potato         83.0%       13.9%          3.1%   ← WEAK
```

Cucumber and potato have lower RELIABLE rates — the model is more often uncertain about these. This matches the lower per-vegetable delta scores above. If high coverage for cucumber/potato is needed, the confidence thresholds could be relaxed, at the cost of slightly lower RELIABLE accuracy.

---

## Phase 7 — Single Image Prediction

This is what happens every time you run `predict_cli.py --image photo.jpg`.

### Step 1 — Pre-flight checks

Before any feature extraction:
- **Blur check:** Laplacian variance < 28 → reject as unfocused
- **Brightness check:** Mean pixel brightness outside [30, 220] → reject as too dark/bright
- **Coverage check:** Largest contour < 40% of frame → warn (but do not reject)

If any hard check fails, the system returns UNRELIABLE immediately without running the classifier.

### Step 2 — Feature extraction

The image is resized to 224×224. EfficientNetB0 produces 1280 deep features. The 32 handcrafted features are computed. The 1312 values are concatenated.

### Step 3 — Preprocessing pipeline (must match training exactly)

The fitted `VarianceThreshold`, `StandardScaler`, and feature index selector from Phase 3 are applied. The result is a 100-element vector. It is critical that these objects are the same ones fitted during Phase 3 — fitting new ones on the test image would produce meaningless results.

### Step 4 — Vegetable identification

The vegetable SVM produces probabilities for all 5 classes. Example:
```
banana:   96.3%
apple:     2.1%
capsicum:  0.9%
cucumber:  0.5%
potato:    0.2%
```
Top-1 confidence = 96.3%, confidence gap = 96.3% − 2.1% = 94.2%

### Step 5 — Centroid consistency check (NEW — C3 fix)

This step now runs **before** the normalization bounds are selected. The sample's position in feature space is compared to the centroid of each vegetable class. If the sample is far from its predicted vegetable's centroid relative to the second-closest class, it is flagged as class-inconsistent and global bounds are used instead of banana bounds.

This prevents a failure mode where: a capsicum is confidently misidentified as a banana → banana bounds are applied → the capsicum score is meaningless.

### Step 6 — Normalization bounds selection

If vegetable confidence ≥ 70%, gap ≥ 15%, and centroid consistency passes → per-vegetable bounds are used. Otherwise → global bounds are used.

### Step 7 — Freshness scoring

```
raw = fresh_svm.decision_function(Xfinal)   → e.g. +0.85 (clearly fresh side)
score = (raw - p5) / (p95 - p5) × 100       → clipped to [0, 100]
```

Example with banana bounds (p5 = −2.43, p95 = +2.11):
```
score = (0.85 − (−2.43)) / (2.11 − (−2.43)) × 100
      = 3.28 / 4.54 × 100
      ≈ 72.2
```

### Step 8 — Mahalanobis OOD check

The 100-feature vector's Mahalanobis distance from the training centroid is computed. Distance > 16.85 → UNRELIABLE.

### Step 9 — Augmentation instability (if enabled)

6 augmented versions of the image are scored. If score range > 32.72 AND the raw decision crosses zero → UNRELIABLE. If score range > 49.08 alone → TENTATIVE with sensitivity warning. (Currently disabled: `use_augmentation_gate = false`.)

### Step 10 — Reliability classification

The system assigns one of three states:

**RELIABLE:**
- Score is valid
- `fresh_label` is shown (Fresh / Rotten)
- `freshness_confidence_band` is shown (High / Medium / Low / Very Low)

**TENTATIVE:**
- Score is valid and shown
- `fresh_label` is withheld (too close to boundary, or low vegetable confidence, or centroid inconsistency)
- `freshness_confidence_band` is withheld

**UNRELIABLE:**
- Nothing shown
- Either OOD, or true augmentation instability, or unreadable image

### Step 11 — Confidence band

The score maps to a confidence band (not a quality grade):

| Score | Band | Meaning |
|-------|------|---------|
| ≥ 85 | **High** | Model is strongly in the fresh region |
| 65–84 | **Medium** | Model is comfortably in the fresh region |
| 40–64 | **Low** | Model is in an ambiguous zone |
| < 40 | **Very Low** | Model is in the rotten region |

**Important:** these bands describe the model's confidence in its freshness classification. They do not directly measure biological freshness or shelf life. A score of 72 (Medium) means the model predicts fresh with reasonable confidence — not that the vegetable will stay fresh for a specific number of days.

### Example terminal output

```
Vegetable : banana (96.30%,  gap=94.20%)
State     : RELIABLE
Score     : 72.20  range=±4.80 / 100
Norm      : per-veg
Freshness : Fresh
Confidence: Medium
Mahal     : 8.234  [trusted]
```

---

## Design Decisions and Honest Limitations

### What this system does well

- Classifies vegetable type with 98.94% accuracy on 2,539 held-out test images
- Classifies fresh vs rotten with 97.99% accuracy
- Produces a continuous score that ranks fresh above rotten with 99.79% AUC reliability
- Provides calibrated uncertainty estimates to know when to trust predictions
- Detects out-of-distribution images and withholds predictions on them
- Uses per-vegetable normalization so scores are comparable within a species

### What this system does not do

- **It does not measure biological freshness.** The score is the SVM's geometric confidence, derived from colour and texture patterns. A vegetable that has internal decay but looks fresh on the outside would score high.
- **It does not guarantee intra-class ordering.** A score of 80 is reliably above a score of 20 (one is fresh, one is rotten). But a score of 80 vs 75 both on fresh bananas — the ordering is not guaranteed. Only the fresh/rotten separation is validated.
- **Scores are not comparable across vegetables.** A banana score of 80 and a potato score of 80 both mean "high relative to that vegetable's training distribution" — but not the same absolute freshness level. This is a consequence of per-vegetable normalization.
- **It does not handle unseen vegetables.** A mango fed to this system would produce an arbitrary prediction for one of the 5 vegetable types with no indication that the vegetable is unknown.
- **It assumes consistent imaging.** Unusual lighting, strong shadows, or unusual camera angles will degrade score reliability.

---

## System Flowchart

```
                         ┌─────────────────────────────┐
                         │  INPUT: vegetable photo.jpg  │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │      PRE-FLIGHT CHECKS       │
                         │  blur / brightness / coverage│
                         └──────────────┬──────────────┘
                                        │
                           fail?        │         pass?
                    ┌───────────────────┤
                    ▼                   │
             ┌──────────┐              │
             │UNRELIABLE│              ▼
             │ (return) │   ┌──────────────────────┐
             └──────────┘   │  FEATURE EXTRACTION  │
                            │  1280 deep (EffNetB0) │
                            │  +  32 handcrafted    │
                            │  =  1312 numbers      │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   PREPROCESSING       │
                            │  VarianceThreshold   │
                            │  → StandardScaler    │
                            │  → select top 100    │
                            └──────────┬───────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
         ┌──────────▼──────────┐              ┌───────────▼─────────┐
         │  VEGETABLE SVM      │              │  FRESHNESS SVM      │
         │  RBF, probability   │              │  RBF, no prob       │
         │  5-class            │              │  binary (0/1)       │
         └──────────┬──────────┘              └───────────┬─────────┘
                    │                                      │
         name, conf%, gap%                    decision_function value
                    │                                      │
                    └─────────────┬────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  CENTROID CONSISTENCY      │
                    │  check BEFORE bounds       │
                    │  dist_pred / dist_2nd      │
                    │  > per-class threshold?    │
                    └─────────────┬─────────────┘
                                  │
                    veg_confident AND consistent?
                    ┌─────────────┴──────────────┐
                   YES                           NO
                    │                             │
             per-veg bounds               global bounds
              (p5, p95 per veg)          (p5=-2.42, p95=2.09)
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      SCORE NORMALIZATION     │
                    │  score = (raw - p5) /        │
                    │          (p95 - p5) × 100    │
                    │  clipped to [0, 100]         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   MAHALANOBIS OOD CHECK      │
                    │   dist > 16.85 → OOD flag    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   RELIABILITY GATE           │
                    │                              │
                    │   is_ood?  → UNRELIABLE      │
                    │   |raw| < 0.05?→ TENTATIVE   │
                    │   veg_conf < 70%?→ TENTATIVE │
                    │   class_inconsistent?        │
                    │              → TENTATIVE     │
                    │                              │
                    │   otherwise → RELIABLE       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────────────────┐
                    │              │                           │
             ┌──────▼──────┐ ┌────▼───────┐          ┌───────▼───────┐
             │ UNRELIABLE  │ │ TENTATIVE  │          │   RELIABLE    │
             │ no output   │ │ score only │          │ score +       │
             │ + warning   │ │ no label   │          │ fresh_label + │
             └─────────────┘ └────────────┘          │ confidence_   │
                                                      │ band          │
                                                      └───────────────┘
                                                              │
                                              ┌───────────────▼──────────────┐
                                              │   CONFIDENCE BAND            │
                                              │   score ≥ 85 → High          │
                                              │   score ≥ 65 → Medium        │
                                              │   score ≥ 40 → Low           │
                                              │   score  < 40 → Very Low     │
                                              └──────────────────────────────┘
```

---

## Summary of Numbers to Remember

| Metric | Value |
|--------|-------|
| Total images used | 12,642 |
| Training samples | 8,883 |
| Validation samples | 1,269 |
| Test samples | 2,539 |
| Features per image | 1312 → 1304 → 100 (after cleaning and selection) |
| Vegetable accuracy | 98.94% |
| Freshness accuracy | 97.99% |
| Freshness ROC-AUC | 0.9979 |
| Fresh mean score | 85.95 / 100 |
| Rotten mean score | 17.76 / 100 |
| Score delta | 68.20 points |
| RELIABLE predictions | 92.7% of test images |
| TENTATIVE predictions | 6.4% |
| UNRELIABLE (OOD) | 0.9% |
| Boundary threshold | 0.05 |
| OOD threshold | 16.85 (Mahalanobis) |
| Instability threshold | 32.72 (score range across augmentations) |