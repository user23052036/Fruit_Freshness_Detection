# SVM Vegetable Freshness — Complete System Workflow

> EfficientNetB0 + Handcrafted Features → SVM Classification + Decision-Function Grading

This document is the complete updated workflow. It covers every stage from raw dataset to single-image prediction, including all fixes applied during development: per-vegetable normalization, test-set calibration, augmentation uncertainty, OOD detection, and score validation.

---

## Pipeline Overview (7 Steps)

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `extract_dataset_features.py` | `vegetable_Dataset/` | `Features/X.npy`, `y_veg.npy`, `y_fresh.npy` |
| 2 | `train_split.py` | `Features/*.npy` | `models/X_train.npy`, `X_test.npy`, ... |
| 3 | `preprocess_and_rank.py` | `models/X_train.npy` | `variance.joblib`, `scaler.joblib`, `selected_features.npy` |
| 4 | `train_svm.py` | `models/X_train.npy` + artifacts | `veg_svm.joblib`, `fresh_svm.joblib`, `scoring_config.json` |
| 5 | `evaluate_models.py` | `models/X_test.npy` | Classification + Score validation printed |
| 6 | `visualize_results.py` | `models/*` | Confusion matrix + feature importance plots |
| 7 | `predict_cli.py` | Single image path | Vegetable, Freshness, Score ± std, Grade |

---

## Step 1 — Feature Extraction

```bash
python src/extract_dataset_features.py
```

### What it does

Scans `vegetable_Dataset/`, filters only target vegetables (apple, banana, capsicum, cucumber, potato), loads images in parallel threads, converts BGR→RGB, resizes to 224×224, runs EfficientNetB0 in batches for 1280 deep features, then computes 32 handcrafted features per image. Concatenates both into a 1312-dimensional feature vector and saves the full matrix.

### Feature breakdown

| Source | Features | What it captures |
|--------|----------|-----------------|
| EfficientNetB0 (imagenet weights) | 1280 | High-level visual patterns: texture, colour, shape via CNN |
| RGB mean/std | 6 | Average colour and spread across red, green, blue channels |
| HSV mean/std | 6 | Hue, saturation, value — directly relates to freshness colour |
| Grayscale mean/std | 2 | Overall brightness and contrast |
| Edge density | 1 | Fraction of edge pixels — fresh = crisp edges, rotten = soft |
| Laplacian variance | 1 | Image sharpness — rotten produce becomes blurry/mushy |
| Luminance histogram | 8 | 8-bin intensity distribution — encodes discolouration |
| Padding | 8 | Zero-padded to reach exactly 32 handcrafted features |
| **TOTAL** | **1312** | **Final feature vector per image** |

### Expected console

```
Scanning dataset...
Total images: 12642
Extracting features...
Batches: 100%|████████████████| 99/99 [04:28<00:00]
Saved feature matrix: (12642, 1312)
```

### Files created

```
Features/
 ├ X.npy        # shape (N, 1312) — full feature matrix
 ├ y_veg.npy    # shape (N,) — vegetable name strings
 └ y_fresh.npy  # shape (N,) — 1=fresh, 0=rotten
```

### Check after run

```bash
python -c "import numpy as np; X=np.load('Features/X.npy'); print(X.shape)"
# Expected: (N, 1312)
```

---

## Step 2 — Train / Test Split

```bash
python src/train_split.py
```

### What it does

Performs a stratified 80/20 split using a combined `veg_fresh` label (e.g., `banana_1`, `potato_0`) to ensure both vegetable type and freshness class are evenly distributed in both splits. Default `random_state=42`.

### Why stratify on combined label?

A simple stratify on freshness alone would not preserve the per-vegetable balance. For example, if capsicum has very few rotten samples, a non-combined stratify might put all of them in train and none in test — making evaluation unreliable for that class.

### Expected console

```
[INFO] Loading Features...
[SUCCESS] Train/test split created. Train=10113 Test=2529
```

### Files created in `models/`

```
X_train.npy       X_test.npy
y_veg_train.npy   y_veg_test.npy
y_fresh_train.npy y_fresh_test.npy
```

### ⚠ Critical check before Step 3

```bash
ls models/X_train.npy || echo "STOP — run train_split.py first"
```

If `X_train.npy` is missing and you run `preprocess_and_rank.py`, it will fall through to `Features/X.npy` (full dataset) → **DATA LEAKAGE**.

---

## Step 3 — Preprocess & Rank Features

```bash
python src/preprocess_and_rank.py
```

### What it does

Loads `X_train.npy` **only** (never the full dataset). Applies three preprocessing steps in sequence, then ranks features by importance using XGBoost. All fitted objects are saved as artifacts and reused identically in every downstream step.

### Preprocessing pipeline

| Step | Operation | Why |
|------|-----------|-----|
| 1 | `VarianceThreshold(threshold=0.0)` | Remove features that are constant across all training samples — they carry zero information |
| 2 | `StandardScaler` (fit on train only) | Normalize feature distribution to mean=0, std=1. Prevents large-range features from dominating the SVM kernel |
| 3 | XGBoost feature ranking (gain) | Train a gradient boosted classifier on combined veg+freshness label. Use gain importance to rank the top 100 most discriminative features |
| 4 | Select top 100 features | Save indices to `selected_features.npy`. These same indices are used in every subsequent script |

### Expected console

```
[INFO] Using 10113 samples for ranking   ← must be train count, NOT 12642
[INFO] VarianceThreshold removed -> 1312 -> ~1304
[DONE] Selected top 100 features
```

**If it prints 12642 — STOP. You have data leakage. Rerun `train_split.py` first.**

### Files created in `models/`

```
variance.joblib          # VarianceThreshold fitted on train
scaler.joblib            # StandardScaler fitted on train
selected_features.npy    # indices of top 100 features — shape (100,)
feature_importances.npy  # XGBoost gain scores — used by visualize_results.py
```

### Check

```bash
python -c "import numpy as np; print(np.load('models/selected_features.npy').shape)"
# Expected: (100,)
```

---

## Step 4 — Train SVMs + Scoring Config

```bash
python src/train_svm.py
```

### What it does

Trains two independent RBF SVMs on the preprocessed training features. After training, loads the held-out test set to calibrate all scoring thresholds honestly — the model never saw these samples during training. Saves everything to `scoring_config.json`.

### Two SVMs trained

**SVM 1 — Vegetable classifier (multiclass)**
- Input: `X_train_final` shape `(N, 100)`
- Target: vegetable name encoded to integer by `LabelEncoder`
- Saved: `veg_svm.joblib` + `label_encoder.joblib`
- Example encoding: apple=0, banana=1, capsicum=2, cucumber=3, potato=4

**SVM 2 — Freshness classifier (binary)**
- Input: `X_train_final` shape `(N, 100)`
- Target: `y_fresh` (0=rotten, 1=fresh)
- Saved: `fresh_svm.joblib`
- No separate label encoder needed — already binary

### Scoring config generation (post-training calibration)

After both SVMs are trained, `train_svm.py` loads `X_test.npy` and computes `fresh_svm.decision_function()` on it. This held-out data is used to calibrate thresholds so they generalise — the model never saw these samples during training.

| Key | How computed | Purpose |
|-----|-------------|---------|
| `global_bounds` | p5/p95/hard_min/hard_max of training decisions (all samples) | Fallback normalization when per-veg is unavailable |
| `per_veg_bounds` | p5/p95/hard_min/hard_max computed separately per vegetable type from training decisions | Ensures banana and potato are scored on their own scale, not a shared global scale |
| `boundary_threshold` | Calibrated on **test set**: sweep abs(decision) thresholds, find where misclassification rate first exceeds 10% | Replaces arbitrary hardcoded 0.3 |
| `unstable_std_thresh` | Derived from data distribution of test set decision spread | Replaces arbitrary hardcoded 8.0 |
| `veg_confidence_threshold` | Fixed at 0.70 | If veg classifier confidence < 70%, use global bounds to avoid wrong normalization from misclassification |

### Expected console

```
[INFO] Feature matrix after selection: (10113, 100)

[INFO] Training vegetable classifier...
[DONE] Vegetable classifier saved

[INFO] Training freshness classifier...
[DONE] Freshness classifier saved

[INFO] Per-vegetable bounds computed:
  apple        p5=X.XXXX  p95=X.XXXX  min=X.XXXX  max=X.XXXX
  banana       p5=X.XXXX  p95=X.XXXX  min=X.XXXX  max=X.XXXX
  capsicum     p5=X.XXXX  p95=X.XXXX  min=X.XXXX  max=X.XXXX
  cucumber     p5=X.XXXX  p95=X.XXXX  min=X.XXXX  max=X.XXXX
  potato       p5=X.XXXX  p95=X.XXXX  min=X.XXXX  max=X.XXXX

[INFO] Loading test set for threshold calibration...
[INFO] Calibrated boundary threshold (test set): X.XXXX
[INFO] Calibrated unstable_std threshold: X.XXXX
[DONE] Scoring config saved → models/scoring_config.json
```

### `scoring_config.json` structure

```json
{
  "global_bounds": {
    "p5": -2.1, "p95": 2.1, "hard_min": -3.6, "hard_max": 3.4
  },
  "per_veg_bounds": {
    "apple":    { "p5": ..., "p95": ..., "hard_min": ..., "hard_max": ... },
    "banana":   { "p5": ..., "p95": ..., "hard_min": ..., "hard_max": ... },
    "capsicum": { "p5": ..., "p95": ..., "hard_min": ..., "hard_max": ... },
    "cucumber": { "p5": ..., "p95": ..., "hard_min": ..., "hard_max": ... },
    "potato":   { "p5": ..., "p95": ..., "hard_min": ..., "hard_max": ... }
  },
  "boundary_threshold":       0.35,
  "unstable_std_thresh":      7.40,
  "veg_confidence_threshold": 0.70
}
```

### Files created in `models/`

```
veg_svm.joblib         # vegetable classifier
fresh_svm.joblib       # freshness classifier
label_encoder.joblib   # vegetable name <-> integer mapping
scoring_config.json    # all normalization + calibration values
```

### Check

```bash
python -c "import json; c=json.load(open('models/scoring_config.json')); print(list(c.keys()))"
# Expected: ['global_bounds', 'per_veg_bounds', 'boundary_threshold',
#            'unstable_std_thresh', 'veg_confidence_threshold']
```

---

## Step 5 — Evaluate Models (mandatory)

```bash
python src/evaluate_models.py
```

### What it does

Runs the complete evaluation suite on the held-out test set. Produces two categories of output: classification performance and score validation.

### Section A — Classification metrics (both SVMs)

For both vegetable SVM and freshness SVM: Accuracy, Classification Report (precision/recall/F1), Confusion Matrix.

Expected results from your run:

```
========== Vegetable Classification ==========
Accuracy: 0.9949

              precision  recall  f1-score  support
       apple       0.99    1.00      1.00      896
      banana       1.00    1.00      1.00      940
    capsicum       1.00    0.99      1.00      218
    cucumber       1.00    0.98      0.99      182
      potato       0.98    0.99      0.98      293
    accuracy                         0.99     2529

========== Freshness Classification ==========
Accuracy: 0.9763

           0       0.98    0.98      0.98     1314
           1       0.97    0.98      0.98     1215
    accuracy                         0.98     2529
```

### Section B — Freshness Score Validation

| Test | What it measures | What it proves |
|------|-----------------|----------------|
| **A. Inter-class pairwise accuracy** | Fraction of (fresh, rotten) pairs where score_fresh > score_rotten | EASY TEST: score separates the two classes. Expected > 0.90. Does NOT prove intra-class ordering. |
| **B. Score distribution + Delta** | mean/std/min/max within fresh and rotten. **Delta = mean_fresh − mean_rotten** | Delta is the key defensible number. Large Delta = strong separation. Low overlap = clean boundary. |
| **C. Intra-class spread** | Score range and fraction of pairs differing by >10 pts within each class | NON-COLLAPSE check: proves score is continuous. Does NOT prove correct ordering within class. |
| **D. Per-vegetable table** | Pairwise acc, FreshMean, RottenMean, Delta per vegetable type | Shows whether grading works equally well for all vegetables. |

### Section C — Threshold statistics

```
Boundary threshold (calibrated on test set) : 0.XXXX
Near-boundary fraction                      : 0.0XXX  (XX/2529 samples flagged)
OOD fraction (hard bounds)                  : 0.0000
Unstable std threshold                      : X.XXXX
```

### Honest limitation printed at end of every run

```
The score is a proxy derived from the SVM decision boundary distance.
It proves class separation and continuous spread, but cannot guarantee
correct intra-class ordering without continuous ground-truth labels
(e.g., decay-day annotations). It should be interpreted as a relative
freshness indicator, not an absolute freshness measurement.
```

---

## Step 6 — Visualization (optional)

```bash
python src/visualize_results.py
```

Produces three matplotlib plots: vegetable confusion matrix (Blues), freshness confusion matrix (Greens), top-20 XGBoost feature importances bar chart. Requires `matplotlib`, `seaborn`.

---

## Step 7 — Single Image Prediction

```bash
# With uncertainty (default) — runs EfficientNet 6 extra times. ~10-15s on CPU.
python src/predict_cli.py --image path/to/image.jpg

# Without uncertainty (fast) — skips augmentation, no ± score.
python src/predict_cli.py --image path/to/image.jpg --no-uncertainty
```

### Full prediction pipeline

```
Input image
    ↓
EfficientNetB0 → 1280 deep features
    +
32 handcrafted features (RGB/HSV/grayscale/edge/Laplacian/histogram)
    ↓
Concatenate → 1312 feature vector
    ↓
variance.transform()          remove same constant features as training
    ↓
scaler.transform()            normalize using training mean/std
    ↓
X[:, selected_features]       keep top 100 features only
    ↓
veg_svm.predict_proba()       → vegetable type + confidence %
    ↓
fresh_svm.predict()           → Fresh or Rotten label
    ↓
fresh_svm.decision_function() → raw signed distance from hyperplane
    ↓
normalize_score(raw, per_veg_bounds)  → Score [0–100]
    ↓
6× augmentations → score std  → input sensitivity estimate
    ↓
grade_from_score(score)       → Grade label
    ↓
Print result + warning flags
```

### Score normalization

Raw SVM decision value = signed geometric distance from the hyperplane.
- Positive → fresh side, Negative → rotten side
- Larger magnitude → further from boundary → more confident

```python
score = (raw - p5) / (p95 - p5) * 100
score = clip(score, 0, 100)
# epsilon guard: if abs(p95 - p5) < 1e-6 → return 50.0
```

Normalization uses **per-vegetable** p5/p95 bounds from training decisions. If vegetable confidence < 70%, falls back to global bounds.

### Grade thresholds

| Score | Grade | Meaning |
|-------|-------|---------|
| ≥ 85 | Truly Fresh | Very far from rotten boundary |
| 65 – 84 | Fresh | Clearly on fresh side |
| 40 – 64 | Moderate | Near the decision boundary |
| < 40 | Rotten | Firmly on rotten side |

### Warning flags (four independent checks)

| Flag | Condition | Type | Meaning |
|------|-----------|------|---------|
| Low veg confidence | `veg_conf < 70%` | Info | Using global normalization instead of per-veg |
| **MODEL UNCERTAINTY** | `abs(raw) < boundary_threshold` | Model | Classifier is unsure — small feature changes could flip the result |
| **INPUT SENSITIVITY** | `score_std > unstable_std_thresh` | Input | Score changes under imaging augmentation — sensitive to lighting/angle |
| **OOD WARNING** | `raw < hard_min` or `raw > hard_max` | OOD | Outside training range. Only catches extreme outliers — distribution shift within training range is NOT detected |

MODEL UNCERTAINTY and INPUT SENSITIVITY are separate signals with different causes and different messages.

### Sample outputs

Normal case:
```
Vegetable : banana (99.93%)
Freshness : Fresh
Score     : 76.60 ± 3.21 / 100
Grade     : Fresh
Norm      : per-veg
```

With warnings:
```
Vegetable : banana (99.93%)
Freshness : Fresh
Score     : 52.10 ± 11.40 / 100
Grade     : Moderate
Norm      : per-veg

[!] MODEL UNCERTAINTY — score near decision boundary (|raw|=0.18 < threshold=0.35).
    The classifier itself is unsure. This result could change with a different model or features.

[!] INPUT SENSITIVITY — score std=11.40 across augmentations (threshold=7.40).
    The score is sensitive to lighting/angle variation.
    Consider re-capturing on a neutral background.
```

---

## What Changed From Original Workflow

| Area | Before | Now |
|------|--------|-----|
| Freshness score | `predict_proba` (class confidence) | `decision_function` (geometric distance) |
| Score normalization | min-max (outlier sensitive) | p5/p95 percentile per vegetable type |
| Boundary threshold | hardcoded 0.3 | calibrated on test set (data-driven) |
| Unstable std threshold | hardcoded 8.0 | calibrated from data distribution |
| Uncertainty output | single point estimate | score ± std with separated warning types |
| OOD detection | none | hard-bound check + explicit limitation note |
| Veg confidence gating | none | fallback to global bounds if conf < 70% |
| Evaluation | accuracy + confusion matrix only | + inter/intra-class score validation + delta |
| Scoring config | `fresh_decision_bounds.npy` | `scoring_config.json` (all values) |
| Data leakage (step 3) | preprocess_and_rank used full dataset | fixed — uses X_train only |

---

## Final Artifacts — Complete File List

```
Features/
 ├ X.npy                  # (N, 1312) — full feature matrix
 ├ y_veg.npy              # (N,) — vegetable name strings
 └ y_fresh.npy            # (N,) — 1=fresh, 0=rotten

models/
 ├ X_train.npy            # (10113, 1312)
 ├ X_test.npy             # (2529, 1312)
 ├ y_veg_train.npy        # (10113,)
 ├ y_veg_test.npy         # (2529,)
 ├ y_fresh_train.npy      # (10113,)
 ├ y_fresh_test.npy       # (2529,)
 ├ variance.joblib        # VarianceThreshold fitted on train
 ├ scaler.joblib          # StandardScaler fitted on train
 ├ selected_features.npy  # top 100 feature indices — shape (100,)
 ├ feature_importances.npy
 ├ veg_svm.joblib         # vegetable classifier
 ├ fresh_svm.joblib       # freshness classifier
 ├ label_encoder.joblib   # vegetable name <-> integer mapping
 └ scoring_config.json    # all normalization + calibration values
```

---

## Failure Modes & Critical Checks

**1. Data leakage (most dangerous)**
Step 3 must print `Using 10113 samples`. If it prints `12642`, `X_train.npy` was missing when you ran it.
```bash
ls models/X_train.npy || echo "STOP — run train_split.py first"
```

**2. Missing `scoring_config.json`**
Both `predict_cli.py` and `evaluate_models.py` require this file. If missing, rerun `train_svm.py`.
```bash
ls models/scoring_config.json || echo "STOP — run train_svm.py first"
```

**3. Shape mismatch**
Never mix artifacts from different runs. If you rerun `preprocess_and_rank.py`, always follow with `train_svm.py` and `evaluate_models.py`.

**4. EfficientNet colour bug**
Feature extractor converts BGR→RGB. If you see sudden accuracy drops, verify `extract_features.py` still does `cv2.cvtColor(img, COLOR_BGR2RGB)`.

**5. Slow prediction**
Default runs 6 augmented inferences for uncertainty estimate. ~10-15 seconds on CPU. Use `--no-uncertainty` for fast inference.

**6. OOD inside training range**
Hard-bound OOD only catches extreme outliers. High `score_std` is the soft signal for in-distribution but imaging-sensitive samples.

---

## Minimal Re-run Checklist

Only changed model hyperparameters (no new features):
```bash
python src/train_svm.py
python src/evaluate_models.py
```

Only changed feature selection (`top_k`):
```bash
# edit top_k in preprocess_and_rank.py first
python src/preprocess_and_rank.py
python src/train_svm.py
python src/evaluate_models.py
```

Added new images to dataset:
```bash
python src/extract_dataset_features.py
python src/train_split.py
python src/preprocess_and_rank.py
python src/train_svm.py
python src/evaluate_models.py
```