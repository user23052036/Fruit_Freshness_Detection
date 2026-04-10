# VarianceThreshold — Study Notes

`VarianceThreshold` is the very first data cleaning step in the pipeline. It takes the raw 1312-dimensional feature matrix and removes any feature that is completely constant across the entire training set. This document explains why constant features are harmful, what variance actually measures, how the step is implemented in this project, and exactly where it sits in the preprocessing chain.

---

## 1. The Core Idea — What Is Variance?

Variance is a measure of how much a set of numbers spreads out around its mean. The formula for the variance of a feature across N training samples is:

```
         1   N
Var(x) = ─  Σ  (x_i − μ)²
         N  i=1

where μ = mean of x across all samples
      x_i = value of the feature for sample i
```

Reading this formula in plain terms: for each sample, compute how far the feature value is from the mean, square that gap, and average all the squared gaps together.

**When variance is large:** the feature takes very different values across samples — some images have a high value, others have a low value. This variation is exactly what a model can learn from.

**When variance is zero:** every single sample has the same value. The squared gaps are all zero, the average is zero. The feature carries no discriminative information whatsoever.

---

## 2. Why Zero-Variance Features Are Harmful

Suppose feature 472 has a value of 0.0 for all 8,883 training images. When your SVM or XGBoost model asks "does this feature help me decide between fresh and rotten?", the answer is always no — because this feature is identical for every image regardless of what the image shows.

Worse, zero-variance features cause a concrete numerical problem in the next step. `StandardScaler` normalises each feature to mean = 0, std = 1 by computing:

```
x_scaled = (x − mean) / std
```

If a feature is constant, its standard deviation is zero. The formula becomes:

```
x_scaled = (x − mean) / 0  →  division by zero → NaN or inf
```

This would silently corrupt the entire feature matrix. `VarianceThreshold` prevents this by removing these features before `StandardScaler` ever sees them.

---

## 3. Where It Fits in the Pipeline

```
preprocess_and_rank.py — execution order:

  X_train  (8,883 × 1312)   ← loaded from models/X_train.npy
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  VarianceThreshold(threshold=0.0)                   │
  │                                                     │
  │  1. Compute variance of each feature across         │
  │     all 8,883 training samples                      │
  │                                                     │
  │  2. Build a boolean mask:                           │
  │     keep[j] = True  if Var(feature_j) > 0.0         │
  │     keep[j] = False if Var(feature_j) = 0.0         │
  │                                                     │
  │  3. Drop the False columns                          │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  X_reduced  (8,883 × 1304)   ← 8 zero-variance features removed
        │
        ▼
  StandardScaler              ← now safe: no std=0 columns remain
        │
        ▼
  X_scaled  (8,883 × 1304)    ← mean=0, std=1 per feature
        │
        ▼
  XGBoost feature ranking     ← operates on 1304-dimensional scaled space
        │
        ▼
  Union feature set (349)     ← top-200 per task, unioned
        │
        ▼
  SVM training and calibration
```

The fitted `VarianceThreshold` object is saved to `models/variance.joblib` so that validation, test, and inference all apply the exact same column mask — no column index mismatch can occur.

---

## 4. The Actual Code

```python
# preprocess_and_rank.py

vt        = VarianceThreshold(threshold=0.0)
X_reduced = vt.fit_transform(X_train)
print(f"[INFO] VarianceThreshold: {X_train.shape[1]} → {X_reduced.shape[1]}")
```

`threshold=0.0` means: remove features with variance **exactly equal to zero**. Features with any positive variance, no matter how small, are kept. This is the most conservative setting — you are only discarding features that contribute literally nothing.

The output from the actual training run:

```
[INFO] VarianceThreshold: 1312 → 1304
```

Eight features were removed. The other 1304 all had at least some variation across the 8,883 training images and are passed to `StandardScaler`.

---

## 5. What Was Removed — Likely Causes

The 8 removed features come from the 32 handcrafted features (indices 1280–1311), not from the 1280 EfficientNetB0 features. EfficientNet's GlobalAveragePooling output is unlikely to produce truly constant values — the ImageNet-pretrained filters respond differently to every image. The handcrafted block, however, includes:

**Zero-padding (indices 1304–1311):** The handcrafted feature vector is explicitly padded to exactly 32 values with zeros:

```python
# extract_features.py

while len(feats) < 32:
    feats.append(0.0)

return np.array(feats[:32], dtype=np.float32)
```

If the natural handcrafted features produce 24 values (6 RGB + 6 HSV + 2 grayscale + 1 edge + 1 Laplacian + 8 histogram = 24), then 8 padding zeros are appended to reach 32. These 8 padding columns are 0.0 for every single image in the dataset — perfect candidates for removal by `VarianceThreshold`. This is precisely the 8 removed features.

---

## 6. Critical Rule — Fit on Train Only, Transform Everywhere

The `VarianceThreshold` is fit **exclusively on training data**. It is then applied identically to all other splits:

```python
# Fit and transform on training data
vt        = VarianceThreshold(threshold=0.0)
X_reduced = vt.fit_transform(X_train)        # learns which columns to remove

# Transform only on validation — no re-fitting
X_val_scaled = scaler.transform(vt.transform(X_val))

# At inference (predict_cli.py):
vt = load_model("models/variance.joblib")    # loads the fitted object
X  = vt.transform(np.array([feats]))         # applies the same column mask
```

Why this rule matters: if you fit `VarianceThreshold` on the full dataset (or on validation or test data), you risk removing a feature that is constant in the training set but has variance in the test set. When training data sees that feature as all zeros but test data has real values, you get a dimension mismatch and corrupted representations. Fitting on train only ensures the column mask is derived entirely from the training distribution and applied consistently everywhere.

---

## 7. VarianceThreshold Is Not Feature Selection

This is an important conceptual distinction to keep clear.

**Feature selection** (what XGBoost ranking does later) chooses features that are *predictive of the target* — freshness or vegetable class. It requires knowing the labels and ranking features by how much they help distinguish classes.

**VarianceThreshold** requires no labels at all. It does not know or care whether a feature is useful for predicting freshness. It only asks one question: *does this feature vary at all?* It removes features that fail that threshold.

The two steps are complementary and sequential:

```
Step 1: VarianceThreshold     → remove features that CANNOT be informative
                                 (zero variance = no discrimination possible)

Step 2: StandardScaler        → equalise feature scales

Step 3: XGBoost ranking       → rank features by how informative they ARE
                                 (requires labels, uses gain importance)

Step 4: Union feature set     → keep top-200 per task
```

VarianceThreshold cleans. XGBoost selects. They do entirely different jobs.

---

## 8. The Mathematics, Worked Through

Take a small example with 5 training samples and 4 features:

```
         feat_0   feat_1   feat_2   feat_3
img_1     0.32     0.00     1.41     0.78
img_2     0.19     0.00     0.93     0.44
img_3     0.56     0.00     2.10     0.91
img_4     0.41     0.00     1.67     0.62
img_5     0.28     0.00     1.22     0.55
```

Computing variance per feature:

```
feat_0:  mean = 0.352
         deviations² = (0.032)² + (0.162)² + (0.208)² + (0.058)² + (0.072)²
                     =  0.001 +  0.026 +  0.043 +  0.003 +  0.005
         variance    = 0.078 / 5 = 0.0156   ← positive → KEEP

feat_1:  all values = 0.00
         mean = 0.0
         deviations² = all zeros
         variance = 0.0000   ← exactly zero → REMOVE

feat_2:  mean = 1.466
         deviations² = (0.056)² + (0.536)² + (0.634)² + (0.204)² + (0.246)²
                     = large positive values
         variance > 0   → KEEP

feat_3:  mean = 0.660
         variance > 0   → KEEP
```

After `VarianceThreshold(threshold=0.0)`:

```
         feat_0   feat_2   feat_3
img_1     0.32     1.41     0.78
img_2     0.19     0.93     0.44
img_3     0.56     2.10     0.91
img_4     0.41     1.67     0.62
img_5     0.28     1.22     0.55

feat_1 removed.  4 features → 3 features.
```

In the actual project: 1312 features → 1304 features (8 removed, all zero-padding columns from the handcrafted feature block).

---

## 9. Why `threshold=0.0` and Not a Higher Value

Setting `threshold=0.0` removes only features with **exactly zero** variance. It is the minimum meaningful intervention.

You could set a higher threshold, such as `threshold=0.01`, which would additionally remove features with very low but non-zero variance. This could be useful if there are features that barely vary — perhaps a histogram bin that is nearly always zero but occasionally has a tiny value.

The pipeline uses `threshold=0.0` because the XGBoost ranking step that follows is already a principled, label-aware method for eliminating uninformative features. VarianceThreshold's job is only to remove the ones that are *mathematically guaranteed* to be useless (zero variance) and to prevent the numerical failure mode (std=0 in StandardScaler). Filtering at a higher threshold would be over-reaching — it would be discarding features based on variance alone, without knowing whether they correlate with freshness labels.

---

## 10. Quick Reference Summary

| Property | Value |
|----------|-------|
| Class | `sklearn.feature_selection.VarianceThreshold` |
| Setting | `threshold=0.0` |
| Input shape | (8,883 × 1312) — training features before any processing |
| Output shape | (8,883 × 1304) — 8 zero-variance columns removed |
| Features removed | 8 (all zero-padding columns from handcrafted block, indices 1304–1311) |
| Labels needed? | No — purely unsupervised, no target information used |
| Fit on | Training data only (`X_train`) |
| Applied to | Train, val, test, and every inference call identically |
| Saved as | `models/variance.joblib` |
| Purpose | Remove constant features + prevent division-by-zero in StandardScaler |
| What it is NOT | Feature importance ranking (that is XGBoost's job) |
| Position in pipeline | Step 1 — before StandardScaler, before XGBoost, before SVM |