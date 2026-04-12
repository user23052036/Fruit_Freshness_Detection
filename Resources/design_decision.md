# Design Decisions and Architectural Rationale

This document records the design-phase reasoning for this project — the original approach that was considered, the problems identified in it, and the decisions made that led to the current implementation. Reading this in order gives a clear picture of *why* the pipeline is built the way it is, not just *what* it does.

---

## 1. Where the Pipeline Starts — Scripts and Their Jobs

Before any design decisions, it helps to be clear about what each script actually does. These are the two foundation scripts that everything else depends on.

### `extract_dataset_features.py`

**Single responsibility:** Convert images into numbers.

The model never reads images again after this step. Every image in `vegetable_Dataset/` is passed through EfficientNetB0 (1280 deep features) and the handcrafted extractor (32 features), producing a 1312-dimensional feature vector per image. The output is a matrix of numbers:

```
vegetable_Dataset/ (12,691 images)
        │
        ▼
EfficientNetB0 + handcrafted extractor
        │
        ▼
Features/X.npy           shape (12691, 1312)   float32
Features/y_veg.npy       vegetable labels
Features/y_fresh.npy     freshness labels  (1=fresh, 0=rotten)
Features/image_paths.npy path per row (required for augmentation stats later)
```

From this point, every downstream script operates on `.npy` arrays. The images themselves are never re-read except during the augmentation calibration in `train_svm.py`.

### `train_split.py`

**Single responsibility:** Divide the feature matrix into three disjoint sets.

```
X.npy (12,691 × 1312)
        │
        ▼  stratified on "{vegetable}_{freshness}" composite label
        │
        ├── 70% → Train (8,883)  — model fitting only
        ├── 10% → Val  (1,269)   — all calibration
        └── 20% → Test (2,539)   — final evaluation, never touched earlier
```

The reason for three sets (not two) is covered in depth below.

---

## 2. The Original Approach — Teacher's Recommended Pipeline

The first version of the pipeline followed a specific pipeline structure:

```
Image
→ EfficientNet + handcrafted → 1312 features
→ VarianceThreshold (remove constant features)
→ StandardScaler
→ ElasticNet (feature selection)  ← choose features with non-zero coefficients
→ XGBoost (rank selected features by gain)
→ Weighted score = Σ (weight_i × z_i)  where z = standardized features
→ Map to 0–100 via percentile → grade
```

The appeal of this pipeline is interpretability: ElasticNet coefficients are readable numbers, XGBoost gains are rankings, and the weighted sum produces an explicit formula for the score. You can point to specific features and say exactly why an image received a given score.

### What ElasticNet Does

ElasticNet is a linear regression model with two regularisation terms:

```
ElasticNet loss = MSE + α × [l1_ratio × |β| + (1 − l1_ratio) × β²]
                               └── L1 term ──┘  └─── L2 term ────┘
```

The L1 term drives some coefficients exactly to zero (sparsity — feature selection). The L2 term prevents remaining coefficients from becoming unstable. The result: ElasticNet selects a subset of features and assigns each a coefficient indicating how much that feature contributes linearly to the prediction.

---

## 3. The Core Problem With ElasticNet as a Filter

ElasticNet is a **linear model**. It can only detect linear relationships between a feature and the target. If a feature is only useful in combination with another feature — an interaction — ElasticNet cannot detect it.

### The Interaction Problem

Suppose:

```
freshness depends on:  colour × texture
                        ─────────────────
                        non-linear interaction
```

ElasticNet evaluates colour and texture **individually** against the freshness label:

```
colour alone   → weak linear signal   → coefficient ≈ 0 → MAY BE DROPPED
texture alone  → weak linear signal   → coefficient ≈ 0 → MAY BE DROPPED
```

If both are dropped, XGBoost never sees them. It cannot learn the interaction it was never given. The information is permanently lost — there is no recovery once features are discarded.

This is the critical failure mode: **ElasticNet can drop features that are individually weak but jointly strong**. Because it is a linear filter applied before a non-linear model, it makes irreversible decisions with incomplete information.

### When This Is and Is Not a Problem

ElasticNet as a filter is not always harmful. In high-dimensional spaces like this one (1312 features), many features are:
- Genuinely noisy (random patterns in EfficientNet channels that don't respond to vegetables)
- Strongly correlated with other features (redundant)
- Constant or near-constant

Removing these saves compute and reduces overfitting risk. If ElasticNet removes only genuine garbage and keeps all signal-carrying features, the downstream XGBoost is unaffected.

The risk is calibration: if ElasticNet regularisation is set too aggressively, it removes signal-carrying features alongside the garbage. This failure is silent — there is no error, just reduced accuracy with no clear explanation.

---

## 4. The Fix — Dual XGBoost Ranking Without ElasticNet

The solution implemented in the current pipeline removes ElasticNet entirely and uses XGBoost directly for feature ranking. This preserves all non-linear signal because XGBoost builds interactions through tree splits:

```
Feature: colour (value = 0.65)
Feature: texture (value = 0.42)

XGBoost tree node:
  if colour > 0.60 AND texture > 0.40:
      → fresh
  else:
      → rotten

The interaction colour × texture is learned from the data,
not pre-specified. XGBoost discovers it automatically.
```

XGBoost is also better for ranking in this pipeline because it uses **gain importance** — how much each feature improves the model's predictions across all tree splits — which directly measures discriminative value rather than linear correlation with the label.

### The Second Problem — Single-Label Ranking

The original XGBoost ranking step was run with a **combined label** encoding both vegetable identity and freshness:

```python
# Original approach
y = [f"{veg}_{fresh}" for veg, fresh in zip(y_veg, y_fresh)]
# e.g. ["banana_1", "potato_0", "apple_1", ...]
# 10 classes: one per (vegetable, freshness) combination
```

This mixes two distinct tasks into one signal. Features that strongly discriminate *vegetable type* get high gain scores even if they carry no freshness signal. Those features then enter the feature set used by the freshness SVM, adding noise without benefit.

Conversely, the previous version used freshness-only labels for ranking, which meant vegetable-discriminative features were underweighted — and the vegetable SVM was denied features it needed.

**The implemented fix: two independent rankings with a union.**

```python
# Current implementation in preprocess_and_rank.py

# Freshness ranking — sees only freshness signal
avg_imp_fresh = compute_full_ranking(X_scaled, y_fresh_train.astype(int))

# Vegetable ranking — sees only vegetable signal
avg_imp_veg = compute_full_ranking(X_scaled, y_veg_enc)

# Each task's top-200 features selected independently
selected_fresh = top_200(avg_imp_fresh)   # best for freshness
selected_veg   = top_200(avg_imp_veg)     # best for vegetable

# Union: both SVMs share one space that serves both tasks
union_set = selected_fresh ∪ selected_veg  # = 349 features
```

Result from the training run:

```
Fresh-specific features : 149
Veg-specific features   : 149
Shared features         : 51
Union size              : 349
```

149 features are valuable only for freshness (e.g., colour shift toward brown, Laplacian variance decrease). 149 features are valuable only for vegetable identity (shape-sensitive EfficientNet channels, overall colour profiles). 51 features are informative for both tasks.

---

## 5. The Calibration Contamination Problem

The earliest version of the pipeline used the test set for calibrating thresholds. `train_svm.py` would load `X_test.npy` to compute the boundary threshold and normalization bounds. `evaluate_models.py` would then report accuracy on the same data.

This is **calibration contamination**: the thresholds are tuned to the test set, and the evaluation runs on the same set. The reported performance is not a measurement of generalisation — it is a measurement of how well the system was tuned to those specific samples.

### Why Training Bounds Are Wrong for Normalization

The freshness score is computed by normalising the SVM's raw margin using p5/p95 percentile bounds:

```
score = clip( (raw − p5) / (p95 − p5) × 100, 0, 100 )
```

If p5 and p95 are computed from training margins, the bounds are inflated. The SVM was fitted to maximise separation on training samples, so training margins are wider than the margins the model produces on new, unseen data.

**Numerical illustration:**

Suppose the actual validation bounds for banana are p5 = −2.017 and p95 = +1.822 (the real values from this project). If training bounds were used instead (plausibly p5 ≈ −3.1, p95 ≈ +2.9, approximately 50% wider):

```
Strongly fresh banana: raw margin = +1.82

With training bounds:   score = (+1.82 − (−3.1)) / (2.9 − (−3.1)) × 100 = 82
With validation bounds: score = (+1.82 − (−2.017)) / (1.822 − (−2.017)) × 100 ≈ 100
```

Same sample, same margin — score of 82 vs 100. Training bounds compress all scores toward the middle, making strongly fresh produce appear only moderately fresh and strongly rotten produce appear only moderately rotten.

**The fix:** A true validation split (10% of data, never used for model fitting) provides the honest reference distribution. Normalization bounds computed on validation data reflect what the model produces on genuinely unseen images.

---

## 6. The SVM Probability Calibration Problem

The vegetable SVM was initially configured with `probability=True`:

```python
# Old approach
veg_model = SVC(kernel="rbf", probability=True, ...)
veg_conf  = veg_model.predict_proba(X)[0].max() * 100
# Gate: if veg_conf >= 70% → use per-veg normalization bounds
```

`probability=True` in scikit-learn uses **Platt scaling** — a sigmoid function fit on cross-validated SVM scores. Platt scaling is known to produce poorly calibrated probabilities, especially in multi-class settings. The 70% confidence threshold was set without verifying that a Platt-scaled score of 70% actually corresponds to 70% empirical accuracy on this dataset. It very likely does not.

**The practical consequence:** the gate that decides whether to apply per-vegetable or global normalization bounds — a gate that directly determines the freshness score — was being triggered by an uncalibrated number.

**The fix:** `CalibratedClassifierCV` with isotonic regression, fit on a held-out `cal_val` subset (50% of the validation set, disjoint from the threshold selection subset):

```python
# Current implementation in train_svm.py

veg_base  = SVC(kernel="rbf", probability=False, ...)
gs.fit(X_train, y_veg)   # GridSearchCV to find best C, gamma
veg_base  = gs.best_estimator_

veg_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(veg_base),
    method="isotonic"    # non-parametric — learns actual shape of score→probability curve
)
veg_model.fit(X_cal_val, y_veg_cal)   # fits calibration only, does not retrain SVC
```

Isotonic regression is more flexible than Platt scaling — it learns the actual empirical shape of the probability curve rather than assuming it is sigmoidal. `FrozenEstimator` ensures the underlying SVC weights are never changed; only the calibration layer is fit on the validation data.

---

## 7. What the Current Pipeline Looks Like

All five problems described above have been addressed. The current pipeline:

```
extract_dataset_features.py
  images → 1312 features per image (EfficientNet 1280 + handcrafted 32)

train_split.py
  stratified 70/10/20 split on composite label → Train / Val / Test

preprocess_and_rank.py
  VarianceThreshold (1312 → 1304, removes 8 zero-padding columns)
  StandardScaler (fit on train only)
  Dual XGBoost ranking:
    Freshness task → avg_imp_fresh   (5 seeds, y_fresh labels)
    Vegetable task → avg_imp_veg     (5 seeds, y_veg labels)
  Two-phase k-selection (proxy LinearSVC sweep + RBF confirmation)
  Union of top-200 per task → 349 feature indices saved

train_svm.py
  Val split: cal_val (634) | thr_val (635)  ← prevents calibration leakage
  Vegetable SVM: GridSearchCV → CalibratedClassifierCV(isotonic, cal_val)
  Freshness SVM: GridSearchCV on X_train
  Normalization bounds: p5/p95 on full val set (not training, not test)
  Mahalanobis OOD: LedoitWolf on X_train, thresholds from P90/P99 of training distances
  Formal threshold selection on thr_val: maximise Coverage subject to Risk ≤ 10%
  Test set: never loaded in this file

evaluate_models.py
  Test set opened here for the first time
  Reports accuracy, inversion rates, gate ablation, state distribution
```

---

## 8. Alternative Approaches That Were Considered

During the design phase, four alternative feature selection strategies were evaluated. These are recorded here for completeness.

### Option A — XGBoost-first (what was implemented)

```
features → XGBoost ranking → top features → dual SVM
```

Preserves non-linear interactions, gives robust importances via gain, no prior feature elimination step that can drop useful features. This is the current approach.

### Option B — PCA → XGBoost

```
features → PCA (reduce to 95% variance) → XGBoost
```

PCA reduces dimensionality without discarding features — it creates new axes that are linear combinations of all original features. Information is preserved but mixed together. Individual feature contributions are no longer interpretable. Useful if compute or memory is the primary constraint.

### Option C — Mutual Information → XGBoost

```
features → MI ranking (top K) → XGBoost
```

Mutual information measures statistical dependence without assuming linearity. It can detect non-linear relationships between individual features and the label, making it better than linear correlation for ranking. More computationally expensive than XGBoost gain; MI estimation can be noisy for small samples.

### Option D — End-to-End Deep Learning

```
features → attention layer + MLP regressor → continuous freshness score
```

Best potential accuracy if training data is abundant and GPU is available. A small attention network over EfficientNet features could learn which features to weight for each vegetable class automatically. Requires significantly more engineering, is much less interpretable, and needs substantially more labelled examples to train stably. Not suitable for this dataset size or deployment context.

---

## 9. The Score — What It Is and What It Is Not

A critical clarification that emerged during the design phase:

**What the score is:** A per-vegetable calibrated proxy derived from the SVM's decision function margin. It measures how far the sample sits from the decision boundary within the context of that vegetable's training distribution. A score of 80 for banana means "this banana's feature vector sits near the top 20% of the banana training distribution in terms of freshness margin distance."

**What the score is not:**
- Not a physical freshness measurement (not correlated to Brix, firmness, or decay-day annotations)
- Not cross-vegetable comparable (banana 80 ≠ potato 80 in absolute freshness terms)
- Not an intra-class ordering guarantee (a score of 80 is reliably fresher than a score of 20; it is not guaranteed to be fresher than a score of 78)

**What the confidence bands are:** Heuristic presentation tiers (High / Medium / Low / Very Low at thresholds 85 / 65 / 40), not externally validated decision thresholds. They partition the [0, 100] range into four regions that align broadly with the fresh/rotten class distributions but should not be used to make operational decisions (e.g., "safe to sell") without biological ground truth.

These limitations are explicitly documented in `evaluate_models.py` under the Limitations section:

```
Score is a per-vegetable calibrated SVM margin proxy.
It proves class separation and ordering reliability between classes.
It does not prove correct intra-class ordering without continuous
ground-truth freshness labels (e.g. decay-day annotations).
```

---

## 10. The Defensible System Claim

The smallest version of the system's claim that holds up under scrutiny:

> This system is a calibrated, per-vegetable freshness proxy. It classifies vegetable type with 99.61% accuracy and freshness state with 98.94% accuracy on held-out test data. It produces a continuous score normalised within each vegetable's training distribution and formally certifies predictions as RELIABLE, TENTATIVE, or UNRELIABLE based on margin proximity, OOD distance, vegetable confidence, and class consistency. It does not claim to measure spoilage progression, produce cross-vegetable comparable grades, or guarantee intra-class ordering without continuous freshness ground-truth labels.

Everything beyond this requires either external biological ground truth labels or additional probability calibration validation against empirical accuracy at each confidence level.