# Mahalanobis OOD Detection

The system needs to know when an input image is so far from the training distribution that the model's prediction cannot be trusted — regardless of how confident the prediction looks. This is the out-of-distribution (OOD) problem. This document explains how the Mahalanobis distance detector solves it, exactly how it is calibrated in this pipeline, what its actual behaviour is on the test set, and where its limits are.

---

## 1. The Problem Being Solved

The SVM freshness classifier was trained on images of five specific vegetables under controlled conditions. Given any input image, it will always produce a decision function value and a predicted class. It has no built-in mechanism to say "I have never seen anything like this."

This matters because the failure mode is silent. An image of a mango, a photograph of a vegetable under extreme lighting, or an image with heavy background clutter will all produce a freshness score as if nothing is wrong. The score may even be high — nowhere near the decision boundary — giving false confidence in a prediction that is meaningless.

The OOD detector catches this. Its job is to measure how far a new image's feature vector lies from the region of feature space that the training data occupies. If the distance exceeds a calibrated threshold, the prediction is marked UNRELIABLE regardless of the SVM's output.

---

## 2. Why Euclidean Distance Is Insufficient

The simplest approach to measuring "distance from training data" is Euclidean distance to the training centroid (mean feature vector). This fails for two reasons.

**Features have different variances.** In the 349-dimensional feature space, some dimensions vary widely across the training set (high-variance EfficientNet channels that respond to background patterns) while others are tightly clustered (HSV mean values that are consistent within each vegetable class). Euclidean distance treats all 349 dimensions equally, so a small deviation in a high-variance dimension registers as a large distance even though it is completely normal.

**Features are correlated.** EfficientNet's 1280 output dimensions (before feature selection) are highly correlated — adjacent filter responses activate together for the same texture patterns. After selecting 349 features, correlations remain. Euclidean distance ignores these correlations entirely. A sample can be within the normal value range of every individual dimension but still be anomalous if its feature *combination* is unusual — a pattern that never appears in the training set even though each individual component does.

Mahalanobis distance solves both problems by scaling each direction by its variance and accounting for the covariance structure between features.

---

## 3. The Mahalanobis Distance Formula

For a single sample feature vector **x** (shape 349):

```
diff       = x − train_mean                    shape (349,)
mahal_dist = sqrt( diffᵀ · precision · diff )  scalar
```

For a batch of N samples (shape N × 349):

```python
# train_svm.py / evaluate_models.py
diff  = X - m                                    # (N, 349)
dists = np.sqrt(np.einsum("ij,jk,ik->i", diff, P, diff))   # (N,)
```

`precision` is the inverse of the covariance matrix of the training feature vectors. It encodes both the variance of each individual feature and the correlation structure between features.

What `precision` does geometrically: it stretches and rotates the feature space so that directions of high variance in the training data become "short" and directions of low variance become "long". After this transformation, Euclidean distance in the transformed space equals Mahalanobis distance in the original space. A sample that deviates from the training mean along a high-variance direction registers a small Mahalanobis distance. A sample that deviates along a direction that is tight and consistent in training registers a large distance.

---

## 4. Why LedoitWolf Shrinkage Is Used

The covariance matrix of a 349-dimensional dataset has 349 × 349 = 121,801 entries to estimate. With 8,883 training samples, the sample covariance matrix is estimated from a dataset where the number of observations is only 25× the number of parameters. This is a well-known regime where the sample covariance matrix is unreliable: its eigenvalues are spread too widely, its smallest eigenvalues are close to zero, and inverting it (to get the precision matrix) amplifies these small eigenvalues into numerically unstable large values.

LedoitWolf shrinkage addresses this by pulling the sample covariance matrix toward a structured estimate (a scaled identity matrix):

```python
# train_svm.py
lw        = LedoitWolf().fit(X_train)
precision = lw.precision_.astype(np.float32)   # shape (349, 349)
```

The shrinkage coefficient is computed analytically from the data (no cross-validation required). The result is a regularised precision matrix that is positive-definite, numerically stable, and produces well-behaved Mahalanobis distances across the full 349-dimensional space.

Using a raw sample covariance inverse — or any distance measure that does not account for the covariance structure — in 349 dimensions would produce distances dominated by numerical noise from the unstable eigenvalues.

---

## 5. What Gets Fitted and Saved

The detector is fitted entirely on training data:

```python
# train_svm.py — fit on X_train[:, union_349] only

train_mean = X_train.mean(axis=0).astype(np.float32)    # shape (349,)
lw         = LedoitWolf().fit(X_train)
precision  = lw.precision_.astype(np.float32)            # shape (349, 349)

np.save("models/train_mean.npy",      train_mean)
np.save("models/train_precision.npy", precision)
```

Fitting on training data only is critical. Fitting on val or test data would let the validation or test distribution influence the definition of "normal" — the detector would then be miscalibrated toward the wrong population.

The fitted mean and precision are loaded unchanged at every inference call (`predict_cli.py`), at threshold calibration (`train_svm.py`), and at evaluation (`evaluate_models.py`). They represent the training distribution and never change after training.

---

## 6. Threshold Calibration

After fitting the covariance model, Mahalanobis distances are computed for every training sample:

```python
# train_svm.py

train_dists = mahal(X_train, train_mean, precision)   # shape (8883,)

thresh_caution = float(np.percentile(train_dists, 90))   # = 24.167
thresh_ood     = float(np.percentile(train_dists, 99))   # = 30.438
```

The thresholds are **percentiles of the training distribution**, not of the validation distribution. This is the correct choice: the training distribution defines what "normal" means. Setting thresholds from the training distances means:

- `thresh_caution = 24.167` corresponds to P90 of training distances. Exactly 10% of training samples have a Mahalanobis distance above this value.
- `thresh_ood = 30.438` corresponds to P99 of training distances. Exactly 1% of training samples have a distance above this value.

By construction, the expected OOD flag rate on data drawn from the same distribution as training is 1%. The observed rate on the validation set (23 of 1,269 samples = 1.81%) and test set (62 of 2,539 samples = 2.44%) are both close to this — confirming the thresholds transfer stably.

```
[INFO] Mahalanobis thresh_caution=24.167  thresh_ood=30.438
[INFO] OOD rate on validation: 0.0181

OOD rate — validation : 0.0181
OOD rate — test       : 0.0244
[OK] OOD rates consistent across splits.  (difference = 0.63% < 5% threshold)
```

---

## 7. The Three Zones

Every input image is assigned to one of three zones based on its Mahalanobis distance:

```
distance(x) = sqrt( (x − mean)ᵀ · precision · (x − mean) )

          0              24.167              30.438
          │─────────────────│──────────────────│──────────────────→
          │    TRUSTED       │    CAUTION       │       OOD
          │  dist < 24.167   │ 24.167 ≤ d < 30.438  │  dist ≥ 30.438
          │                  │                  │
          │ No effect on     │ Warning flag     │ score_unreliable
          │ state            │ appended to      │ = True
          │                  │ result dict      │ → UNRELIABLE
```

```python
# predict_cli.py

def mahal_zone(dist, thresh_caution, thresh_ood):
    if dist >= thresh_ood:      return "ood"      # → UNRELIABLE
    if dist >= thresh_caution:  return "caution"  # → warning only
    return "trusted"                               # → no effect
```

**Trusted zone (< 24.167):** The sample lies within the dense region of the training distribution. OOD detection places no constraint on the prediction. Other gates (boundary proximity, vegetable confidence, centroid consistency) still apply.

**Caution zone (24.167 to 30.438):** The sample is in the outer 10% of the training distribution but has not exceeded the OOD threshold. A warning is appended to the result dict but the prediction state is not changed. This is the expected zone for unusual but not untrustworthy samples — for example, a vegetable photographed at an unusual angle or under slightly different lighting.

**OOD zone (≥ 30.438):** The sample lies outside the P99 boundary of the training distribution. `score_unreliable` is set to `True`, the prediction state is forced to UNRELIABLE, and no score or fresh_label is returned. The warning message includes the exact distance and threshold.

---

## 8. How It Appears at Inference

At runtime in `predict_cli.py`:

```python
# predict_cli.py

train_mean      = np.load("models/train_mean.npy")       # (349,)
train_precision = np.load("models/train_precision.npy")  # (349, 349)

# After preprocessing image to Xfinal shape (1, 349):
dist = mahalanobis_dist(Xfinal, train_mean, train_precision)
zone = mahal_zone(dist, cfg["mahal_thresh_caution"], cfg["mahal_thresh_ood"])
is_ood = (zone == "ood")

# Gate effect:
score_unreliable = unstable or is_ood
```

An example OOD output:

```
Vegetable : apple (88.10%,  gap=62.30%)
State     : UNRELIABLE
Mahal     : 32.15  [ood]
[!] OOD — Mahalanobis dist=32.15 > threshold=30.438.
    Outside training distribution.
```

Note that the vegetable prediction (88.1% confidence) and the raw freshness score are both computed — they are just not returned. The model was confident; the OOD gate overrides that confidence. This is the intended behaviour: a high-confidence prediction on an OOD sample is a false confidence, not a reason to trust the output.

A caution-zone example:

```
Vegetable : potato (71.20%,  gap=18.40%)
State     : TENTATIVE
Score     : 49.10 / 100
Norm      : per-veg
Mahal     : 26.83  [caution]
[!] CAUTION — Mahalanobis dist=26.83 in caution zone [24.167, 30.438].
```

The prediction is not suppressed — the sample is not OOD — but the caution warning signals to the caller that the sample is in the outer tail of the training distribution.

---

## 9. Why a Single Global Distribution (Not Per-Class)

The OOD detector fits one covariance model to all 8,883 training samples across all five vegetable classes. It does not fit separate models per vegetable. This is the correct design for two reasons.

**The goal is to detect genuinely novel inputs, not misclassifications.** A cucumber that looks like a potato is not an OOD sample — it is an in-distribution sample that is being misclassified. The centroid consistency gate (`G4`) handles that case. The Mahalanobis gate's role is to detect inputs whose feature vectors lie outside the entire occupied region of the training distribution, regardless of which class they would have been assigned to.

**Per-class models would create coverage gaps.** If separate covariance models were fit per class, a sample that falls between two vegetable clusters — outside both per-class ellipses but inside the global training region — would appear OOD when it is not. The global model correctly covers the full region that the training data occupies, including the space between classes.

The per-class centroid thresholds used by G4 (the centroid consistency gate) serve the complementary function of catching within-distribution misclassifications.

---

## 10. Gate Ablation Result (Test Set)

The gate ablation study in `evaluate_models.py` measures the actual cost and benefit of the OOD gate on the held-out test set:

```
  Gate     Fires  Fire%  Catch_W  Block_C   Δ_acc    Δ_cov   Verdict
  ────────────────────────────────────────────────────────────────────
  G1_OOD     62   2.4%      1       61    −0.0004  +0.0760   REVIEW

  Baseline: acc=0.9898  coverage=0.923
```

62 test samples (2.4%) exceeded `thresh_ood = 30.438`. Of those 62:
- 1 had an incorrect freshness prediction — the gate correctly blocked this error.
- 61 had correct freshness predictions — the gate unnecessarily withheld these results.

Disabling the gate would expand coverage by 7.6 percentage points (from 92.3% to ~99.9%) while reducing RELIABLE accuracy by only 0.04 percentage points. The verdict is **REVIEW** rather than KEEP because the in-distribution accuracy benefit is small.

The REVIEW verdict does not mean the gate should be removed. The 62 OOD-flagged samples genuinely lie outside the P99 of the training Mahalanobis distribution. Their freshness predictions happened to be correct on this particular test set, but these are exactly the samples where the model's calibration guarantees break down under distribution shift. The gate provides deployment robustness — protection against inputs that are subtly different from training conditions in ways that may not affect this test set but could affect a different one.

The practical decision depends on deployment context:

| Deployment condition | Recommendation |
|---------------------|----------------|
| Tightly controlled imaging (fixed camera, background, lighting) | Consider downgrading to caution-only (warning without UNRELIABLE) to recover 7.6% coverage |
| Open-ended deployment (varied cameras, backgrounds, conditions) | Keep the OOD gate as-is |
| Increasing OOD rate over time (detected via monitoring) | Tighten `thresh_ood` or collect more training data for the problematic conditions |

---

## 11. OOD Rate Monitoring

The evaluation reports OOD rates on val and test as a consistency check:

```
OOD rate — validation : 0.0181  (23 / 1,269 samples)
OOD rate — test       : 0.0244  (62 / 2,539 samples)
Difference            : 0.0063  (0.63%)  < 5% stability threshold  [OK]
```

A discrepancy larger than 5 percentage points would suggest the Mahalanobis threshold does not transfer stably between splits — either because the val and test distributions differ significantly, or because the threshold was accidentally overfit to the val distribution. The 0.63% difference here confirms stable transfer.

In deployment, monitoring the rolling OOD rate on production images serves as a distributional alarm: a sudden increase in OOD flags signals that incoming images have changed in a way the model was not trained on, which is actionable information even before prediction errors become apparent.

---

## 12. Summary of Configuration

All Mahalanobis parameters are stored in `models/scoring_config.json` and `models/train_mean.npy` / `models/train_precision.npy`:

| Parameter | Value | Source |
|-----------|-------|--------|
| Feature space | 349 dimensions (union feature set) | `selected_union_features.npy` |
| Covariance method | LedoitWolf shrinkage | `sklearn.covariance.LedoitWolf` |
| Fitted on | `X_train` (8,883 samples) | Training split only |
| `train_mean` | shape (349,), float32 | `models/train_mean.npy` |
| `train_precision` | shape (349, 349), float32 | `models/train_precision.npy` |
| `thresh_caution` | **24.167** | P90 of training distances |
| `thresh_ood` | **30.438** | P99 of training distances |
| OOD rate on val | 1.81% | 23 / 1,269 samples |
| OOD rate on test | 2.44% | 62 / 2,539 samples |
| OOD gate fires (test) | 62 (2.4%) | Gate ablation |
| Catch_W (wrong predictions blocked) | 1 | Gate ablation |
| Block_C (correct predictions withheld) | 61 | Gate ablation |
| Verdict | REVIEW | Δ_acc=−0.0004, Δ_cov=+0.0760 |