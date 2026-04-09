# Centroid Consistency Gate

The centroid consistency gate is a geometric cross-check that runs after the vegetable classifier produces a prediction. Its purpose is to detect cases where the SVM has predicted a class with high probability, but the sample's position in feature space is geometrically inconsistent with that class — a situation where the probability score is misleading. When the gate fires, the prediction is downgraded to TENTATIVE and global normalization bounds replace per-vegetable bounds.

---

## 1. The Problem It Solves

The freshness score is normalized using per-vegetable p5/p95 bounds — a separate calibration for each vegetable class. For this normalization to be valid, the predicted vegetable must be correct. If a capsicum is misclassified as a banana, the system applies banana's normalization bounds to a capsicum's raw freshness margin. The resulting score is numerically valid but calibrated to the wrong reference distribution — it is meaningless.

The vegetable confidence gate (G3) catches cases where the SVM is visibly uncertain: low top-1 probability or a small gap between top-1 and top-2. But the SVM can produce a high-confidence wrong prediction. For example:

```
veg_svm.predict_proba() → [apple: 0.01, banana: 0.98, capsicum: 0.01, ...]

veg_conf = 98%    ← passes the confidence gate
conf_gap = 97%    ← passes the gap gate

but the image is actually a capsicum
```

The confidence score tells you how certain the SVM is given its learned decision boundary. It does not tell you whether the sample geometrically belongs to the predicted class. The centroid gate provides this second check.

---

## 2. What a Centroid Is

For each vegetable class, the centroid is the mean feature vector across all training samples of that class:

```python
# train_svm.py

class_centroids = {}
for i, veg in enumerate(le.classes_):
    mask = (yveg_encoded == i)
    class_centroids[veg] = X_train[mask].mean(axis=0).tolist()
```

This gives five centroid vectors, each of shape (349,), representing the geometric centre of each vegetable's cluster in the 349-dimensional union feature space.

```
Feature space (simplified to 2D for illustration):

            ●  ●  ●                   ○  ○
         ●  ●  ●  ●  ●             ○  ○  ○  ○
            ●  ●  ●                   ○  ○
              (×)                       (×)
          banana                      apple
         centroid                    centroid

              ■  ■                ▲  ▲  ▲
           ■  ■  ■  ■          ▲  ▲  ▲  ▲  ▲
              ■  ■                ▲  ▲  ▲
               (×)                  (×)
            potato               capsicum
           centroid              centroid
```

The centroids are saved to `models/class_centroids.npy` as a (5, 349) float32 array, indexed in the same order as `LabelEncoder.classes_`.

---

## 3. The Centroid Ratio

At inference, for a new sample with feature vector `x` (shape 349), the system computes L2 distances to all five class centroids and then forms a ratio:

```python
# predict_cli.py

x_flat = Xfinal.flatten()                                         # shape (349,)
dists_to_centroids = np.linalg.norm(class_centroids - x_flat, axis=1)  # shape (5,)

d_pred   = dists_to_centroids[veg_idx]           # distance to predicted class centroid
d_second = next(dists_to_centroids[j]
                for j in sorted_centroid_idx if j != veg_idx)    # nearest other centroid

centroid_ratio = d_pred / (d_second + 1e-9)
```

The ratio compares how far the sample is from the predicted class centroid relative to how far it is from the nearest *other* class centroid:

```
centroid_ratio = d_pred / d_second

ratio < 1.0  →  sample is closer to its predicted class than to any other
ratio = 1.0  →  sample is equidistant between predicted class and next nearest
ratio > 1.0  →  sample is closer to another class than to the one it was predicted as
```

---

## 4. Why a Ratio and Not Raw Distance

Raw distance alone is not informative because the scale of the feature space varies. A raw distance of 10 might be very small if the class clusters are spread far apart (meaning the sample is well inside its cluster), or very large if the clusters are compact. The ratio removes this scale dependence by expressing the distance to the predicted centroid *relative to* the distance to the nearest alternative.

```
Example A — compact clusters (raw distances are small):
  d_pred = 4.1   d_second = 8.7
  ratio  = 0.47  ← clearly inside the predicted cluster

Example B — diffuse clusters (raw distances are large):
  d_pred = 42.3  d_second = 89.1
  ratio  = 0.47  ← same ratio, same geometric interpretation

Both cases: the sample is roughly half as far from its predicted centroid as from
the nearest alternative. Both should be treated equivalently.
```

In contrast, raw distance thresholds would need separate calibration for each class because cluster sizes differ.

---

## 5. Threshold Calibration (P95 of Correct Val Predictions)

Each vegetable class has its own threshold, calibrated from the validation set:

```python
# train_svm.py — computed on the full val set

for i, veg in enumerate(le.classes_):
    correct_mask = (veg_pred_full == i) & (yveg_val_enc_full == i)
    # correct_mask: val samples that were predicted as this class AND are this class

    per_class_ratio_thresh[veg] = float(
        np.percentile(all_ratios[correct_mask], 95)
    )
```

For each class, the threshold is the **P95 of centroid ratios among correctly classified validation samples**. This means: among all the validation images the model correctly identified as (say) banana, 95% of them had a centroid ratio at or below this threshold. A new sample exceeding this threshold is in the outer 5% of the geometric distribution of correctly classified bananas — geometrically suspicious even if the SVM says "banana: 98%".

**Actual calibrated thresholds from the training run:**

```
[INFO] Per-class centroid ratio thresholds (P95 of correct val predictions):
  apple         threshold=1.0220  (n=448 correct val predictions)
  banana        threshold=0.9552  (n=469)
  capsicum      threshold=0.9973  (n=114)
  cucumber      threshold=1.0257  (n=91)
  potato        threshold=0.9740  (n=147)
```

Notice that banana and potato have thresholds below 1.0 (0.9552 and 0.9740). This means that for these classes, even correctly classified validation samples were predominantly *closer* to their own centroid than to any other — the clusters are compact. A banana with a ratio of 0.96 would pass the gate for apple (threshold 1.0220) but fail the gate for banana (threshold 0.9552).

Apple and cucumber have thresholds above 1.0 (1.0220 and 1.0257), meaning correctly classified samples from these classes can be slightly closer to another centroid than to their own and still be geometrically consistent. This reflects less compact or more overlapping clusters for these classes in the 349-dimensional feature space.

---

## 6. What Happens When the Gate Fires

```python
# predict_cli.py

class_inconsistent = centroid_ratio > centroid_ratio_thresh

use_per_veg = veg_confident and not class_inconsistent
bounds = per_veg.get(veg_name, globl) if use_per_veg else globl
```

When `class_inconsistent = True`, two things change:

**Per-vegetable normalization bounds are replaced by global bounds.** The system cannot safely apply banana's p5/p95 calibration to a sample that is geometrically inconsistent with the banana cluster. Global bounds, derived from all training samples, are used instead. This produces a less precise score but one that is not contaminated by wrong-class calibration.

**`decision_unreliable` is set to True, producing TENTATIVE.** The fresh_label is withheld. The score is shown but marked as uncertain. The warning message includes the exact ratio and threshold:

```
[!] CLASS INCONSISTENCY — centroid ratio=1.083 (threshold=0.9552).
    Sample is not clearly in the banana cluster.
    Global normalization bounds applied.
```

The gate does **not** produce UNRELIABLE — it produces TENTATIVE. The sample has a valid score (computed with global bounds) but the system is not confident enough in the vegetable identity to provide a fresh_label.

---

## 7. A Worked Example

Suppose a capsicum image is submitted and the vegetable SVM predicts banana with 96% confidence.

```
veg_svm output: banana=0.96, capsicum=0.03, apple=0.01, ...
veg_conf = 96.0%   → passes G3 confidence gate (threshold 70%)
conf_gap = 93.0%   → passes G3 gap gate (threshold 15%)
```

Now the centroid check runs:

```
x_flat = feature vector of the capsicum image [349 values]

Distances to centroids:
  apple    : 31.2
  banana   : 28.4   ← predicted class
  capsicum : 26.8   ← nearest other centroid
  cucumber : 35.1
  potato   : 33.7

d_pred   = 28.4  (distance to banana centroid)
d_second = 26.8  (distance to capsicum centroid — nearest alternative)

centroid_ratio = 28.4 / 26.8 = 1.060

banana threshold = 0.9552

1.060 > 0.9552  →  class_inconsistent = True
```

Despite the SVM's 96% confidence, the sample is geometrically closer to the capsicum centroid than to the banana centroid. The gate fires. Global bounds are applied, the output is TENTATIVE, and the fresh_label is withheld.

```
Vegetable : banana (96.00%,  gap=93.00%)
State     : TENTATIVE
Score     : 54.3 / 100    (computed with global bounds)
Norm      : global
[!] CLASS INCONSISTENCY — centroid ratio=1.060 (threshold=0.9552).
    Sample is not clearly in the banana cluster.
    Global normalization bounds applied.
```

---

## 8. Relation to the OOD Gate

The centroid gate and the Mahalanobis OOD gate serve different purposes and catch different failure modes.

| Property | Centroid Gate (G4) | OOD Gate (G1) |
|----------|-------------------|----------------|
| What it measures | Distance to predicted class centroid vs. next nearest centroid | Distance from the entire training distribution centroid |
| What it catches | Vegetable misclassifications (in-distribution but wrong class) | Genuinely novel inputs outside all training classes |
| Effect when fires | TENTATIVE (global bounds, score shown, no label) | UNRELIABLE (no score, no label) |
| Threshold source | P95 of correct val predictions, per class | P90/P99 of training Mahalanobis distances |
| Metric used | Euclidean L2 ratio (centroid-relative) | Mahalanobis distance (covariance-weighted) |

A sample can fail both gates simultaneously — it can be both geometrically inconsistent with its predicted class AND outside the training distribution overall. In that case, G1 (OOD) takes precedence because it produces UNRELIABLE rather than TENTATIVE.

**Test-set wrong-veg detection breakdown:**

```
  Total veg misclassifications   : 10
  Caught by OOD gate only        :  5
  Caught by centroid gate only   :  2
  Caught by both                 :  0
  Missed by both (blind spots)   :  3
    Of blind spots, freshness also wrong: 0  ← zero catastrophic failures
```

The centroid gate uniquely caught 2 vegetable misclassifications that the OOD gate missed — samples that were in-distribution overall but landed in the wrong class's region. Without the centroid gate, those 2 samples would have reached RELIABLE with wrong per-vegetable normalization bounds applied to their freshness scores.

---

## 9. Configuration

All centroid gate parameters are stored in `models/scoring_config.json` and `models/class_centroids.npy`:

| Parameter | Value | Source |
|-----------|-------|--------|
| Centroids | shape (5, 349) float32 | Mean of X_train per class |
| Fitted on | X_train (8,883 samples) | Training split only |
| Thresholds | Per-class, see table above | P95 of correct val ratios |
| Effect when fires | TENTATIVE, global bounds | `class_inconsistent=True` |
| Key in scoring_config | `centroid_ratio_thresholds` | dict of {veg: threshold} |