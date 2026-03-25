# Full Pipeline Audit — Vegetable Freshness Grading

---

## 1. Executive Summary

The pipeline is **partially solid**. Data handling is clean: no leakage found, calibration is done on the validation split, and test metrics are correctly measured on held-out test data. The OOD safety stack (Mahalanobis, boundary threshold, centroid consistency, augmentation instability) is well-designed. However, the feature selection is task-mismatched for the vegetable classifier, no hyperparameter search was run, the RBF kernel is assumed not justified, the grade thresholds are arbitrary design constants with no empirical validation, and there are minor but real inconsistencies between `evaluate_models.py` and `predict_cli.py` around centroid gating.

---

## 2. Doubt 1 — Feature Sharing Across Two SVMs

### What the code does

**Where selection happens:** `preprocess_and_rank.py`, function `rank_features()`.

```python
def load_training_features():
    X = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
    y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_train.npy"))
    return X, y_fresh

# rank_features uses y_fresh ONLY
clf = xgb.XGBClassifier(...)
clf.fit(X_scaled, y_fresh.astype(int))
```

The XGBoost gain importance is computed using **freshness labels only**. `y_veg` is never passed to this function.

**How the indices propagate:**

```python
# preprocess_and_rank.py
np.save(os.path.join(MODEL_DIR, "selected_features.npy"), selected_idx)

# train_svm.py  _load_split()
X_f = X_s[:, selected]   # same selected for both classifiers
```

Both `veg_model.fit(X_train, yveg_encoded)` and `fresh_model.fit(X_train, y_fresh_train)` operate on the identical 100-column matrix.

### Validity assessment

**The assumption:** The 100 features most discriminative of fresh vs rotten also contain sufficient signal to discriminate apple vs banana vs capsicum vs cucumber vs potato.

This assumption holds **empirically** (98.94% vegetable accuracy) but is never checked methodologically. The code acknowledges it:

```python
# evaluate_models.py
print("  FEATURE SELECTION NOTE: XGBoost ranks on freshness-only labels.")
print("  Vegetable classifier reuses the same top-100 subset.")
print("  Union-of-top-features fix requires full retrain.")
print("  Deferred while veg accuracy remains 99.13%.")
```

**Failure mode:** If the dataset distribution shifts such that freshness-discriminating features (texture, edge density, Laplacian variance, color variance) no longer overlap with vegetable-discriminating features (shape, structural color), the vegetable accuracy degrades silently. A new vegetable category, unusual lighting, or a domain shift can expose this. The freshness classifier would remain unaffected.

### Minimal fix

Compute a second XGBoost importance ranking using `y_veg` labels. Use the union (or each separately) for the respective SVM:

```python
# In preprocess_and_rank.py, add:
selected_veg, _ = rank_features(X_scaled, y_veg_encoded, seed=42, top_k=100)
np.save(os.path.join(MODEL_DIR, "selected_veg_features.npy"), selected_veg)
np.save(os.path.join(MODEL_DIR, "selected_fresh_features.npy"), selected_idx)
```

Then in `train_svm.py / _load_split()`, pass the relevant feature set to each classifier. This requires storing two separate 100-dim representations and routing inference accordingly in `predict_cli.py`.

**Tradeoff:** This doubles the dimensionality overhead at inference and requires retraining both SVMs. Given 98.94% is already near-ceiling, the empirical gain is likely small on the current dataset.

---

## 3. Doubt 2 — Kernel Choice (RBF)

### What the code does

```python
# train_svm.py
veg_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")
fresh_model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
```

`C=1.0` and `gamma="scale"` (= `1 / (n_features × X.var())`) are scikit-learn defaults. **No grid search was run. No comparison with linear kernel was done in the actual pipeline.**

### What evidence exists

The `dataset_validation.ipynb` uses a linear SVM diagnostically:

```python
svm_val = SVC(kernel='linear', probability=False)
svm_val.fit(X_train_s, y_train)
margins_train = svm_val.decision_function(X_train_s)
# Output: Fresh mean = 3.2039, Rotten mean = -4.0467, Delta = 7.2506  → "PASS"
```

But this is on the **full 1312-feature scaled space**, not the 100-feature selected space used by the actual pipeline. These are different experiments and the linear result does not justify RBF choice for the actual 100-feature SVM.

### Why RBF is not justified here

- The feature space has 100 dimensions and ~8883 training samples. RBF is standard for non-linearly separable problems, but the diagnostic notebook already shows very strong linear separation in the full feature space.
- `gamma="scale"` adapts to feature variance but is not tuned.
- There is no ablation showing RBF outperforms linear on these 100 features.
- `C=1.0` controls the regularization boundary. With a moderately large dataset (8883 samples), an untested `C=1.0` could be either over- or under-regularizing.

**The missing evidence:** A 5-fold cross-validation comparison on the training set of `{linear, rbf} × C ∈ {0.1, 1, 10, 100}` measuring both accuracy and ROC-AUC.

### Minimal experiment

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'],    'C': [0.1, 1, 10, 100],
                           'gamma': ['scale', 0.01, 0.001]}
]
gs = GridSearchCV(
    SVC(class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    refit=False
)
gs.fit(X_train_100features, y_fresh_train)
```

Run once on training data. Report best CV score per kernel type. If linear achieves ≥ 99% of RBF ROC-AUC, use linear (it removes the gamma sensitivity entirely).

**Tradeoff of the fix:** Extra training time (~minutes at this scale). If linear is comparable, inference also becomes faster.

---

## 4. Doubt 3 — Accuracy Mismatch

### What the code does

```python
# train_split.py
def main(val_size=0.10, test_size=0.20, random_state=42):
    train_idx, valtest_idx = train_test_split(
        idx, test_size=val_size + test_size, ...)  # 30% goes to val+test pool
    val_rel = val_size / (val_size + test_size)    # 0.10/0.30 = 0.333
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=1.0 - val_rel, ...)  # 2/3 of pool → test
```

With 12,642 total images:
- Train: ~8,849 (70%)
- Val: ~1,264 (10% of total, 1/3 of pool)
- Test: ~2,528 (20% of total, 2/3 of pool)

The overview confirms: **val = 1,269**, **test = 2,539**.

```python
# evaluate_models.py
X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y_fresh_test = np.load(os.path.join(MODEL_DIR, "y_fresh_test.npy")).astype(int)
# ...
accuracy_score(y_fresh_test, yfresh_pred)  # → 0.9799
```

The freshness confusion matrix totals confirm the test set size:
```
1294 + 29 + 22 + 1194 = 2539  ✓ (test set)
```

### Verdict

**There is no mismatch.** The user's premise was inverted. The actual sizes are val=1,269 and test=2,539. The reported accuracy 0.9799 (freshness) and 0.9894 (vegetable) are correctly measured on the **test set** (2,539 samples). `evaluate_models.py` loads `X_test.npy` and never touches validation data for metrics.

**Exact verification:**
```python
import numpy as np
X_test = np.load("models/X_test.npy")
print(X_test.shape[0])   # must print 2539 (or ~2539)
X_val  = np.load("models/X_val.npy")
print(X_val.shape[0])    # must print 1269 (or ~1269)
```

**No fix needed** for this doubt.

---

## 5. Doubt 4 — Platt Scaling

### What the code does

```python
# train_svm.py
# Vegetable SVM — Platt scaling ON
veg_model = SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, class_weight="balanced")
veg_model.fit(X_train, yveg_encoded)

# Freshness SVM — Platt scaling OFF
# Comment: "probability=True is intentionally omitted here.
# The freshness path uses decision_function() (signed margin)
# for scoring and predict() for the binary label."
fresh_model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
fresh_model.fit(X_train, y_fresh_train.astype(int))
```

### Is this consistent?

**Yes, by design:**

- Vegetable SVM probabilities are used for gating (`veg_conf >= 70%`, gap `>= 15%`).
- Freshness SVM margin is used as the grading signal via `decision_function()`.

Using Platt scaling on freshness would impose probability semantics on a margin-based scoring system, which the code explicitly rejects.

### Where is Platt calibration performed?

Scikit-learn's `SVC(probability=True)` performs internal **5-fold cross-validation on the training data** to fit the sigmoid parameters (not on the validation set). The code does not call `CalibratedClassifierCV`. This means:

- The sigmoid is fitted on training data only, not on the independently held-out validation set.
- This is acceptable, but calibration on a separate held-out set would be more reliable.

### Is sigmoid assumption validated?

**NOT FOUND IN FILES.** No calibration curve, reliability diagram, or Brier score is computed anywhere for the vegetable probability estimates. This means there is no evidence that `predict_proba()` returns well-calibrated probability values.

### Does miscalibration matter?

The gating thresholds `veg_conf_thresh=0.70` and `veg_gap_thresh=0.15` are **design constants**, not data-calibrated. Even if Platt probabilities are off by ±15%, the gating behavior changes only at the threshold boundary. For high-confidence predictions (>90%), Platt miscalibration is irrelevant.

### Minimal fix

If precise probability calibration is required:

```python
# After training veg_model WITHOUT probability=True:
base_svc = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
base_svc.fit(X_train, yveg_encoded)

from sklearn.calibration import CalibratedClassifierCV
veg_model = CalibratedClassifierCV(base_svc, cv='prefit', method='sigmoid')
veg_model.fit(X_val, y_veg_val_enc)  # Calibrate on held-out val
```

**Tradeoff:** Requires a dedicated calibration pass on validation data, slightly reducing the signal available for threshold calibration. Given the design-constant gating thresholds, this is low priority unless the vegetable confidence values are being used for user-facing probability estimates.

---

## 6. Doubt 5 — Grading Method

### Full step-by-step trace

**Step 1 — Raw SVM margin:**
```python
# predict_cli.py
raw = float(fresh_svm.decision_function(Xfinal)[0])
```
`decision_function` returns the signed distance from the SVM hyperplane. Positive = fresh side, negative = rotten side.

**Step 2 — Bounds selection:**
```python
# predict_cli.py
use_per_veg = veg_confident and not class_inconsistent
bounds = per_veg.get(veg_name, globl) if use_per_veg else globl
```

Bounds come from `scoring_config.json`, calibrated in `train_svm.py` on the validation set:
```python
val_decisions = fresh_model.decision_function(X_val)
global_bounds = {
    "p5" : float(np.percentile(val_decisions, 5)),   # −2.4208
    "p95": float(np.percentile(val_decisions, 95)),  # +2.0882
}
per_veg_bounds = compute_per_veg_bounds(val_decisions, y_veg_val, veg_classes)
```

**Step 3 — Normalization:**
```python
def normalize_score(raw, bounds):
    p5, p95 = bounds["p5"], bounds["p95"]
    denom = p95 - p5
    return float(np.clip((raw - p5) / denom * 100.0, 0.0, 100.0))
```

Mathematically:
```
score = clip( (raw − p5) / (p95 − p5) × 100,  0, 100 )
```

**Step 4 — Grade band mapping:**
```python
# utils.py
if score >= 85: return "High"
if score >= 65: return "Medium"
if score >= 40: return "Low"
return "Very Low"
```

### Concrete numerical example (from overview / scoring_config values)

**Input:** banana image, `raw = +0.85`

Banana per-veg bounds: `p5 = −2.4324`, `p95 = +2.1062`

```
denom = 2.1062 − (−2.4324) = 4.5386
score = (0.85 − (−2.4324)) / 4.5386 × 100
      = 3.2824 / 4.5386 × 100
      ≈ 72.3
Band: 65 ≤ 72.3 < 85  →  "Medium"
```

**Second example:** rotten sample, `raw = −1.8`, global bounds (`p5=−2.42`, `p95=2.09`)

```
denom = 2.09 − (−2.42) = 4.51
score = (−1.8 − (−2.42)) / 4.51 × 100
      = 0.62 / 4.51 × 100
      ≈ 13.7
Band: < 40  →  "Very Low"
```

### Mathematical issues in the grading method

**Issue 1: Decision boundary ≠ score 50**

The normalization anchors at p5 and p95 of the **combined** validation distribution (fresh + rotten mixed). The decision boundary (`raw = 0`) maps to:

```
Using global bounds p5=−2.42, p95=2.09:
score at raw=0  =  (0 − (−2.42)) / (2.09 − (−2.42)) × 100
               =  2.42 / 4.51 × 100
               ≈  53.7
```

Score 50 corresponds to `raw = p5 + 0.5 × (p95 − p5) = −2.42 + 2.255 = −0.165`, which is slightly on the **rotten side**. This is never documented but is a direct consequence of the mixed-distribution normalization. It means a "Low" band score (40–64) does not straightforwardly map to "uncertain about freshness" — some Low scores are on the rotten side of the boundary.

**Issue 2: Grade thresholds are arbitrary**

The thresholds 85/65/40 are defined in `scoring_config.json` under `"grade_thresholds"`:
```python
"grade_thresholds": {"truly_fresh": 85, "fresh": 65, "moderate": 40}
```

They were not derived from any calibration. There is no analysis showing that score ≥ 85 implies, say, ≥ 95% precision on the fresh class. The thresholds are design choices with no empirical grounding.

**Issue 3: Scores not comparable across vegetables (by design)**

With per-vegetable normalization, a potato score of 70 and a banana score of 70 use different p5/p95 anchors and thus mean different distances from the hyperplane. The code acknowledges this:

```python
# evaluate_models.py
print(f"  Per-vegetable normalization fits separate p5/p95 bounds per class.")
print(f"  This means scores ARE locally comparable within a vegetable class")
print(f"  but are NOT globally comparable across vegetables by design.")
```

This is correct documentation but should be surfaced to end users.

---

## 7. Assumptions (All Implicit)

| # | Assumption | Where it lives | If violated |
|---|-----------|---------------|-------------|
| A1 | Top-100 freshness features also discriminate vegetable types | `preprocess_and_rank.py` | Vegetable accuracy degrades, freshness scoring is unaffected |
| A2 | RBF kernel with C=1.0, gamma="scale" is near-optimal | `train_svm.py` | Accuracy 2–5% below tuned model |
| A3 | Validation decision-function distribution is representative of deployment distribution | `train_svm.py` calibration | p5/p95 bounds miscalibrated; scores drift |
| A4 | Single global Mahalanobis centroid adequately captures in-distribution boundary for all 5 vegetable classes | `train_svm.py` | False OOD flags for vegetables that are far from the mixed centroid |
| A5 | Platt sigmoid maps SVM scores to valid probabilities for vegetable gating | `train_svm.py` | Gating fires at wrong confidence levels |
| A6 | Grade thresholds (85/65/40) have meaningful accuracy interpretation | `utils.py` | Bands are decorative; High≠reliable |
| A7 | Score normalization [0,100] is monotone in freshness quality | `predict_cli.py` | Correct for fresh/rotten separation, not for intra-class ordering (stated limitation) |
| A8 | Augmentation instability (score range across 6 augmentations) correlates with deployment error | `train_svm.py` calibrate_unstable_range_thresh | Gate fires unnecessarily or misses real instability |

---

## 8. Vulnerabilities and Fixes

### V1 — Feature Selection Task Mismatch

**What:** `selected_features.npy` is computed with `y_fresh` labels only. Vegetable SVM uses it unchanged.

**Failure mode:** Under distribution shift where freshness-relevant features diverge from vegetable-relevant features, vegetable accuracy drops without warning.

**Impact:** Medium. Current dataset shows 98.94% veg accuracy, so the assumption holds now.

**Minimal fix:** Separate feature selection runs for each task; route each SVM to its own feature set. (Acknowledged in code, correctly deferred.)

---

### V2 — No Hyperparameter Tuning

**What:** `SVC(C=1.0, gamma="scale")` for both SVMs.

**Failure mode:** Suboptimal decision boundary. `C=1.0` may be over-regularizing on 100 features × 8883 samples. `gamma="scale"` is a heuristic.

**Impact:** Unknown, but measured AUC 0.9979 suggests current parameters are not catastrophically wrong.

**Minimal fix:** 5-fold grid search on training data for `{C: [0.1, 1, 10, 100], kernel: [linear, rbf], gamma: [scale, 0.001, 0.01]}`. Run once before full training.

---

### V3 — Centroid Gating Missing from Evaluation and Calibration

**What:** `predict_cli.py` uses `class_inconsistent` (centroid ratio check) to fall back to global bounds. `evaluate_models.py / get_deployed_scores()` does NOT implement this check:

```python
# evaluate_models.py  get_deployed_scores()
veg_ok = (top1 >= conf_thresh) and (gap >= gap_thresh)
bounds = per_veg.get(veg, globl) if (veg_ok and veg in per_veg) else globl
# ↑ centroid check absent
```

`calibrate_unstable_range_thresh` in `train_svm.py` also omits centroid gating despite the comment claiming "exact mirror of predict_cli.py."

**Failure mode:** Inversion rates and deployed score distributions reported in `evaluate_models.py` are slightly different from actual deployment behavior. Instability threshold is calibrated on a slightly different scoring path than what runs at inference.

**Impact:** Low (vegetable classifier is 98.94% accurate; centroid failures are rare). But the comment claiming exact mirroring is false.

**Minimal fix:**
```python
# In evaluate_models.py  get_deployed_scores():
# After computing veg_ok, add centroid ratio check:
if veg_ok:
    x_f = X[i]
    dists_c = np.linalg.norm(class_centroids_arr - x_f, axis=1)
    pred_idx_int = int(np.argmax(veg_probs[i]))
    sorted_c = np.argsort(dists_c)
    d_pred = dists_c[pred_idx_int]
    d_sec = next(dists_c[j] for j in sorted_c if j != pred_idx_int)
    ratio = d_pred / (d_sec + 1e-9)
    thresh_c = float(per_cls_thresh.get(veg, 1.0))
    if ratio > thresh_c:
        veg_ok = False
bounds = per_veg.get(veg, globl) if veg_ok else globl
```

Apply the same change inside `calibrate_unstable_range_thresh`.

---

### V4 — LedoitWolf Mean Inconsistency

**What:**
```python
lw = LedoitWolf().fit(X_train)
train_mean = X_train.mean(axis=0).astype(np.float32)   # ← separate
precision = lw.precision_.astype(np.float32)
```

`LedoitWolf` computes its own location estimate (`lw.location_`). Using a separately computed `X_train.mean()` is inconsistent — it couples a non-robust mean with a robust precision matrix.

**Failure mode:** Negligible for well-centered data; could matter under heavy class imbalance or outliers.

**Minimal fix:** Replace `X_train.mean(axis=0)` with `lw.location_`.

---

### V5 — OOD Check Uses Single Global Centroid

**What:** Mahalanobis distance is computed from a single mean over all 5 vegetable classes mixed together.

**Failure mode:** The mixed distribution has a wider, multimodal spread. A vegetable type that is at the fringe of the mixed cloud (e.g., potato, which has the weakest separation) may appear OOD even when in-distribution. Acknowledged in `mahalanobis_OOD_detection.md`.

**Impact:** OOD rate validation shows 1.02% val / 0.91% test, which is consistent. But class-specific OOD sensitivity is not validated.

**Minimal fix:** Compute per-class Mahalanobis distance and flag a sample as OOD only if it is far from ALL class centroids (max over classes of normalized distance). This requires per-class precision matrices, which may be rank-deficient if class sample counts are small (n < 100 features). Approximate with LedoitWolf per class.

---

### V6 — Grade Thresholds Have No Accuracy Backing

**What:** Thresholds `85/65/40` are design constants. "High" confidence band does not correspond to any measured error rate.

**Failure mode:** A system integrator interprets "High" as >98% precision and makes operational decisions on that. The code doesn't provide this guarantee.

**Minimal fix:** Add a calibration report to `evaluate_models.py`:
```python
for threshold_name, threshold_val in [('High', 85), ('Medium', 65), ('Low', 40)]:
    mask = (scores_deployed >= threshold_val)
    if mask.sum() > 0:
        acc = accuracy_score(y_fresh_test[mask], yfresh_pred_all[mask])
        print(f"  Score >= {threshold_val} ({threshold_name}): "
              f"n={mask.sum()}, acc={acc:.4f}")
```

---

## 9. Further Optimizations

1. **Separate feature sets per task (V1 fix):** Strongest methodological improvement. Run feature selection on `y_veg` labels separately and compare against the current shared set.

2. **Hyperparameter tuning (V2 fix):** Single grid search (linear vs RBF, C in [0.1–100]) on training data. With 8883 samples and 100 features, this takes minutes with `n_jobs=-1`.

3. **Grade threshold empirical grounding:** Compute precision-at-threshold curves on validation set to replace arbitrary 85/65/40 with data-driven boundaries.

4. **Per-class Mahalanobis OOD:** Replace single-centroid OOD with min-over-classes distance or mixture model approach. Given per-class centroids already exist in `class_centroids.npy`, this is straightforward.

5. **Platt calibration on validation set:** Use `CalibratedClassifierCV(cv='prefit')` on `X_val` for the vegetable SVM if confidence probability values will be user-facing.

6. **Formal threshold selection:** `threshold_selection.py` already implements a coverage-maximizing formal optimization (`select_thresholds`). It is not used in the actual pipeline. Replacing the heuristic sweep in `calibrate_boundary_threshold` with the formal approach would provide guaranteed risk-coverage tradeoffs.

7. **Fix `evaluate_models.py` centroid gating inconsistency (V3 fix):** Small fix, removes an undocumented discrepancy between evaluation and deployment.

---

## 10. Final Verdict

**Partially solid.**

### What is scientifically sound

- Data split is clean: 70/10/20 with stratification, no leakage anywhere in calibration or evaluation
- All calibration (normalization bounds, boundary threshold, centroid ratios, augmentation instability, Mahalanobis thresholds) uses the correct splits
- Test set is preserved and correctly evaluated
- The pipeline has genuine uncertainty quantification (OOD detection, boundary proximity, instability, centroid consistency) rather than just raw accuracy
- The grading formula is mathematically consistent and the bounds are anchored to validation data (not training data, a documented C2 fix)
- The code acknowledges its own limitations clearly

### What is not scientifically defensible as-is

- **Feature selection task mismatch:** freshness-only XGBoost gain decides features for the vegetable classifier. Justified only by empirical result on current dataset.
- **No hyperparameter tuning:** both SVMs use scikit-learn defaults. C=1.0, gamma="scale" are not validated choices.
- **RBF kernel is assumed:** linear kernel is tested diagnostically on different features; no comparison on actual 100-feature space.
- **Grade thresholds are arbitrary:** 85/65/40 have no empirical grounding in error rates or precision.
- **Centroid gating is missing** from `evaluate_models.py`'s deployed score calculation and from augmentation calibration, creating a silent discrepancy with `predict_cli.py`.

### Bottom line

The reliability infrastructure is unusually careful for a project at this scale. The core weakness is that the model parameters were never tuned and the scoring interpretation was never empirically validated. A single well-designed grid search and a threshold calibration pass would make the pipeline scientifically defensible end to end.