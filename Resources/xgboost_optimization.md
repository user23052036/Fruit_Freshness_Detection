# XGBoost Feature Ranking — The Dual-Task Design

This document covers one of the most important design decisions in the pipeline: how feature importance is computed using XGBoost, why ranking on a single label is wrong, what two alternative fixes exist, and which one was implemented and why. Understanding this is essential because a wrong choice here corrupts the feature set used by both SVMs.

---

## 1. Background — Why XGBoost Is Used for Feature Ranking

After `VarianceThreshold` removes zero-variance features, 1304 features remain. Both SVMs — the vegetable classifier and the freshness classifier — will be trained on a subset of these. The question is: which features to keep?

XGBoost is used to rank features by importance. It trains a gradient boosted classifier on the 1304-dimensional scaled feature matrix, and records for each feature how much it contributed to correct splits across all trees — a metric called **gain importance**. Features with high gain are the most discriminative for the classification task. The top-k features by gain are selected.

XGBoost importance is well-suited here because:
- It captures non-linear interactions between features (unlike correlation with the label, which is linear)
- It is fast to compute relative to running an SVM cross-validation sweep across all 1304 features
- Gain importance is stable: features that consistently contribute across many trees rank highly, reducing noise

---

## 2. The Original Bug — Single-Label Ranking

In an earlier version of the pipeline, XGBoost was trained with only the **freshness label** (`y_fresh`) to rank features:

```python
# PREVIOUS (WRONG) VERSION

y = y_fresh_train.astype(int)   # binary: 0=rotten, 1=fresh
clf.fit(X_scaled, y)             # XGBoost sees only freshness signal
importances = clf.get_booster().get_score(importance_type="gain")

top_100 = argsort(importances)[-100:]   # top-100 by freshness gain

# Then BOTH SVMs trained on these same 100 features:
veg_svm.fit(X[:, top_100], y_veg)       # WRONG — veg features may be excluded
fresh_svm.fit(X[:, top_100], y_fresh)   # correct
```

This has a specific, concrete failure mode:

```
Feature 841: responds strongly to banana-vs-potato shape differences
→ high gain for vegetable classification
→ near-zero gain for freshness (shape doesn't predict rot)
→ XGBoost (trained on freshness labels) ranks it low
→ Feature 841 is discarded

Result: vegetable SVM loses its most discriminative feature
        but the freshness SVM doesn't care — it never needed that feature
```

The two SVMs need different features. A single ranking based on one task's label will always deprioritise features that are useful for the other task.

---

## 3. Two Possible Fixes

When this problem was identified, two approaches were considered:

### Approach A — Combined Label Ranking (Proposed but not implemented)

Create a composite label that encodes both vegetable identity and freshness state:

```python
# Approach A: combined label

combined_y = [f"{veg}_{fresh}" for veg, fresh in zip(y_veg_train, y_fresh_train)]
# e.g. ["banana_1", "banana_0", "apple_1", "potato_0", ...]
# 10 classes: banana_fresh, banana_rotten, apple_fresh, apple_rotten, ...

clf.fit(X_scaled, combined_y)   # XGBoost on 10-class problem
```

This forces XGBoost to rank features that help distinguish all 10 (vegetable × freshness) combinations at once. A feature useful for vegetable identity will get gain from separating banana\_fresh from apple\_fresh. A feature useful for freshness will get gain from separating banana\_fresh from banana\_rotten.

This sounds appealing but has a subtle problem: the two signals are mixed into a single gain value, and there is no guarantee they are weighted appropriately. If the vegetable classes are more spread out in feature space than the freshness classes (which is typically true — a banana looks very different from a potato, but a fresh banana and a rotten banana share most visual properties), the vegetable signal will dominate the gain ranking, and freshness-specific features may still be underweighted.

### Approach B — Dual Independent Rankings with Union (Implemented)

Run XGBoost twice independently — once per task, with the correct label for each:

```python
# Approach B: dual ranking (ACTUAL IMPLEMENTATION)

# Freshness ranking — uses freshness labels only
avg_imp_fresh, _ = compute_full_ranking(
    X_scaled, y_fresh_train.astype(int), task_label="freshness"
)

# Vegetable ranking — uses vegetable labels only
avg_imp_veg, _ = compute_full_ranking(
    X_scaled, y_veg_enc, task_label="vegetable"
)

# Each task gets its own top-k, then union is formed
selected_fresh = top_k_indices(avg_imp_fresh, k=200)   # best 200 for freshness
selected_veg   = top_k_indices(avg_imp_veg,   k=200)   # best 200 for vegetable
union_set      = union(selected_fresh, selected_veg)    # 349 total
```

This is the correct approach. Each task's XGBoost sees only the signal that task needs, without the other task's signal contaminating the gain values. The union then ensures both SVMs have access to the features they need.

---

## 4. Why Dual Rankings Is Better Than Combined Labels

The core reason is **signal isolation**:

```
Dual ranking:
  Freshness XGBoost → ranks features by freshness discriminability
                       (colour shift, texture change, edge softening)
  Vegetable XGBoost → ranks features by vegetable discriminability
                       (shape, overall colour, silhouette)

  Each ranking is clean. Neither contaminates the other.


Combined-label ranking:
  10-class XGBoost → gains are a blend of both signals
                     high gain could mean:
                       "useful for vegetable ID" OR
                       "useful for freshness" OR
                       "useful for both"

  You cannot separate them. A feature that is critical for freshness
  but contributes little to vegetable separation may score low overall,
  even though the freshness SVM needs it.
```

Dual ranking is also more principled scientifically. The two tasks require genuinely different information from the feature space:

- Freshness signal lives in: colour shift toward brown/yellow, Laplacian variance decrease (softening), edge density decrease, HSV saturation change
- Vegetable identity signal lives in: EfficientNet channels responding to shape and silhouette, overall colour statistics (capsicum is very different in colour from cucumber)

These do not fully overlap. Keeping them separate ensures neither task cannibalises the other's features.

---

## 5. The 5-Seed Averaging — Stability Against Randomness

XGBoost uses randomness (bootstrapped samples, random split candidates per tree). A single run may place a borderline feature in the top-k one time and outside it another time. This makes the ranking seed-sensitive.

The fix: run five independent XGBoost fits with different random seeds and average the gain vectors:

```python
# preprocess_and_rank.py

RANK_SEEDS = [42, 7, 123, 17, 99]

def compute_full_ranking(X_scaled, y, task_label, seeds=RANK_SEEDS):
    all_imps = []
    for s in seeds:
        imp = _rank_single_seed(X_scaled, y, random_state=s)
        all_imps.append(imp)

    avg_imp = np.mean(all_imps, axis=0)   # average gain per feature
    return avg_imp, all_imps
```

Each seed produces a (1304,) importance vector. Averaging across 5 seeds produces a stable ranking where consistently high-importance features remain high regardless of which bootstrap samples were used in any individual run.

The stability is verified by measuring minimum pairwise overlap between seed rankings:

```python
def check_ranking_stability(all_imps, top_k):
    for each pair of seeds:
        overlap = |top_k(seed_i) ∩ top_k(seed_j)| / k

min_pairwise_overlap must be ≥ 0.80
```

**Actual stability results from the training run:**

```
[Stability 'freshness'] min pairwise overlap=1.000  [OK]
[Stability 'vegetable'] min pairwise overlap=1.000  [OK]
```

All 5 seeds agreed perfectly on the top features for both tasks. An overlap of 1.0 means the top-200 features are identical across every pair of seeds — the ranking is completely stable for this dataset.

---

## 6. Compute Efficiency — Rank Once at Max k, Slice Per Candidate

The pipeline sweeps across five values of k: `{50, 100, 150, 200, 250}`. A naive implementation would re-run XGBoost for each k value. This would mean 5 seeds × 2 tasks × 5 k-values = 50 XGBoost fits.

The optimisation: XGBoost is run **once** at max(k) = 250 to obtain the full importance vector. Selecting the top-k for any k is then a cheap numpy argsort operation on the pre-computed vector:

```python
def rank_features_at_k(avg_imp, top_k):
    order    = np.argsort(avg_imp)[::-1]           # sort descending by gain
    selected = np.sort(order[:top_k])              # top-k indices, sorted
    return selected

# At k=50:  rank_features_at_k(avg_imp_fresh, 50)  → 50 indices
# At k=200: rank_features_at_k(avg_imp_fresh, 200) → 200 indices (first 200 of same order)
# No XGBoost re-run for any k
```

Total XGBoost fits: 5 seeds × 2 tasks = **10 fits total** (down from 50 in the previous version).

---

## 7. The Union Construction

After rankings are computed and the best k is confirmed (see the k-selection sweep in the preprocess_and_rank notes), the union feature set is formed:

```
selected_fresh = top-200 by avg_imp_fresh   → 200 feature indices
selected_veg   = top-200 by avg_imp_veg     → 200 feature indices

union_set = selected_fresh ∪ selected_veg

┌────────────────┬──────────────┬────────────────┐
│  Fresh-only    │    Shared    │   Veg-only     │
│  149 features  │  51 features │  149 features  │
└────────────────┴──────────────┴────────────────┘
         ←─────── 349 features total ──────────→
```

**From the actual training run:**

```
top_k per task          : 200
Fresh-specific features : 149
Veg-specific features   : 149
Shared features         : 51
Union size              : 349
```

149 features were ranked high for freshness but not for vegetable identity. 149 features were ranked high for vegetable identity but not for freshness. 51 features were in the top-200 for both tasks — these are features that simultaneously carry vegetable-discriminative and freshness-discriminative information.

Both SVMs (vegetable and freshness) are then trained on this same 349-feature union. Each SVM has access to the features it needs. Neither SVM is denied features because the other task's signal dominated the ranking.

---

## 8. What Would Have Happened Without the Fix

To make the impact concrete, consider what the pipeline produced before and after the fix:

```
BEFORE (single freshness-only ranking, top-100):

  Feature set: 100 features, all selected for freshness discriminability
  Veg SVM trained on features that were NOT selected for vegetable identity
  → Vegetable accuracy significantly lower
  → Wrong vegetable predictions → wrong normalization bounds applied
  → Freshness score corrupted for misclassified vegetables


AFTER (dual ranking, union of top-200 per task):

  Feature set: 349 features = fresh-specific + veg-specific + shared
  Veg SVM has access to the 200 best vegetable features
  Fresh SVM has access to the 200 best freshness features

  Test results:
    Vegetable accuracy    : 99.61%   ← 10 errors in 2,539 test samples
    Freshness accuracy    : 98.94%
    Freshness ROC-AUC     : 0.9994
    RBF val acc (veg)     : 0.9992   ← during k-selection sweep
    RBF val acc (fresh)   : 0.9858
```

---

## 9. The XGBoost Gain Metric — What It Actually Measures

XGBoost offers three importance metrics: weight (number of times a feature is used in splits), cover (number of samples a feature's splits affect), and gain. Gain is used here because it is the most informative for feature selection.

**Gain** measures the total improvement in model loss attributed to a feature across all splits where it is used. Specifically, for each tree node that uses feature j to split:

```
gain(j, split) = loss_before_split − (loss_left_child + loss_right_child)
               = improvement in prediction accuracy from this split
```

Gain is summed across all nodes across all trees and normalised. A feature with high total gain consistently produces useful splits — it is genuinely informative for the classification task rather than just frequently used (which weight measures) or affecting many samples (which cover measures).

For ranking purposes, gain is the right choice because it directly measures discriminative value, which is what you want feature selection to optimise.

---

## 10. Summary — Before, Proposed, and Implemented

| Design | Label used for XGBoost | Feature set | Problem |
|--------|----------------------|-------------|---------|
| **Original (broken)** | `y_fresh` only (binary) | Top-100 freshness features | Vegetable SVM denied its best features |
| **Proposed fix (not used)** | `y_veg + "_" + y_fresh` (10-class combined) | Top-k combined features | Signals mixed; vegetable dominates gain; freshness may be underweighted |
| **Implemented fix** | Separate: `y_fresh` AND `y_veg` independently | Union of top-200 per task = 349 | Each task has a clean ranking; union guarantees both are served |

The implemented solution is the most principled: it treats the two tasks as genuinely independent, gives each its own clean XGBoost fit with the correct label, and combines the results through a union rather than a blend. The 5-seed averaging ensures the ranking is stable. The result is a 349-feature set that the test set confirms works well for both tasks simultaneously.