`VarianceThreshold` is a **feature filtering technique**. It removes features that **never change across samples**, because such features carry **zero information for prediction**.

In your pipeline, it is used here:

```python
vt = VarianceThreshold(threshold=0.0)
X_sel = vt.fit_transform(X)
```

This happens in **`preprocess_and_rank.py`**. 

---

# 1. Intuition

Variance measures **how much a feature changes across the dataset**.

Example feature values:

| image | feature_1 |
| ----- | --------- |
| img1  | 0         |
| img2  | 0         |
| img3  | 0         |
| img4  | 0         |

Variance:

```
0
```

Meaning:

```
this feature never changes
```

Such a feature **cannot help a model distinguish anything**.

---

# 2. Why it matters in your pipeline

Your feature vector size is:

```
EfficientNet features = 1280
Handcrafted features = 32
Total = 1312
```

Some features may be constant because:

* EfficientNet neurons may never activate for your dataset
* certain handcrafted statistics may always be identical

Example:

```
feature_472 = always 0
feature_918 = always 0
```

These features are useless.

---

# 3. What `VarianceThreshold` does

It removes those useless columns.

Example:

Before:

```
1312 features
```

After:

```
1312 → 1247 features
```

Then the rest of the pipeline continues with the reduced matrix.

---

# 4. Why it must happen **before scaling**

Scaling computes:

```
(x − mean) / std
```

If variance is zero:

```
std = 0
```

which causes:

```
division by zero
```

So removing constant features **prevents numerical issues**.

---

# 5. Mathematical definition

Variance of a feature:

```
Var(X) = (1/n) Σ (xi − μ)²
```

If:

```
Var(X) = 0
```

then

```
xi = μ for every sample
```

Meaning:

```
the feature is constant
```

---

# 6. Example in your vegetable dataset

Suppose you have a handcrafted feature:

```
texture_entropy
```

If all your dataset images have similar textures:

```
0.12
0.12
0.12
0.12
```

Then:

```
variance = 0
```

So it gets removed.

---

# 7. What remains after this step

The pipeline becomes:

```
1312 features
↓
remove constant features
↓
~1200–1300 features
↓
standardization
↓
XGBoost ranking
↓
top 60 features
↓
SVM
```

So `VarianceThreshold` is just a **quick cleanup step before the real feature selection**.

---

# 8. Why it is very cheap but useful

Runtime:

```
O(N × features)
```

For your dataset:

```
36k × 1312
```

This takes **<1 second**.

But it prevents useless work later.

---

# 9. Important clarification

`VarianceThreshold` **does not rank features**.

It only removes features that have:

```
variance ≤ threshold
```

Real feature importance comes later from:

```
XGBoost feature ranking
```

---
