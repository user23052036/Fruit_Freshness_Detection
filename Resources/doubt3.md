
# 1. `extract_dataset_features.py`

**Purpose:**
Convert **images → numbers**.

What it does:

```
image
↓
EfficientNet + handcrafted features
↓
1312 numbers per image
```

For all images in your dataset it creates:

```
Features/X.npy        → feature matrix
Features/y_veg.npy    → vegetable labels
Features/y_fresh.npy  → freshness labels
```

So after this step:

```
36k images
↓
36k rows of features
```

Your model will **never read images again**, only these numbers.

---

# 2. `train_split.py`

**Purpose:**
Divide the dataset into **training data and testing data**.

What it does:

```
X.npy
↓
split
↓
80% training
20% testing
```

It saves:

```
models/X_train.npy
models/X_test.npy
models/y_veg_train.npy
models/y_veg_test.npy
models/y_fresh_train.npy
models/y_fresh_test.npy
```

So you can:

```
train model → training data
test model → unseen test data
```

---

# 3. In one sentence

**extract_dataset_features**

```
images → feature numbers
```

**train_split**

```
split features → train + test
```

---

# 4. Where they appear in your pipeline

```
Images
↓
extract_dataset_features
↓
Feature dataset
↓
train_split
↓
Train set + Test set
↓
Model training
```
