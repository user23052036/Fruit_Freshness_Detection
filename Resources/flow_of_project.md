
# 1. Training Pipeline (Full System)

```text
vegetable_Dataset/
        │
        │
        ▼
Scan folders
(parse Fresh/Rotten + vegetable name)
        │
        ▼
Filter only target vegetables
{tomato, carrot, potato, cucumber, capsicum}
        │
        ▼
Load images (threaded)
        │
        ▼
Resize → 224×224
        │
        ▼
────────────────────────────────
Feature Extraction
────────────────────────────────
        │
        ├── EfficientNetB0
        │       │
        │       ▼
        │   1280 deep features
        │
        └── Handcrafted features
                │
                ├ RGB mean/std
                ├ HSV mean/std
                ├ grayscale stats
                ├ edge density
                ├ Laplacian variance
                └ histogram bins
                        │
                        ▼
                    32 features
        │
        ▼
Concatenate features
        │
        ▼
1312 feature vector per image
        │
        ▼
Feature Matrix
X.npy (N × 1312)
y_veg.npy
y_fresh.npy
```

---

# 2. Dataset Split

```text
X.npy
y_veg.npy
y_fresh.npy
        │
        ▼
train_split.py
        │
        ▼
Stratified Split
(vegetable + freshness)
        │
        ├── X_train.npy
        ├── y_veg_train.npy
        ├── y_fresh_train.npy
        │
        └── X_test.npy
            y_veg_test.npy
            y_fresh_test.npy
```

---

# 3. Feature Optimization

```text
X_train
        │
        ▼
VarianceThreshold
(remove constant features)
        │
        ▼
StandardScaler
(feature normalization)
        │
        ▼
XGBoost Feature Ranking
(label = vegetable + freshness)
        │
        ▼
Feature Importance
        │
        ▼
Select Top 60 Features
        │
        ▼
Save preprocessing artifacts
        │
        ├ variance.joblib
        ├ scaler.joblib
        └ selected_features.npy
```

---

# 4. Model Training

```text
Selected Features
        │
        ▼
Two Separate Models
```

### Vegetable Classifier

```text
Selected Features
        │
        ▼
LabelEncoder
(vegetable names → integers)
        │
        ▼
SVM (multiclass)
        │
        ▼
veg_svm.joblib
```

### Freshness Classifier

```text
Selected Features
        │
        ▼
SVM (binary classifier)
        │
        ▼
fresh_svm.joblib
```

---

# 5. Prediction Pipeline

```text
Input Image
        │
        ▼
extract_features.py
        │
        ▼
EfficientNetB0 → 1280 features
        │
Handcrafted features → 32
        │
        ▼
1312 feature vector
        │
        ▼
VarianceThreshold
(remove same features removed in training)
        │
        ▼
StandardScaler
(using saved scaler)
        │
        ▼
Feature Selection
(top 60 features)
        │
        ▼
────────────────────────
Prediction
────────────────────────
        │
        ├── Vegetable SVM
        │       │
        │       ▼
        │   vegetable type
        │
        └── Freshness SVM
                │
                ▼
           P(fresh)
```

---

# 6. Scoring System

```text
P(fresh)
        │
        ▼
Score = P(fresh) × 100
        │
        ▼
Grade Assignment
```

| Score  | Grade        |
| ------ | ------------ |
| 95–100 | Fully Fresh  |
| 75–95  | Mostly Fresh |
| 50–75  | Medium       |
| <50    | Rotten       |

---

# 7. Final Output

```text
Image
        │
        ▼
Vegetable Type
        │
        ▼
Freshness Probability
        │
        ▼
Score (0–100)
        │
        ▼
Grade
```

Example output:

```text
Vegetable: tomato
Freshness probability: 0.87
Score: 87
Grade: Mostly Fresh
```

---

# 8. Full System (Ultra-Compact View)

```text
Image
 ↓
EfficientNetB0 (1280)
 +
Handcrafted (32)
 ↓
1312 Features
 ↓
VarianceThreshold
 ↓
StandardScaler
 ↓
XGBoost Feature Ranking
 ↓
Top 60 Features
 ↓
 ┌───────────────┬───────────────┐
 │               │               │
 ▼               ▼               │
Vegetable SVM    Freshness SVM   │
 │               │               │
 ▼               ▼               │
Vegetable Type   P(fresh)        │
                 │               │
                 ▼               │
            Score = P×100        │
                 │               │
                 ▼               │
               Grade
```

---
