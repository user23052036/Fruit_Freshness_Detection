# 1. Feature Extraction

```text
vegetable_Dataset/
        │
        ▼
Scan folders
(parse Fresh/Rotten + vegetable name)
        │
        ▼
Filter only target vegetables
{apple, banana, capsicum, cucumber, potato}
        │
        ▼
Load images (threaded, parallel)
        │
        ▼
Resize → 224×224
Convert BGR → RGB
        │
        ▼
────────────────────────────────────────
Feature Extraction (per image)
────────────────────────────────────────
        │
        ├── EfficientNetB0 (imagenet weights)
        │       │
        │       ▼
        │   GlobalAveragePooling
        │       │
        │       ▼
        │   1280 deep features
        │   (texture, shape, colour patterns)
        │
        └── Handcrafted features
                │
                ├ RGB mean / std          → 6 values
                ├ HSV mean / std          → 6 values
                ├ Grayscale mean / std    → 2 values
                ├ Edge density (Canny)    → 1 value
                ├ Laplacian variance      → 1 value
                └ Luminance histogram     → 8 bins (normalized)
                        │
                        ▼
                    32 features
        │
        ▼
Concatenate
[deep(1280) | handcrafted(32)]
        │
        ▼
1312 feature vector per image
        │
        ▼
Feature Matrix saved
        │
        ├ X.npy        # shape (N, 1312)
        ├ y_veg.npy    # vegetable name strings
        └ y_fresh.npy  # 1=fresh, 0=rotten
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
(combined label = vegetable + freshness)
e.g. "banana_1", "potato_0"
Ensures balanced distribution in both sets
        │
        ├────────────────────┬──────────────────────
        │                    │
        ▼                    ▼
   TRAIN (80%)           TEST (20%)
   10113 samples         2529 samples
        │                    │
        ├ X_train.npy        ├ X_test.npy
        ├ y_veg_train.npy    ├ y_veg_test.npy
        └ y_fresh_train.npy  └ y_fresh_test.npy

⚠ Never use test data before evaluation.
  Test set is only touched in step 5 and 6.
```

---

# 3. Feature Optimization

```text
X_train.npy  (10113 × 1312)
        │
        ▼
VarianceThreshold
(remove features constant across all train samples)
        │
        ▼
X_train_vt  (10113 × ~1304)
        │
        ▼
StandardScaler
(fit on train only — normalize to mean=0, std=1)
        │
        ▼
X_train_scaled  (10113 × ~1304)
        │
        ▼
XGBoost Classifier
(label = combined veg+freshness, e.g. "banana_1")
Trains to rank features by gain importance
        │
        ▼
Feature Importance Array
[0.021, 0.004, 0.033, 0.0001, ...]
length = number of features after VarianceThreshold
        │
        ▼
Pick top 100 indices
        │
        ▼
X_train_final  (10113 × 100)
        │
        ▼
Save preprocessing artifacts
        │
        ├ variance.joblib          # VarianceThreshold
        ├ scaler.joblib            # StandardScaler
        ├ selected_features.npy   # top 100 indices
        └ feature_importances.npy # used for visualization only
```

---

# 4. Model Training

```text
X_train_final  (10113 × 100)
        │
        ▼
Two Separate Models
```

### Vegetable Classifier (SVM 1)

```text
X_train_final
        │
        ▼
LabelEncoder
(vegetable names → integers)
apple=0, banana=1, capsicum=2, cucumber=3, potato=4
        │
        ▼
RBF SVM (multiclass, class_weight=balanced)
        │
        ▼
veg_svm.joblib
label_encoder.joblib
```

### Freshness Classifier (SVM 2)

```text
X_train_final
        │
        ▼
RBF SVM (binary, class_weight=balanced)
Target: y_fresh  (0=rotten, 1=fresh)
        │
        ▼
fresh_svm.joblib
```

### Post-Training: Scoring Config (NEW)

```text
fresh_svm
        │
        ▼
decision_function(X_train_final)
→ raw signed distances from hyperplane
  positive = fresh side
  negative = rotten side
        │
        ▼
Compute per-vegetable bounds
(from training decisions, split by veg label)
        │
        ├ apple    → p5, p95, hard_min, hard_max
        ├ banana   → p5, p95, hard_min, hard_max
        ├ capsicum → p5, p95, hard_min, hard_max
        ├ cucumber → p5, p95, hard_min, hard_max
        └ potato   → p5, p95, hard_min, hard_max
        │
        ▼
Compute global bounds
(fallback — from all training decisions)
        │
        ▼
Load X_test → apply pipeline → decision_function(X_test)
(use TEST SET for calibration — not training data)
        │
        ▼
Calibrate boundary_threshold
Sweep abs(decision) 0.05→1.5
Find threshold where misclassification rate ≥ 10%
→ boundary_threshold = 0.05
        │
        ▼
Calibrate unstable_std_thresh
Derived from decision spread of test set
→ unstable_std_thresh = 9.0
        │
        ▼
Save scoring_config.json
        │
        ├ global_bounds
        ├ per_veg_bounds       { apple, banana, capsicum, cucumber, potato }
        ├ boundary_threshold   = 0.05
        ├ unstable_std_thresh  = 9.0
        └ veg_confidence_threshold = 0.70
```

---

# 5. Model Evaluation

```text
X_test.npy  (2529 × 1312)
        │
        ▼
Apply saved pipeline:
variance.transform()
scaler.transform()
X[:, selected_features]
        │
        ▼
X_test_final  (2529 × 100)
        │
        ├────────────────────────────────────────────────────────
        │                                                        │
        ▼                                                        ▼
veg_svm.predict(X_test_final)               fresh_svm.predict(X_test_final)
        │                                                        │
        ▼                                                        ▼
Predicted vegetable labels                  Predicted fresh/rotten labels
        │                                                        │
        ▼                                                        ▼
Accuracy: 0.9937                            Accuracy: 0.9771
Classification Report                       Classification Report
Confusion Matrix                            Confusion Matrix

────────────────────────────────────────────────────────────────
Score Validation (NEW)
────────────────────────────────────────────────────────────────

fresh_svm.decision_function(X_test_final)
→ raw distances for all 2529 test samples
        │
        ▼
normalize using global p5/p95
→ scores[0..100] for all samples
        │
        ├── A. Inter-class pairwise accuracy
        │       Sample (fresh, rotten) pairs
        │       Check: score_fresh > score_rotten
        │       Result: 0.9950  ← 99.5% of pairs correctly ordered
        │
        ├── B. Score distribution + Delta
        │       Fresh  mean=83.92  std=12.52
        │       Rotten mean=17.91  std=13.60
        │       Δ = 83.92 − 17.91 = 66.01 points  ← key metric
        │       Overlap = 0.0008  ← almost zero overlap
        │
        ├── C. Intra-class spread (non-collapse check)
        │       Fresh  range=75.89  std=12.52
        │       Rotten range=90.96  std=13.60
        │       Score is continuous, not collapsed to two values
        │
        └── D. Per-vegetable summary
                Vegetable    PairwiseAcc   FreshMean  RottenMean    Delta
                apple            0.9992       88.62       19.59     69.02
                banana           1.0000       84.52       13.67     70.85
                capsicum         1.0000       82.95        7.54     75.41
                cucumber         0.9632       81.55       24.06     57.50
                potato           0.9654       77.92       27.02     50.91

────────────────────────────────────────────────────────────────
Threshold Statistics
────────────────────────────────────────────────────────────────
        Boundary threshold (calibrated) : 0.05
        Near-boundary fraction          : 0.0055  (14/2529 samples)
        OOD fraction (hard bounds)      : 0.0000
        Unstable std threshold          : 9.0
```

---

# 6. Prediction Pipeline (Single Image)

```text
Input Image
        │
        ▼
extract_features.py
        │
        ▼
EfficientNetB0 → 1280 deep features
        +
Handcrafted    →   32 features
        │
        ▼
1312 feature vector
        │
        ▼
variance.transform()
(remove same constant features as training)
        │
        ▼
scaler.transform()
(normalize using saved training mean/std)
        │
        ▼
X[:, selected_features]
(keep same top 100 features as training)
        │
        ▼
X_final  (1 × 100)
        │
        ├────────────────────────┬────────────────────────
        │                        │
        ▼                        ▼
veg_svm.predict_proba()    fresh_svm.predict()
        │                        │
        ▼                        ▼
vegetable type             Fresh / Rotten label
+ confidence %
e.g. banana (99.97%)
        │                        │
        └──────────┬─────────────┘
                   │
                   ▼
        fresh_svm.decision_function()
                   │
                   ▼
        raw signed distance from hyperplane
        e.g. +0.85
        positive = fresh side
        negative = rotten side
```

---

# 7. Scoring System (NEW)

```text
raw decision value
(e.g. +0.85)
        │
        ▼
Select normalization bounds:
        │
        ├── IF veg_conf ≥ 70%
        │       use per-vegetable bounds
        │       e.g. banana: p5=-2.42  p95=2.52
        │
        └── IF veg_conf < 70%
                use global bounds (fallback)
                avoids wrong scaling from misclassification
        │
        ▼
Percentile normalization
score = (raw − p5) / (p95 − p5) × 100
score = clip(score, 0, 100)

epsilon guard:
if abs(p95 − p5) < 1e-6 → return 50.0

Example:
raw=0.85, p5=-2.42, p95=2.52
score = (0.85 − (−2.42)) / (2.52 − (−2.42)) × 100
score = 3.27 / 4.94 × 100
score = 76.01
        │
        ▼
Grade Assignment
```

| Score   | Grade       |
| ------- | ----------- |
| ≥ 85    | Truly Fresh |
| 65 – 84 | Fresh       |
| 40 – 64 | Moderate    |
| < 40    | Rotten      |

---

# 8. Uncertainty Estimation (NEW)

```text
Same input image
        │
        ▼
Apply 6 lightweight augmentations:
        │
        ├ Brightness +15%
        ├ Brightness −15%
        ├ Horizontal flip
        ├ Gaussian blur
        ├ Rotation +5°
        └ Rotation −5°
        │
        ▼
For each augmented image:
  → extract 1312 features
  → apply pipeline
  → decision_function → score
        │
        ▼
scores = [76, 81, 70, 74, 78, 72]
        │
        ▼
score_std = std(scores)
e.g. score_std = 6.56
        │
        ▼
Compare against unstable_std_thresh (9.0)

score_std < 9.0  → stable   ✅ no warning
score_std ≥ 9.0  → unstable ❌ INPUT SENSITIVITY warning
```

---

# 9. Warning Flags (NEW)

```text
Four independent checks run after every prediction:
        │
        ├── Low veg confidence
        │       IF veg_conf < 70%
        │       → "[!] Using global normalization"
        │
        ├── MODEL UNCERTAINTY
        │       IF abs(raw) < boundary_threshold (0.05)
        │       → "[!] MODEL UNCERTAINTY — near decision boundary"
        │       Meaning: the classifier is unsure
        │       Even a small change in features could flip the result
        │
        ├── INPUT SENSITIVITY
        │       IF score_std > unstable_std_thresh (9.0)
        │       → "[!] INPUT SENSITIVITY — score unstable under augmentation"
        │       Meaning: the score is sensitive to imaging conditions
        │       Consider retaking on neutral background
        │
        └── OOD WARNING
                IF raw < hard_min OR raw > hard_max
                → "[!] OOD — outside training distribution"
                Limitation: only catches extreme outliers
                Distribution shift WITHIN training range is NOT detected

Note:
MODEL UNCERTAINTY  = the model itself is unsure (boundary proximity)
INPUT SENSITIVITY  = the imaging conditions are causing instability
These are different signals. Both can appear independently.
```

---

# 10. Final Output

```text
Image
        │
        ▼
Vegetable type + confidence
        │
        ▼
Freshness label (Fresh / Rotten)
        │
        ▼
Score (0–100) ± std
        │
        ▼
Grade
        │
        ▼
Warning flags (if any)
```

Example output (normal):

```text
Vegetable : banana (99.97%)
Freshness : Fresh
Score     : 76.01 ± 6.56 / 100
Grade     : Fresh
Norm      : per-veg
```

Example output (with warnings):

```text
Vegetable : banana (99.97%)
Freshness : Fresh
Score     : 52.10 ± 11.40 / 100
Grade     : Moderate
Norm      : per-veg

[!] MODEL UNCERTAINTY — score near decision boundary
    (|raw|=0.03 < threshold=0.05)
    The classifier is unsure. Result may change under different features.

[!] INPUT SENSITIVITY — score std=11.40 (threshold=9.0)
    Score is sensitive to lighting/angle.
    Consider re-capturing on a neutral background.
```

---

# 11. Full System (Ultra-Compact View)

```text
Image
 ↓
EfficientNetB0 (1280) + Handcrafted (32)
 ↓
1312 Features
 ↓
VarianceThreshold → ~1304
 ↓
StandardScaler
 ↓
XGBoost Feature Ranking
 ↓
Top 100 Features
 ↓
 ┌──────────────────────┬──────────────────────┐
 │                      │                      │
 ▼                      ▼                      │
Vegetable SVM       Freshness SVM              │
 │                      │                      │
 ▼                      ▼                      │
Vegetable type      Fresh / Rotten label       │
+ confidence %          │                      │
                        ▼                      │
               decision_function()             │
               (raw signed distance)           │
                        │                      │
                        ▼                      │
               Per-veg percentile              │
               normalization                   │
               (p5 → p95 → 0–100)             │
                        │                      │
                        ▼                      │
               Score ± std                     │
               (6× augmentations)              │
                        │                      │
                        ▼                      │
                      Grade                    │
                        │                      │
                        ▼                      │
               Warning flags ──────────────────┘
               (boundary / input / OOD)
```

---

# 12. What Changed From Original (Summary)

```text
ORIGINAL WORKFLOW          →    UPDATED WORKFLOW
──────────────────────────────────────────────────────
predict_proba (confidence) →    decision_function (distance)

Score = P(fresh) × 100    →    Score = (raw − p5) / (p95 − p5) × 100

Single global min-max      →    Per-vegetable p5/p95 percentile bounds

Hardcoded threshold 0.3    →    Calibrated on test set (0.05)

Hardcoded std 8.0          →    Calibrated from data (9.0)

No uncertainty             →    Score ± std via 6 augmentations

No OOD detection           →    Hard-bound check + limitation note

60 features selected       →    100 features selected

No veg confidence check    →    Fallback to global if conf < 70%

Accuracy + confusion only  →    + Pairwise acc, Delta, spread, per-veg table

fresh_decision_bounds.npy  →    scoring_config.json (all values unified)

Leakage in step 3          →    Fixed — X_train only (10113 samples)
```