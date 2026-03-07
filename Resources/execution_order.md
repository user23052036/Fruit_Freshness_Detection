
# 1. Step 1 — Extract Dataset Features

Command:

```bash
python src/extract_dataset_features.py
```

What happens internally:

```
Images
↓
EfficientNetB0 (1280 features)
+
Handcrafted features (32)
↓
1312 feature vector per image
↓
Saved into Features/
```

Expected console output:

```text
Scanning dataset...
Total images: 36000

Extracting features...

Batches: 100%|████████████████████████████████| 282/282 [00:25<00:00]

Saved feature matrix: (36000, 1312)
```

Files created:

```
Features/
 ├ X.npy
 ├ y_veg.npy
 └ y_fresh.npy
```

Example shapes:

```
X.npy → (36000, 1312)
y_veg.npy → (36000,)
y_fresh.npy → (36000,)
```

---

# 2. Step 2 — Train/Test Split

Command:

```bash
python src/train_split.py
```

What happens:

```
Dataset
↓
Stratified split (veg + freshness)
↓
80% train
20% test
```

Expected output:

```text
[INFO] Loading Features...
[SUCCESS] Train/test split created. Train=28800 Test=7200
```

Files created:

```
models/
 ├ X_train.npy
 ├ X_test.npy
 ├ y_veg_train.npy
 ├ y_veg_test.npy
 ├ y_fresh_train.npy
 └ y_fresh_test.npy
```

---

# 3. Step 3 — Feature Optimization

Command:

```bash
python src/preprocess_and_rank.py
```

What happens:

```
1312 features
↓
VarianceThreshold
↓
StandardScaler
↓
XGBoost ranking
↓
Select top 60 features
```

Expected output:

```text
[INFO] Using 28800 samples for ranking
[INFO] VarianceThreshold removed -> 1312 -> 1287
[DONE] Selected top 60 features
```

Files created:

```
models/
 ├ variance.joblib
 ├ scaler.joblib
 ├ selected_features.npy
 └ feature_importances.npy
```

---

# 4. Step 4 — Train Models

Command:

```bash
python src/train_svm.py
```

What happens:

```
Selected features
↓
Train two SVM models
```

Models:

```
Vegetable classifier
Freshness classifier
```

Expected output:

```text
[INFO] Loading features...
[INFO] Applying preprocessing pipeline...
[INFO] Feature matrix after selection: (28800, 60)

[INFO] Training vegetable classifier...
[DONE] Vegetable classifier saved

[INFO] Training freshness classifier...
[DONE] Freshness classifier saved
```

Files created:

```
models/
 ├ veg_svm.joblib
 ├ fresh_svm.joblib
 └ label_encoder.joblib
```

---

# 5. Step 5 — Prediction

Command:

```bash
python src/predict_cli.py --image test_images/tomato.jpg
```

What happens:

```
Image
↓
Feature extraction
↓
Preprocessing pipeline
↓
SVM prediction
↓
Score + grade
```

Expected output example:

```text
Vegetable: tomato (93.14%)
Freshness probability: 0.8723
Score: 87.23
Grade: Mostly Fresh
```

---

# Full Pipeline Summary

```text
Image Dataset
↓
Feature Extraction
↓
X.npy (1312 features)
↓
Train/Test Split
↓
Feature Optimization
↓
Top 60 features
↓
Train SVM models
↓
Prediction
↓
Score + Grade
```

---

# Optional Flags / Variants

These are **optional configurations you can change**.

---

# 1. Change number of selected features

Default:

```
top_k = 60
```

Run:

```bash
python src/preprocess_and_rank.py
```

Modify code to:

```python
main(top_k=100)
```

Expected output:

```text
[DONE] Selected top 100 features
```

Effect:

```
Higher capacity model
Slightly slower SVM
Sometimes +1–2% accuracy
```

---

# 2. Change test split size

Default:

```
test_size = 0.2
```

Run:

```python
main(test_size=0.3)
```

Expected output:

```
Train = 25200
Test = 10800
```

Effect:

```
More reliable evaluation
Less training data
```

---

# 3. Change feature batch size (feature extraction)

Default:

```
BATCH_SIZE = 128
```

If GPU or strong CPU:

```python
BATCH_SIZE = 256
```

Expected output:

```
Batches reduced
Extraction faster
```

---

# 4. Run prediction on multiple images

Example:

```bash
python src/predict_cli.py --image sample_images/carrot1.jpg
python src/predict_cli.py --image sample_images/tomato1.jpg
python src/predict_cli.py --image sample_images/potato1.jpg
```

Example outputs:

```
Vegetable: carrot (91.02%)
Score: 82.4
Grade: Mostly Fresh
```

```
Vegetable: potato (88.11%)
Score: 45.1
Grade: Rotten
```

---

# Final Files Created in Project

```
Features/
 ├ X.npy
 ├ y_veg.npy
 └ y_fresh.npy

models/
 ├ X_train.npy
 ├ X_test.npy
 ├ y_veg_train.npy
 ├ y_veg_test.npy
 ├ y_fresh_train.npy
 ├ y_fresh_test.npy
 ├ variance.joblib
 ├ scaler.joblib
 ├ selected_features.npy
 ├ feature_importances.npy
 ├ veg_svm.joblib
 ├ fresh_svm.joblib
 └ label_encoder.joblib
```

---

