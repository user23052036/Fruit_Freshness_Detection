# REQUIRED execution order (do not skip or reorder)

1. `extract_dataset_features.py` — extract features from images → `Features/`
2. `train_split.py` — create train/test split → `models/` (X_train.npy exists)
3. `preprocess_and_rank.py` — remove constant features, scale, XGBoost ranking → save artifacts in `models/`
4. `train_svm.py` — train the two SVMs using selected features → save models in `models/`
5. `evaluate_models.py` — evaluate on `X_test.npy` (optional but mandatory for validation)
6. `visualize_results.py` — plots confusion matrices / feature importance (optional)
7. `predict_cli.py` — single-image inference

Do this order every time you regenerate Features. If you change `Features/X.npy`, rerun steps 2→5.

---

# Exact commands & expected outputs

> Run from repo root (where `src/` folder sits).

## 1) Extract dataset features

```bash
python src/extract_dataset_features.py
```

What it does: loads images (threaded), converts BGR→RGB, resizes 224×224, runs EfficientNet batches + handcrafted features, saves `Features/X.npy`, `Features/y_veg.npy`, `Features/y_fresh.npy`.

Expected console (example):

```
Scanning dataset...
Total images: N
Extracting features...
Batches: 100%|██████████| M/M [00:25<00:00]
Saved feature matrix: (N, 1312)
```

Files created:

```
Features/
 ├ X.npy          # shape (N,1312)
 ├ y_veg.npy      # shape (N,)
 └ y_fresh.npy    # shape (N,)
```

Checks after run:

* `python -c "import numpy as np; X=np.load('Features/X.npy'); print(X.shape)"`
* Confirm `(N, 1312)`.

---

## 2) Create train/test split

```bash
python src/train_split.py
```

What it does: stratified split by combined label (`veg_fresh`) to preserve distribution and saves train/test arrays in `models/`.

Expected console:

```
[INFO] Loading Features...
[SUCCESS] Train/test split created. Train=TRAIN_COUNT Test=TEST_COUNT
```

Files created in `models/`:

```
X_train.npy, X_test.npy,
y_veg_train.npy, y_veg_test.npy,
y_fresh_train.npy, y_fresh_test.npy
```

Critical: **Do not continue** to step 3 until `models/X_train.npy` exists. If missing, you will leak.

---

## 3) Preprocess & rank features (XGBoost)

```bash
python src/preprocess_and_rank.py
```

What it does:

* Loads `models/X_train.npy` (or `Features/X.npy` if you mistakenly skipped step 2 — don’t),
* VarianceThreshold (removes constant features), StandardScaler (fit on train), XGBoost ranking (gain), selects `top_k` indices, saves artifacts.

Expected console:

```
[INFO] Using TRAIN_COUNT samples for ranking
[INFO] VarianceThreshold removed -> 1312 -> K_REDUCED
[DONE] Selected top 60 features
```

Files created in `models/`:

```
variance.joblib
scaler.joblib
selected_features.npy   # indices into the reduced-space
feature_importances.npy
```

Checks:

* `np.load('models/selected_features.npy').shape` should be `(60,)` if default.
* Confirm `variance.joblib` and `scaler.joblib` exist.

Important: if you want deterministic ranking, `preprocess_and_rank.py` sets `np.random.seed(42)` and XGBoost `random_state=42`.

---

## 4) Train SVMs

```bash
python src/train_svm.py
```

What it does:

* Loads `models/X_train.npy` (or `Features/X.npy` fallback), applies `variance.joblib` → `scaler.joblib` → selects `selected_features.npy`, trains:

  * vegetable SVM (multiclass)
  * freshness SVM (binary)
* Saves `veg_svm.joblib`, `fresh_svm.joblib`, `label_encoder.joblib`.

Expected console:

```
[INFO] Loading features...
[INFO] Applying preprocessing pipeline...
[INFO] Feature matrix after selection: (TRAIN_COUNT, 60)

[INFO] Training vegetable classifier...
[DONE] Vegetable classifier saved

[INFO] Training freshness classifier...
[DONE] Freshness classifier saved
```

Files added in `models/`:

```
veg_svm.joblib
fresh_svm.joblib
label_encoder.joblib
```

Checks:

* `python -c "from joblib import load; load('models/veg_svm.joblib')"` should not error.
* Quick sanity: run `python src/evaluate_models.py` next.

---

## 5) Evaluate on test split (must run)

```bash
python src/evaluate_models.py
```

What it does: loads `models/X_test.npy`, applies preprocessing pipeline, predicts with both models, prints accuracy/precision/recall/F1 and confusion matrices.

Expected console (example):

```
[INFO] Loading test data...
[INFO] Applying preprocessing pipeline...

========== Vegetable Classification ==========
Accuracy: 0.93
Classification Report:
...
Confusion Matrix:
...

========== Freshness Classification ==========
Accuracy: 0.91
Classification Report:
...
Confusion Matrix:
...
```

If scores look **unreasonably high** (e.g., >99%), **stop** — re-check that Step 3 used only train data (no leakage).

---

## 6) Visualization (optional)

```bash
python src/visualize_results.py
```

What it does: shows confusion heatmaps and the top-XGBoost importances plot. Requires `matplotlib`, `seaborn`.

Expected: three plots open (vegetable CM, freshness CM, top feature importances).

---

## 7) Single-image prediction

```bash
python src/predict_cli.py --image path/to/image.jpg
```

Expected output example:

```
Vegetable: tomato (93.14%)
Freshness probability: 0.8723
Score: 87.23
Grade: Mostly Fresh
```

---

# Optional flags / variants & how to use them

> Many scripts currently take parameters via editing the `main(...)` call or function arguments. Quick options:

### Change number of selected features (`top_k`)

* Edit at top of `preprocess_and_rank.py` or call `main(top_k=100)` inside `if __name__ == "__main__":` before running.
* Output: `[DONE] Selected top 100 features` and `models/selected_features.npy` length = 100.
* Tradeoff: larger `top_k` → slower SVM training, sometimes slightly better accuracy.

### Change split ratio (`test_size`) and seed

* Edit `train_split.py` `main(test_size=..., random_state=...)`.
* Run split again to regenerate `models/X_train.npy` & test files.
* Use `random_state=42` for reproducibility.

### Change extraction batch size (speed)

* Edit `BATCH_SIZE` in `src/extract_dataset_features.py` (default 128). Increase if you have RAM & CPU to speed up extraction.
* `NUM_WORKERS = os.cpu_count() or 4` controls threads for loading images.

### PCA — when and how to enable (ONLY if necessary)

* **Do not enable PCA by default.**
* Use PCA only if you face memory or speed problems (e.g., cannot train SVM because `selected_features` would be >200 or SVM time/memory is prohibitive).
* If you must use PCA, integrate it **after StandardScaler** and **before** the model — but **do not** run PCA **before** XGBoost ranking if you want interpretable ranking. There are two valid designs:

  * **If you keep XGBoost ranking**: do **not** PCA — ranking expects original features.
  * **If you switch to PCA compression**: remove XGBoost ranking and train SVM on PCA components:

    ```
    VarianceThreshold → StandardScaler → PCA → SVM
    ```
* How to add PCA quickly:

  * `from sklearn.decomposition import PCA`
  * `pca = PCA(n_components=100, random_state=42) ; Xp = pca.fit_transform(Xs)`
  * Save `pca.joblib` and load it in `predict_cli.py`.

### Determinism

* `np.random.seed(42)` is included in `preprocess_and_rank.py`. Keep `random_state=42` in any downstream training for reproducible results.

---

# Failure modes & checks (be ruthless)

1. **Data leakage**: always confirm `models/X_train.npy` exists before running `preprocess_and_rank.py`. If `preprocess_and_rank.py` uses `Features/X.npy`, that’s leakage. Quick check:

   ```bash
   ls models/X_train.npy || echo "Run train_split.py first"
   ```

2. **Mismatched lengths**: before splitting, `train_split.py` validates `len(X)==len(y_veg)==len(y_fresh)`. If it fails, abort and inspect `Features/y_*.npy` — extractor silently skips unreadable images but must keep label arrays in sync.

3. **EfficientNet color bug**: extractor converts BGR→RGB; if you changed that, re-run feature extraction. If you see sudden accuracy drops, confirm `extract_features.py` returns shape `(1312,)`.

4. **Model loading error**: `predict_cli.py` should not call `ensure_dirs()` — if models missing, raise error rather than creating empty models. If missing, re-run training.

5. **Memory spikes during extraction**: lower `BATCH_SIZE` or increase swap. Monitor `htop`.

6. **Unusually high evaluation scores (>98%)**: very likely leakage. Re-run steps and ensure proper order.

---

# Final artifacts (what you should see at the end)

```
Features/
 ├ X.npy             # (N,1312)
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

# Final suggested short checklist before step 3 (preprocess_and_rank)

1. `python -c "import numpy as np; print(np.load('Features/X.npy').shape)"`
2. `python -c "import numpy as np; print(np.load('Features/y_veg.npy').shape, np.load('Features/y_fresh.npy').shape)"`
3. Run `python src/train_split.py`
4. Confirm `ls models/X_train.npy models/X_test.npy`
5. Only then run `python src/preprocess_and_rank.py`

---


