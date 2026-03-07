# Project overview — what each file does (short, exact)

**`extract_features.py`**

* Low-level feature extraction for a single image and batch helpers.
* Exposes:

  * `model` — EfficientNetB0 (include_top=False, pooling='avg')
  * `preprocess_input` — EfficientNet preprocessing function
  * `extract_handcrafted(img)` — returns 32 handcrafted features from an RGB image array
  * `extract_deep_batch(paths, batch_size)` — run EfficientNet on a list of image paths (batched) → returns `(N,1280)` array
  * `extract_features(path)` — convenience: reads a single path → deep(1280) + handcrafted(32) → 1312-d vector
* Input: image file path(s).
* Output: NumPy arrays of features.

**`extract_dataset_features.py`**

* Full dataset feature building script (deterministic, robust).
* Scans `vegetable_Dataset/`, filters by `TARGET_VEGETABLES`, loads images (parallelized with threads), runs batched EfficientNet inference using `extract_deep_batch` and computes handcrafted features, concatenates them → saves:

  * `Features/X.npy` (shape: N × 1312)
  * `Features/y_veg.npy` (N strings)
  * `Features/y_fresh.npy` (N ints: 1 fresh, 0 rotten)
* Key behaviors: batching, thread-based image IO, filters non-image files, skips bad images.

**`train_split.py`**

* Create and save a reproducible stratified train/test split (recommended once).
* Loads `Features/X.npy`, `y_veg`, `y_fresh`. Builds combined stratify label `f"{veg}_{fresh}"` to keep both vegetable and freshness balance. Saves split arrays into `models/`:

  * `models/X_train.npy`, `models/X_test.npy`
  * `models/y_veg_train.npy`, `models/y_veg_test.npy`
  * `models/y_fresh_train.npy`, `models/y_fresh_test.npy`
* Use when you want stable evaluation and to avoid reloading full dataset repeatedly.

**`preprocess_and_rank.py`**

* Feature preprocessing + XGBoost ranking on *standardized raw features* (not PCA-first).
* Steps (using training split if available, else full Features):

  1. `VarianceThreshold` — remove constant features (fit → saves `variance.joblib`)
  2. `StandardScaler` — fit on reduced features (saves `scaler.joblib`)
  3. Fit `xgboost.XGBClassifier` on standardized reduced features vs `y_fresh`
  4. Compute importances → select top_k indices (saves `selected_features.npy`)
  5. Save `feature_importances.npy` for inspection
* Outputs in `models/`: `variance.joblib`, `scaler.joblib`, `selected_features.npy`, `feature_importances.npy`.

**`train_svm.py`**

* Final model training script. Uses preprocessing artifacts to produce optimized inputs and save SVM models:

  * Load train data (`models/X_train.npy` if present, else `Features/X.npy`)
  * Apply `variance.joblib` → `scaler.joblib` → select `selected_features.npy` → get `X_sel` (final features)
  * Train LabelEncoder on `y_veg` and save `label_encoder.joblib`
  * Train two SVM pipelines (each includes an inner `StandardScaler` then `SVC(probability=True, class_weight='balanced')`):

    * `veg_svm.joblib` (multiclass vegetable type)
    * `fresh_svm.joblib` (binary fresh/rotten with probabilities)
* Saves models to `models/`.

**`predict_cli.py`**

* Inference CLI for a single image. Workflow:

  1. `extract_features(path)` → 1312 features
  2. Apply `variance.joblib` → `scaler.joblib` → select features by `selected_features.npy`
  3. `veg_svm.predict_proba` → choose top class and confidence (LabelEncoder decode)
  4. `fresh_svm.predict_proba` → `P(fresh)` → `score = P(fresh) * 100`
  5. `grade = grade_from_score(score)` (utils provides thresholds)
* Prints: vegetable, veg confidence, freshness probability, score, grade.

**`utils.py`**

* Small helpers and constants:

  * `TARGET_VEGETABLES` set (tomato, carrot, potato, cucumber, capsicum)
  * `save_model` / `load_model` wrappers (joblib), `ensure_dirs`
  * `grade_from_score(score)` — mapping score → grade per thresholds.

---

# Artifacts produced (files saved on disk)

```
Features/
    X.npy                # N x 1312 feature matrix
    y_veg.npy            # N vegetable labels (strings)
    y_fresh.npy          # N freshness labels (0/1)
models/
    X_train.npy          # optional (if you ran train_split)
    X_test.npy
    y_veg_train.npy
    y_veg_test.npy
    y_fresh_train.npy
    y_fresh_test.npy

    variance.joblib      # VarianceThreshold fitted on raw features
    scaler.joblib        # StandardScaler fitted on reduced raw features
    selected_features.npy# indices (into reduced space) selected by XGBoost
    feature_importances.npy
    label_encoder.joblib
    veg_svm.joblib
    fresh_svm.joblib
```

---

# Exact workflow mind-map (ASCII, top → bottom)

```
[ IMAGE FOLDERS ]
vegetable_Dataset/
  ├─ FreshTomato/
  ├─ RottenTomato/
  ├─ FreshCarrot/
  └─ ...

         │
         │  (1) extract_dataset_features.py
         │     - scan dataset (sorted)
         │     - parallel image loading (threads)
         │     - batch deep feature inference (EfficientNetB0)
         │     - handcrafted features per image
         ▼
  Features/X.npy  (N × 1312)   Features/y_veg.npy   Features/y_fresh.npy
         │
         │  (optional: one-time) train_split.py
         │     - stratified split (veg + fresh)
         │     -> models/X_train.npy, X_test.npy, y_*-train/test.npy
         ▼
  models/X_train.npy  (or Features/X.npy if you skip split)
         │
         │  (2) preprocess_and_rank.py
         │     - VarianceThreshold (remove constant features)
         │     - StandardScaler (fit on reduced raw features)
         │     - XGBoost on standardized raw features vs y_fresh
         │     - select top_k feature indices
         │     -> models/variance.joblib, scaler.joblib, selected_features.npy
         ▼
  selected raw features (reduced → standardized → selected)
         │
         │  (3) train_svm.py
         │     - load X (train split or full)
         │     - apply variance.joblib -> scaler.joblib -> selected indices
         │     - train LabelEncoder on y_veg --> save label_encoder.joblib
         │     - train veg_svm (multiclass) --> save veg_svm.joblib
         │     - train fresh_svm (binary, probability=True) --> save fresh_svm.joblib
         ▼
  models/veg_svm.joblib   models/fresh_svm.joblib   label_encoder.joblib
         │
         │  (4) predict_cli.py  (inference for a single image)
         │     - extract_features(path) -> 1312 vector
         │     - vt.transform -> scaler.transform -> selected indices
         │     - veg_svm.predict_proba -> veg name + confidence
         │     - fresh_svm.predict_proba -> P(fresh)
         │     - score = P(fresh) * 100
         │     - grade_from_score(score)
         ▼
  Output: Vegetable, Veg confidence, Freshness probability, Score, Grade
```

---

# Short run commands (exact order to run once)

```bash
# 1. extract features (slow; run once)
python src/extract_dataset_features.py

# 2. create train/test split (optional but recommended)
python src/train_split.py

# 3. preprocess + feature ranking (fit on train split if exists)
python src/preprocess_and_rank.py

# 4. train final SVMs
python src/train_svm.py

# 5. predict on single image
python src/predict_cli.py --image path/to/image.jpg
```

---

# Edge cases, important notes & failure modes (be explicit)

* **Missing EfficientNet weights / TensorFlow**: first run will download weights. If TF not installed, extraction fails. Confirm `python -c "import tensorflow as tf; print(tf.__version__)"`.
* **Batch size and RAM**: `extract_dataset_features.py` uses batching and threads. If you run out of RAM, reduce `BATCH_SIZE` and/or `NUM_WORKERS`. On your machine (i7, 16GB) recommended start: `BATCH_SIZE=64`.
* **Dataset folder naming**: code expects folders named like `FreshTomato` / `RottenTomato` (case-insensitive). If your dataset uses underscores or different naming, update `parse_folder()` accordingly.
* **Corrupt images**: extractor skips corrupt images but keeps label arrays in parallel — ensure labels / paths remain aligned. The provided scripts handle bad images by skipping and still saving aligned arrays (they keep the same ordering; if an image is skipped your script currently drops that sample — confirm label arrays are trimmed the same way).
* **PCA**: currently PCA is **not** used by default. If you want further dimension reduction after selection we can add PCA as an optional step.
* **Interpretability**: XGBoost ranks standardized **raw** reduced features (not PCA components) so you can inspect which original features mattered.

---

