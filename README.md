# Vegetable Freshness & Variety Classification System

A production-quality machine learning pipeline that classifies **vegetable identity** and **freshness state** from images, equipped with a multi-level **Reliability Gating System** that formally certifies when a prediction can be trusted — and when it cannot.

Rather than returning raw predictions with uncalibrated confidence scores, this system produces one of three output states (`RELIABLE`, `TENTATIVE`, `UNRELIABLE`) backed by formal threshold calibration, out-of-distribution detection, and perturbation stability analysis.

---

## Performance (Held-Out Test Set — 2,539 images)

| Metric | Value |
|--------|-------|
| Vegetable classification accuracy | **99.61%** |
| Freshness classification accuracy | **98.94%** |
| Freshness ROC-AUC (margin-based) | **0.9994** |
| Fresh score mean | 86.84 / 100 |
| Rotten score mean | 16.10 / 100 |
| Score delta (fresh − rotten) | 70.73 pts |
| Overlap (rotten > fresh mean) | 0.00% |
| RELIABLE predictions | **92.3%** of test images |
| TENTATIVE predictions | 5.3% |
| UNRELIABLE / OOD | 2.4% |
| RELIABLE-only freshness accuracy | 98.98% |
| Catastrophic silent failures | **0** |

---

## Table of Contents

1. [Supported Vegetables](#supported-vegetables)
2. [Requirements](#requirements)
3. [System Overview](#system-overview)
4. [Feature Extraction](#feature-extraction)
5. [Dataset Splitting](#dataset-splitting)
6. [Feature Selection](#feature-selection)
7. [Model Training & Calibration](#model-training--calibration)
8. [Threshold Selection](#threshold-selection)
9. [Inference & Reliability Gating](#inference--reliability-gating)
10. [Evaluation](#evaluation)
11. [Directory Structure](#directory-structure)
12. [Pipeline Execution](#pipeline-execution)
13. [Output Reference](#output-reference)
14. [Design Principles & Known Limitations](#design-principles--known-limitations)

---

## Supported Vegetables

| Vegetable | Freshness States | RELIABLE Rate | RELIABLE Accuracy |
|-----------|-----------------|---------------|-------------------|
| Apple | Fresh / Rotten | 95.5% | 99.07% |
| Banana | Fresh / Rotten | 92.1% | 99.88% |
| Capsicum | Fresh / Rotten | 91.6% | 100.00% |
| Cucumber | Fresh / Rotten | 87.9% | 96.88% |
| Potato | Fresh / Rotten | 86.1% | 96.05% |

---

## Requirements

```bash
pip install tensorflow scikit-learn xgboost opencv-python numpy tqdm joblib seaborn matplotlib
```

| Package | Purpose |
|---------|---------|
| `tensorflow` | EfficientNetB0 feature extraction |
| `scikit-learn` | SVM, calibration, preprocessing, metrics |
| `xgboost` | Feature importance ranking |
| `opencv-python` | Image loading, handcrafted feature extraction |
| `numpy` | Array operations |
| `tqdm` | Progress bars during batch extraction |
| `joblib` | Model serialisation |
| `seaborn`, `matplotlib` | Visualisation (`visualize_results.py` only) |

All scripts live in `src/` and are run from the **project root**:

```bash
cd ~/Desktop/mini-project
python src/extract_dataset_features.py
```

---

## System Overview

The pipeline has seven sequential stages, each implemented in a dedicated script under `src/`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FULL PIPELINE OVERVIEW                             │
└─────────────────────────────────────────────────────────────────────────────┘

   src/extract_dataset_features.py       src/train_split.py
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │  For every image in      │          │  Stratified 70/10/20     │
   │  vegetable_Dataset/:     │───────►  │  split on (veg, fresh)   │
   │  EfficientNetB0 [1280]   │          │  composite label         │
   │  + Handcrafted  [32]     │          │  → X_train / X_val /     │
   │  = Feature vec  [1312]   │          │    X_test  (.npy)        │
   └──────────────────────────┘          └──────────────────────────┘
                                                       │
                                                       ▼
   src/preprocess_and_rank.py                 src/train_svm.py
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │  VarianceThreshold       │          │  GridSearchCV (RBF SVM)  │
   │  + StandardScaler        │───────►  │  Vegetable SVM +         │
   │  XGBoost ranking         │          │  Freshness SVM           │
   │  (5 seeds × 2 tasks)     │          │  Isotonic calibration    │
   │  Union feature set       │          │  OOD + bounds + gates    │
   └──────────────────────────┘          └──────────────────────────┘
                                                       │
                                                       ▼
   src/evaluate_models.py                 src/predict_cli.py
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │  Held-out test set only  │          │  Full inference path:    │
   │  Classification metrics  │          │  Preflight → Centroid →  │
   │  Inversion rates         │          │  OOD → Boundary →        │
   │  Gate ablation study     │          │  RELIABLE / TENTATIVE /  │
   │  State distribution      │          │  UNRELIABLE              │
   └──────────────────────────┘          └──────────────────────────┘
```

---

## Feature Extraction

**Script:** `src/extract_dataset_features.py` | **Feature engine:** `src/extract_features.py`

Each image is converted into a **1312-dimensional feature vector** by concatenating two complementary representations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                           │
│                                                                     │
│   Input Image (any size, JPEG / PNG)                                │
│         │                                                           │
│         ▼  resize to 224×224, BGR→RGB                               │
│   ┌─────────────────────┐   ┌───────────────────────────────────┐   │
│   │   EfficientNetB0    │   │       Handcrafted Features        │   │
│   │   (ImageNet weights)│   │                                   │   │
│   │   include_top=False │   │  RGB channel means      [3]       │   │
│   │   pooling="avg"     │   │  RGB channel stds       [3]       │   │
│   │                     │   │  HSV channel means      [3]       │   │
│   │   Global Average    │   │  HSV channel stds       [3]       │   │
│   │   Pooling of the    │   │  Grayscale mean + std   [2]       │   │
│   │   final conv layer  │   │  Edge density (Canny)   [1]       │   │
│   │                     │   │  Laplacian variance     [1]       │   │
│   │   Output: [1280]    │   │  Luminance histogram    [8]       │   │
│   └─────────────────────┘   │  Zero-padding to 32    [7]        │   │
│           │                 │                                   │   │
│           │                 │  Output: [32]                     │   │
│           │                 └───────────────────────────────────┘   │
│           └──────────┬────────────────┘                             │
│                      ▼                                              │
│           Concatenate → [1312]                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation details:**
- Images loaded in parallel via `ThreadPoolExecutor` with `os.cpu_count()` workers
- EfficientNet inference runs in batches of 128 for cache-efficient CPU execution
- Failed image loads are silently filtered; surviving paths are saved row-aligned to the feature matrix
- `image_paths.npy` is required by `src/train_svm.py` to run real EfficientNet augmentations during threshold calibration

---

## Dataset Splitting

**Script:** `src/train_split.py`

The dataset is split **once** into three disjoint sets using stratified sampling on a **composite label** `"{vegetable}_{freshness}"` to ensure each vegetable×freshness combination is proportionally represented in every split.

```
┌──────────────────────────────────────────────────────────────────┐
│                   STRATIFIED SPLITTING STRATEGY                  │
│                                                                  │
│  Composite label: "banana_fresh", "potato_rotten", ...           │
│                                                                  │
│  Full Dataset  (12,691 images)                                   │
│  │                                                               │
│  ├── 70%  →  Train (8,883)   model fitting only                  │
│  ├── 10%  →  Val   (1,269)   ALL calibration — thresholds,       │
│  │                            bounds, OOD, augmentation stats    │
│  └── 20%  →  Test  (2,539)   LOCKED until evaluate_models.py     │
│                                                                  │
│  The val set is NEVER used for final accuracy reporting.         │
│  The test set is NEVER opened before evaluate_models.py.         │
└──────────────────────────────────────────────────────────────────┘
```

The val set is divided further inside `src/train_svm.py` into disjoint **cal_val** (634 samples) and **thr_val** (635 samples) halves to prevent calibration leakage.

---

## Feature Selection

**Script:** `src/preprocess_and_rank.py`

Feature selection reduces the 1312-dimensional space to a compact **union feature set** used by both SVMs.

### Preprocessing

```
X_train [8,883 × 1312]
      │
      ▼  VarianceThreshold(threshold=0.0)   — fit on train only
      │  removes 8 zero-variance columns (zero-padding from handcrafted block)
      ▼
X_reduced [8,883 × 1304]
      │
      ▼  StandardScaler                     — fit on train only
      │  zero mean, unit variance per feature
      ▼
X_scaled [8,883 × 1304]
```

### XGBoost Feature Ranking (5-seed average, 2 independent tasks)

XGBoost is run separately for each task. A freshness-only ranking denies the vegetable classifier the features it needs; a combined label mixes the two signals. Dual independent rankings with a union is the correct approach:

```
┌─────────────────────────────────────────────────────────────────┐
│              DUAL TASK FEATURE RANKING                          │
│                                                                 │
│  Task 1: FRESHNESS  (y_fresh labels)                            │
│    Seeds [42, 7, 123, 17, 99] → XGBoost(100 trees) each         │
│    Average gain importance → avg_imp_fresh [1304]               │
│                                                                 │
│  Task 2: VEGETABLE  (y_veg labels, same procedure)              │
│    5 seeds → average → avg_imp_veg [1304]                       │
│                                                                 │
│  Rankings computed ONCE at max(k)=250, sliced per candidate.    │
│  Total XGBoost fits: 5 × 2 = 10  (not 50)                       │
│  Stability: min pairwise seed overlap = 1.000  [OK]             │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Phase k Selection

```
k_candidates = {50, 100, 150, 200, 250}

Phase 1 — Proxy Sweep (LinearSVC):
  union_k = top-k(fresh) ∪ top-k(veg) for each k
  Proxy winner: k=250  (combined val acc=0.9807)

Phase 2 — RBF SVM Confirmation (3-fold sweep, 5-fold refit on winner):
  RBF winner: k=200  (combined=0.9925)
  Note: proxy k=250 corrected to k=200 — larger sets add noise for RBF

  5-fold refit:  veg C=10.0, γ=0.001  |  fresh C=10.0, γ='scale'
```

### Final Union Feature Set

```
  selected_fresh = top-200 by avg_imp_fresh
  selected_veg   = top-200 by avg_imp_veg

  ┌────────────────┬──────────────┬────────────────┐
  │  Fresh-only    │   Shared     │   Veg-only     │
  │  149 features  │  51 features │  149 features  │
  └────────────────┴──────────────┴────────────────┘
             ←──────── 349 features ────────→

  Both SVMs trained on this same union set.
```

**Artifacts saved:** `variance.joblib`, `scaler.joblib`, `selected_union_features.npy`, `feature_importances_fresh.npy`, `feature_importances_veg.npy`, `feature_selection_report.json`

---

## Model Training & Calibration

**Script:** `src/train_svm.py`

### Val Set Disjoint Split (Calibration Leakage Prevention)

```
┌─────────────────────────────────────────────────────────────────┐
│               VAL SET DISJOINT SPLIT                            │
│                                                                 │
│  X_val (1,269 samples, stratified by y_fresh)                   │
│    │                                                            │
│    ├── 50% → cal_val (634)  → CalibratedClassifierCV            │
│    │                           (isotonic on veg_base)           │
│    │                                                            │
│    └── 50% → thr_val (635)  → select_thresholds()               │
│                                augmentation stability stats     │
│                                formal gate calibration          │
│                                                                 │
│  WITHOUT THIS SPLIT: isotonic calibration sees the same data    │
│  as threshold selection → thresholds overfit to calibration     │
│  distribution → fail on genuinely new images                    │
└─────────────────────────────────────────────────────────────────┘
```

### Dual SVM Training

```
VEGETABLE CLASSIFIER
  GridSearchCV: 5-fold StratifiedKFold, 150 fits total
  Base: SVC(kernel="rbf", class_weight="balanced", probability=False)
  Grid: C ∈ {1e-3, 1e-2, 0.1, 1, 10, 100}
        γ ∈ {1e-4, 1e-3, 1e-2, 0.1, "scale"}
  Best: C=10.0, γ=0.001   CV acc=0.9958
  Calibrate: CalibratedClassifierCV(FrozenEstimator(veg_base), method="isotonic")
             fit on cal_val only → veg_svm.joblib
  Provides: predict_proba()

FRESHNESS CLASSIFIER
  Same GridSearchCV procedure, binary target (0=rotten, 1=fresh)
  Best: C=10.0, γ="scale"   CV acc=0.9865
  Saved: fresh_svm.joblib
  Provides: decision_function() (raw margin)
            predict() (Fresh / Rotten)
```

### Normalization Bounds (Per-Vegetable, Full Val Set)

The raw SVM margin is converted to a 0–100 score:

```
score = clip( (raw − p5_veg) / (p95_veg − p5_veg) × 100, 0, 100 )

Actual bounds from training run:
  apple     p5=−2.5635  p95=+2.1198  spread=4.6833
  banana    p5=−2.0173  p95=+1.8217  spread=3.8390
  capsicum  p5=−1.2853  p95=+1.8389  spread=3.1241
  cucumber  p5=−1.6697  p95=+1.6762  spread=3.3460
  potato    p5=−1.8869  p95=+1.6565  spread=3.5434
  global    p5=−2.2678  p95=+1.9306  (fallback)
```

Bounds use the **full val set** — not training data (which produces inflated margins due to SVM optimisation) and not a half-split (which may leave thin classes below 50 samples). A missing class raises a hard `RuntimeError` with no silent fallback.

### Mahalanobis OOD Detector

```
  train_mean    = X_train.mean(axis=0)              shape (349,)
  precision     = LedoitWolf().fit(X_train).precision_  shape (349, 349)
  mahal_dist(x) = sqrt( (x − mean)ᵀ · precision · (x − mean) )

  thresh_caution = P90 of training distances = 24.167
  thresh_ood     = P99 of training distances = 30.438

  OOD rate: val=1.81%  test=2.44%  difference=0.63%  [OK < 5%]
```

### Per-Class Centroid Consistency Gate

```
  For each class: centroid = mean(X_train rows for that class)

  At inference:
    d_pred   = L2(x, predicted_centroid)
    d_second = L2(x, nearest other centroid)
    ratio    = d_pred / d_second
    threshold = P95 of ratio on correctly classified val samples

  Thresholds (from training run):
    apple=1.0220  banana=0.9552  capsicum=0.9973
    cucumber=1.0257  potato=0.9740

  class_inconsistent = (ratio > threshold)
  Effect: forces global bounds → decision_unreliable=True → TENTATIVE
```

---

## Threshold Selection

**Script:** `src/threshold_selection.py`

`T_boundary` and `T_instability` are selected by constrained optimisation on `thr_val`:

```
┌─────────────────────────────────────────────────────────────────┐
│  RELIABILITY FORMULA (fixed contract)                           │
│                                                                 │
│  RELIABLE_i = (                                                 │
│    NOT is_ood_i                                                 │
│    AND NOT (crosses_boundary_i AND aug_range_i > T_instability) │
│    AND abs(decision_i) > T_boundary                             │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  OPTIMISATION                                                   │
│                                                                 │
│  Maximise:   Coverage = P(RELIABLE)                             │
│  Subject to: Risk = P(error | RELIABLE) ≤ ε = 0.10              │
│              n_reliable ≥ n_min                                 │
│                                                                 │
│  Grid: T_boundary    ∈ [0.0, 3.0]  step 0.05  (61 values)       │
│        T_instability ∈ [0.0, max_aug_range]  step 0.5           │
│                                                                 │
│  Result: T_boundary=0.0000   T_instability=36.0000              │
│          Risk=0.0188  Coverage=97.89%  n_reliable=372           │
│          feasible=True                                          │
│                                                                 │
│  T_boundary=0.0 is the formal result — the base model is        │
│  accurate enough that no margin cutoff is needed to keep        │
│  Risk ≤ 10%. The boundary gate is effectively inactive.         │
└─────────────────────────────────────────────────────────────────┘
```

If no feasible pair exists, `diagnose_infeasibility()` classifies the failure:

| Case | Description | Recommended Action |
|------|-------------|-------------------|
| **Case (a)** | Risk flat across margin quantiles. Margin has no predictive power for errors. | Remove T_boundary gate |
| **Case (b)** | Risk decreases with margin but never reaches ε. Base model error is binding. | Lower ε or improve the freshness classifier |
| **insufficient_data** | Fewer than 2 quantiles with n_reliable > 0 | Reduce n_min or collect more data |

---

## Inference & Reliability Gating

**Script:** `src/predict_cli.py`

Every prediction passes through a sequential gate pipeline. The first gate to fail determines the output state.

```
Input: path/to/image.jpg
         │
         ▼
┌─────────────────────┐
│  STAGE 1            │
│  Preflight Checks   │  Laplacian variance < 28.0         →  UNRELIABLE
│                     │  Mean brightness ∉ [30, 220]       →  UNRELIABLE
│                     │  Object coverage < 0.40            →  warning only
└────────┬────────────┘
         │  pass
         ▼
┌─────────────────────┐
│  STAGE 2            │
│  Feature Extraction │  EfficientNetB0 → [1280]
│  + Preprocessing    │  Handcrafted → [32]  |  Concat → [1312]
│                     │  VarianceThreshold → StandardScaler → union[349]
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 3            │
│  Vegetable SVM      │  predict_proba() → top-1 label + veg_conf% + gap%
│                     │  veg_confident = (conf ≥ 70%) AND (gap ≥ 15%)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 4            │
│  Centroid Check     │  ratio = d_pred / d_second
│                     │  class_inconsistent = ratio > per-class P95 threshold
│                     │  → forces global bounds if inconsistent
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 5            │
│  Freshness SVM      │  decision_function() → raw margin
│                     │  normalize → score [0–100]
│                     │  (per-veg bounds if confident & consistent,
│                     │   else global bounds)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 6            │
│  OOD Gate           │  dist ≥ 30.438  →  score_unreliable=True  →  UNRELIABLE
│  (Mahalanobis)      │  zone: trusted / caution [24.167–30.438] / ood
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 7 (optional) │
│  Augmentation Gate  │  6 augmented views: ±15% brightness, flip,
│  (disabled by       │  Gaussian blur, ±5° rotation
│   default)          │  score_range = max(scores) − min(scores)
│                     │  crosses_boundary = min(raw) < 0 AND max(raw) > 0
│                     │  unstable = (score_range ≥ 36.0) AND crosses_boundary
│                     │  → score_unreliable=True if unstable
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 8            │
│  Boundary Gate      │  near_boundary = abs(raw) < T_boundary
│                     │  (currently inactive — T_boundary=0.0)
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  RELIABILITY DECISION                                            │
│                                                                  │
│  score_unreliable    = is_ood OR unstable                        │
│  decision_unreliable = near_boundary OR NOT veg_confident OR     │
│                        class_inconsistent OR conf_gap < 10%      │
│                                                                  │
│  High-confidence override: veg_conf > 95% AND not near_boundary  │
│  AND not crosses_boundary AND not is_ood AND not inconsistent    │
│  → force RELIABLE regardless of decision_unreliable              │
│                                                                  │
│  score_unreliable=True   →  UNRELIABLE  (no score, no label)     │
│  decision_unreliable=True →  TENTATIVE  (score, no fresh_label)  │
│  neither                 →  RELIABLE   (score + label + band)    │
└──────────────────────────────────────────────────────────────────┘
```

### Output States

| State | Score | Fresh Label | Confidence Band | When |
|-------|-------|-------------|-----------------|------|
| `RELIABLE` | ✅ | ✅ | ✅ | All gates passed |
| `TENTATIVE` | ✅ | ❌ | ❌ | Near boundary, low veg confidence, or centroid mismatch |
| `UNRELIABLE` | ❌ | ❌ | ❌ | Image quality failed, OOD detected, or severe aug instability |

### Freshness Confidence Bands

| Band | Score | Interpretation |
|------|-------|----------------|
| **High** | ≥ 85 | Strongly in the fresh region of feature space |
| **Medium** | 65–84 | Moderately fresh signal |
| **Low** | 40–64 | Weakly fresh; marginal |
| **Very Low** | < 40 | Strongly rotten signal |

> **Important:** The score is a per-vegetable calibrated proxy derived from the SVM decision margin. It reflects model certainty relative to the training distribution — not an absolute freshness measurement. A score of 80 for banana and 80 for potato both mean "high relative to that vegetable's training distribution" — they are not on a shared scale. The `fresh_label` field (`Fresh` / `Rotten`) is the primary actionable output.

---

## Evaluation

**Script:** `src/evaluate_models.py`

The test set is loaded **for the first time** here. No calibration step has seen this data.

```
┌────────────────────────────────────────────────────────────────┐
│                  EVALUATION REPORT SECTIONS                    │
│                                                                │
│  1. Classification Metrics                                     │
│     Vegetable accuracy + per-class precision/recall/F1         │
│     Freshness accuracy + confusion matrix                      │
│     Freshness ROC-AUC (SVM margin as ranking signal)           │
│                                                                │
│  2. Inversion Rate Diagnostics                                 │
│     Raw margin / global-norm / deployed-norm inversion         │
│     Per-vegetable gate delta                                   │
│     STABLE if: global delta < 0.01 AND max per-veg < 0.02      │
│                                                                │
│  3. Gate Ablation Study                                        │
│     G1 OOD, G2 near-boundary, G3 low-veg-confidence            │
│     Δ_acc = accuracy change when gate disabled                 │
│     Δ_cov = coverage change when gate disabled                 │
│     Verdict: KEEP / REMOVE / REVIEW / NEVER FIRES              │
│                                                                │
│  4. State Distribution                                         │
│     Global + per-vegetable RELIABLE/TENTATIVE/UNRELIABLE       │
│                                                                │
│  5. RELIABLE-Subset Accuracy                                   │
│     RELIABLE acc must be ≥ overall acc (gate sanity check)     │
│     Per-vegetable RELIABLE accuracy vs 98.98% baseline         │
│                                                                │
│  6. OOD Rate Consistency  (val vs test — must be < 5% diff)    │
│                                                                │
│  7. Score Distribution                                         │
│     Fresh mean=86.84  Rotten mean=16.10  Delta=70.73 pts       │
│     Overlap (rotten > fresh mean) = 0.0000                     │
│                                                                │
│  8. Silent Failure Analysis                                    │
│     Veg wrong but RELIABLE → 3 (all "accidental correct")      │
│     Catastrophic (veg+fresh both wrong) → 0                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
mini-project/
│
├── src/
│   ├── extract_features.py          # EfficientNetB0 + handcrafted feature engine (1312-d)
│   ├── extract_dataset_features.py  # Batch extraction over full dataset → Features/
│   ├── train_split.py               # Stratified 70/10/20 split → models/
│   ├── preprocess_and_rank.py       # VarianceThreshold, scaler, XGBoost, union set
│   ├── train_svm.py                 # Dual RBF SVM, calibration, gate fitting
│   ├── threshold_selection.py       # Formal T_boundary / T_instability selection
│   ├── predict_cli.py               # Full inference with 8-stage reliability gating
│   ├── evaluate_models.py           # Held-out test set evaluation + gate ablation
│   ├── visualize_results.py         # Confusion matrices + feature importance plots
│   └── utils.py                     # I/O helpers, confidence_band(), TARGET_VEGETABLES
│
├── Features/                        # Generated by extract_dataset_features.py
│   ├── X.npy                        # [12691 × 1312] float32
│   ├── y_veg.npy                    # Vegetable label strings
│   ├── y_fresh.npy                  # Freshness labels (0=rotten, 1=fresh)
│   └── image_paths.npy              # Row-aligned image paths (required by train_svm.py)
│
├── models/                          # Generated by training scripts
│   ├── X_train.npy                  # [8883 × 1312]
│   ├── X_val.npy                    # [1269 × 1312]
│   ├── X_test.npy                   # [2539 × 1312]  LOCKED until evaluate_models.py
│   ├── y_veg_train/val/test.npy
│   ├── y_fresh_train/val/test.npy
│   ├── val_image_paths.npy          # Required for aug stats in train_svm.py
│   ├── variance.joblib              # VarianceThreshold (fit on train only)
│   ├── scaler.joblib                # StandardScaler (fit on train only)
│   ├── selected_union_features.npy  # [349] column indices used by both SVMs
│   ├── selected_fresh_features.npy  # [200] freshness-task top-200
│   ├── selected_veg_features.npy    # [200] vegetable-task top-200
│   ├── feature_importances_fresh.npy
│   ├── feature_importances_veg.npy
│   ├── feature_selection_report.json
│   ├── veg_svm_base.joblib          # Raw GridSearchCV SVC (pre-calibration)
│   ├── veg_svm.joblib               # CalibratedClassifierCV (final)
│   ├── fresh_svm.joblib             # Freshness SVC
│   ├── label_encoder.joblib         # Vegetable name ↔ integer
│   ├── train_mean.npy               # Mahalanobis centroid [349]
│   ├── train_precision.npy          # LedoitWolf precision [349 × 349]
│   ├── class_centroids.npy          # Per-class centroids [5 × 349]
│   └── scoring_config.json          # All thresholds + calibration metadata
│
└── vegetable_Dataset/               # Source images (not committed)
    ├── freshapple/
    ├── rottenapple/
    ├── freshbanana/
    ├── rottenbanana/
    └── ...                          # one folder per vegetable × freshness
```

---

## Pipeline Execution

Run all commands from the **project root** (`~/Desktop/mini-project`).

### Step 1 — Extract Features

```bash
python src/extract_dataset_features.py
```

Scans `vegetable_Dataset/`, extracts 1312-d feature vectors for all images, saves to `Features/`. Takes approximately 5 minutes on CPU (Intel i7 12th-gen). TensorFlow startup warnings (oneDNN, CUDA) are informational — the CUDA error is expected on CPU-only machines.

```
Total images: 12691
Batches: 100%|████████████████████| 100/100 [04:49<00:00,  2.90s/it]
Saved feature matrix: (12691, 1312)
```

### Step 2 — Split Dataset

```bash
python src/train_split.py
```

Creates stratified 70/10/20 splits, saves all feature matrices and labels to `models/`, including `val_image_paths.npy` which is required by Step 4.

```
[SUCCESS] Split created — Train=8883  Val=1269  Test=2539
[INFO] Test set must remain untouched until evaluate_models.py
```

### Step 3 — Select Features

```bash
python src/preprocess_and_rank.py
```

Fits preprocessing on training data only, runs dual XGBoost ranking (10 fits total), runs two-phase k-selection sweep (proxy + RBF confirmation), saves 349-feature union set.

```
[INFO] VarianceThreshold: 1312 → 1304
  [NOTE] RBF confirmation changed best_k: 250 → 200  (combined=0.9925)
  Union size: 349  (Fresh-only: 149, Shared: 51, Veg-only: 149)
```

### Step 4 — Train & Calibrate

```bash
python src/train_svm.py
```

Tunes both SVMs via GridSearchCV (150 fits each), calibrates vegetable probabilities with isotonic regression on `cal_val`, computes per-vegetable normalization bounds on full val set, fits Mahalanobis OOD detector, runs augmentation statistics on `thr_val`, runs formal threshold selection, saves `scoring_config.json`. **Test set is never loaded in this step.**

```
[INFO] val split: cal_val=634  thr_val=635
[INFO] Formal thresholds — T_boundary=0.0000  T_instability=36.0000
       Risk=0.0188  Coverage=0.9789  n_reliable=372
```

### Step 5 — Evaluate

```bash
python src/evaluate_models.py
```

Opens the test set for the first time. Produces the full evaluation report.

### Step 6 — Predict

```bash
# Full pipeline with reliability gating
python src/predict_cli.py --image path/to/vegetable.jpg

# Fast mode — skips augmentation gate (saves ~6× EfficientNet passes)
python src/predict_cli.py --image path/to/vegetable.jpg --no-uncertainty
```

### Step 7 — Visualize (optional)

```bash
python src/visualize_results.py
```

Renders four matplotlib figures using the test set: vegetable confusion matrix (Blues), freshness confusion matrix (Greens), top-20 XGBoost feature importances for the freshness task, top-20 for the vegetable task.

### Partial Re-run Checklist

| What changed | Commands needed |
|---|---|
| SVM hyperparameters only | `train_svm.py` → `evaluate_models.py` |
| Feature selection (k, seeds, XGBoost settings) | `preprocess_and_rank.py` → `train_svm.py` → `evaluate_models.py` |
| New images added to dataset | All 5 steps in order |
| Gate thresholds in `scoring_config.json` only | Edit config → `evaluate_models.py` |

---

## Output Reference

### `predict_cli.py` — Return Value Structure

**RELIABLE:**
```json
{
  "state": "RELIABLE",
  "veg": "banana",
  "veg_conf": 98.4,
  "score": 77.2,
  "score_range": 0.0,
  "raw": 1.843,
  "fresh_label": "Fresh",
  "freshness_confidence_band": "Medium",
  "norm_source": "per-veg",
  "mahal_dist": 14.2,
  "mahal_zone": "trusted",
  "warnings": []
}
```

**TENTATIVE** (score present; no fresh_label):
```json
{
  "state": "TENTATIVE",
  "veg": "potato",
  "veg_conf": 61.3,
  "score": 42.1,
  "raw": 0.217,
  "fresh_label": null,
  "freshness_confidence_band": null,
  "norm_source": "global",
  "mahal_dist": 18.7,
  "mahal_zone": "caution",
  "warnings": [
    "Low veg confidence (61.3%, gap=8.2%) — using global normalization.",
    "CAUTION — Mahalanobis dist=18.7 in caution zone [24.167, 30.438]."
  ]
}
```

**UNRELIABLE** (image quality failure):
```json
{
  "state": "UNRELIABLE",
  "reason": "Image out of focus (lap_var=12.3 < 28.0)",
  "score": null,
  "raw": null,
  "fresh_label": null,
  "freshness_confidence_band": null
}
```

### `scoring_config.json` — Key Fields

| Key | Value | Description |
|-----|-------|-------------|
| `boundary_threshold` | 0.0 | abs(raw) must exceed this for RELIABLE |
| `unstable_range_thresh` | 36.0 | T_instability (aug gate; currently disabled) |
| `use_augmentation_gate` | false | Aug gate inactive at inference by default |
| `mahal_thresh_caution` | 24.167 | P90 of training Mahalanobis distances |
| `mahal_thresh_ood` | 30.438 | P99 of training Mahalanobis distances |
| `veg_confidence_threshold` | 0.70 | Min top-1 probability for per-veg bounds |
| `veg_gap_threshold` | 0.15 | Min top-1 minus top-2 probability gap |
| `per_veg_bounds` | per-class dict | p5/p95 normalization per vegetable |
| `global_bounds` | fallback dict | Used when veg confidence is low |
| `centroid_ratio_thresholds` | per-class dict | Centroid consistency gate thresholds |
| `threshold_selection_result` | dict | Formal outcome: feasible, risk, coverage |
| `svm_best_params` | dict | Best GridSearchCV params for each SVM |
| `calibration_note` | string | Provenance record for all design decisions |

---

## Design Principles & Known Limitations

### Design Principles

**Dual task separation.** XGBoost rankings are computed independently for freshness and vegetable tasks. A freshness-only ranking denies the vegetable classifier the features it needs; a combined label mixes the two signals. Independent rankings with a union is the correct design.

**No calibration leakage.** The val set is split into disjoint `cal_val` and `thr_val` halves before any calibration step. Isotonic regression sees only `cal_val`; threshold selection sees only `thr_val`. Sharing data between these steps causes thresholds to overfit to the calibration distribution.

**Hard errors over silent fallbacks.** Missing per-vegetable normalization bounds raise a `RuntimeError` at training time rather than silently falling back to global bounds. A silent fallback would mask thin-class failures and produce miscalibrated scores with no visible warning.

**Evaluation mirrors deployment exactly.** `evaluate_models.py` uses *predicted* (not ground-truth) vegetable labels to select normalization bounds when computing deployed scores. Using ground truth would silently inflate evaluation scores by ensuring the correct bounds are always chosen.

**Formal threshold selection over heuristics.** T_boundary and T_instability are found by constrained optimisation (maximise Coverage subject to Risk ≤ ε) on held-out `thr_val` data — not by manual tuning or arbitrary percentile rules.

### Known Limitations

- **Scores are not cross-vegetable comparable.** A banana score of 80 and a potato score of 80 both mean "high relative to that vegetable's training distribution" — not the same absolute freshness level. Cross-vegetable comparison requires using `global_bounds` throughout.

- **Intra-class ordering is unvalidated.** The system proves class-level separation (fresh vs rotten) and ordering reliability between classes. Intra-class ordering (e.g., "this banana is 3 days older than that one") is not validated without continuous decay-day annotations.

- **Augmentation gate requires 6× inference cost.** The instability gate runs EfficientNetB0 six times per image. Currently disabled (`use_augmentation_gate=False`). T_instability=36.0 is formally calibrated and stored. Reactivate by setting `use_augmentation_gate: true` in `scoring_config.json`.

- **Imaging condition sensitivity.** All calibration assumes consistent imaging conditions similar to training. Significant departures in lighting, background, or camera quality will increase OOD flags. The OOD rate discrepancy between val and test is monitored; a > 5% difference flags a warning in `evaluate_models.py`.

- **Score is not a physical freshness measurement.** A vegetable with internal decay that looks visually fresh will receive a high score. The system classifies visual appearance — the `fresh_label` field is the primary actionable output; the numeric score quantifies model confidence, not biological freshness state.