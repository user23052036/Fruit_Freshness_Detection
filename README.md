# Vegetable Freshness & Variety Classification System

A production-quality machine learning pipeline that classifies **vegetable identity** and **freshness state** from images, equipped with a multi-level **Reliability Gating System** that formally certifies when a prediction can be trusted — and when it cannot.

Rather than returning raw predictions with uncalibrated confidence scores, this system produces one of three output states (`RELIABLE`, `TENTATIVE`, `UNRELIABLE`) backed by formal threshold calibration, out-of-distribution detection, and perturbation stability analysis.

---

## Table of Contents

1. [Supported Vegetables](#supported-vegetables)
2. [System Overview](#system-overview)
3. [Feature Extraction](#feature-extraction)
4. [Dataset Splitting](#dataset-splitting)
5. [Feature Selection](#feature-selection)
6. [Model Training & Calibration](#model-training--calibration)
7. [Threshold Selection](#threshold-selection)
8. [Inference & Reliability Gating](#inference--reliability-gating)
9. [Evaluation](#evaluation)
10. [Directory Structure](#directory-structure)
11. [Pipeline Execution](#pipeline-execution)
12. [Output Reference](#output-reference)
13. [Design Principles & Known Limitations](#design-principles--known-limitations)

---

## Supported Vegetables

The system is trained and calibrated specifically for five vegetable classes:

| Vegetable  | Freshness States |
|------------|-----------------|
| Apple      | Fresh / Rotten  |
| Banana     | Fresh / Rotten  |
| Capsicum   | Fresh / Rotten  |
| Cucumber   | Fresh / Rotten  |
| Potato     | Fresh / Rotten  |

---

## System Overview

The pipeline consists of six sequential stages, each implemented in a dedicated script:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FULL PIPELINE OVERVIEW                             │
└─────────────────────────────────────────────────────────────────────────────┘

   extract_dataset_features.py          train_split.py
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  For every image in      │         │  Stratified 70/10/20     │
   │  vegetable_Dataset/:     │──────►  │  split on (veg, fresh)   │
   │  EfficientNetB0 [1280]   │         │  composite label         │
   │  + Handcrafted  [32]     │         │  → X_train / X_val /     │
   │  = Feature vec  [1312]   │         │    X_test  (.npy)        │
   └──────────────────────────┘         └──────────────────────────┘
                                                      │
                                                      ▼
   preprocess_and_rank.py                    train_svm.py
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  VarianceThreshold       │         │  GridSearchCV (RBF SVM)  │
   │  + StandardScaler        │──────►  │  Vegetable SVM +         │
   │  XGBoost ranking         │         │  Freshness SVM           │
   │  (5 seeds × 2 tasks)     │         │  Isotonic calibration    │
   │  Union feature set       │         │  OOD + bounds + gates    │
   └──────────────────────────┘         └──────────────────────────┘
                                                      │
                                                      ▼
   predict_cli.py / app.py              evaluate_models.py
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  Full inference path:    │         │  Held-out test set only  │
   │  Preflight → OOD →       │         │  Classification metrics  │
   │  Boundary → Augmentation │         │  Inversion rates         │
   │  → RELIABLE/TENTATIVE/   │         │  Gate ablation study     │
   │    UNRELIABLE            │         │  State distribution      │
   └──────────────────────────┘         └──────────────────────────┘
```

---

## Feature Extraction

**Script:** `extract_dataset_features.py` | **Feature engine:** `extract_features.py`

Each image is converted into a **1312-dimensional feature vector** by concatenating two complementary representations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                           │
│                                                                     │
│   Input Image (any size, JPEG / PNG)                                │
│         │                                                           │
│         ▼  resize to 224×224, BGR→RGB                               │
│   ┌─────────────────────┐   ┌───────────────────────────────────┐  │
│   │   EfficientNetB0    │   │       Handcrafted Features        │  │
│   │   (ImageNet weights)│   │                                   │  │
│   │   include_top=False │   │  RGB channel means      [3]       │  │
│   │   pooling="avg"     │   │  RGB channel stds       [3]       │  │
│   │                     │   │  HSV channel means      [3]       │  │
│   │   Global Average    │   │  HSV channel stds       [3]       │  │
│   │   Pooling of the    │   │  Grayscale mean + std   [2]       │  │
│   │   final conv layer  │   │  Edge density (Canny)   [1]       │  │
│   │                     │   │  Laplacian variance     [1]       │  │
│   │   Output: [1280]    │   │  Luminance histogram    [8]       │  │
│   └─────────────────────┘   │  Zero-padding to 32    [7]       │  │
│           │                 │                                   │  │
│           │                 │  Output: [32]                     │  │
│           │                 └───────────────────────────────────┘  │
│           │                           │                            │
│           └──────────┬────────────────┘                            │
│                      ▼                                             │
│           Concatenate → [1312]                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Batch processing details:**
- Images are loaded in parallel with `ThreadPoolExecutor`
- EfficientNet inference runs in batches of 128 for GPU/memory efficiency
- Failed image loads are silently filtered; final paths are saved aligned to the feature matrix
- Outputs saved to `Features/`: `X.npy`, `y_veg.npy`, `y_fresh.npy`, `image_paths.npy`

---

## Dataset Splitting

**Script:** `train_split.py`

The dataset is split **once** into three disjoint sets using stratified sampling on a **composite label** `"{vegetable}_{freshness}"` to ensure each vegetable×freshness combination is proportionally represented in every split.

```
┌──────────────────────────────────────────────────────────────────┐
│                   STRATIFIED SPLITTING STRATEGY                  │
│                                                                  │
│  Composite label: "{vegetable}_{fresh/rotten}"                   │
│  e.g. "banana_fresh", "potato_rotten", "cucumber_fresh", ...     │
│                                                                  │
│  Full Dataset  (N samples)                                       │
│  │                                                               │
│  ├── 70%  →  Train Set     (model fitting)                       │
│  ├── 10%  →  Val Set       (ALL calibration — thresholds,        │
│  │                          bounds, OOD, augmentation stats)     │
│  └── 20%  →  Test Set      ← LOCKED until evaluate_models.py    │
│                                                                  │
│  IMPORTANT: The val set is NEVER used for final reporting.       │
│  The test set is NEVER touched until evaluation.                 │
└──────────────────────────────────────────────────────────────────┘
```

The val set is later split further inside `train_svm.py` into disjoint **cal_val** (50%) and **thr_val** (50%) halves to prevent calibration leakage (see [Model Training](#model-training--calibration)).

---

## Feature Selection

**Script:** `preprocess_and_rank.py`

Feature selection reduces the 1312-dimensional space to a compact **union feature set** used by both SVMs. The process runs in two phases.

### Preprocessing

```
X_train [N × 1312]
      │
      ▼  VarianceThreshold(threshold=0.0)   — fit on train only
      │  removes zero-variance columns
      ▼
X_reduced [N × M]   (M ≤ 1312)
      │
      ▼  StandardScaler                     — fit on train only
      │  zero mean, unit variance
      ▼
X_scaled [N × M]
```

### XGBoost Feature Ranking (5-seed average, 2 tasks)

XGBoost is run separately for each task to avoid using freshness-derived importance ranks for the vegetable classifier (or vice versa):

```
┌─────────────────────────────────────────────────────────────────┐
│              DUAL TASK FEATURE RANKING                          │
│                                                                 │
│  Task 1: FRESHNESS ranking                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Seed 42  → XGBoost(100 trees) → gain importance [M]    │   │
│  │  Seed  7  → XGBoost(100 trees) → gain importance [M]    │   │
│  │  Seed 123 → XGBoost(100 trees) → gain importance [M]    │   │
│  │  Seed 17  → XGBoost(100 trees) → gain importance [M]    │   │
│  │  Seed 99  → XGBoost(100 trees) → gain importance [M]    │   │
│  │                    Average → avg_imp_fresh [M]           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Task 2: VEGETABLE ranking  (same procedure)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  5 seeds → average → avg_imp_veg [M]                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Rankings computed ONCE at max(k), sliced cheaply per k.        │
│  Total XGBoost fits: 5 seeds × 2 tasks = 10   (not 50)         │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Phase k Selection

```
k_candidates = {50, 100, 150, 200, 250}

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Proxy Sweep (LinearSVC, fast)                        │
│                                                                 │
│  For each k in {50, 100, 150, 200, 250}:                        │
│    sel_fresh = top-k from avg_imp_fresh                         │
│    sel_veg   = top-k from avg_imp_veg                           │
│    union_k   = sel_fresh ∪ sel_veg                              │
│                                                                 │
│    LinearSVC → val acc (freshness)                              │
│    LinearSVC → val acc (vegetable)                              │
│    combined  = (fresh_acc + veg_acc) / 2                        │
│                                                                 │
│    proxy_best_k = argmax(combined)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — RBF SVM Confirmation (corrects proxy bias)           │
│                                                                 │
│  Sweep (3-fold GridSearchCV, all k candidates):                 │
│    RBF SVM on union_k → val acc (freshness + vegetable)         │
│    Param grid: C ∈ {1e-3,1e-2,0.1,1,10,100}                    │
│               γ ∈ {1e-4,1e-3,1e-2,0.1,"scale"}                 │
│                                                                 │
│  Refit winner (5-fold GridSearchCV):                            │
│    best_k_final = argmax(combined RBF val accuracy)             │
│                                                                 │
│  Stability check: pairwise seed overlap ≥ 0.80 required        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  FINAL UNION FEATURE SET                                        │
│                                                                 │
│  selected_fresh = top-best_k from avg_imp_fresh                 │
│  selected_veg   = top-best_k from avg_imp_veg                   │
│                                                                 │
│  union_set = selected_fresh ∪ selected_veg                      │
│                                                                 │
│  ┌────────────┬────────────┬────────────┐                       │
│  │ Fresh-only │  Shared    │  Veg-only  │  = Union set          │
│  │  features  │  features  │  features  │                       │
│  └────────────┴────────────┴────────────┘                       │
│                                                                 │
│  Both SVMs (vegetable + freshness) are trained on this          │
│  SAME union set. Task-specific ranks are used only for          │
│  selection — the shared feature space preserves signal          │
│  for both classifiers.                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Artifacts saved:** `variance.joblib`, `scaler.joblib`, `selected_union_features.npy`, `feature_importances_fresh.npy`, `feature_importances_veg.npy`, `feature_selection_report.json`

---

## Model Training & Calibration

**Script:** `train_svm.py`

### Val Set Split (Leakage Prevention)

The validation set is split 50/50 into two disjoint halves **before** any calibration occurs. This prevents the isotonic calibration from seeing the same data used for threshold selection:

```
┌─────────────────────────────────────────────────────────────────┐
│               VAL SET DISJOINT SPLIT  (Issue-2 Fix)            │
│                                                                 │
│  X_val  (stratified by y_fresh)                                 │
│    │                                                            │
│    ├── 50%  →  cal_val   →  CalibratedClassifierCV             │
│    │                        (isotonic regression on veg_base)   │
│    │                                                            │
│    └── 50%  →  thr_val   →  select_thresholds()               │
│                              augmentation stability stats       │
│                              formal gate calibration            │
│                                                                 │
│  WITHOUT THIS SPLIT: isotonic calibration sees threshold data   │
│  → calibrated probs overfit to thr_val → thresholds too loose  │
└─────────────────────────────────────────────────────────────────┘
```

### Dual SVM Training

```
┌──────────────────────────────────────────────────────────────────┐
│                    DUAL SVM TRAINING                             │
│                                                                  │
│  Input: X_train[:, union_set]                                    │
│                                                                  │
│  ┌────────────────────────────────────────────┐                  │
│  │         VEGETABLE CLASSIFIER               │                  │
│  │                                            │                  │
│  │  GridSearchCV (5-fold, StratifiedKFold)    │                  │
│  │  Base: SVC(kernel="rbf", class_weight=     │                  │
│  │            "balanced", probability=False)  │                  │
│  │  Grid: C × γ (30 combinations)            │                  │
│  │  → veg_base (best refitted estimator)      │                  │
│  │                                            │                  │
│  │  CalibratedClassifierCV(                   │                  │
│  │    FrozenEstimator(veg_base),              │                  │
│  │    method="isotonic"                       │                  │
│  │  ).fit(X_cal_val, y_veg_cal)              │                  │
│  │  → veg_svm.joblib                          │                  │
│  │  Outputs: predict_proba() for gate logic   │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                  │
│  ┌────────────────────────────────────────────┐                  │
│  │         FRESHNESS CLASSIFIER               │                  │
│  │                                            │                  │
│  │  GridSearchCV (5-fold, StratifiedKFold)    │                  │
│  │  Same RBF SVC, same param grid             │                  │
│  │  → fresh_svm.joblib                        │                  │
│  │  Outputs: decision_function() (raw margin) │                  │
│  │           predict() (Fresh/Rotten label)   │                  │
│  └────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

### Normalization Bounds (Per-Vegetable)

The raw SVM margin (a real number) is converted to a 0–100 score using per-vegetable p5/p95 percentile bounds computed on the **full val set**:

```
score = clip( (raw - p5_veg) / (p95_veg - p5_veg) × 100, 0, 100 )
```

Why the full val set (not just cal_val or thr_val)? The 50/50 split can leave individual vegetable classes below the 50-sample minimum required for stable percentile estimates. Using the full val set for bounds is safe because p5/p95 is a fixed linear transform that cannot encode label information.

Each vegetable must have ≥ 50 val samples for its own bounds; missing entries raise a hard `RuntimeError` — there is no silent fallback to global bounds at training time.

### Mahalanobis OOD Detector

A LedoitWolf-shrinkage covariance estimate is fit on training features to define the training distribution:

```
  train_mean      = X_train.mean(axis=0)
  precision       = LedoitWolf().fit(X_train).precision_
  mahal_dist(x)   = sqrt( (x - mean)ᵀ · precision · (x - mean) )

  thresh_caution  = P90 of mahal distances on X_train
  thresh_ood      = P99 of mahal distances on X_train

  zone(x) = "trusted"   if dist < thresh_caution
           = "caution"  if thresh_caution ≤ dist < thresh_ood
           = "ood"      if dist ≥ thresh_ood
```

### Per-Class Centroid Consistency Check

A centroid ratio gate catches vegetable misclassifications that the OOD detector misses (i.e., samples in-distribution overall but placed in the wrong cluster):

```
  centroids  = mean of X_train per vegetable class
  d_pred     = L2 distance from x to predicted class centroid
  d_second   = L2 distance from x to nearest other centroid

  ratio      = d_pred / d_second
  threshold  = P95 of ratio on correctly classified val samples (per class)

  class_inconsistent = (ratio > threshold)
```

---

## Threshold Selection

**Script:** `threshold_selection.py`

This module formally selects the two reliability gate thresholds on `thr_val` (the held-out half of val):

```
┌─────────────────────────────────────────────────────────────────┐
│              RELIABILITY FORMULA (fixed contract)               │
│                                                                 │
│  RELIABLE_i = (                                                 │
│    NOT is_ood_i                                                 │
│    AND NOT (crosses_boundary_i AND aug_range_i > T_instability) │
│    AND abs(decision_i) > T_boundary                             │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              OPTIMISATION PROBLEM                               │
│                                                                 │
│  Find (T_boundary*, T_instability*) that:                       │
│                                                                 │
│    Maximise:  Coverage = P(RELIABLE)                            │
│    Subject to: Risk = P(error | RELIABLE) ≤ ε   (ε = 0.10)    │
│                n_reliable ≥ n_min                               │
│                                                                 │
│  Grid search:                                                   │
│    T_boundary    ∈ [0.0, 3.0]  step 0.05                       │
│    T_instability ∈ [0.0, max_aug_range]  step 0.5              │
│                                                                 │
│  Tie-break: max coverage → min T_boundary                       │
└─────────────────────────────────────────────────────────────────┘
```

If no feasible `(T_b, T_i)` pair satisfies the constraints, `diagnose_infeasibility()` runs a diagnostic sweep to classify the failure as:

| Case | Description | Recommended Action |
|------|-------------|-------------------|
| **Case (a)** | Risk is flat across margin quantiles (< 2 pp total drop). Margin has no predictive power for errors. | Remove T_boundary gate; consider augmentation-range gate only |
| **Case (b)** | Risk decreases with margin but never reaches ε. Base model error is the binding constraint. | Lower ε to an achievable target, or improve the freshness classifier |
| **insufficient_data** | Fewer than 2 quantiles with n_reliable > 0 | Reduce n_min or collect more data |

---

## Inference & Reliability Gating

**Script:** `predict_cli.py`

Every prediction passes through a sequential gate pipeline. The first gate to fail determines the output state.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE (predict_cli.py)                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: image path
         │
         ▼
┌─────────────────────┐
│  STAGE 1            │
│  Preflight Checks   │  ── Laplacian variance  < 28.0  →  UNRELIABLE
│  (image quality)    │  ── Mean brightness not in [30, 220]  →  UNRELIABLE
│                     │  ── Object coverage < 0.40  →  warning (not rejected)
└────────┬────────────┘
         │  OK
         ▼
┌─────────────────────┐
│  STAGE 2            │
│  Feature Extraction │  EfficientNetB0 + handcrafted → [1312]
│  + Preprocessing    │  VarianceThreshold → StandardScaler → union_set slice
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 3            │
│  Vegetable SVM      │  predict_proba() → top-1 label + confidence + gap
│                     │  veg_confident = (conf ≥ 70%) AND (gap ≥ 15%)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 4            │
│  Centroid           │  centroid_ratio = d_pred / d_second
│  Consistency Check  │  class_inconsistent = ratio > per-class P95 threshold
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 5            │
│  Freshness SVM      │  decision_function() → raw margin
│                     │  normalize → 0-100 score (per-veg bounds if confident
│                     │             and class-consistent, else global bounds)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 6            │
│  OOD Gate           │  Mahalanobis distance > thresh_ood  →  score_unreliable=True
│  (Mahalanobis)      │  zone: trusted / caution / ood
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐   use_augmentation_gate=True only
│  STAGE 7 (optional) │
│  Augmentation       │  6 augmented views: ±15% brightness, flip,
│  Instability Gate   │  blur, ±5° rotation
│                     │  score_range = max(scores) - min(scores)
│                     │  crosses_boundary = min(raw) < 0 AND max(raw) > 0
│                     │  unstable = (score_range ≥ T_instability) AND crosses_boundary
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STAGE 8            │
│  Boundary Gate      │  near_boundary = abs(raw) < T_boundary
│                     │  decision_unreliable |= near_boundary
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RELIABILITY DECISION                          │
│                                                                  │
│  score_unreliable    = unstable OR is_ood                        │
│  decision_unreliable = near_boundary OR low_veg_conf OR          │
│                        class_inconsistent OR conf_gap < 10%      │
│                                                                  │
│  HIGH-CONFIDENCE OVERRIDE: if veg_conf > 95% AND not near        │
│  boundary AND not crosses_boundary AND not OOD AND not           │
│  class_inconsistent → force RELIABLE regardless of above         │
│                                                                  │
│  score_unreliable  = True  →  UNRELIABLE  (no score, no label)  │
│  decision_unreliable = True →  TENTATIVE  (score shown,         │
│                                 no fresh_label)                  │
│  neither           = True  →  RELIABLE   (score + label +       │
│                                 confidence band)                 │
└──────────────────────────────────────────────────────────────────┘
```

### Output States

| State | Score | Fresh Label | Freshness Band | When |
|-------|-------|-------------|----------------|------|
| `RELIABLE` | ✅ | ✅ | ✅ | All gates passed; prediction trustworthy |
| `TENTATIVE` | ✅ | ❌ | ❌ | Score valid but decision uncertain (near boundary, ambiguous veg, centroid mismatch) |
| `UNRELIABLE` | ❌ | ❌ | ❌ | Image quality failed, OOD detected, or severe augmentation instability |

### Freshness Confidence Bands

The 0–100 normalized score is mapped to a qualitative band for `RELIABLE` outputs:

| Band | Score Range | Interpretation |
|------|-------------|----------------|
| **High** | ≥ 85 | Strongly in the fresh region of feature space |
| **Medium** | 65–84 | Moderately fresh signal |
| **Low** | 40–64 | Weakly fresh; marginal |
| **Very Low** | < 40 | Strongly rotten signal |

> **Important:** The score is a calibrated proxy derived from the SVM decision margin. It reflects **model certainty** in the fresh/rotten classification relative to the training distribution — not an absolute freshness measurement. The `fresh_label` field (`Fresh` / `Rotten`) is the primary actionable decision.

---

## Evaluation

**Script:** `evaluate_models.py`

The test set is loaded **for the first time** here. The evaluation produces a comprehensive report covering:

```
┌────────────────────────────────────────────────────────────────┐
│                  EVALUATION REPORT SECTIONS                    │
│                                                                │
│  1. Classification Metrics                                     │
│     • Vegetable accuracy + per-class precision/recall/F1       │
│     • Freshness accuracy + confusion matrix                    │
│     • Freshness ROC-AUC (SVM margin as ranking signal)         │
│                                                                │
│  2. Inversion Rate Diagnostics                                 │
│     • Raw margin inversion   (model signal quality)            │
│     • Global-norm inversion  (normalization layer)             │
│     • Deployed-norm inversion (mirrors CLI gate exactly)       │
│     • Per-vegetable gate delta (catches localized failures)    │
│       → STABLE if global delta < 0.01 AND max per-veg < 0.02  │
│                                                                │
│  3. Gate Ablation Study                                        │
│     For each gate: G1 OOD, G2 near-boundary, G3 low-veg-conf  │
│     Reports: fires%, catches_wrong, blocks_correct             │
│     Δ_acc = accuracy change when gate disabled                 │
│     Δ_cov = coverage change when gate disabled                 │
│     Verdict: KEEP / REMOVE / REVIEW / NEVER FIRES              │
│                                                                │
│  4. State Distribution (RELIABLE / TENTATIVE / UNRELIABLE)     │
│     • Global distribution on test set                          │
│     • Per-vegetable breakdown (catches per-class collapse)     │
│                                                                │
│  5. RELIABLE-subset Accuracy                                   │
│     • Overall freshness accuracy                               │
│     • RELIABLE-only freshness accuracy                         │
│     • Expected: RELIABLE acc ≥ overall acc                     │
│     • Per-vegetable RELIABLE accuracy vs baseline              │
│                                                                │
│  6. OOD Rate Consistency  (val vs test — should be < 5% diff)  │
│                                                                │
│  7. Score Distribution Summary                                 │
│     • Fresh mean/std/range vs rotten                           │
│     • Delta (fresh mean − rotten mean)                         │
│     • Overlap fraction (rotten > fresh mean)                   │
│                                                                │
│  8. Silent Failure Analysis                                    │
│     • Veg wrong but RELIABLE (blind-spot count)                │
│     • Catastrophic = veg wrong AND fresh wrong                 │
│     • Centroid vs OOD coverage of veg misclassifications       │
└────────────────────────────────────────────────────────────────┘
```

### Inversion Rate Explained

The inversion rate measures ordering reliability: what fraction of (fresh, rotten) pairs have `score_fresh < score_rotten`? A perfect system has inversion rate = 0.

```
  Pair sampling: 10,000 random (i_fresh, j_rotten) pairs
  Inversion = (score_fresh_i < score_rotten_j).mean()

  Three layers:
    Raw margin   → tests the model signal before normalization
    Global norm  → tests the linear rescaling step
    Deployed     → tests the full CLI path (per-veg bounds, gate logic)

  Layer deltas tell you exactly where ordering degrades.
```

---

## Directory Structure

```
.
├── extract_features.py          # EfficientNetB0 + handcrafted feature engine (1312-d)
├── extract_dataset_features.py  # Batch extraction over full dataset → Features/
├── train_split.py               # Stratified 70/10/20 split → models/
├── preprocess_and_rank.py       # Variance filter, scaler, XGBoost ranking, union set
├── train_svm.py                 # Dual RBF SVM training, calibration, gate fitting
├── threshold_selection.py       # Formal T_boundary / T_instability selection
├── predict_cli.py               # Full inference with 8-stage reliability gating
├── evaluate_models.py           # Held-out test set evaluation + ablation
├── visualize_results.py         # Confusion matrices + feature importance plots
├── utils.py                     # I/O helpers, confidence_band(), TARGET_VEGETABLES
│
├── Features/                    # Raw extracted features (generated)
│   ├── X.npy                    # Feature matrix [N × 1312]
│   ├── y_veg.npy                # Vegetable labels
│   ├── y_fresh.npy              # Freshness labels (0=rotten, 1=fresh)
│   └── image_paths.npy          # Aligned image paths
│
├── models/                      # All trained artifacts (generated)
│   ├── X_train.npy              # Preprocessed train split
│   ├── X_val.npy                # Preprocessed val split
│   ├── X_test.npy               # Preprocessed test split (locked)
│   ├── variance.joblib          # VarianceThreshold (fit on train)
│   ├── scaler.joblib            # StandardScaler (fit on train)
│   ├── selected_union_features.npy  # Final feature indices
│   ├── feature_importances_fresh.npy
│   ├── feature_importances_veg.npy
│   ├── feature_selection_report.json
│   ├── veg_svm_base.joblib      # Raw tuned SVC (before calibration)
│   ├── veg_svm.joblib           # CalibratedClassifierCV (final)
│   ├── fresh_svm.joblib         # Freshness SVC
│   ├── label_encoder.joblib     # LabelEncoder for vegetable classes
│   ├── train_mean.npy           # Mahalanobis centroid
│   ├── train_precision.npy      # LedoitWolf precision matrix
│   ├── class_centroids.npy      # Per-class L2 centroids
│   ├── val_image_paths.npy      # Val image paths (for augmentation stats)
│   └── scoring_config.json      # All gate thresholds + calibration metadata
│
└── vegetable_Dataset/           # Source images
    ├── freshapple/
    ├── rottenapple/
    ├── freshbanana/
    ├── rottenbanana/
    └── ...                      # (one folder per vegetable × freshness)
```

---

## Pipeline Execution

Run the following scripts in order. Each script depends on outputs from the previous one.

### Step 1 — Extract Features

```bash
python extract_dataset_features.py
```

Scans `vegetable_Dataset/`, extracts 1312-d feature vectors for all images, saves to `Features/`.

### Step 2 — Split Dataset

```bash
python train_split.py
```

Creates stratified 70/10/20 train/val/test splits. Saves feature matrices and labels to `models/`.

### Step 3 — Select Features

```bash
python preprocess_and_rank.py
```

Fits VarianceThreshold + StandardScaler on training data, ranks features per task via XGBoost (5 seeds), runs two-phase k-selection sweep, saves union feature set and preprocessing artifacts.

### Step 4 — Train & Calibrate

```bash
python train_svm.py
```

Tunes and fits both SVMs via GridSearchCV, calibrates vegetable probabilities (isotonic), fits Mahalanobis OOD detector, computes per-vegetable normalization bounds, runs formal threshold selection, saves `scoring_config.json`.

### Step 5 — Evaluate

```bash
python evaluate_models.py
```

Loads the held-out test set for the first time. Produces the full evaluation report: classification metrics, inversion diagnostics, gate ablation, state distribution, silent failure analysis.

### Step 6 — Predict

```bash
python predict_cli.py --image path/to/vegetable.jpg
```

Runs the full 8-stage inference pipeline on a single image.

```bash
# Skip augmentation stability check (faster, no 6× EfficientNet passes)
python predict_cli.py --image path/to/vegetable.jpg --no-uncertainty
```

### Step 7 — Visualize (optional)

```bash
python visualize_results.py
```

Renders confusion matrices for both tasks and top-20 XGBoost feature importance plots.

---

## Output Reference

### `predict_cli.py` Return Value

```python
result = predict("image.jpg")
```

**RELIABLE output:**
```json
{
  "state": "RELIABLE",
  "veg": "banana",
  "veg_conf": 98.4,
  "score": 77.2,
  "score_range": 3.1,
  "raw": 1.843,
  "fresh_label": "Fresh",
  "freshness_confidence_band": "Medium",
  "norm_source": "per-veg",
  "mahal_dist": 14.2,
  "mahal_zone": "trusted",
  "warnings": []
}
```

**TENTATIVE output** (score present; no fresh_label):
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
    "CAUTION — Mahalanobis dist=18.7 in caution zone."
  ]
}
```

**UNRELIABLE output** (image quality failure):
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

### `scoring_config.json` Key Fields

| Key | Description |
|-----|-------------|
| `boundary_threshold` | T_boundary — abs(raw) must exceed this for RELIABLE |
| `unstable_range_thresh` | T_instability — augmentation score range threshold |
| `mahal_thresh_ood` | P99 Mahalanobis distance on training set |
| `mahal_thresh_caution` | P90 Mahalanobis distance on training set |
| `per_veg_bounds` | Per-vegetable p5/p95 normalization bounds |
| `global_bounds` | Fallback normalization bounds (full val set) |
| `veg_confidence_threshold` | 0.70 — minimum top-1 probability for veg gate |
| `veg_gap_threshold` | 0.15 — minimum top-1 minus top-2 probability gap |
| `centroid_ratio_thresholds` | Per-class P95 centroid ratio for consistency gate |
| `threshold_selection_result` | Formal selection outcome: feasible, risk, coverage |
| `infeasibility_diagnosis` | Case classification + risk/coverage curve (if infeasible) |
| `svm_best_params` | Best GridSearchCV params for each SVM |
| `calibration_note` | Human-readable provenance string for all design decisions |

---

## Design Principles & Known Limitations

### Design Principles

**Dual task separation.** XGBoost rankings are computed independently for freshness and vegetable tasks. Using a single freshness-derived feature set for the vegetable classifier would be scientifically invalid — vegetable-discriminative features may not be freshness-discriminative and vice versa.

**No calibration leakage.** The val set is split into disjoint halves before any calibration step. Isotonic regression sees `cal_val`; threshold selection sees `thr_val`. If both used the same data, thresholds would overfit to the calibration distribution.

**Hard errors over silent fallbacks.** Missing per-vegetable bounds raise a `RuntimeError` at training time rather than silently falling back to global bounds. A silent fallback would mask thin-class failures and produce miscalibrated scores for underrepresented vegetables.

**Evaluation mirrors deployment.** `evaluate_models.py` uses the same gating logic as `predict_cli.py`, including using *predicted* (not ground-truth) vegetable labels to select normalization bounds. Using ground truth here would silently inflate evaluation scores.

**Formal threshold selection over heuristics.** T_boundary and T_instability are found by constrained optimisation (Risk ≤ ε, coverage maximised) on held-out data — not by arbitrary percentile rules or manual tuning.

### Known Limitations

- **Score is not absolute freshness.** The 0–100 score is a per-vegetable calibrated SVM margin proxy. A score of 80 for banana and 80 for potato both mean "high relative to that vegetable's training distribution" — they are not on a common scale. If cross-vegetable comparability is required, use `global_bounds` throughout.

- **Augmentation gate requires 6× inference.** The instability gate runs EfficientNetB0 six times per image. When `use_augmentation_gate=False` (the current default), `T_instability` from formal threshold selection is still stored in `scoring_config.json` but the gate is not applied at runtime.

- **OOD detection in high-dimensional space.** The Mahalanobis detector uses a LedoitWolf-shrunk precision matrix. In very high-dimensional feature spaces, the detector may become over-aggressive. The OOD rate on val vs test should be monitored (a > 5% discrepancy is flagged by `evaluate_models.py`).

- **No intra-class freshness ground truth.** The system proves class-level separation (fresh vs rotten) and ordering reliability between classes. Intra-class ordering (e.g., "this banana is 3 days older than that one") is not validated because no continuous decay-day annotations exist in the training data.

- **Imaging condition sensitivity.** All calibration assumes consistent imaging conditions similar to the training set (lighting, distance, background). Significant departures will increase OOD flags and should be handled by retraining or domain adaptation.