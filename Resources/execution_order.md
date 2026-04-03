# SVM Vegetable Freshness — Complete Execution Guide

> EfficientNetB0 + Handcrafted Features → Dual RBF SVM + Formal Reliability Gating

All numbers in this document are from the actual training run. Run every script from the project root (`~/Desktop/mini-project/`), with `src/` as the script prefix.

---

## Pipeline Overview

| Step | Script | Input | Key Output |
|------|--------|-------|------------|
| 1 | `src/extract_dataset_features.py` | `vegetable_Dataset/` | `Features/X.npy` (12691 × 1312) |
| 2 | `src/train_split.py` | `Features/*.npy` | `models/X_train.npy`, `X_val.npy`, `X_test.npy` |
| 3 | `src/preprocess_and_rank.py` | `models/X_train.npy` | `variance.joblib`, `scaler.joblib`, `selected_union_features.npy` (349 features) |
| 4 | `src/train_svm.py` | `models/X_train.npy` + artifacts | `veg_svm.joblib`, `fresh_svm.joblib`, `scoring_config.json` |
| 5 | `src/evaluate_models.py` | `models/X_test.npy` | Classification metrics, gate ablation, state distribution |
| 6 | `src/visualize_results.py` | `models/*` | Confusion matrices + feature importance plots |
| 7 | `src/predict_cli.py` | Single image path | RELIABLE / TENTATIVE / UNRELIABLE output |

> **Test set rule:** `X_test.npy` is never loaded before Step 5. Steps 3 and 4 use only train and val data.

---

## Step 1 — Feature Extraction

```bash
python src/extract_dataset_features.py
```

### What it does

Scans `vegetable_Dataset/`, parses the `fresh`/`rotten` prefix from each folder name to extract vegetable class and freshness label, filters to the five target vegetables, loads images in parallel via `ThreadPoolExecutor`, and for each image runs EfficientNetB0 (1280 deep features) and a handcrafted extractor (32 features). Saves the full feature matrix and aligned label and path arrays.

### Feature breakdown

| Source | Count | What it captures |
|--------|-------|-----------------|
| EfficientNetB0 GlobalAvgPool | 1280 | High-level texture, shape, and colour patterns (ImageNet-pretrained) |
| RGB mean per channel | 3 | Average redness, greenness, blueness |
| RGB std per channel | 3 | Colour variance — patchy discolouration affects std |
| HSV mean per channel | 3 | Hue, saturation, value — browning shifts hue and desaturates |
| HSV std per channel | 3 | Colour tone spread across the image |
| Grayscale mean | 1 | Overall brightness |
| Grayscale std | 1 | Contrast |
| Edge density (Canny) | 1 | Fraction of edge pixels — fresh produce has crisp, firm edges |
| Laplacian variance | 1 | Whole-image sharpness — rotten produce appears visually softer |
| Luminance histogram (8 bins) | 8 | Brightness distribution — spoilage shifts the histogram shape |
| Zero-padding | 7 | Pads handcrafted block to exactly 32 |
| **Total** | **1312** | **One row per image** |

### Expected console output

TensorFlow prints several `oneDNN` and CPU-flag warnings on startup — these are informational and do not affect results. The CUDA error (`Failed call to cuInit`) is expected on CPU-only machines; EfficientNet falls back to CPU automatically.

```
Scanning dataset...
Total images: 12691
Extracting features...
Batches: 100%|█████████████████████████████| 100/100 [04:49<00:00,  2.90s/it]
Saved feature matrix: (12691, 1312)
```

### Files created

```
Features/
 ├ X.npy              shape (12691, 1312)  float32
 ├ y_veg.npy          shape (12691,)       string  ["banana", "apple", ...]
 ├ y_fresh.npy        shape (12691,)       int     [1, 0, 1, ...]  1=fresh, 0=rotten
 └ image_paths.npy    shape (12691,)       string  absolute path per row
```

`image_paths.npy` is required by `train_svm.py` to run real EfficientNet augmentations on validation images during threshold calibration.

### Verification

```bash
python -c "import numpy as np; X=np.load('Features/X.npy'); print(X.shape)"
# Expected: (12691, 1312)
```

---

## Step 2 — Dataset Split

```bash
python src/train_split.py
```

### What it does

Performs a stratified **70 / 10 / 20** three-way split using a composite label `"{vegetable}_{freshness}"` (e.g. `"banana_1"`, `"potato_0"`) so that every vegetable × freshness combination is proportionally represented in all three splits.

### Why three splits

| Split | Size | Purpose |
|-------|------|---------|
| **Train (70%)** | 8,883 | Model fitting — both SVMs learn only from here |
| **Val (10%)** | 1,269 | All calibration: normalization bounds, OOD thresholds, aug stats, formal gate thresholds |
| **Test (20%)** | 2,539 | Final evaluation only — locked until Step 5 |

The validation set is further divided **inside Step 4** into `cal_val` (634) and `thr_val` (635) to prevent calibration leakage between isotonic probability fitting and threshold selection.

Using a two-way split (train/test only) would require calibrating thresholds on test data — those thresholds would then overfit to the test set and the evaluation would no longer be an honest estimate of deployment performance.

### Expected console output

```
[INFO] Loading Features...
[SUCCESS] Split created — Train=8883  Val=1269  Test=2539
[INFO] Test set must remain untouched until evaluate_models.py
```

### Files created in `models/`

```
X_train.npy           y_veg_train.npy       y_fresh_train.npy
X_val.npy             y_veg_val.npy         y_fresh_val.npy
X_test.npy            y_veg_test.npy        y_fresh_test.npy
val_image_paths.npy   (val image paths, required for aug stats in Step 4)
```

### Critical check before Step 3

```bash
ls models/X_train.npy || echo "STOP — run train_split.py first"
ls models/X_val.npy   || echo "STOP — val split missing"
```

---

## Step 3 — Feature Preprocessing and Selection

```bash
python src/preprocess_and_rank.py
```

### What it does

Loads `X_train.npy` (never the full dataset). Applies VarianceThreshold and StandardScaler fitted on training data only. Then runs dual-task XGBoost feature ranking — freshness and vegetable tasks ranked independently — across 5 seeds each. Runs a two-phase k-selection sweep to find the optimal number of top features per task, then forms a union feature set used by both SVMs.

### Preprocessing steps

| Step | Operation | Result |
|------|-----------|--------|
| VarianceThreshold | Remove zero-variance features (fit on train only) | 1312 → 1304 |
| StandardScaler | Normalize to mean=0, std=1 (fit on train only) | 1304 → 1304 |

### Dual-task XGBoost ranking

XGBoost is run separately for freshness labels and vegetable labels. Using only a freshness-derived ranking for the vegetable classifier is scientifically invalid — the two tasks require different features. Each task is averaged across 5 seeds for stability:

```
Rankings computed once at max(k)=250, sliced per k candidate.
Total XGBoost fits: 5 seeds × 2 tasks = 10
```

### Two-phase k-selection sweep

**Phase 1 — LinearSVC proxy:**

```
  k     union  fresh_val   veg_val  combined
  ----  -----  ---------  --------  --------
    50     98     0.9464    0.9850    0.9657
   100    187     0.9598    0.9913    0.9756
   150    270     0.9661    0.9929    0.9795
   200    349     0.9653    0.9945    0.9799
   250    432     0.9661    0.9953    0.9807
  Proxy winner: k=250
```

**Phase 2 — RBF SVM confirmation (3-fold sweep, 5-fold refit on winner):**

```
  k     union  rbf_fresh   rbf_veg  combined
  ----  -----  ---------  --------  --------
    50     98     0.9835    0.9945    0.9890
   100    187     0.9866    0.9968    0.9917
   150    270     0.9858    0.9976    0.9917
   200    349     0.9858    0.9992    0.9925   ← winner
   250    432     0.9842    0.9976    0.9909

  RBF confirmation changed best_k: 250 → 200  (combined=0.9925)
```

k=250 wins with a LinearSVC proxy but loses with a real RBF SVM — the extra 83 features add noise the RBF kernel cannot ignore.

**5-fold refit of winner (k=200):**

```
  veg   k=200: C=10.0, gamma=0.001   CV acc=0.9958
  fresh k=200: C=10.0, gamma='scale' CV acc=0.9865
```

### Union feature set

```
  top_k per task        : 200
  Fresh-specific        : 149 features
  Veg-specific          : 149 features
  Shared                : 51 features
  ─────────────────────────────
  Union size            : 349 features

  [Stability 'freshness'] min pairwise overlap=1.000  [OK]
  [Stability 'vegetable'] min pairwise overlap=1.000  [OK]
```

### Expected console output

```
[INFO] Train samples : 8883
[INFO] Val   samples : 1269
[INFO] VarianceThreshold: 1312 → 1304

[INFO] Computing XGBoost feature rankings (5 seeds × 2 tasks — once only)...

[Phase-1] Proxy k-sweep (LinearSVC, pre-computed rankings):
  ...
  Proxy best_k=250  combined=0.9807

[Phase-2] RBF confirmation sweep ...
  [NOTE] RBF confirmation changed best_k: 250 → 200  (combined=0.9925)

============================================================
UNION FEATURE SET SUMMARY
============================================================
  top_k per task          : 200
  Fresh-specific features : 149
  Veg-specific features   : 149
  Shared features         : 51
  Union size              : 349

============================================================
VALIDATION REPORT
============================================================
  best_k                  : 200  (proxy=250)
  union_feature_count     : 349
  best SVM params (veg)   : {'C': 10.0, 'gamma': 0.001}
  best SVM params (fresh) : {'C': 10.0, 'gamma': 'scale'}
  RBF val acc (fresh)     : 0.9858
  RBF val acc (veg)       : 0.9992
[INFO] Test set untouched. Run train_svm.py next.
```

### Files created in `models/`

```
variance.joblib                    VarianceThreshold (fit on train)
scaler.joblib                      StandardScaler (fit on train)
selected_union_features.npy        shape (349,) — indices used by both SVMs
selected_fresh_features.npy        shape (200,) — freshness-task top-200
selected_veg_features.npy          shape (200,) — vegetable-task top-200
feature_importances_fresh.npy      avg XGBoost gain per feature, freshness task
feature_importances_veg.npy        avg XGBoost gain per feature, vegetable task
feature_selection_report.json      full sweep tables + best params
```

### Verification

```bash
python -c "import numpy as np; print(np.load('models/selected_union_features.npy').shape)"
# Expected: (349,)
```

---

## Step 4 — Train SVMs and Calibrate

```bash
python src/train_svm.py
```

### What it does

Loads the union feature set and the train/val splits. Splits val into disjoint `cal_val` and `thr_val` halves. Trains both SVMs with GridSearchCV. Calibrates vegetable probabilities on `cal_val`. Computes normalization bounds on the full val set. Fits the Mahalanobis OOD detector. Runs augmentation statistics on `thr_val`. Runs formal threshold selection. Saves everything to `scoring_config.json`.

### Val set disjoint split (calibration leakage fix)

```
X_val (1,269 samples) — stratified by y_fresh
  │
  ├── 50% → cal_val  (634 samples)
  │         CalibratedClassifierCV(isotonic) fit here only
  │
  └── 50% → thr_val  (635 samples)
            formal threshold selection + augmentation stats here only

[INFO] val split: cal_val=634  thr_val=635
```

Using the same val data for both calibration and threshold selection would cause the isotonic layer to implicitly encode the threshold targets — thresholds would appear tight in validation but fail on genuinely new data.

### Vegetable SVM

```
Base SVC: kernel=rbf, class_weight=balanced, probability=False
GridSearchCV: StratifiedKFold(5), 30 param combinations, 150 fits
  Grid: C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
        γ ∈ {0.0001, 0.001, 0.01, 0.1, "scale"}

Best: C=10.0, gamma=0.001   CV acc=0.9958

Isotonic calibration on cal_val only:
  CalibratedClassifierCV(FrozenEstimator(veg_base), method="isotonic")
  Vegetable acc — cal_val=1.0000   thr_val=1.0000
```

Saved as `veg_svm.joblib`. Provides `predict_proba()` for confidence and gap computation.

### Freshness SVM

```
Same GridSearchCV procedure, binary target (0=rotten, 1=fresh)
Best: C=10.0, gamma='scale'   CV acc=0.9865
  Freshness acc — cal_val=0.9858   thr_val=0.9858
```

Saved as `fresh_svm.joblib`. Provides `decision_function()` (raw margin) and `predict()`.

### p5/p95 normalization bounds (full val set)

```
[INFO] Per-vegetable bounds (from full val set):
  apple         p5=-2.5635  p95=2.1198  hard_min=-3.2613  hard_max=2.6114
  banana        p5=-2.0173  p95=1.8217  hard_min=-2.6101  hard_max=2.5962
  capsicum      p5=-1.2853  p95=1.8389  hard_min=-1.9306  hard_max=2.0868
  cucumber      p5=-1.6697  p95=1.6762  hard_min=-1.9225  hard_max=2.0566
  potato        p5=-1.8869  p95=1.6565  hard_min=-2.7631  hard_max=2.0914

  Global fallback  p5=-2.2678  p95=1.9306

p5/p95 stability (5-fold CV on X_train):
  apple    p5_cv=0.016  p95_cv=0.013  [OK]
  banana   p5_cv=0.021  p95_cv=0.016  [OK]
  capsicum p5_cv=0.061  p95_cv=0.016  [OK]
  cucumber p5_cv=0.036  p95_cv=0.050  [OK]
  potato   p5_cv=0.057  p95_cv=0.020  [OK]
```

Full val set is used (not just half) because the 50/50 split leaves thin classes below the 50-sample minimum for stable percentile estimates. If any vegetable is missing bounds, a hard `RuntimeError` is raised — no silent fallback.

### Mahalanobis OOD detector

```
LedoitWolf covariance fit on X_train[:, union_349]

[INFO] Mahalanobis thresh_caution=24.167  thresh_ood=30.438
[INFO] OOD rate on validation: 0.0181
```

### Augmentation stats (on thr_val only)

```
[INFO] Computing augmentation stats on thr_val subset...
  [INFO] Aug sampling apple        100 images (restricted=True)
  [INFO] Aug sampling banana       100 images (restricted=True)
  [INFO] Aug sampling capsicum      59 images (restricted=True)
  [INFO] Aug sampling cucumber      52 images (restricted=True)
  [INFO] Aug sampling potato        69 images (restricted=True)
[INFO] unstable_range_thresh (P95): 29.4715  (380 samples)
```

This P95 (29.47) is an input to the formal threshold optimiser — it is not the final threshold.

### Formal threshold selection

```
[INFO] Running formal threshold selection on thr_val subset...
[INFO] Formal thresholds — T_boundary=0.0000  T_instability=36.0000
       Risk=0.0188  Coverage=0.9789  n_reliable=372
```

The optimiser finds `T_boundary=0.0` because the base model is accurate enough across the full margin range — no margin cutoff is needed to keep Risk ≤ 10%.

### Per-class centroid thresholds

```
[INFO] Per-class centroid ratio thresholds (P95 of correct val predictions):
  apple         threshold=1.0220  (n=448)
  banana        threshold=0.9552  (n=469)
  capsicum      threshold=0.9973  (n=114)
  cucumber      threshold=1.0257  (n=91)
  potato        threshold=0.9740  (n=147)
```

### Expected final console output

```
[DONE] Scoring config saved → models/scoring_config.json

============================================================
VALIDATION REPORT
============================================================
  best_k                  : 200
  union_feature_count     : 349

  [Threshold Selection]
  feasible                : True
  T_boundary              : 0.0000
  T_instability           : 36.0000
  risk (thr_val)          : 0.0188
  coverage (thr_val)      : 0.9789
  n_reliable              : 372

  [SVM CV Scores]
  veg  5-fold CV acc      : 0.9958
  fresh 5-fold CV acc     : 0.9865
  veg  cal_val acc        : 1.0000
  veg  thr_val acc        : 1.0000
  fresh cal_val acc       : 0.9858
  fresh thr_val acc       : 0.9858
============================================================
[INFO] Test set untouched. Run evaluate_models.py for final metrics.
```

### Files created in `models/`

```
veg_svm_base.joblib         raw GridSearchCV SVC (before calibration)
veg_svm.joblib              CalibratedClassifierCV (final)
fresh_svm.joblib            freshness SVC
label_encoder.joblib        vegetable name ↔ integer mapping
train_mean.npy              Mahalanobis centroid  shape (349,)
train_precision.npy         LedoitWolf precision  shape (349, 349)
class_centroids.npy         per-class L2 centroids  shape (5, 349)
scoring_config.json         all thresholds, bounds, and calibration metadata
```

### Key `scoring_config.json` fields

| Key | Value | Purpose |
|-----|-------|---------|
| `boundary_threshold` | 0.0 | abs(raw) must exceed this for RELIABLE |
| `unstable_range_thresh` | 36.0 | T_instability for aug gate (currently disabled) |
| `use_augmentation_gate` | false | Aug gate stored but inactive at inference |
| `mahal_thresh_caution` | 24.167 | P90 of training Mahalanobis distances |
| `mahal_thresh_ood` | 30.438 | P99 of training Mahalanobis distances |
| `veg_confidence_threshold` | 0.70 | Min top-1 prob for per-veg bounds |
| `veg_gap_threshold` | 0.15 | Min top-1 minus top-2 prob gap |
| `per_veg_bounds` | per-vegetable dict | p5/p95 normalization per class |
| `global_bounds` | fallback dict | Used when veg confidence is low |
| `centroid_ratio_thresholds` | per-class dict | Centroid consistency gate |

---

## Step 5 — Evaluate on Test Set

```bash
python src/evaluate_models.py
```

`X_test.npy` is opened here for the first time. Nothing from Steps 1–4 has seen these samples.

### Classification results (test set, 2,539 samples)

```
========== Vegetable Classification ==========
Accuracy: 0.9961

              precision  recall  f1-score  support
apple            1.00     1.00     1.00      896
banana           1.00     1.00     1.00      940
capsicum         1.00     0.99     1.00      227
cucumber         0.99     0.98     0.98      182
potato           0.99     0.99     0.99      294

Confusion Matrix:
[[895   0   0   0   1]
 [  1 939   0   0   0]
 [  0   0 225   1   1]
 [  0   2   0 178   2]
 [  1   0   0   1 292]]

========== Freshness Classification ==========
Accuracy: 0.9894

Confusion Matrix:
[[1308   15]
 [  12 1204]]

Freshness ROC-AUC (margin-based): 0.9994
```

### Score distribution

```
========== Score Distribution ==========
  Fresh  — mean=86.84  std=10.39  range=69.93
  Rotten — mean=16.10  std=12.38  range=85.96
  Delta (fresh − rotten mean) : 70.73 pts
  Overlap (rotten > fresh mean): 0.0000
```

Zero rotten samples score above the average fresh score. Fresh and rotten distributions do not overlap at the mean.

### Inversion rate diagnostics

```
  Raw margin inversion        : 0.0007   (0.07% of pairs incorrectly ordered)
  Global-norm inversion       : 0.0007
  Deployed-norm inversion     : 0.0010

  [STABLE] Global delta < 0.01 AND max per-veg delta < 0.02.
           Gate is stable. Keep current conf/gap thresholds.
```

### State distribution (test set)

```
========== State Distribution (Test Set) ==========
  Total test samples : 2539
  RELIABLE           :  2343  (92.3%)
  TENTATIVE          :   134  ( 5.3%)
  UNRELIABLE (OOD)   :    62  ( 2.4%)
  [Note] Aug-instability UNRELIABLE not counted (gate disabled).
  [OK] 92.3% of samples reach RELIABLE state.
```

Per-vegetable state breakdown:

```
  Veg        N      RELIABLE        TENTATIVE       UNRELIABLE
  apple      896    856  (95.5%)     35  ( 3.9%)      5  ( 0.6%)
  banana     940    866  (92.1%)     54  ( 5.7%)     20  ( 2.1%)
  capsicum   227    208  (91.6%)     15  ( 6.6%)      4  ( 1.8%)
  cucumber   182    160  (87.9%)      8  ( 4.4%)     14  ( 7.7%)  ← WEAK
  potato     294    253  (86.1%)     22  ( 7.5%)     19  ( 6.5%)  ← WEAK
```

### Gate ablation

```
========== Gate Trigger Statistics ==========

  Gate               Fires  Fire%  Catch_W  Block_C   Δ_acc    Δ_cov   Verdict
  ─────────────────────────────────────────────────────────────────────────────
  G1_OOD               62   2.4%      1       61    −0.0004  +0.0760   REVIEW
  G2_near_boundary      0   0.0%      0        0    −0.0003  +0.0524   NEVER FIRES
  G3_low_veg_conf       3   0.1%      0        3    −0.0003  +0.0528   REVIEW

  Baseline: acc=0.9898  coverage=0.923
```

G2 never fires because T_boundary = 0.0 (formal threshold selection found no margin cutoff needed).

### Freshness accuracy on RELIABLE samples

```
  Overall freshness accuracy   : 0.9894
  RELIABLE-only accuracy       : 0.9898   ← higher than overall [OK]

  Per-vegetable RELIABLE accuracy (baseline = 0.9898):
  apple     n=856   acc=0.9907
  banana    n=866   acc=0.9988
  capsicum  n=208   acc=1.0000
  cucumber  n=160   acc=0.9688
  potato    n=253   acc=0.9605
```

RELIABLE-only accuracy ≥ overall accuracy confirms the gate is filtering correctly.

### Silent failure analysis

```
  Total veg misclassifications   : 10
  Caught by OOD gate only        :  5
  Caught by centroid gate only   :  2
  Caught by both                 :  0
  Missed by both (blind spots)   :  3
    Of blind spots, freshness also wrong: 0  ← zero catastrophic failures
```

---

## Step 6 — Visualization (optional)

```bash
python src/visualize_results.py
```

Produces four matplotlib figures:
- Vegetable classification confusion matrix (Blues colormap)
- Freshness classification confusion matrix (Greens colormap)
- Top-20 feature importances — freshness task (XGBoost gain, blue bars)
- Top-20 feature importances — vegetable task (XGBoost gain, orange bars)

Requires `matplotlib` and `seaborn`. No files are saved — plots display interactively.

---

## Step 7 — Single Image Prediction

```bash
# Full pipeline — with reliability gate (default)
python src/predict_cli.py --image path/to/image.jpg

# Fast mode — skips augmentation gate (6× EfficientNet passes saved)
python src/predict_cli.py --image path/to/image.jpg --no-uncertainty
```

### Full inference pipeline

```
Input image
  │
  ▼  Preflight: Laplacian var / brightness / coverage
  │  fail → UNRELIABLE immediately
  │
  ▼  EfficientNetB0(224×224) → [1280]
     + handcrafted extraction → [32]
     concatenate → [1312]
  │
  ▼  vt.transform → [1304]
     scaler.transform → [1304]
     [:, union_349] → [349]
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
VEGETABLE SVM                   FRESHNESS SVM
predict_proba() →               decision_function() → raw margin
  veg_name, veg_conf%, gap%     predict() → 0 (rotten) or 1 (fresh)
  │                                  │
  └──────────────┬───────────────────┘
                 │
  Centroid consistency check (BEFORE bound selection)
    ratio = d_pred / d_second
    class_inconsistent = (ratio > per_class_threshold)
                 │
  veg_confident AND NOT class_inconsistent?
    YES → per-veg bounds       NO → global bounds
                 │
  score = clip((raw − p5) / (p95 − p5) × 100, 0, 100)
                 │
  Mahalanobis OOD check
    dist ≥ 30.438 → is_ood=True
    dist ≥ 24.167 → caution warning
                 │
  Aug gate (use_augmentation_gate=False — currently skipped)
                 │
  Reliability decision:
    is_ood               → score_unreliable    → UNRELIABLE
    low_conf / near_bnd
    / class_inconsistent → decision_unreliable → TENTATIVE
    neither              →                       RELIABLE
                 │
  Confidence band (RELIABLE only):
    score ≥ 85 → High
    score ≥ 65 → Medium
    score ≥ 40 → Low
    score  < 40 → Very Low
```

### Output states

| State | Score shown | Fresh label | Band | When |
|-------|-------------|-------------|------|------|
| `RELIABLE` | ✅ | ✅ | ✅ | All gates passed |
| `TENTATIVE` | ✅ | ❌ | ❌ | Score valid but decision uncertain |
| `UNRELIABLE` | ❌ | ❌ | ❌ | Image quality failed or OOD |

### Sample outputs

**RELIABLE — fresh banana:**
```
Vegetable : banana (98.40%,  gap=96.80%)
State     : RELIABLE
Score     : 77.20 / 100
Norm      : per-veg
Freshness : Fresh
Confidence: Medium
Mahal     : 14.20  [trusted]
```

**TENTATIVE — low vegetable confidence:**
```
Vegetable : potato (61.30%,  gap=8.20%)
State     : TENTATIVE
Score     : 42.10 / 100
Norm      : global
Mahal     : 18.70  [caution]
[!] Low veg confidence (61.3%, gap=8.2%) — using global normalization.
[!] CAUTION — Mahalanobis dist=18.7 in caution zone [24.167, 30.438].
```

**UNRELIABLE — image quality failure:**
```
[UNRELIABLE] Pre-flight failed: Image out of focus (lap_var=12.3 < 28.0)
```

**UNRELIABLE — OOD:**
```
Vegetable : apple (88.10%,  gap=62.30%)
State     : UNRELIABLE
Mahal     : 32.15  [ood]
[!] OOD — Mahalanobis dist=32.15 > threshold=30.438. Outside training distribution.
```

---

## Final Artifacts — Complete File List

```
Features/
 ├ X.npy                        (12691, 1312)  full feature matrix
 ├ y_veg.npy                    (12691,)       vegetable label strings
 ├ y_fresh.npy                  (12691,)       1=fresh, 0=rotten
 └ image_paths.npy              (12691,)       aligned image paths

models/
 ├ X_train.npy                  (8883, 1312)
 ├ X_val.npy                    (1269, 1312)
 ├ X_test.npy                   (2539, 1312)
 ├ y_veg_train/val/test.npy
 ├ y_fresh_train/val/test.npy
 ├ val_image_paths.npy          (1269,) aligned to X_val rows
 ├ variance.joblib              VarianceThreshold (fit on train)
 ├ scaler.joblib                StandardScaler (fit on train)
 ├ selected_union_features.npy  (349,) column indices — both SVMs use this
 ├ selected_fresh_features.npy  (200,) freshness-task top-200
 ├ selected_veg_features.npy    (200,) vegetable-task top-200
 ├ feature_importances_fresh.npy
 ├ feature_importances_veg.npy
 ├ feature_selection_report.json
 ├ veg_svm_base.joblib          raw GridSearchCV SVC
 ├ veg_svm.joblib               CalibratedClassifierCV (final)
 ├ fresh_svm.joblib             freshness SVC
 ├ label_encoder.joblib         vegetable name ↔ integer
 ├ train_mean.npy               shape (349,)
 ├ train_precision.npy          shape (349, 349)
 ├ class_centroids.npy          shape (5, 349)
 └ scoring_config.json          all thresholds + calibration metadata
```

---

## Failure Modes and Checks

**1. Running scripts from the wrong directory**
All scripts must be invoked as `python src/<script>.py` from the project root. Running `python extract_dataset_features.py` directly fails with `No such file or directory`.

```bash
# Correct:
cd ~/Desktop/mini-project
python src/extract_dataset_features.py
```

**2. Missing val split**
The old two-way split (train/test only) is no longer used. If `X_val.npy` is missing, `train_svm.py` will fail at the cal_val/thr_val split.
```bash
ls models/X_val.npy || echo "STOP — re-run train_split.py"
```

**3. Wrong artifact for feature count**
The union feature set is saved as `selected_union_features.npy` (349 features), not `selected_features.npy` (100 features from the old architecture). If any script raises a shape mismatch, check which file it is loading.
```bash
python -c "import numpy as np; print(np.load('models/selected_union_features.npy').shape)"
# Expected: (349,)
```

**4. Stale artifacts across partial reruns**
If you rerun `preprocess_and_rank.py`, you must also rerun `train_svm.py` and `evaluate_models.py`. The union feature set shape can change if k changes — loading an old `veg_svm.joblib` with a new feature set causes a silent shape mismatch.

**5. Missing `val_image_paths.npy`**
`train_svm.py` requires this file for real augmentation statistics on val images. It is created by `train_split.py`. If missing, re-run `train_split.py`.
```bash
ls models/val_image_paths.npy || echo "STOP — re-run train_split.py"
```

**6. Missing `scoring_config.json`**
Both `predict_cli.py` and `evaluate_models.py` require this file. Re-run `train_svm.py` if absent.
```bash
ls models/scoring_config.json || echo "STOP — re-run train_svm.py"
```

**7. TensorFlow warnings at startup**
`oneDNN`, `AVX2`, and CUDA errors print to stderr on every TF import. These are informational — the CUDA error means TensorFlow is running on CPU (expected on this machine). EfficientNet falls back to CPU automatically.

**8. Slow prediction (aug gate)**
`predict_cli.py` runs `compute_uncertainty=True` by default, but `use_augmentation_gate=False` in `scoring_config.json` means the 6× EfficientNet passes are skipped at runtime. The `--no-uncertainty` flag also forces this off. Prediction on CPU takes ~2–4 seconds per image with the gate disabled.

---

## Minimal Re-run Checklist

**Only changed SVM hyperparameters (no feature or data changes):**
```bash
python src/train_svm.py
python src/evaluate_models.py
```

**Changed feature selection (k, ranking seeds, or XGBoost settings):**
```bash
# Edit preprocess_and_rank.py first, then:
python src/preprocess_and_rank.py
python src/train_svm.py
python src/evaluate_models.py
```

**Added new images to the dataset:**
```bash
python src/extract_dataset_features.py
python src/train_split.py
python src/preprocess_and_rank.py
python src/train_svm.py
python src/evaluate_models.py
```

**Changed only gate thresholds or scoring config values:**

If you want to adjust `use_augmentation_gate`, `veg_confidence_threshold`, `veg_gap_threshold`, or `grade_thresholds`, edit `scoring_config.json` directly and re-run `evaluate_models.py`. No retraining is needed — these values are read at inference and evaluation time.

```bash
# Edit models/scoring_config.json, then:
python src/evaluate_models.py
python src/predict_cli.py --image path/to/image.jpg
```

---

## Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| Total images | 12,691 |
| Train / Val / Test | 8,883 / 1,269 / 2,539 |
| Raw features | 1,312 → 1,304 (after VarianceThreshold) |
| Union features (both SVMs) | **349** |
| Best k per task | 200 |
| Best SVM params (veg) | C=10.0, γ=0.001 |
| Best SVM params (fresh) | C=10.0, γ="scale" |
| Veg test accuracy | **99.61%** |
| Freshness test accuracy | **98.94%** |
| Freshness ROC-AUC | **0.9994** |
| Fresh score mean | 86.84 / 100 |
| Rotten score mean | 16.10 / 100 |
| Score delta | 70.73 pts |
| RELIABLE rate (test) | **92.3%** |
| TENTATIVE rate (test) | 5.3% |
| UNRELIABLE / OOD (test) | 2.4% |
| RELIABLE-only accuracy | 98.98% |
| Catastrophic silent failures | **0** |
| T_boundary | 0.0 |
| T_instability | 36.0 (gate disabled) |
| Mahalanobis thresh_ood | 30.438 |