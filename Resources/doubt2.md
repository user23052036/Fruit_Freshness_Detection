
# Teacher’s recommended pipeline

1. **Image → Feature extraction**

   * EfficientNet deep features (e.g. 1280-d) + 32 handcrafted features → **raw feature vector**.
2. **Feature matrix**

   * Build `X` (N × D) and `y` (freshness labels).
3. **Remove constant / zero-variance features**

   * Variance threshold (e.g. `< 1e-8`).
4. **Standardization**

   * Fit `StandardScaler` on training features; save mean & std.
5. **ElasticNet feature selection**

   * Fit ElasticNet (CV): logistic/linear with L1+L2, tune `alpha` and `l1_ratio`. Drop features with coefficient = 0.
6. **XGBoost gain ranking (optional)**

   * Train XGBoost on ElasticNet-selected features; compute `gain` importance; optionally choose top-K features.
7. **Build score**

   * Weight features by normalized gains `w_i` → raw score `S_raw = Σ w_i · z_i` (z = standardized features).
   * Map `S_raw` → percentile → scale 0–100.
8. **Grading**

   * Apply thresholds → grade labels.
9. **Deliverables**

   * Save scaler, selected features, XGBoost model, `weights.json`, thresholds, explanation output.

**Pros**

* Interpretable: ElasticNet coefficients + XGBoost gains easy to explain.
* Compact feature set if ElasticNet selects well.

**Cons / Risk**

* ElasticNet can drop features that are only useful via interactions → irreversible loss.
* If ElasticNet is aggressive, XGBoost won’t be able to recover those nonlinear signals.

---

# Recommended pipeline (Option A — **XGBoost-first**)

**Why this one:** preserves nonlinear structure, gives robust importances, is simple to implement, and is data-efficient on tabular features including deep embeddings.

1. **Image → Feature extraction**

   * EfficientNet features + 32 handcrafted features → raw features `X`.
2. **Remove constant features**

   * Variance threshold.
3. **Optional: PCA on deep features only**

   * If deep features are extremely high-dim and slow, run PCA on EfficientNet part (retain 95% variance) then concatenate with handcrafted features.
4. **Standardize (for features you will use with linear scoring/calibration)**

   * Fit & save `StandardScaler`.
5. **Train XGBoost classifier / regressor**

   * If labels are discrete classes (teacher’s grades): train XGBoost classifier.
   * If you have continuous freshness scores: train XGBoost regressor.
   * Suggested params (start): `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`.
6. **Extract feature importances (gain)**

   * `gain_i = total_gain(feature_i)`. Normalize `w_i = gain_i / Σ gain`.
7. **Compute score**

   * `S_raw = Σ w_i · z_i` (z = standardized selected features).
   * Or use XGBoost prediction probability `P(y=fresh)` as alternate score. Prefer `S_raw` for explainability.
   * Map `S_raw` distribution on validation → percentile → 0–100.
8. **Grading**

   * Use teacher ranges or tune thresholds on validation to maximize grade-F1.
9. **Explainability**

   * For an image, show top-k contributors by `abs(w_i * z_i)` and optionally SHAP values from XGBoost for per-instance explanation.

**Pros**

* Preserves nonlinear interactions
* Robust feature importances
* Good accuracy on tabular+deep features
* Fast and stable
* Easy to produce top-k contributing features and SHAP explanations

**Cons**

* Slightly less “classically linear” interpretable than ElasticNet coefficients (but SHAP fixes this)
* If features extremely high-dim, XGBoost training can be heavier (mitigate with PCA)

---

# Alternative pipelines (short) — compare quickly

### Option B — PCA → XGBoost

* PCA reduces deep-feature dim before XGBoost.
* Good when EfficientNet outputs are huge and redundant.
* Preserves variance structure but PCA components are hard to interpret.
* Use when compute/memory is a constraint.

### Option C — Mutual Information → XGBoost

* Rank features by mutual information (nonlinear), pick top-K, then XGBoost.
* Captures nonlinear dependencies without assuming linearity.
* More computationally expensive; MI estimation noisy for small N.

### Option D — End-to-end learn-to-score (EfficientNet + attention + MLP regressor)

* Train a small network (attention over features + MLP) to predict continuous scores.
* Best potential accuracy if you have many labeled examples and GPU.
* Requires more engineering, hyperparameter tuning, and is less interpretable unless you use attention/SHAP.

---

# Comparison matrix (concise)

| Criterion                        | Teacher (ElasticNet→XGB) | Recommended (XGBoost-first) |                PCA→XGB |              MI→XGB |                End-to-end |
| -------------------------------- | -----------------------: | --------------------------: | ---------------------: | ------------------: | ------------------------: |
| Preserves nonlinear interactions |      ⚠️ risky (can lose) |                      ✅ good |                 ✅ good |              ✅ good |                    ✅ best |
| Interpretability                 |       ✅ (coeffs + gains) |            ✅ (gains + SHAP) |   ⚠️ (components hard) | ✅ (MI ranks + SHAP) | ⚠️ (needs SHAP/attention) |
| Implementation effort            |               Low-medium |                         Low |             Low-medium |              Medium |                      High |
| Compute cost                     |                      Low |                  Low-medium | Lower (if reduce dims) |         Medium-high |                High (GPU) |
| Risk of dropping useful features |     High (if aggressive) |                         Low |                    Low |                 Low |                       Low |
| Best for small data              |            ✅ (but risky) |                           ✅ |                      ✅ |                   ✅ |       ❌ (needs more data) |

---

# Failure modes & edge cases (stress-test)

* **ElasticNet too aggressive**: kills features that only appear in interaction — fix: lower regularization or run ElasticNet in parallel as advisory only.
* **Domain shift** (lighting/background): color features fail → rely more on texture/deep features; add color normalization (gray-world) or augment training.
* **Duplicate/leaky examples**: inflate importance — use your duplicate detection scripts before training.
* **Imbalanced grades**: calibrate thresholds with stratified CV; consider ordinal loss if grades are ordered.
* **Small sample size vs high-dim features**: overfitting → use PCA or regularized tree settings (shrinkage, column sampling).

---

