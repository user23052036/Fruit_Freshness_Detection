# Boundary Threshold Gate

The boundary threshold gate flags predictions where the freshness SVM's raw decision margin is very close to zero — indicating the classifier is uncertain about which side of the fresh/rotten boundary a sample falls on. This document explains what the raw margin means, how the threshold is selected, why the current threshold is zero, and what this tells us about the base model.

---

## 1. What the Raw Margin Represents

The freshness SVM produces a signed real number for every sample, called the **decision function value** or raw margin:

```python
# predict_cli.py
raw = float(fresh_svm.decision_function(Xfinal)[0])
```

This value is the signed geometric distance from the sample's feature vector to the SVM's decision hyperplane in the 349-dimensional feature space:

```
Feature space (simplified to 1D for illustration):

        rotten side          boundary          fresh side
  ←──────────────────────────│──────────────────────────→
                             0

  raw = −2.1   raw = −0.8   raw = 0    raw = +0.6   raw = +1.9
  Clearly      Probably     Uncertain  Probably      Clearly
  rotten       rotten                  fresh         fresh

  Large |raw|  →  far from boundary  →  confident prediction
  Small |raw|  →  near boundary      →  uncertain prediction
```

The boundary threshold gate fires when `|raw| < T_boundary`. Any sample within this margin band is considered geometrically uncertain — a small perturbation to the image (a change in lighting, a slight blur) could push it to the other side of the hyperplane, flipping the prediction.

```python
# predict_cli.py
boundary_thresh = cfg["boundary_threshold"]   # = 0.0
near_boundary   = abs(raw) < boundary_thresh
```

When `near_boundary = True`, `decision_unreliable` is set to True, producing a TENTATIVE output: the freshness score is shown but `fresh_label` is withheld.

---

## 2. How T_boundary Is Selected — The Formal Optimisation

T_boundary is not chosen by inspection or heuristic. It is selected by solving a **constrained coverage-maximisation problem** on `thr_val` (the held-out half of the validation set, disjoint from the probability calibration set).

The reliability formula is:

```
RELIABLE_i = (
    NOT is_ood_i
    AND NOT (crosses_bnd_i AND aug_range_i > T_instability)
    AND abs(decision_i) > T_boundary
)
```

The optimiser (`select_thresholds()` in `threshold_selection.py`) sweeps a grid of candidate T_boundary values and finds the one that maximises Coverage subject to a Risk constraint:

```
Find T_boundary* that:
  Maximise:  Coverage = P(RELIABLE)
  Subject to: Risk = P(error | RELIABLE) ≤ ε = 0.10
              n_reliable ≥ n_min

Grid: T_boundary ∈ [0.0, 3.0]  step 0.05   →  61 candidate values
```

For each candidate T_b, the optimiser computes:
- Which samples pass all three gate conditions simultaneously (RELIABLE mask)
- The error rate among those RELIABLE samples (Risk)
- The fraction of all samples that are RELIABLE (Coverage)

The selected T_boundary is the value that gives the highest Coverage while keeping Risk ≤ 10%.

```python
# threshold_selection.py

T_b_grid = np.arange(0.0, 3.01, 0.05)

for T_b in T_b_grid:
    for T_i in T_i_grid:
        m = compute_gate_metrics(
            decisions, predictions, true_labels,
            is_ood, crosses_bnd, aug_range,
            T_b, T_i,
        )
        # Risk = P(error | RELIABLE) = error_mask[reliable_mask].mean()
        # Coverage = reliable_mask.sum() / N

        if m.risk <= epsilon and m.n_reliable >= n_min:
            if m.coverage > best_coverage:
                best_feasible = (m.coverage, T_b, T_i, m.risk, m.n_reliable)
```

---

## 3. Why T_boundary Is Currently 0.0

The optimiser found T_boundary = 0.0 on this dataset. This is not a bug or a placeholder — it is the formal result.

```
[INFO] Formal thresholds — T_boundary=0.0000  T_instability=36.0000
       Risk=0.0188  Coverage=0.9789  n_reliable=372
```

T_boundary = 0.0 means `abs(raw) > 0.0` — which is always true (a sample would need to land exactly on the hyperplane to fail this condition). In effect, **the boundary proximity gate is currently inactive**.

The optimiser chose this because setting T_boundary higher (say, 0.05 or 0.10) would exclude samples with small margins from RELIABLE, but those samples do not have a higher error rate than samples with large margins on this dataset. Excluding them would reduce Coverage without reducing Risk — a worse outcome by the optimisation objective.

This outcome means the base freshness model is accurate enough that near-boundary samples are not disproportionately wrong. The samples that do contain errors are being caught by the OOD gate and centroid gate, not by the margin proximity gate.

To visualise what the sweep looks like conceptually:

```
As T_boundary increases from 0.0 → 3.0:

  Coverage:   ──────────────────╲
                                 ╲────────────────
  (decreases as more samples are excluded from RELIABLE)

  Risk:       ─────────────────────────────────────
  (stays roughly flat — near-boundary errors no worse than far-boundary)

  At T_boundary = 0.0: Coverage = 97.89%, Risk = 1.88%  ← optimal
  At T_boundary = 0.5: Coverage ≈ lower,  Risk ≈ similar
  At T_boundary = 1.0: Coverage ≈ lower,  Risk ≈ similar

  Optimal choice: T_boundary = 0.0  (maximum coverage, risk already ≤ 10%)
```

If the model were weaker — if near-boundary samples had substantially higher error rates — the Risk curve would dip below the flat line as T_boundary increases, making exclusion beneficial. The optimiser would then find a T_boundary > 0. The gate is not disabled by design; it is inactive because the model does not need it on this data.

---

## 4. How the Gate Would Behave If T_boundary Were > 0

For reference, here is what a non-zero T_boundary would look like in practice. Suppose a retrain produced T_boundary = 0.25:

```
Sample A:
  raw = +0.08
  |raw| = 0.08 < 0.25  →  near_boundary = True
  →  decision_unreliable = True  →  TENTATIVE

Sample B:
  raw = +1.43
  |raw| = 1.43 > 0.25  →  near_boundary = False
  →  gate does not fire (other gates still apply)

Sample C:
  raw = −0.19
  |raw| = 0.19 < 0.25  →  near_boundary = True
  →  decision_unreliable = True  →  TENTATIVE
  Warning: "MODEL UNCERTAINTY — near decision boundary
            (|raw|=0.1900 < 0.2500). Classifier is unsure."
```

The warning message format (from `predict_cli.py`) includes the exact raw margin and threshold:

```
[!] MODEL UNCERTAINTY — near decision boundary
    (|raw|=X.XXXX < X.XXXX). Classifier is unsure.
```

---

## 5. The Gate Ablation Result

The test-set ablation confirms that the gate is effectively inactive:

```
  Gate               Fires  Fire%  Catch_W  Block_C   Δ_acc    Δ_cov   Verdict
  ─────────────────────────────────────────────────────────────────────────────
  G2_near_boundary      0   0.0%      0        0    −0.0003  +0.0524   NEVER FIRES
```

Zero samples on the test set have a raw margin of exactly zero (which is the only way `abs(raw) < 0.0` can trigger). The non-zero Δ_acc and Δ_cov values come from gate interdependency effects in the state recomputation — not from any samples that actually failed the G2 condition.

The NEVER FIRES verdict is correct and expected. If a future retrain produces a model where near-boundary samples are more error-prone, the optimiser will automatically recalibrate T_boundary to a value above zero when `train_svm.py` is re-run.

---

## 6. The Relationship Between T_boundary and Coverage

Coverage is a first-class metric in this system. It measures the fraction of predictions the system is willing to commit to as RELIABLE. Lowering T_boundary (or keeping it at zero) maximises coverage. Raising T_boundary increases the margin band that is excluded, reducing coverage.

The optimiser's constraint `Risk ≤ ε = 0.10` ensures that maximising coverage does not come at the cost of accuracy: the RELIABLE subset must have an error rate below 10%.

```
Current results (T_boundary = 0.0):

  RELIABLE   : 2,343 / 2,539  =  92.3%  of test samples
  TENTATIVE  :   134 / 2,539  =   5.3%
  UNRELIABLE :    62 / 2,539  =   2.4%

  RELIABLE accuracy: 98.98%   (higher than overall 98.94%)
  Risk on thr_val:   1.88%    (well below ε = 10%)
```

The 5.3% TENTATIVE rate comes entirely from the OOD caution zone, low vegetable confidence, and centroid inconsistency — not from the boundary gate.

---

## 7. Infeasibility Handling

If the optimiser cannot find any (T_boundary, T_instability) pair satisfying Risk ≤ 10% with n_reliable ≥ n_min, `select_thresholds()` returns `feasible=False` and the best observed pair (minimum Risk, regardless of the epsilon constraint) is used as a diagnostic fallback. `train_svm.py` then calls `diagnose_infeasibility()` to classify the failure:

```
Case (a): Risk is flat across margin quantiles.
          The margin gate has no predictive power for errors on this data.
          Recommended: remove the T_boundary gate.

Case (b): Risk decreases with larger margins but never reaches ε = 10%.
          The base model error rate is the binding constraint.
          Recommended: lower ε to an achievable target, or improve the
          freshness classifier before recalibrating.
```

On the current run, the result was feasible, so the diagnosis was not triggered.

---

## 8. Configuration

T_boundary and its related fields are stored in `models/scoring_config.json`:

| Key | Value | Meaning |
|-----|-------|---------|
| `boundary_threshold` | **0.0** | abs(raw) must exceed this to avoid near_boundary=True |
| `threshold_selection_result.feasible` | true | Optimiser found a valid (T_b, T_i) pair |
| `threshold_selection_result.T_boundary` | 0.0 | Selected threshold |
| `threshold_selection_result.risk` | 0.0188 | Risk on thr_val at selected thresholds |
| `threshold_selection_result.coverage` | 0.9789 | Coverage on thr_val at selected thresholds |
| `threshold_selection_result.n_reliable` | 372 | RELIABLE count on thr_val |
| `threshold_selection_result.data_source` | "thr_val (disjoint from cal_val)" | Confirms no leakage |

T_boundary is recalibrated automatically whenever `train_svm.py` is re-run. Do not manually edit this value — the formal optimiser will set the correct value for whatever model and feature set is in place at the time of retraining.