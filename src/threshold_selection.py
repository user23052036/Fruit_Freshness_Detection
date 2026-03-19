"""
threshold_selection.py
======================
Threshold calibration for the RELIABLE gate.

Pipeline contract (fixed — do not change):
    RELIABLE_i = (
        NOT is_ood_i
        AND NOT (crosses_bnd_i AND aug_range_i > T_instability)
        AND abs(decisions_i) > T_boundary
    )

Score = normalize(raw, global_bounds)  — always, regardless of veg prediction.

Public API
----------
compute_gate_metrics(decisions, predictions, true_labels,
                     is_ood, crosses_bnd, aug_range,
                     T_boundary, T_instability) -> GateMetrics

select_thresholds(decisions, predictions, true_labels,
                  is_ood, crosses_bnd, aug_range,
                  epsilon, n_min, T_b_grid, T_i_grid) -> ThresholdResult

evaluate_on_test(decisions, predictions, true_labels,
                 is_ood, crosses_bnd, aug_range,
                 T_boundary, T_instability, cal_risk) -> EvalResult

diagnose_infeasibility(decisions, predictions, true_labels,
                       is_ood, crosses_bnd, aug_range,
                       epsilon, quantiles) -> None
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class GateMetrics(NamedTuple):
    reliable_mask : np.ndarray   # bool[N]
    error_mask    : np.ndarray   # bool[N]
    risk          : float        # P(error | RELIABLE); nan if n_reliable == 0
    coverage      : float        # P(RELIABLE); 0.0 if N == 0
    n_reliable    : int


class ThresholdResult(NamedTuple):
    T_boundary    : float        # T_b*
    T_instability : float        # T_i*
    risk          : float        # achieved Risk on calibration set
    coverage      : float        # achieved Coverage on calibration set
    n_reliable    : int
    feasible      : bool         # False if constraint could not be satisfied


class EvalResult(NamedTuple):
    risk       : float
    coverage   : float
    n_reliable : int
    n_total    : int
    risk_delta : float           # risk_test - cal_risk  (positive = worse)


# ---------------------------------------------------------------------------
# 1.  compute_gate_metrics
# ---------------------------------------------------------------------------

def compute_gate_metrics(
    decisions     : np.ndarray,  # float[N]  raw SVM decision_function values
    predictions   : np.ndarray,  # int[N]    fresh_model.predict(X) in {0,1}
    true_labels   : np.ndarray,  # int[N]    ground truth           in {0,1}
    is_ood        : np.ndarray,  # bool[N]   mahal_dist > thresh_ood
    crosses_bnd   : np.ndarray,  # bool[N]   min_aug_raw<0 AND max_aug_raw>0
    aug_range     : np.ndarray,  # float[N]  max(aug_scores) - min(aug_scores)
    T_boundary    : float,
    T_instability : float,
) -> GateMetrics:
    """
    Apply the RELIABLE formula at fixed (T_boundary, T_instability) and
    return metrics for one dataset split.

    Empty-input guard: if N == 0, returns zero-length masks, nan risk,
    0.0 coverage, and 0 n_reliable without performing any division.
    """

    decisions   = np.asarray(decisions,   dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    is_ood      = np.asarray(is_ood,      dtype=bool)
    crosses_bnd = np.asarray(crosses_bnd, dtype=bool)
    aug_range   = np.asarray(aug_range,   dtype=float)

    N = len(decisions)

    # ------------------------------------------------------------------
    # Empty-input guard — no division performed when N == 0
    # ------------------------------------------------------------------
    if N == 0:
        empty = np.empty(0, dtype=bool)
        return GateMetrics(
            reliable_mask = empty,
            error_mask    = empty,
            risk          = float("nan"),
            coverage      = 0.0,
            n_reliable    = 0,
        )

    # ------------------------------------------------------------------
    # RELIABLE  =  NOT is_ood
    #              AND NOT (crosses_bnd AND aug_range > T_instability)
    #              AND abs_margin > T_boundary
    # ------------------------------------------------------------------
    reliable_mask = (
        (~is_ood)
        & ~(crosses_bnd & (aug_range > T_instability))
        & (np.abs(decisions) > T_boundary)
    )

    error_mask = (predictions != true_labels)   # bool[N], full split

    n_reliable = int(reliable_mask.sum())
    coverage   = n_reliable / N              # N > 0 guaranteed past guard

    risk = (
        float(error_mask[reliable_mask].mean())
        if n_reliable > 0
        else float("nan")
    )

    return GateMetrics(
        reliable_mask = reliable_mask,
        error_mask    = error_mask,
        risk          = risk,
        coverage      = coverage,
        n_reliable    = n_reliable,
    )


# ---------------------------------------------------------------------------
# 2.  select_thresholds
# ---------------------------------------------------------------------------

def select_thresholds(
    decisions   : np.ndarray,
    predictions : np.ndarray,
    true_labels : np.ndarray,
    is_ood      : np.ndarray,
    crosses_bnd : np.ndarray,
    aug_range   : np.ndarray,
    epsilon     : float             = 0.10,
    n_min       : int               = 20,
    T_b_grid    : np.ndarray | None = None,
    T_i_grid    : np.ndarray | None = None,
) -> ThresholdResult:
    """
    Find (T_boundary*, T_instability*) that maximises Coverage subject to:

        Risk(T_b, T_i)                  <= epsilon
        RELIABLE_mask(T_b, T_i).sum()   >= n_min

    Fallback logic — two independent trackers
    ------------------------------------------
    best_feasible
        Tuple (coverage, T_b, T_i, risk, n_reliable).
        Updated only when BOTH risk <= epsilon AND n_reliable >= n_min.
        Tie-break: maximum coverage, then minimum T_boundary.

    best_min_risk
        Tuple (risk, coverage, T_b, T_i, n_reliable).
        Updated for any grid pair where n_reliable >= 1 (risk is defined),
        regardless of whether n_min or epsilon is satisfied.
        Tracks the globally minimum risk seen in the sweep.
        Used as the diagnostic fallback when no feasible pair exists.

    Return priority
    ---------------
    1. best_feasible is not None   → return it,  feasible=True
    2. best_feasible is None AND
       best_min_risk  is not None  → return it,  feasible=False
    3. both are None               → return sentinel (inf thresholds,
                                     nan risk, 0 coverage), feasible=False
    """

    decisions   = np.asarray(decisions,   dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    is_ood      = np.asarray(is_ood,      dtype=bool)
    crosses_bnd = np.asarray(crosses_bnd, dtype=bool)
    aug_range   = np.asarray(aug_range,   dtype=float)

    # ------------------------------------------------------------------
    # Default grids
    # ------------------------------------------------------------------
    if T_b_grid is None:
        T_b_grid = np.arange(0.0, 3.01, 0.05)

    if T_i_grid is None:
        max_range = float(aug_range.max()) if len(aug_range) > 0 else 100.0
        T_i_grid  = np.arange(0.0, max_range + 1.0, 0.5)

    # ------------------------------------------------------------------
    # Two independent fallback trackers
    # ------------------------------------------------------------------
    best_feasible : tuple | None = None   # (coverage, T_b, T_i, risk, n_rel)
    best_min_risk : tuple | None = None   # (risk, coverage, T_b, T_i, n_rel)

    for T_b in T_b_grid:
        for T_i in T_i_grid:

            m = compute_gate_metrics(
                decisions, predictions, true_labels,
                is_ood, crosses_bnd, aug_range,
                T_b, T_i,
            )

            # n_reliable == 0  →  risk is nan; skip both trackers
            if np.isnan(m.risk):
                continue

            # ---- best_min_risk: updated for any pair with risk defined --
            # Condition: n_reliable >= 1 (guaranteed by the nan check above)
            # No epsilon or n_min constraint applied here.
            if (
                best_min_risk is None
                or m.risk < best_min_risk[0]
                or (m.risk == best_min_risk[0]
                    and m.coverage > best_min_risk[1])
            ):
                best_min_risk = (m.risk, m.coverage, T_b, T_i, m.n_reliable)

            # ---- n_min guard for feasibility ----------------------------
            if m.n_reliable < n_min:
                continue

            # ---- epsilon guard for feasibility --------------------------
            if m.risk > epsilon:
                continue

            # ---- best_feasible: max coverage, tie-break min T_boundary --
            if (
                best_feasible is None
                or m.coverage > best_feasible[0]
                or (m.coverage == best_feasible[0]
                    and T_b < best_feasible[1])
            ):
                best_feasible = (m.coverage, T_b, T_i, m.risk, m.n_reliable)

    # ------------------------------------------------------------------
    # Return priority  (1 → 2 → 3)
    # ------------------------------------------------------------------
    if best_feasible is not None:
        cov, T_b, T_i, risk, n_rel = best_feasible
        return ThresholdResult(
            T_boundary    = float(T_b),
            T_instability = float(T_i),
            risk          = risk,
            coverage      = cov,
            n_reliable    = n_rel,
            feasible      = True,
        )

    if best_min_risk is not None:
        risk, cov, T_b, T_i, n_rel = best_min_risk
        return ThresholdResult(
            T_boundary    = float(T_b),
            T_instability = float(T_i),
            risk          = risk,
            coverage      = cov,
            n_reliable    = n_rel,
            feasible      = False,
        )

    # Priority 3: every grid pair gives n_reliable == 0
    return ThresholdResult(
        T_boundary    = float("inf"),
        T_instability = float("inf"),
        risk          = float("nan"),
        coverage      = 0.0,
        n_reliable    = 0,
        feasible      = False,
    )


# ---------------------------------------------------------------------------
# 3.  evaluate_on_test
# ---------------------------------------------------------------------------

def evaluate_on_test(
    decisions     : np.ndarray,
    predictions   : np.ndarray,
    true_labels   : np.ndarray,
    is_ood        : np.ndarray,
    crosses_bnd   : np.ndarray,
    aug_range     : np.ndarray,
    T_boundary    : float,
    T_instability : float,
    cal_risk      : float,
) -> EvalResult:
    """
    Apply fixed thresholds (calibrated on a separate split) to a held-out
    test set and return Risk, Coverage, and risk_delta.

    risk_delta = risk_test - cal_risk
        positive → test set is harder than calibration set;
                   the epsilon constraint may not hold in deployment.
        negative → test set is easier than calibration set.
        nan      → one or both risks are undefined (n_reliable == 0).
    """

    m = compute_gate_metrics(
        decisions, predictions, true_labels,
        is_ood, crosses_bnd, aug_range,
        T_boundary, T_instability,
    )

    if np.isnan(m.risk) or np.isnan(cal_risk):
        risk_delta = float("nan")
    else:
        risk_delta = m.risk - cal_risk

    return EvalResult(
        risk       = m.risk,
        coverage   = m.coverage,
        n_reliable = m.n_reliable,
        n_total    = len(decisions),
        risk_delta = risk_delta,
    )


# ---------------------------------------------------------------------------
# 4.  diagnose_infeasibility
# ---------------------------------------------------------------------------

def diagnose_infeasibility(
    decisions   : np.ndarray,
    predictions : np.ndarray,
    true_labels : np.ndarray,
    is_ood      : np.ndarray,
    crosses_bnd : np.ndarray,
    aug_range   : np.ndarray,
    epsilon     : float = 0.10,
    quantiles   : tuple = (0.50, 0.75, 0.90, 0.95, 0.99),
) -> None:
    """
    Call when select_thresholds returns feasible=False.

    Sweeps T_boundary at several quantiles of |decisions| with
    T_instability=inf (isolates the margin gate only).

    Prints a diagnostic table and ends with an explicit conclusion:

        Case (a): Risk is flat across margin quantiles.
                  The margin gate has no predictive power for errors.
                  Errors are uniformly distributed across margin magnitude.

        Case (b): Risk decreases as quantile rises but never reaches epsilon.
                  The base model error rate is the binding constraint.
                  No margin-based gating can achieve Risk <= epsilon on
                  this data.

    The flat/decreasing classification uses FLAT_THRESHOLD = 0.02 pp
    total drop.  Adjust if needed for your dataset scale.
    """

    decisions   = np.asarray(decisions,   dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)
    is_ood      = np.asarray(is_ood,      dtype=bool)
    crosses_bnd = np.asarray(crosses_bnd, dtype=bool)
    aug_range   = np.asarray(aug_range,   dtype=float)

    N = len(decisions)
    if N == 0:
        print("diagnose_infeasibility: empty input — nothing to diagnose.")
        return

    abs_margin = np.abs(decisions)
    base_error = float((predictions != true_labels).mean())

    print(f"Base error rate (whole split) : {base_error:.4f}")
    print(f"Target epsilon                : {epsilon:.4f}")
    print()
    print(
        f"{'Quantile':>10}  {'T_boundary':>12}  {'n_reliable':>10}  "
        f"{'Risk|RELIABLE':>14}  {'Coverage':>10}"
    )
    print("-" * 64)

    risk_values: list[float] = []

    for q in quantiles:
        T_b = float(np.quantile(abs_margin, q))

        m = compute_gate_metrics(
            decisions, predictions, true_labels,
            is_ood, crosses_bnd, aug_range,
            T_boundary    = T_b,
            T_instability = float("inf"),  # isolate margin gate only
        )

        risk_str = f"{m.risk:.4f}" if not np.isnan(m.risk) else "     nan"
        print(
            f"{q:>10.2f}  {T_b:>12.4f}  {m.n_reliable:>10d}  "
            f"{risk_str:>14}  {m.coverage:>10.4f}"
        )
        risk_values.append(m.risk)

    # ------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------
    print()
    print("--- Conclusion ---")

    defined = [
        (q, r) for q, r in zip(quantiles, risk_values)
        if not np.isnan(r)
    ]

    if len(defined) < 2:
        print(
            "Insufficient data points with n_reliable > 0 to classify "
            "the infeasibility case.  Reduce n_min or collect more data."
        )
        return

    first_risk = defined[0][1]
    last_risk  = defined[-1][1]
    drop       = first_risk - last_risk      # positive if risk falls with T_b

    FLAT_THRESHOLD = 0.02                    # < 2 pp total drop → treated as flat

    reaches_epsilon = any(r <= epsilon for _, r in defined)

    if reaches_epsilon:
        # Guard: should not normally be reached if called after feasible=False
        print(
            "Risk reaches epsilon at one or more margin quantiles.  "
            "The constraint may be satisfiable with a finer T_b grid or "
            "a larger n_min value.  Re-run select_thresholds with a denser "
            "T_b_grid around the quantile where risk first drops below epsilon."
        )
    elif drop < FLAT_THRESHOLD:
        print(
            f"Case (a): Risk is flat across margin quantiles "
            f"(total drop = {drop:.4f} pp, threshold = {FLAT_THRESHOLD:.4f} pp).\n"
            "The margin gate has no predictive power for errors on this data.\n"
            "Errors are uniformly distributed across margin magnitude.\n"
            "Recommended action: remove the T_boundary gate, or replace it "
            "with a gate whose statistic is actually correlated with error "
            "(e.g. augmentation range)."
        )
    else:
        print(
            f"Case (b): Risk decreases as margin quantile rises "
            f"(total drop = {drop:.4f} pp) but never reaches "
            f"epsilon = {epsilon:.4f}.\n"
            "The base model error rate is the binding constraint.\n"
            "No margin-based gating can achieve Risk <= epsilon on this data.\n"
            f"Recommended action: lower epsilon to a feasible target "
            f"(minimum observed risk ≈ {last_risk:.4f}), or improve the "
            "underlying freshness classifier before re-running calibration."
        )


# ---------------------------------------------------------------------------
# 5.  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("SMOKE TEST — threshold_selection.py")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N   = 600

    # Synthetic data: decisions centred on 0; true label = sign(decision)
    decisions   = rng.normal(0, 2.5, N)
    true_labels = (decisions > 0).astype(int)

    # 12 % label noise
    flip        = rng.random(N) < 0.12
    predictions = np.where(flip, 1 - true_labels, true_labels)

    is_ood      = rng.random(N) < 0.04
    crosses_bnd = rng.random(N) < 0.18
    aug_range   = np.abs(rng.normal(8, 4, N)).clip(0)

    # ------------------------------------------------------------------
    # Guard test: empty arrays
    # ------------------------------------------------------------------
    print("\n--- Empty-input guard ---")
    m_empty = compute_gate_metrics(
        np.array([]), np.array([]), np.array([]),
        np.array([]), np.array([]), np.array([]),
        T_boundary=0.5, T_instability=20.0,
    )
    print(f"  N=0  risk={m_empty.risk}  coverage={m_empty.coverage}"
          f"  n_reliable={m_empty.n_reliable}")
    assert np.isnan(m_empty.risk),   "Expected nan risk for empty input"
    assert m_empty.coverage == 0.0,  "Expected 0.0 coverage for empty input"
    assert m_empty.n_reliable == 0,  "Expected 0 n_reliable for empty input"
    print("  PASS")

    # ------------------------------------------------------------------
    # compute_gate_metrics at a single threshold pair
    # ------------------------------------------------------------------
    print("\n--- compute_gate_metrics (T_b=0.5, T_i=20.0) ---")
    m = compute_gate_metrics(
        decisions, predictions, true_labels,
        is_ood, crosses_bnd, aug_range,
        T_boundary=0.5, T_instability=20.0,
    )
    print(f"  n_reliable : {m.n_reliable}")
    print(f"  risk       : {m.risk:.4f}")
    print(f"  coverage   : {m.coverage:.4f}")

    # ------------------------------------------------------------------
    # Cal / test split
    # ------------------------------------------------------------------
    idx     = rng.permutation(N)
    cal_idx = idx[:480]
    tst_idx = idx[480:]

    # ------------------------------------------------------------------
    # select_thresholds — feasible case
    # ------------------------------------------------------------------
    print("\n--- select_thresholds (epsilon=0.10, n_min=20) ---")
    result = select_thresholds(
        decisions[cal_idx], predictions[cal_idx], true_labels[cal_idx],
        is_ood[cal_idx], crosses_bnd[cal_idx], aug_range[cal_idx],
        epsilon = 0.10,
        n_min   = 20,
    )
    print(f"  feasible       : {result.feasible}")
    print(f"  T_boundary*    : {result.T_boundary:.4f}")
    print(f"  T_instability* : {result.T_instability:.4f}")
    print(f"  risk (cal)     : {result.risk:.4f}")
    print(f"  coverage (cal) : {result.coverage:.4f}")
    print(f"  n_reliable     : {result.n_reliable}")

    # ------------------------------------------------------------------
    # select_thresholds — infeasible case (epsilon=0.001 forces failure)
    # ------------------------------------------------------------------
    print("\n--- select_thresholds (epsilon=0.001 — expected infeasible) ---")
    result_inf = select_thresholds(
        decisions[cal_idx], predictions[cal_idx], true_labels[cal_idx],
        is_ood[cal_idx], crosses_bnd[cal_idx], aug_range[cal_idx],
        epsilon = 0.001,
        n_min   = 20,
    )
    print(f"  feasible       : {result_inf.feasible}  (min-risk fallback)")
    print(f"  T_boundary     : {result_inf.T_boundary:.4f}")
    print(f"  T_instability  : {result_inf.T_instability:.4f}")
    print(f"  risk           : {result_inf.risk:.4f}")
    print(f"  n_reliable     : {result_inf.n_reliable}")

    # ------------------------------------------------------------------
    # diagnose_infeasibility
    # ------------------------------------------------------------------
    print("\n--- diagnose_infeasibility (epsilon=0.001) ---")
    diagnose_infeasibility(
        decisions[cal_idx], predictions[cal_idx], true_labels[cal_idx],
        is_ood[cal_idx], crosses_bnd[cal_idx], aug_range[cal_idx],
        epsilon = 0.001,
    )

    # ------------------------------------------------------------------
    # evaluate_on_test
    # ------------------------------------------------------------------
    print("\n--- evaluate_on_test ---")
    eval_r = evaluate_on_test(
        decisions[tst_idx], predictions[tst_idx], true_labels[tst_idx],
        is_ood[tst_idx], crosses_bnd[tst_idx], aug_range[tst_idx],
        T_boundary    = result.T_boundary,
        T_instability = result.T_instability,
        cal_risk      = result.risk,
    )
    print(f"  risk (test)    : {eval_r.risk:.4f}")
    print(f"  coverage(test) : {eval_r.coverage:.4f}")
    print(f"  n_reliable     : {eval_r.n_reliable} / {eval_r.n_total}")
    delta_sign = "worse" if eval_r.risk_delta > 0 else "better"
    print(f"  risk_delta     : {eval_r.risk_delta:+.4f}  ({delta_sign} than cal)")

    print("\nSmoke test complete.")