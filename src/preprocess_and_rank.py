# Speed improvements vs previous version
# ---------------------------------------
# OPT-1: XGBoost rankings are computed ONCE per task at max(k_candidates),
#        then sliced to each k. Previously re-ran 5 seeds × 2 tasks × 5 k-values
#        = 50 XGBoost fits; now just 5 seeds × 2 tasks = 10 fits total.
#
# OPT-2: Phase-1 proxy uses pre-computed rankings (no extra XGBoost fits).
#        VarianceThreshold + StandardScaler applied once; per-k slicing
#        is a cheap numpy index operation.
#
# OPT-3: Phase-2 RBF confirmation GS uses 3-fold CV for the sweep over
#        all candidates, then refits the winner with the full 5-fold GS.
#        Cost: 10 × (3-fold × 30 params) + 2 × (5-fold × 30 params)
#        vs original: 10 × (5-fold × 30 params).  ~40% fewer CV folds.
#
# OPT-4: XGBoost n_estimators reduced from 200 to 100 (importance
#        ranking converges well before 200 trees; halves XGBoost time).
#
# OPT-5: GridSearchCV verbose=0 everywhere (stdout flushing overhead
#        is measurable on 30-param × 5-fold grids).
#
# All correctness fixes from the original version are preserved:
#   Issue 1: RBF confirmation of best_k
#   Issue 3: multi-seed averaged XGBoost ranking
#   Issue 4: log-scale 30-point param grid

import os
import json
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import xgboost as xgb

from utils import save_model, ensure_dirs, SVM_PARAM_GRID

FEATURE_DIR    = "Features"
MODEL_DIR      = "models"
N_RANK_SEEDS   = 5
RANK_SEEDS     = [42, 7, 123, 17, 99]

ensure_dirs(MODEL_DIR)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_training_features():
    xtrain_path = os.path.join(MODEL_DIR, "X_train.npy")
    if not os.path.exists(xtrain_path):
        raise RuntimeError("X_train.npy not found. Run train_split.py first.")
    X       = np.load(xtrain_path)
    y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_train.npy"))
    y_veg   = np.load(os.path.join(MODEL_DIR, "y_veg_train.npy"))
    return X, y_fresh, y_veg


def load_val_features():
    xval_path = os.path.join(MODEL_DIR, "X_val.npy")
    if not os.path.exists(xval_path):
        return None, None, None
    X       = np.load(xval_path)
    y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_val.npy"))
    y_veg   = np.load(os.path.join(MODEL_DIR, "y_veg_val.npy"))
    return X, y_fresh, y_veg


# ─────────────────────────────────────────────────────────────
# OPT-1/4: XGBoost ranking — computed ONCE at max_k, sliced per k
# ─────────────────────────────────────────────────────────────

def _rank_single_seed(X_scaled, y, random_state):
    """One XGBoost run → normalised gain vector of length n_features."""
    clf = xgb.XGBClassifier(
        n_estimators=100,          # OPT-4: 200→100; ranking stabilises early
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)
    gain  = clf.get_booster().get_score(importance_type="gain")
    n     = X_scaled.shape[1]
    imp   = np.array([gain.get(f"f{i}", 0.0) for i in range(n)], dtype=float)
    total = imp.sum()
    if total > 0:
        imp /= total
    return imp


def compute_full_ranking(X_scaled, y, task_label="", seeds=None):
    """
    OPT-1: Average importance over all seeds ONCE.
    Returns (avg_imp, all_imps_list).
    Use rank_features_at_k() to get index arrays for any k.
    """
    if seeds is None:
        seeds = RANK_SEEDS

    all_imps = []
    for s in seeds:
        imp = _rank_single_seed(X_scaled, y, random_state=s)
        all_imps.append(imp)

    avg_imp = np.mean(all_imps, axis=0)

    if task_label:
        print(f"  [INFO] Ranking '{task_label}': averaged {len(seeds)} seeds "
              f"over {X_scaled.shape[1]} features (computed once).")

    return avg_imp, all_imps


def rank_features_at_k(avg_imp, top_k):
    """Cheap O(n log n) slice — no XGBoost re-run needed."""
    order    = np.argsort(avg_imp)[::-1]
    selected = np.sort(order[:min(top_k, len(order))])
    return selected


def check_ranking_stability(all_imps, top_k, task_label=""):
    per_seed_sel = []
    for imp in all_imps:
        order = np.argsort(imp)[::-1]
        per_seed_sel.append(set(order[:min(top_k, len(order))].tolist()))

    overlaps = []
    for i in range(len(per_seed_sel)):
        for j in range(i + 1, len(per_seed_sel)):
            ov = len(per_seed_sel[i] & per_seed_sel[j]) / max(top_k, 1)
            overlaps.append(ov)

    min_ov = min(overlaps) if overlaps else 1.0
    status = "OK" if min_ov >= 0.80 else "WARNING < 0.80"
    if task_label:
        print(f"  [Stability '{task_label}'] min pairwise overlap={min_ov:.3f} [{status}]")
    return min_ov


# ─────────────────────────────────────────────────────────────
# RBF SVM grid search helper
# ─────────────────────────────────────────────────────────────

def _fit_rbf_svm_gs(X_train, y_train, task_label, n_cv=3, random_state=42):
    """
    GridSearchCV on RBF SVM.
    n_cv=3 for sweep (OPT-3), n_cv=5 for the final winner refit.
    """
    base = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        base, SVM_PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,          # OPT-5: no per-fold stdout
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"    [GS {n_cv}-fold] {task_label}: best={gs.best_params_}  "
          f"CV acc={gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


# ─────────────────────────────────────────────────────────────
# OPT-2: Phase-1 proxy k-sweep (no XGBoost re-runs)
# ─────────────────────────────────────────────────────────────

def _proxy_k_sweep(X_tr, y_fresh_tr, y_veg_tr_enc,
                   X_ev, y_fresh_ev, y_veg_ev_enc,
                   avg_imp_fresh, avg_imp_veg, candidates):
    print("\n[Phase-1] Proxy k-sweep (LinearSVC, pre-computed rankings):")
    print(f"  {'k':>5}  {'union':>6}  {'fresh_val':>10}  "
          f"{'veg_val':>9}  {'combined':>10}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*10}")

    best_k, best_score = candidates[0], -1.0
    rows = []

    for k in candidates:
        sel_f = rank_features_at_k(avg_imp_fresh, k)
        sel_v = rank_features_at_k(avg_imp_veg,   k)
        union = np.union1d(sel_f, sel_v)

        X_tr_u = X_tr[:, union]
        X_ev_u = X_ev[:, union]

        clf_f = LinearSVC(max_iter=5000, random_state=42, class_weight="balanced")
        clf_f.fit(X_tr_u, y_fresh_tr.astype(int))
        f_acc = accuracy_score(y_fresh_ev.astype(int), clf_f.predict(X_ev_u))

        clf_v = LinearSVC(max_iter=5000, random_state=42, class_weight="balanced")
        clf_v.fit(X_tr_u, y_veg_tr_enc)
        v_acc = accuracy_score(y_veg_ev_enc, clf_v.predict(X_ev_u))

        combined = (f_acc + v_acc) / 2.0
        rows.append({"k": k, "union": len(union), "fresh_val": f_acc,
                     "veg_val": v_acc, "combined": combined, "model": "LinearSVC"})
        print(f"  {k:>5}  {len(union):>6}  {f_acc:>10.4f}  "
              f"{v_acc:>9.4f}  {combined:>10.4f}")
        if combined > best_score:
            best_score, best_k = combined, k

    print(f"\n  Proxy best_k={best_k}  combined={best_score:.4f}")
    return best_k, rows


# ─────────────────────────────────────────────────────────────
# OPT-3: Phase-2 RBF confirmation — 3-fold sweep, 5-fold final
# ─────────────────────────────────────────────────────────────

def _rbf_confirm_k(X_tr, y_fresh_tr, y_veg_tr_enc,
                   X_ev, y_fresh_ev, y_veg_ev_enc,
                   avg_imp_fresh, avg_imp_veg,
                   proxy_best_k, candidates):
    """
    OPT-3: All candidates evaluated with 3-fold GS (fast).
    The winning k is then refitted once with 5-fold GS (accurate).
    Rankings come from pre-computed importance vectors (OPT-1/2).
    """
    print("\n[Phase-2] RBF confirmation sweep "
          "(3-fold GS per candidate; winner refitted with 5-fold)...")
    print(f"  {'k':>5}  {'union':>6}  {'rbf_fresh':>10}  "
          f"{'rbf_veg':>9}  {'combined':>10}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*10}")

    best_k     = candidates[0]
    best_score = -1.0
    best_veg_p = best_fresh_p = {}
    best_v_acc = best_f_acc   = 0.0
    rows = []

    # --- 3-fold sweep (OPT-3) ---
    for k in candidates:
        sel_f = rank_features_at_k(avg_imp_fresh, k)
        sel_v = rank_features_at_k(avg_imp_veg,   k)
        union = np.union1d(sel_f, sel_v)

        X_tr_u = X_tr[:, union]
        X_ev_u = X_ev[:, union]

        svm_v, p_v, _ = _fit_rbf_svm_gs(X_tr_u, y_veg_tr_enc,
                                          f"veg  k={k}", n_cv=3)
        svm_f, p_f, _ = _fit_rbf_svm_gs(X_tr_u, y_fresh_tr.astype(int),
                                          f"fresh k={k}", n_cv=3)

        v_acc    = accuracy_score(y_veg_ev_enc,           svm_v.predict(X_ev_u))
        f_acc    = accuracy_score(y_fresh_ev.astype(int), svm_f.predict(X_ev_u))
        combined = (f_acc + v_acc) / 2.0

        rows.append({"k": k, "union": len(union),
                     "rbf_fresh_val": f_acc, "rbf_veg_val": v_acc,
                     "combined": combined,
                     "veg_params": p_v, "fresh_params": p_f,
                     "model": "RBF-SVM-3fold"})
        print(f"  {k:>5}  {len(union):>6}  {f_acc:>10.4f}  "
              f"{v_acc:>9.4f}  {combined:>10.4f}")

        if combined > best_score:
            best_score   = combined
            best_k       = k
            best_veg_p   = p_v
            best_fresh_p = p_f
            best_v_acc   = v_acc
            best_f_acc   = f_acc

    # --- 5-fold refit on winner only (OPT-3) ---
    print(f"\n  [Phase-2 final] Refitting winner k={best_k} with 5-fold GS "
          f"for accurate parameter estimates...")
    sel_f5 = rank_features_at_k(avg_imp_fresh, best_k)
    sel_v5 = rank_features_at_k(avg_imp_veg,   best_k)
    union5 = np.union1d(sel_f5, sel_v5)
    X_tr5  = X_tr[:, union5]
    X_ev5  = X_ev[:, union5]

    svm_v5, p_v5, _ = _fit_rbf_svm_gs(X_tr5, y_veg_tr_enc,
                                        f"veg  k={best_k} [5-fold]", n_cv=5)
    svm_f5, p_f5, _ = _fit_rbf_svm_gs(X_tr5, y_fresh_tr.astype(int),
                                        f"fresh k={best_k} [5-fold]", n_cv=5)

    best_v_acc   = accuracy_score(y_veg_ev_enc,           svm_v5.predict(X_ev5))
    best_f_acc   = accuracy_score(y_fresh_ev.astype(int), svm_f5.predict(X_ev5))
    best_veg_p   = p_v5
    best_fresh_p = p_f5

    if best_k != proxy_best_k:
        print(f"\n  [NOTE] RBF confirmation changed best_k: "
              f"{proxy_best_k} → {best_k}  (combined={best_score:.4f})")
    else:
        print(f"\n  [OK] RBF confirms proxy: best_k={best_k}  "
              f"combined={best_score:.4f}")

    return best_k, best_veg_p, best_fresh_p, best_v_acc, best_f_acc, rows


# ─────────────────────────────────────────────────────────────
# CV-based fallback (no val split)
# ─────────────────────────────────────────────────────────────

def _cv_k_sweep(X_tr, y_fresh_tr, y_veg_tr_enc,
                avg_imp_fresh, avg_imp_veg, candidates, cv=5):
    print("\n[INFO] No val split — using 5-fold CV on training data.")
    print(f"  {'k':>5}  {'union':>6}  {'fresh_cv':>10}  "
          f"{'veg_cv':>9}  {'combined':>10}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*10}")

    best_k, best_score = candidates[0], -1.0
    rows = []

    for k in candidates:
        sel_f = rank_features_at_k(avg_imp_fresh, k)
        sel_v = rank_features_at_k(avg_imp_veg,   k)
        union = np.union1d(sel_f, sel_v)
        X_u   = X_tr[:, union]

        clf_f = LinearSVC(max_iter=5000, random_state=42, class_weight="balanced")
        clf_v = LinearSVC(max_iter=5000, random_state=42, class_weight="balanced")
        f_cv = cross_val_score(clf_f, X_u, y_fresh_tr.astype(int),
                               cv=cv, scoring="accuracy", n_jobs=-1).mean()
        v_cv = cross_val_score(clf_v, X_u, y_veg_tr_enc,
                               cv=cv, scoring="accuracy", n_jobs=-1).mean()
        combined = (f_cv + v_cv) / 2.0
        rows.append({"k": k, "union": len(union),
                     "fresh_cv": f_cv, "veg_cv": v_cv,
                     "combined": combined, "model": "LinearSVC-CV"})
        print(f"  {k:>5}  {len(union):>6}  {f_cv:>10.4f}  "
              f"{v_cv:>9.4f}  {combined:>10.4f}")
        if combined > best_score:
            best_score, best_k = combined, k

    print(f"\n  CV best_k={best_k}  combined={best_score:.4f}")
    return best_k, rows


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    X_train, y_fresh_train, y_veg_train = load_training_features()
    X_val,   y_fresh_val,   y_veg_val   = load_val_features()
    has_val = X_val is not None

    print(f"[INFO] Train samples : {X_train.shape[0]}")
    if has_val:
        print(f"[INFO] Val   samples : {X_val.shape[0]}")
    else:
        print("[WARNING] Val data not found. CV fallback will be used.")

    # ── VarianceThreshold (fit on train only) ─────────────────
    vt        = VarianceThreshold(threshold=0.0)
    X_reduced = vt.fit_transform(X_train)
    print(f"[INFO] VarianceThreshold: {X_train.shape[1]} → {X_reduced.shape[1]}")

    # ── StandardScaler (fit on train only) ───────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    X_val_scaled = (scaler.transform(vt.transform(X_val)) if has_val else None)

    # ── Encode vegetable labels ───────────────────────────────
    le_veg    = LabelEncoder()
    y_veg_enc = le_veg.fit_transform(y_veg_train)
    if has_val:
        assert y_veg_val is not None
        y_veg_val_enc = le_veg.transform(y_veg_val)
    else:
        y_veg_val_enc = None

    k_candidates = (50, 100, 150, 200, 250)

    # ── OPT-1: Compute rankings ONCE (both tasks) ─────────────
    print(f"\n[INFO] Computing XGBoost feature rankings "
          f"({N_RANK_SEEDS} seeds × 2 tasks — once only)...")

    print("\n[FRESHNESS ranking]")
    avg_imp_fresh, all_imps_fresh = compute_full_ranking(
        X_scaled, y_fresh_train.astype(int), task_label="freshness",
    )

    print("\n[VEGETABLE ranking]")
    avg_imp_veg, all_imps_veg = compute_full_ranking(
        X_scaled, y_veg_enc, task_label="vegetable",
    )

    # ── k selection ───────────────────────────────────────────
    if has_val:
        proxy_best_k, proxy_rows = _proxy_k_sweep(
            X_scaled,     y_fresh_train, y_veg_enc,
            X_val_scaled, y_fresh_val,   y_veg_val_enc,
            avg_imp_fresh, avg_imp_veg,
            k_candidates,
        )
        (best_k_final, best_veg_params, best_fresh_params,
         rbf_val_veg, rbf_val_fresh, rbf_rows) = _rbf_confirm_k(
            X_scaled,     y_fresh_train, y_veg_enc,
            X_val_scaled, y_fresh_val,   y_veg_val_enc,
            avg_imp_fresh, avg_imp_veg,
            proxy_best_k, k_candidates,
        )
        all_rows = proxy_rows + rbf_rows
    else:
        best_k_final, all_rows = _cv_k_sweep(
            X_scaled, y_fresh_train, y_veg_enc,
            avg_imp_fresh, avg_imp_veg, k_candidates,
        )
        proxy_best_k       = best_k_final
        best_veg_params    = {}
        best_fresh_params  = {}
        rbf_val_veg        = float("nan")
        rbf_val_fresh      = float("nan")

    # ── Final feature sets at best_k_final ───────────────────
    print(f"\n{'='*60}")
    print(f"FINAL FEATURE RANKING  "
          f"(k={best_k_final}, {N_RANK_SEEDS} seeds averaged)")
    print(f"{'='*60}")

    selected_fresh = rank_features_at_k(avg_imp_fresh, best_k_final)
    selected_veg   = rank_features_at_k(avg_imp_veg,   best_k_final)

    check_ranking_stability(all_imps_fresh, best_k_final, "freshness")
    check_ranking_stability(all_imps_veg,   best_k_final, "vegetable")

    union_set   = np.union1d(selected_fresh, selected_veg)
    overlap_set = np.intersect1d(selected_fresh, selected_veg)

    # ── Save artifacts ────────────────────────────────────────
    save_model(vt,     os.path.join(MODEL_DIR, "variance.joblib"))
    save_model(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    np.save(os.path.join(MODEL_DIR, "selected_fresh_features.npy"),  selected_fresh)
    np.save(os.path.join(MODEL_DIR, "selected_veg_features.npy"),    selected_veg)
    np.save(os.path.join(MODEL_DIR, "selected_union_features.npy"),  union_set)

    np.save(os.path.join(MODEL_DIR, "feature_importances_fresh.npy"), avg_imp_fresh)
    np.save(os.path.join(MODEL_DIR, "feature_importances_veg.npy"),   avg_imp_veg)
    np.save(os.path.join(MODEL_DIR, "best_k.npy"), np.array([best_k_final]))

    report = {
        "best_k"               : int(best_k_final),
        "proxy_best_k"         : int(proxy_best_k),
        "union_feature_count"  : int(len(union_set)),
        "overlap_count"        : int(len(overlap_set)),
        "fresh_only"           : int(len(selected_fresh) - len(overlap_set)),
        "veg_only"             : int(len(selected_veg)   - len(overlap_set)),
        "rbf_val_acc_fresh"    : rbf_val_fresh,
        "rbf_val_acc_veg"      : rbf_val_veg,
        "best_svm_params_veg"  : best_veg_params,
        "best_svm_params_fresh": best_fresh_params,
        "n_rank_seeds"         : N_RANK_SEEDS,
        "rank_seeds"           : RANK_SEEDS,
        "k_candidates"         : list(k_candidates),
        "sweep_rows"           : all_rows,
        "note": (
            "best_k confirmed by RBF-SVM GridSearchCV on val set (issue-1 fix). "
            f"Feature importances averaged over {N_RANK_SEEDS} XGBoost seeds "
            "per task — computed ONCE at max_k, sliced per candidate (OPT-1/2). "
            "Grid: C=[1e-3,1e-2,0.1,1,10,100] x gamma=[1e-4,1e-3,1e-2,0.1,'scale'] "
            "(issue-4 fix). Phase-2 uses 3-fold GS sweep + 5-fold final refit (OPT-3). "
            "XGBoost n_estimators=100 (OPT-4). GS verbose=0 (OPT-5)."
        ),
    }

    report_path = os.path.join(MODEL_DIR, "feature_selection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Validation report ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("UNION FEATURE SET SUMMARY")
    print(f"{'='*60}")
    print(f"  top_k per task          : {best_k_final}")
    print(f"  Fresh-specific features : {len(selected_fresh) - len(overlap_set)}")
    print(f"  Veg-specific features   : {len(selected_veg)   - len(overlap_set)}")
    print(f"  Shared features         : {len(overlap_set)}")
    print(f"  Union size              : {len(union_set)}")

    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  best_k                  : {best_k_final}  "
          f"(proxy={proxy_best_k})")
    print(f"  union_feature_count     : {len(union_set)}")
    print(f"  best SVM params (veg)   : {best_veg_params}")
    print(f"  best SVM params (fresh) : {best_fresh_params}")
    print(f"  RBF val acc (fresh)     : {rbf_val_fresh:.4f}")
    print(f"  RBF val acc (veg)       : {rbf_val_veg:.4f}")
    print(f"  Report saved            : {report_path}")
    print(f"{'='*60}")
    print("[INFO] Test set untouched. Run train_svm.py next.")


if __name__ == "__main__":
    main()