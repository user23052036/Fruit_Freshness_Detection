import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb

from utils import save_model, ensure_dirs

FEATURE_DIR = "Features"
MODEL_DIR   = "models"

ensure_dirs(MODEL_DIR)


def load_training_features():
    xtrain_path = os.path.join(MODEL_DIR, "X_train.npy")
    if not os.path.exists(xtrain_path):
        raise RuntimeError(
            "X_train.npy not found. Run train_split.py first."
        )
    X       = np.load(xtrain_path)
    y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_train.npy"))
    return X, y_fresh


def rank_features(X_scaled, y_fresh, random_state, top_k):
    """
    XGBoost feature ranking using FRESHNESS-ONLY labels.
    Vegetable identity is NOT mixed into the ranking label.
    """
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        verbosity=0,
        random_state=random_state,
    )
    clf.fit(X_scaled, y_fresh.astype(int))
    gain = clf.get_booster().get_score(importance_type="gain")
    n    = X_scaled.shape[1]
    importances = np.array(
        [gain.get(f"f{i}", 0.0) for i in range(n)], dtype=float
    )
    order = np.argsort(importances)[::-1]
    return order[:min(top_k, len(order))], importances


def feature_selection_stability(X_scaled, y_fresh, top_k,
                                 seeds=(42, 7, 123)):
    """
    Run XGBoost ranking across multiple seeds.
    Report pairwise overlap of top-k feature sets.
    Warns if min overlap < 0.80.
    """
    selections = []
    for s in seeds:
        idx, _ = rank_features(X_scaled, y_fresh, random_state=s, top_k=top_k)
        selections.append(set(idx.tolist()))

    print("\n[INFO] Feature selection stability across seeds:")
    overlaps = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ov = len(selections[i] & selections[j]) / top_k
            overlaps.append(ov)
            print(f"  Seed {seeds[i]} vs {seeds[j]}  overlap = {ov:.2f}")

    min_ov = min(overlaps)
    if min_ov < 0.80:
        print(
            f"[WARNING] Min overlap {min_ov:.2f} < 0.80. "
            "Feature selection is seed-sensitive. Pipeline may be fragile."
        )
    else:
        print(f"[OK] Min overlap {min_ov:.2f} >= 0.80. Selection is stable.")

    return overlaps


def top_k_sweep(X_scaled, y_fresh, candidates=(50, 100, 150, 200), seed=42):
    """
    Sweep top_k values to check whether k=100 is justified.

    For each k: train a quick linear SVM on the selected features
    and report cross-val accuracy. This is a diagnostic only —
    the final selection still uses k=100 (canonical).

    Interpretation:
      performance same across k  → k=100 is arbitrary (could use fewer)
      performance peaks at k<100 → current k is over-selected
      performance peaks at k>100 → current k may be under-selected
    """
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score

    print("\n[INFO] top_k sweep (diagnostic — does not change final selection):")
    print(f"  {'k':>5}  {'CV_acc (5-fold)':>16}")
    print(f"  {'-'*5}  {'-'*16}")

    for k in candidates:
        idx, _ = rank_features(X_scaled, y_fresh, random_state=seed, top_k=k)
        X_k    = X_scaled[:, idx]
        clf    = LinearSVC(max_iter=2000, random_state=seed)
        scores = cross_val_score(clf, X_k, y_fresh.astype(int), cv=5,
                                 scoring="accuracy", n_jobs=-1)
        print(f"  {k:>5}  {scores.mean():.4f} ± {scores.std():.4f}")

    print("  [NOTE] If accuracy is flat across k values, k=100 is arbitrary.")
    print("         This does not change the selected features — review only.")


def main(top_k=100):
    np.random.seed(42)

    X, y_fresh = load_training_features()
    print(f"[INFO] Using {X.shape[0]} train samples for ranking.")

    # ── VarianceThreshold ────────────────────────────────────────
    vt = VarianceThreshold(threshold=0.0)
    X_reduced = vt.fit_transform(X)
    print(f"[INFO] VarianceThreshold: {X.shape[1]} → {X_reduced.shape[1]}")

    # ── StandardScaler ───────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # ── Seed stability diagnostic ────────────────────────────────
    feature_selection_stability(X_scaled, y_fresh, top_k)

    # ── top_k sweep diagnostic ───────────────────────────────────
    top_k_sweep(X_scaled, y_fresh)

    # ── Final selection on canonical seed ────────────────────────
    selected_idx, importances = rank_features(
        X_scaled, y_fresh, random_state=42, top_k=top_k
    )

    # ── Save artifacts ───────────────────────────────────────────
    save_model(vt,     os.path.join(MODEL_DIR, "variance.joblib"))
    save_model(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    np.save(os.path.join(MODEL_DIR, "selected_features.npy"),  selected_idx)
    np.save(os.path.join(MODEL_DIR, "feature_importances.npy"), importances)

    print(f"[DONE] Selected top {len(selected_idx)} features (freshness-only ranking).")


if __name__ == "__main__":
    main()