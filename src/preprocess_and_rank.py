# src/preprocess_and_rank.py

import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

from utils import save_model, ensure_dirs

FEATURE_DIR = "Features"
MODEL_DIR = "models"

ensure_dirs(MODEL_DIR)


def load_training_features():

    xtrain_path = os.path.join(MODEL_DIR, "X_train.npy")

    if not os.path.exists(xtrain_path):
        raise RuntimeError(
            "X_train.npy not found. Run train_split.py before preprocess_and_rank.py"
        )

    # Load only training split — never full dataset (would cause data leakage)
    X = np.load(xtrain_path)
    y_veg = np.load(os.path.join(MODEL_DIR, "y_veg_train.npy"))
    y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_train.npy"))

    # Combine veg + freshness into single label for XGBoost feature ranking
    combined_labels = np.array([f"{v}_{f}" for v, f in zip(y_veg, y_fresh)])

    le = LabelEncoder()
    y = le.fit_transform(combined_labels)

    return X, y


def main(top_k=100):

    np.random.seed(42)

    X, y = load_training_features()

    print(f"[INFO] Using {X.shape[0]} samples for ranking")

    # ---------------------------------
    # Remove constant features
    # ---------------------------------

    vt = VarianceThreshold(threshold=0.0)

    X_reduced = vt.fit_transform(X)

    print("[INFO] VarianceThreshold removed ->", X.shape[1], "->", X_reduced.shape[1])

    # ---------------------------------
    # Standardize features
    # ---------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_reduced)

    # ---------------------------------
    # Train XGBoost for feature ranking
    # ---------------------------------

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        verbosity=0,
        random_state=42
    )

    clf.fit(X_scaled, y)

    # ---------------------------------
    # Get gain-based feature importance
    # ---------------------------------

    booster = clf.get_booster()

    gain_dict = booster.get_score(importance_type="gain")

    n_features = X_scaled.shape[1]

    importances = np.array(
        [gain_dict.get(f"f{i}", 0.0) for i in range(n_features)],
        dtype=float
    )

    # ---------------------------------
    # Select top features
    # ---------------------------------

    order = np.argsort(importances)[::-1]

    selected_idx = order[:min(top_k, len(order))]

    # ---------------------------------
    # Save preprocessing artifacts
    # ---------------------------------

    save_model(vt, os.path.join(MODEL_DIR, "variance.joblib"))
    save_model(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    np.save(os.path.join(MODEL_DIR, "selected_features.npy"), selected_idx)
    np.save(os.path.join(MODEL_DIR, "feature_importances.npy"), importances)

    print(f"[DONE] Selected top {len(selected_idx)} features")


if __name__ == "__main__":
    main()