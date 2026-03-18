import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from utils import save_model, load_model, ensure_dirs

MODEL_DIR = "models"
FEATURE_DIR = "Features"

ensure_dirs(MODEL_DIR)


def load_features_for_training():

    # Prefer train split if available
    x_train_path = os.path.join(MODEL_DIR, "X_train.npy")

    if os.path.exists(x_train_path):

        X = np.load(x_train_path)
        y_veg = np.load(os.path.join(MODEL_DIR, "y_veg_train.npy"))
        y_fresh = np.load(os.path.join(MODEL_DIR, "y_fresh_train.npy"))

    else:

        X = np.load(os.path.join(FEATURE_DIR, "X.npy"))
        y_veg = np.load(os.path.join(FEATURE_DIR, "y_veg.npy"))
        y_fresh = np.load(os.path.join(FEATURE_DIR, "y_fresh.npy"))

    return X, y_veg, y_fresh


def main():

    print("[INFO] Loading features...")

    X, y_veg, y_fresh = load_features_for_training()

    # Load preprocessing artifacts
    vt = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_features.npy"))

    print("[INFO] Applying preprocessing pipeline...")

    # Remove constant features
    X_reduced = vt.transform(X)

    # Standardize
    X_scaled = scaler.transform(X_reduced)

    # Select top features
    X_final = X_scaled[:, selected]

    print("[INFO] Feature matrix after selection:", X_final.shape)

    # Encode vegetable labels
    le = LabelEncoder()
    yveg_encoded = le.fit_transform(y_veg)

    save_model(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # ---------------------------
    # Vegetable classifier
    # ---------------------------

    print("[INFO] Training vegetable classifier...")

    veg_model = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced")

    veg_model.fit(X_final, yveg_encoded)

    save_model(
        veg_model,
        os.path.join(MODEL_DIR, "veg_svm.joblib")
    )

    print("[DONE] Vegetable classifier saved")

    # ---------------------------
    # Freshness classifier
    # ---------------------------

    print("[INFO] Training freshness classifier...")

    fresh_model = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced")

    fresh_model.fit(X_final, y_fresh)

    save_model(
        fresh_model,
        os.path.join(MODEL_DIR, "fresh_svm.joblib")
    )

    print("[DONE] Freshness classifier saved")

    # ---------------------------
    # Compute and save decision
    # function normalization bounds

    print("[INFO] Computing decision function bounds for freshness grading...")

    # decision_function returns signed distance from hyperplane
    # positive = fresh side, negative = rotten side
    train_decisions = fresh_model.decision_function(X_final)

    bounds = np.array([train_decisions.min(), train_decisions.max()])

    np.save(os.path.join(MODEL_DIR, "fresh_decision_bounds.npy"), bounds)

    print(f"[DONE] Decision bounds saved — min: {bounds[0]:.4f}, max: {bounds[1]:.4f}")


if __name__ == "__main__":
    main()