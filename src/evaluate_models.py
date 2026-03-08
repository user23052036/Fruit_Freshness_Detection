import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import load_model

MODEL_DIR = "models"

def main():

    print("[INFO] Loading test data...")

    X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_veg_test = np.load(os.path.join(MODEL_DIR, "y_veg_test.npy"))
    y_fresh_test = np.load(os.path.join(MODEL_DIR, "y_fresh_test.npy"))

    vt = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_features.npy"))

    veg_model = load_model(os.path.join(MODEL_DIR, "veg_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    le = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print("[INFO] Applying preprocessing pipeline...")

    X = vt.transform(X_test)
    X = scaler.transform(X)
    X = X[:, selected]

    # -----------------------------
    # Vegetable classifier
    # -----------------------------

    print("\n========== Vegetable Classification ==========")

    yveg_encoded = le.transform(y_veg_test)

    yveg_pred = veg_model.predict(X)

    veg_accuracy = accuracy_score(yveg_encoded, yveg_pred)

    print(f"Accuracy: {veg_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(yveg_encoded, yveg_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(yveg_encoded, yveg_pred))


    # -----------------------------
    # Freshness classifier
    # -----------------------------

    print("\n========== Freshness Classification ==========")

    yfresh_pred = fresh_model.predict(X)

    fresh_accuracy = accuracy_score(y_fresh_test, yfresh_pred)

    print(f"Accuracy: {fresh_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_fresh_test, yfresh_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_fresh_test, yfresh_pred))


if __name__ == "__main__":
    main()