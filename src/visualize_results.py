import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from utils import load_model

MODEL_DIR = "models"


def main():

    print("[INFO] Loading test data...")

    X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_veg_test = np.load(os.path.join(MODEL_DIR, "y_veg_test.npy"))
    y_fresh_test = np.load(os.path.join(MODEL_DIR, "y_fresh_test.npy"))

    vt = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_union_features.npy"))

    veg_model = load_model(os.path.join(MODEL_DIR, "veg_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    le = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # Load both importance arrays (fresh-task and veg-task)
    importances_fresh = np.load(os.path.join(MODEL_DIR, "feature_importances_fresh.npy"))
    importances_veg   = np.load(os.path.join(MODEL_DIR, "feature_importances_veg.npy"))

    print("[INFO] Applying preprocessing pipeline...")

    X = vt.transform(X_test)
    X = scaler.transform(X)
    X = X[:, selected]

    # ---------------------------------------
    # Vegetable confusion matrix
    # ---------------------------------------

    yveg_encoded = le.transform(y_veg_test)
    yveg_pred = veg_model.predict(X)

    cm_veg = confusion_matrix(yveg_encoded, yveg_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm_veg,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.title("Vegetable Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # Freshness confusion matrix
    # ---------------------------------------

    yfresh_pred = fresh_model.predict(X)

    cm_fresh = confusion_matrix(y_fresh_test, yfresh_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm_fresh,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Rotten","Fresh"],
        yticklabels=["Rotten","Fresh"]
    )

    plt.title("Freshness Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # Feature importance plot — fresh task
    # ---------------------------------------

    order_fresh = np.argsort(importances_fresh)[::-1][:20]
    top_imp_fresh = importances_fresh[order_fresh]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top_imp_fresh)), top_imp_fresh)
    plt.title("Top 20 Feature Importances — Freshness Task (XGBoost Gain)")
    plt.xlabel("Feature Rank")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # Feature importance plot — vegetable task
    # ---------------------------------------

    order_veg = np.argsort(importances_veg)[::-1][:20]
    top_imp_veg = importances_veg[order_veg]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top_imp_veg)), top_imp_veg, color="orange")
    plt.title("Top 20 Feature Importances — Vegetable Task (XGBoost Gain)")
    plt.xlabel("Feature Rank")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()