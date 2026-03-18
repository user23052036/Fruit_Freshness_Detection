import argparse
import os
import numpy as np

from extract_features import extract_features
from utils import load_model, grade_from_score

MODEL_DIR = "models"


def load_pipeline_artifacts():

    vt = load_model(os.path.join(MODEL_DIR, "variance.joblib"))
    scaler = load_model(os.path.join(MODEL_DIR, "scaler.joblib"))
    selected = np.load(os.path.join(MODEL_DIR, "selected_features.npy"))

    veg_svm = load_model(os.path.join(MODEL_DIR, "veg_svm.joblib"))
    fresh_svm = load_model(os.path.join(MODEL_DIR, "fresh_svm.joblib"))
    le = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    bounds = np.load(os.path.join(MODEL_DIR, "fresh_decision_bounds.npy"))

    return vt, scaler, selected, veg_svm, fresh_svm, le, bounds


def compute_freshness_score(fresh_svm, X_final, bounds) -> float:
    """
    Compute freshness score [0, 100] using SVM decision function.

    decision_function returns the signed distance of the sample
    from the SVM decision hyperplane:
      - Positive value → fresh side of boundary
      - Negative value → rotten side of boundary
      - Larger magnitude → further from boundary → more confident

    We normalize this raw distance using the min/max observed
    across the training set so the score is always in [0, 100].
    Score = 100 means maximally fresh, 0 means maximally rotten.
    """

    raw = float(fresh_svm.decision_function(X_final)[0])

    min_train, max_train = float(bounds[0]), float(bounds[1])

    score = (raw - min_train) / (max_train - min_train) * 100.0

    return float(np.clip(score, 0.0, 100.0))


def predict(image_path: str):

    vt, scaler, selected, veg_svm, fresh_svm, le, bounds = load_pipeline_artifacts()

    feats = extract_features(image_path)  # (1312,)

    X = vt.transform(np.array([feats]))
    Xs = scaler.transform(X)
    Xfinal = Xs[:, selected]

    # ---------------------------
    # Vegetable prediction
    # ---------------------------

    veg_probs = veg_svm.predict_proba(Xfinal)[0]

    veg_idx = int(np.argmax(veg_probs))

    veg_name = le.inverse_transform([veg_idx])[0]

    veg_conf = float(veg_probs[veg_idx]) * 100.0

    # ---------------------------
    # Freshness prediction
    # ---------------------------

    # Class label: 0 = rotten, 1 = fresh (from extract_dataset_features.py)
    fresh_class = int(fresh_svm.predict(Xfinal)[0])
    fresh_label = "Fresh" if fresh_class == 1 else "Rotten"

    # ---------------------------
    # Freshness score via
    # decision function geometry
    # ---------------------------

    score = compute_freshness_score(fresh_svm, Xfinal, bounds)

    grade = grade_from_score(score)

    print(f"Vegetable : {veg_name} ({veg_conf:.2f}%)")
    print(f"Freshness : {fresh_label}")
    print(f"Score     : {score:.2f} / 100")
    print(f"Grade     : {grade}")

    return {
        "veg": veg_name,
        "veg_conf": veg_conf,
        "fresh_label": fresh_label,
        "score": score,
        "grade": grade
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    predict(args.image)