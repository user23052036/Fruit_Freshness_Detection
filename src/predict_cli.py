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

    return vt, scaler, selected, veg_svm, fresh_svm, le


def predict(image_path: str):

    vt, scaler, selected, veg_svm, fresh_svm, le = load_pipeline_artifacts()

    feats = extract_features(image_path)  # (1312,)

    X = vt.transform(np.array([feats]))
    Xs = scaler.transform(X)
    Xfinal = Xs[:, selected]

    # vegetable prediction
    veg_probs = veg_svm.predict_proba(Xfinal)[0]

    veg_idx = int(np.argmax(veg_probs))

    veg_name = le.inverse_transform([veg_idx])[0]

    veg_conf = float(veg_probs[veg_idx]) * 100.0

    # freshness prediction
    fresh_prob = float(fresh_svm.predict_proba(Xfinal)[0][1])

    score = fresh_prob * 100.0

    grade = grade_from_score(score)

    print(f"Vegetable: {veg_name} ({veg_conf:.2f}%)")
    print(f"Freshness probability: {fresh_prob:.4f}")
    print(f"Score: {score:.2f}")
    print(f"Grade: {grade}")

    return {
        "veg": veg_name,
        "veg_conf": veg_conf,
        "fresh_prob": fresh_prob,
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