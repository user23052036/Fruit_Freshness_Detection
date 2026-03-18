import os
import joblib

TARGET_VEGETABLES = {
    "apple",
    "banana",
    "potato",
    "cucumber",
    "capsicum"
}

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)

def grade_from_score(score: float) -> str:
    """
    Score in [0, 100] derived from SVM decision function distance.
    Higher score = further into the fresh region of feature space.
    """
    if score >= 85:
        return "Truly Fresh"
    if score >= 65:
        return "Fresh"
    if score >= 40:
        return "Moderate"
    return "Rotten"