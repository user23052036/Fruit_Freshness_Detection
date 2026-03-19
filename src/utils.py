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

def grade_from_score(score: float, config=None) -> str:
    """
    Score in [0, 100] derived from SVM decision function distance.
    Higher score = further into the fresh region of feature space.
    """
    thr = (config or {}).get("grade_thresholds", {})
    t1  = thr.get("truly_fresh", 85)
    t2  = thr.get("fresh",       65)
    t3  = thr.get("moderate",    40)
    if score >= t1:
        return "Truly Fresh"
    if score >= t2:
        return "Fresh"
    if score >= t3:
        return "Moderate"
    return "Rotten"