import os
import joblib

TARGET_VEGETABLES = {
    "tomato",
    "carrot",
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
    """Score in [0,100]."""
    if score >= 95:
        return "Fully Fresh"
    if score >= 75:
        return "Mostly Fresh"
    if score >= 50:
        return "Medium"
    return "Rotten"