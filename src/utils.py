import os
import joblib
import numpy as np

SVM_PARAM_GRID = {
    "C"    : [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    "gamma": [1e-4, 1e-3, 1e-2, 1e-1, "scale"],
}

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

def confidence_band(score: float, config=None) -> str:
    """
    Score in [0, 100] derived from SVM decision function distance.
    Higher score = further into the fresh region of feature space.

    Returns a confidence band reflecting model certainty in the
    fresh/rotten classification — NOT a validated freshness quality grade.
    The binary fresh_label field is the primary actionable decision.

    Bands (thresholds read from scoring_config["grade_thresholds"]):
        High     >= truly_fresh  (default 85)
        Medium   >= fresh        (default 65)
        Low      >= moderate     (default 40)
        Very Low  < moderate
    """
    thr = (config or {}).get("grade_thresholds", {})
    if score >= thr.get("truly_fresh", 85):
        return "High"
    if score >= thr.get("fresh", 65):
        return "Medium"
    if score >= thr.get("moderate", 40):
        return "Low"
    return "Very Low"

def normalize_score(raw, bounds):
    p5, p95 = bounds["p5"], bounds["p95"]
    denom   = p95 - p5
    if abs(denom) < 1e-6:
        return 50.0
    return float(np.clip((raw - p5) / denom * 100.0, 0.0, 100.0))