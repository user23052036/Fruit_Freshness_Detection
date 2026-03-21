"""
app.py — FastAPI backend for the Vegetable Freshness Grader web app
====================================================================
Run:
    uvicorn app:app --reload --port 8000
    Open  http://localhost:8000
"""

import os
import sys
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Make src/ importable without modifying any ML source files ──
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from predict_cli import (
    compute_object_coverage,
    load_pipeline_artifacts,
    mahalanobis_dist,
    mahal_zone,
    normalize_score,
    preflight_checks,
)
from extract_features import extract_features
from utils import confidence_band

# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Vegetable Freshness Grader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE    = os.path.dirname(__file__)
_STATIC  = os.path.join(_BASE, "static")
_TMPL    = os.path.join(_BASE, "templates")

app.mount("/static", StaticFiles(directory=_STATIC), name="static")
templates = Jinja2Templates(directory=_TMPL)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────
# Upload endpoint
# ──────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    ext = os.path.splitext(file.filename or "img.jpg")[-1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        ext = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _predict_extended(tmp_path)
        return JSONResponse(result)
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────
# Extended predict — mirrors predict_cli.predict() exactly;
# captures intermediate values for the pipeline UI.
# ──────────────────────────────────────────────────────────────

def _predict_extended(image_path: str) -> dict:

    (vt, scaler, selected, veg_svm, fresh_svm, le,
     train_mean, train_precision, class_centroids, cfg) = load_pipeline_artifacts()

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return {
            "state": "UNRELIABLE", "reason": "Image could not be read",
            "warnings": ["Image unreadable"], "pipeline": None,
        }

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    blur_val       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_val = float(gray.mean())
    coverage_val   = float(compute_object_coverage(gray))

    min_lap = cfg.get("min_laplacian_variance", 28.0)
    min_b   = cfg.get("min_brightness",         30.0)
    max_b   = cfg.get("max_brightness",         220.0)
    min_cov = cfg.get("min_coverage",           0.40)

    preflight_status, preflight_reason = preflight_checks(image_path, cfg)

    preflight = {
        "status":          preflight_status,
        "reason":          preflight_reason,
        "blur_val":        round(blur_val, 1),
        "blur_thresh":     min_lap,
        "blur_pass":       blur_val >= min_lap,
        "brightness_val":  round(brightness_val, 1),
        "brightness_min":  min_b,
        "brightness_max":  max_b,
        "brightness_pass": min_b <= brightness_val <= max_b,
        "coverage_val":    round(coverage_val, 3),
        "coverage_min":    min_cov,
        "coverage_warn":   coverage_val < min_cov,
    }

    if preflight_status == "UNRELIABLE":
        return {
            "state": "UNRELIABLE", "reason": preflight_reason,
            "warnings": [preflight_reason],
            "pipeline": {"preflight": preflight},
        }

    preflight_warnings = (
        [preflight_reason] if preflight_status == "OK_LOW_COVERAGE" else []
    )

    feats  = extract_features(image_path)
    X_vt   = vt.transform(np.array([feats]))
    X_scl  = scaler.transform(X_vt)
    Xfinal = X_scl[:, selected]

    features_info = {
        "total":     int(len(feats)),
        "deep_dims": 1280,
        "hand_dims": 32,
        "after_vt":  int(X_vt.shape[1]),
        "after_sel": int(Xfinal.shape[1]),
    }

    veg_probs_arr = veg_svm.predict_proba(Xfinal)[0]
    sorted_probs  = np.sort(veg_probs_arr)[::-1]
    veg_idx       = int(np.argmax(veg_probs_arr))
    veg_name      = le.inverse_transform([veg_idx])[0]
    veg_conf      = float(sorted_probs[0]) * 100.0
    conf_gap      = float(sorted_probs[0] - sorted_probs[1]) * 100.0

    veg_conf_t    = cfg.get("veg_confidence_threshold", 0.70) * 100.0
    veg_gap_t     = cfg.get("veg_gap_threshold",        0.15) * 100.0
    veg_confident = (veg_conf >= veg_conf_t) and (conf_gap >= veg_gap_t)

    veg_all_probs = {
        le.classes_[i]: round(float(veg_probs_arr[i]) * 100, 2)
        for i in range(len(le.classes_))
    }

    raw             = float(fresh_svm.decision_function(Xfinal)[0])
    fresh_class     = int(fresh_svm.predict(Xfinal)[0])
    fresh_label_raw = "Fresh" if fresh_class == 1 else "Rotten"

    x_flat         = Xfinal.flatten()
    dists_to_c     = np.linalg.norm(class_centroids - x_flat, axis=1)
    sorted_c_idx   = np.argsort(dists_to_c)
    d_pred         = float(dists_to_c[veg_idx])
    d_second       = float(next(dists_to_c[j] for j in sorted_c_idx if j != veg_idx))
    centroid_ratio = d_pred / (d_second + 1e-9)

    per_class_t      = cfg.get("centroid_ratio_thresholds", {})
    centroid_ratio_t = float(per_class_t.get(veg_name, 1.0))
    class_inconsistent = centroid_ratio > centroid_ratio_t

    centroid_info = {
        "ratio":      round(centroid_ratio,   4),
        "threshold":  round(centroid_ratio_t, 4),
        "consistent": not class_inconsistent,
        "d_pred":     round(d_pred,   3),
        "d_second":   round(d_second, 3),
    }

    per_veg     = cfg["per_veg_bounds"]
    globl       = cfg["global_bounds"]
    use_per_veg = veg_confident and not class_inconsistent
    bounds      = per_veg.get(veg_name, globl) if use_per_veg else globl
    norm_source = "per-veg" if (use_per_veg and veg_name in per_veg) else "global"

    bounds_info = {
        "source":   norm_source,
        "p5":       round(bounds["p5"],  4),
        "p95":      round(bounds["p95"], 4),
        "veg_name": veg_name,
    }

    score      = normalize_score(raw, bounds)
    score_info = {
        "raw":   round(raw, 4),
        "p5":    round(bounds["p5"],  4),
        "p95":   round(bounds["p95"], 4),
        "score": round(score, 2),
    }

    dist   = mahalanobis_dist(Xfinal, train_mean, train_precision)
    zone   = mahal_zone(dist, cfg["mahal_thresh_caution"], cfg["mahal_thresh_ood"])
    is_ood = (zone == "ood")

    mahal_info = {
        "dist":           round(dist, 3),
        "zone":           zone,
        "thresh_caution": round(cfg["mahal_thresh_caution"], 3),
        "thresh_ood":     round(cfg["mahal_thresh_ood"],     3),
    }

    use_aug_gate     = cfg.get("use_augmentation_gate", False)
    crosses_boundary = False
    unstable         = False
    sensitive_only   = False

    boundary_thresh = cfg["boundary_threshold"]
    near_boundary   = abs(raw) < boundary_thresh

    high_conf_override = (
        veg_conf > 95.0
        and not near_boundary
        and not crosses_boundary
        and not is_ood
        and not class_inconsistent
    )

    if high_conf_override:
        score_unreliable    = False
        decision_unreliable = False
    else:
        score_unreliable    = unstable or is_ood
        decision_unreliable = (
            near_boundary
            or sensitive_only
            or (not veg_confident)
            or class_inconsistent
            or (conf_gap < 10)
        )

    state = ("UNRELIABLE" if score_unreliable
             else "TENTATIVE" if decision_unreliable
             else "RELIABLE")

    warnings = list(preflight_warnings)
    if high_conf_override:
        warnings.append(
            f"HIGH-CONFIDENCE OVERRIDE — veg_conf={veg_conf:.1f}%, forced RELIABLE.")
    if class_inconsistent:
        warnings.append(
            f"CLASS INCONSISTENCY — centroid ratio={centroid_ratio:.3f} "
            f"(threshold={centroid_ratio_t:.3f}). Global bounds applied.")
    if not veg_confident:
        warnings.append(
            f"Low veg confidence ({veg_conf:.1f}%, gap={conf_gap:.1f}%) "
            "— using global normalization.")
    if near_boundary:
        warnings.append(
            f"MODEL UNCERTAINTY — near decision boundary "
            f"(|raw|={abs(raw):.4f} < {boundary_thresh:.4f}).")
    if is_ood:
        warnings.append(
            f"OOD — Mahalanobis dist={dist:.3f} > "
            f"threshold={cfg['mahal_thresh_ood']:.3f}.")
    if zone == "caution":
        warnings.append(
            f"CAUTION — Mahalanobis dist={dist:.3f} in caution zone "
            f"[{cfg['mahal_thresh_caution']:.3f}, {cfg['mahal_thresh_ood']:.3f}].")

    band = confidence_band(score, cfg) if state == "RELIABLE" else None

    gates = [
        {
            "name":   "OOD Gate",
            "fired":  is_ood,
            "reason": (f"dist={dist:.3f} > {cfg['mahal_thresh_ood']:.3f}"
                       if is_ood
                       else f"dist={dist:.3f} — within distribution"),
        },
        {
            "name":   "Boundary Gate",
            "fired":  near_boundary,
            "reason": (f"|raw|={abs(raw):.4f} < {boundary_thresh:.4f}"
                       if near_boundary
                       else f"|raw|={abs(raw):.4f} — clear of boundary"),
        },
        {
            "name":   "Veg Confidence Gate",
            "fired":  not veg_confident,
            "reason": (f"conf={veg_conf:.1f}% gap={conf_gap:.1f}% — low"
                       if not veg_confident
                       else f"conf={veg_conf:.1f}% gap={conf_gap:.1f}%"),
        },
        {
            "name":   "Centroid Gate",
            "fired":  class_inconsistent,
            "reason": (f"ratio={centroid_ratio:.3f} > {centroid_ratio_t:.3f}"
                       if class_inconsistent
                       else f"ratio={centroid_ratio:.3f} — consistent"),
        },
    ]

    return {
        "state":                     state,
        "veg":                       veg_name,
        "veg_conf":                  round(veg_conf, 2),
        "conf_gap":                  round(conf_gap, 2),
        "score":                     round(score, 2),
        "raw":                       round(raw,   4),
        "fresh_label":               fresh_label_raw if state == "RELIABLE" else None,
        "freshness_confidence_band": band,
        "norm_source":               norm_source,
        "mahal_dist":                round(dist,  3),
        "mahal_zone":                zone,
        "warnings":                  warnings,
        "pipeline": {
            "preflight":          preflight,
            "features":           features_info,
            "veg_probs":          veg_all_probs,
            "veg_conf_thresh":    round(veg_conf_t, 1),
            "veg_gap_thresh":     round(veg_gap_t,  1),
            "veg_confident":      veg_confident,
            "freshness_raw":      round(raw, 4),
            "boundary_thresh":    boundary_thresh,
            "centroid":           centroid_info,
            "bounds":             bounds_info,
            "score_norm":         score_info,
            "mahal":              mahal_info,
            "gates":              gates,
            "high_conf_override": high_conf_override,
            "aug_gate_enabled":   use_aug_gate,
        },
    }