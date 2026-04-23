# src/extract_features.py
# EfficientNetB0 deep features (1280) + 32 handcrafted features -> 1312 total

import numpy as np
import cv2

# TensorFlow EfficientNet
try:
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
except Exception as e:
    raise ImportError("TensorFlow must be installed: pip install tensorflow") from e


# ----------------------------------------------------
# Global EfficientNet model (loaded once)
# ----------------------------------------------------

_DEEP_MODEL = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

_DEEP_INPUT_SIZE = (224, 224)


# ----------------------------------------------------
# Image loading
# ----------------------------------------------------

def _read_rgb(path: str, target_size=_DEEP_INPUT_SIZE) -> np.ndarray:

    img = cv2.imread(path)

    if img is None:
        raise IOError(f"Unable to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)

    return img  # uint8 RGB


# ----------------------------------------------------
# Handcrafted feature extraction
# ----------------------------------------------------

def extract_handcrafted_from_array(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract ~32 handcrafted features from RGB image.
    """

    h, w, _ = img_rgb.shape

    # normalize RGB
    arr = img_rgb.astype(np.float32) / 255.0

    # HSV + grayscale
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    feats = []

    # ---------------------------------
    # RGB statistics
    # ---------------------------------

    feats.extend(arr.mean(axis=(0, 1)).tolist())
    feats.extend(arr.std(axis=(0, 1)).tolist())

    # ---------------------------------
    # HSV statistics
    # ---------------------------------

    feats.extend(hsv.mean(axis=(0, 1)).tolist())
    feats.extend(hsv.std(axis=(0, 1)).tolist())

    # ---------------------------------
    # grayscale statistics
    # ---------------------------------

    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))

    # ---------------------------------
    # edge density
    # ---------------------------------

    edges = cv2.Canny((gray * 255).astype('uint8'), 100, 200)

    feats.append(float((edges > 0).sum()) / (h * w))

    # ---------------------------------
    # Laplacian variance (sharpness)
    # ---------------------------------

    lap = cv2.Laplacian((gray * 255).astype('uint8'), cv2.CV_64F)

    feats.append(float(lap.var()))

    # ---------------------------------
    # luminance histogram (8 bins)
    # ---------------------------------

    hist = cv2.calcHist(
        [(gray * 255).astype('uint8')],
        [0],
        None,
        [8],
        [0, 256]
    ).flatten()

    if hist.sum() > 0:
        hist = hist / hist.sum()
    else:
        hist = np.zeros_like(hist)

    feats.extend(hist.tolist())

    # pad to exactly 32 features
    while len(feats) < 32:
        feats.append(0.0)

    return np.array(feats[:32], dtype=np.float32)


# ----------------------------------------------------
# Single-image feature extraction
# ----------------------------------------------------

def extract_features(path: str) -> np.ndarray:
    """
    Extract full 1312 feature vector for one image
    """

    img = _read_rgb(path, _DEEP_INPUT_SIZE)

    deep_in = effnet_preprocess(np.expand_dims(img.astype(np.float32), 0))

    deep_feat = _DEEP_MODEL.predict(deep_in, verbose=0)[0].astype(np.float32)

    handcrafted = extract_handcrafted_from_array(img)

    return np.concatenate([deep_feat, handcrafted], axis=0)


# ----------------------------------------------------
# Public aliases used by other scripts
# ----------------------------------------------------

model = _DEEP_MODEL
preprocess_input = effnet_preprocess
extract_handcrafted = extract_handcrafted_from_array

__all__ = [
    "model",
    "preprocess_input",
    "extract_handcrafted",
    "extract_features",
]