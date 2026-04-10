# EfficientNetB0 — Feature Extraction Architecture

This document covers three things: what EfficientNetB0 is and why it was chosen over larger variants, how it is used in this pipeline to produce deep features, and how the batch extraction architecture in `extract_dataset_features.py` keeps the 12,691-image extraction run under five minutes on CPU.

---

## 1. What EfficientNet Is

EfficientNet is a family of convolutional neural networks introduced in 2019 (Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"). The key contribution is **compound scaling**: rather than making a network deeper, wider, or higher-resolution in isolation, the method scales all three dimensions simultaneously using a fixed ratio. This produces a series of models (B0 through B7) where each step up increases parameters and FLOPs by a consistent factor.

B0 is the baseline — found by Neural Architecture Search (NAS) on ImageNet. B1 through B7 are progressively scaled versions of B0 using the compound coefficient.

---

## 2. B0 Through B3 — Specification Comparison

The first four variants are the practically relevant ones for transfer learning on CPU hardware.

| Model | Input size | Parameters | FLOPs | ImageNet Top-1 |
|-------|-----------|------------|-------|----------------|
| **EfficientNet-B0** | **224 × 224** | **~5.3 M** | **0.39 B** | **77.1%** |
| EfficientNet-B1 | 240 × 240 | ~7.8 M | 0.70 B | 79.1% |
| EfficientNet-B2 | 260 × 260 | ~9.1 M | 1.00 B | 80.1% |
| EfficientNet-B3 | 300 × 300 | ~12.2 M | 1.80 B | 81.6% |

What these numbers mean in practice:

**Parameters** scale the memory footprint. B3 has 2.3× the parameters of B0. On an Intel i7 12th-gen with 16 GB RAM, both fit comfortably, but B3 adds loading time at startup and heavier intermediate activations during inference.

**FLOPs** scale the per-image inference time. B3 requires 4.6× the floating-point operations of B0 per forward pass. At `BATCH_SIZE = 128`, this difference is substantial across 12,691 images.

**Input resolution** determines the detail EfficientNet can resolve in a single image. B3 at 300×300 can distinguish finer texture gradients than B0 at 224×224. For tasks like medical imaging where sub-millimetre detail matters, the extra resolution is valuable. For vegetable freshness from standard camera photos, colour statistics and surface texture at 224×224 fully capture the relevant signal — browning, discolouration, softening of edges — without requiring B3's resolution.

**Top-1 accuracy** on ImageNet measures classification performance on a 1000-class problem. This is useful context but not directly predictive of transfer learning performance on a 5-class binary task. The ImageNet accuracy gap between B0 (77.1%) and B3 (81.6%) is 4.5 percentage points on a very different problem. The actual transfer gap on vegetable freshness is smaller because the task is simpler.

---

## 3. Why B0 Was Chosen for This Project

The pipeline uses B0 as a **feature extractor**, not as a classifier. The classification head is discarded; only the global average pooling output is used. This means the accuracy gap between B0 and B3 on ImageNet is largely irrelevant — what matters is whether the internal representations are discriminative enough for the downstream SVM to separate fresh from rotten produce.

Three practical reasons support B0:

**Inference cost.** The extraction run processes 12,691 images. B0 takes approximately 4 minutes 49 seconds on an Intel i7 12th-gen CPU (from the actual run: 100 batches at 2.90s/it). B3 would require roughly 4.6× longer — over 22 minutes — for the same dataset, with the same SVM accuracy result, because the SVM operates on 1280-dimensional pooled features in both cases.

**Feature dimension is identical.** Both B0 and B3 use global average pooling over the last convolutional block, producing a 1280-dimensional vector regardless of which variant is used. The SVM sees 1280 numbers either way. The difference is only in what patterns those 1280 numbers capture — and for produce freshness (colour shifts, texture degradation, edge softening), B0's representations are sufficient.

**Augmentation re-inference cost.** During threshold calibration in `train_svm.py`, EfficientNet is run six additional times per sampled validation image (one per augmentation). This runs 380 images × 6 augmentations = 2,280 forward passes. With B3, this calibration step would take approximately 4.6× longer with no measurable improvement to the gate thresholds, since the augmentation gate evaluates score stability — not absolute accuracy.

The result: vegetable test accuracy of **99.61%**, freshness test accuracy of **98.94%**, and freshness ROC-AUC of **0.9994** — all achieved with B0 in under 5 minutes of extraction time.

---

## 4. How EfficientNetB0 Is Used in This Pipeline

The model is configured without its classification head. The global average pooling layer becomes the output:

```python
# extract_features.py

_DEEP_MODEL = EfficientNetB0(
    weights="imagenet",       # pretrained on ImageNet — weights are frozen
    include_top=False,        # remove the 1000-class Dense + Softmax head
    pooling="avg"             # GlobalAveragePooling2D becomes the output layer
)
```

```
EfficientNetB0 architecture as used in this pipeline:

  Input [batch × 224 × 224 × 3]
        │
  Stem Conv (3×3, stride 2)                    →  112 × 112 × 32
        │
  MBConv Block 1  (3×3, ×1, expand=1)          →  112 × 112 × 16
        │
  MBConv Block 2  (3×3, ×2, expand=6)          →   56 × 56 × 24
        │
  MBConv Block 3  (5×5, ×2, expand=6)          →   28 × 28 × 40
        │
  MBConv Block 4  (3×3, ×3, expand=6)          →   14 × 14 × 80
        │
  MBConv Block 5  (5×5, ×3, expand=6)          →   14 × 14 × 112
        │
  MBConv Block 6  (5×5, ×4, expand=6)          →    7 × 7 × 192
        │
  MBConv Block 7  (3×3, ×1, expand=6)          →    7 × 7 × 320
        │
  Top Conv (1×1)                                →    7 × 7 × 1280
        │
  Global Average Pooling    ← OUTPUT LAYER HERE
        │
  [1280-dimensional vector per image]
        │
  (Dense 1000 + Softmax)    ← REMOVED (include_top=False)
```

The 1280 output values are not individual features with interpretable meaning. They are the spatial average of 1280 learned filter responses across the 7×7 final feature map. What the model has learned — through ImageNet pretraining — is to detect edges at various orientations, colour gradients, texture patterns (smooth, rough, mottled), and surface characteristics that distinguish object classes. These general visual features transfer well to distinguishing fresh vegetables (uniform colour, crisp edges, smooth surface) from rotten ones (patchy discolouration, soft edges, irregular texture).

### EfficientNet preprocessing

Images cannot be fed directly to EfficientNetB0 as raw pixel values. The model was trained with a specific normalisation scheme, and inference must use the same scheme:

```python
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# Applies the EfficientNet-specific normalisation:
# rescales [0, 255] uint8 to the range expected by the model
batch_preprocessed = effnet_preprocess(batch_np)
```

This is separate from `StandardScaler`, which is applied later to the full 1312-dimensional feature vector. The EfficientNet preprocessing happens before the model sees the image. `StandardScaler` normalises the extracted features before the SVM.

---

## 5. The Full 1312-Dimensional Feature Vector

EfficientNet's 1280 values are concatenated with 32 handcrafted features that capture domain-specific freshness signals:

```
Feature vector construction per image:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  DEEP FEATURES [0 : 1280]                                               │
  │  EfficientNetB0 GlobalAvgPool output                                    │
  │  Captures: texture, shape, surface appearance, colour gradients          │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  HANDCRAFTED FEATURES [1280 : 1312]                                     │
  │                                                                         │
  │  [1280–1282]  RGB mean  (R̄, Ḡ, B̄)  — average colour per channel        │
  │  [1283–1285]  RGB std   (σR, σG, σB) — colour variance per channel      │
  │  [1286–1288]  HSV mean  (H̄, S̄, V̄)  — hue, saturation, brightness      │
  │  [1289–1291]  HSV std   (σH, σS, σV) — variation in tone                │
  │  [1292]       Grayscale mean          — overall brightness              │
  │  [1293]       Grayscale std           — contrast                        │
  │  [1294]       Edge density            — Canny(100,200) / total pixels   │
  │  [1295]       Laplacian variance      — whole-image sharpness           │
  │  [1296–1303]  Luminance histogram     — 8-bin brightness distribution   │
  │  [1304–1311]  Zero-padding            — pads to exactly 32              │
  └─────────────────────────────────────────────────────────────────────────┘
                              │
                    np.concatenate(axis=0)
                              │
                    [1312-dimensional float32 vector]
```

### Why handcrafted features alongside deep features

EfficientNet is optimised for classifying object identity across 1000 ImageNet categories. Freshness-specific signals — the shift from saturated green to dull yellow in a cucumber, the decrease in Laplacian variance as a potato softens — are not the primary signals EfficientNet was trained to detect. Adding handcrafted features that directly measure these properties gives the downstream SVM explicit access to signals that may be diffuse or entangled in the 1280 deep feature dimensions.

Concretely: a fresh cucumber has high edge density (firm skin, clear ridges), high Laplacian variance (sharp), and a narrow HSV std (uniform green). A rotten cucumber has low edge density (softened skin), low Laplacian variance (blurry texture), and a wider HSV std (yellowing patches). These are direct, computable measurements that the SVM can use without needing to discover them implicitly in 1280-dimensional space.

---

## 6. Batch Extraction Architecture

The most computationally expensive part of the pipeline is running EfficientNet across all 12,691 images. The architecture separates this into two distinct phases — I/O-bound image loading and compute-bound inference — and handles each optimally.

### Why the separation matters

Loading an image from disk involves a disk read, JPEG decoding, BGR-to-RGB conversion, and a resize. These operations are I/O-bound and memory-bound, not compute-bound. Running them sequentially means the CPU spends most of its time waiting for disk I/O while the model sits idle.

Running EfficientNet on a single image at a time is also inefficient: the model's internal matrix multiplications (convolutions are implemented as matrix-matrix products via `im2col`) are most efficient when operating on large matrices. A single 224×224×3 image produces a very small matrix. A batch of 128 images produces a matrix 128× larger, allowing the MKL/oneDNN libraries that TensorFlow uses on CPU to apply SIMD vectorization and cache-optimised execution paths.

### The actual pipeline

```
vegetable_Dataset/
  ├── freshapple/      ├── rottenpotato/
  ├── rottenapple/     ├── freshcapsicum/ ...
  └── freshbanana/ ...
        │
        ▼
  PHASE 1 — Scanning (single-threaded, fast)
  ─────────────────────────────────────────
  Iterate sorted folder names
  parse_folder("freshbanana") → ("banana", 1)
  parse_folder("rottenpotato") → ("potato", 0)
  Filter: veg not in TARGET_VEGETABLES → skip
  Collect image_paths[], y_veg[], y_fresh[]
  Result: 12,691 paths with labels
        │
        ▼
  PHASE 2 — Parallel image loading (ThreadPoolExecutor)
  ──────────────────────────────────────────────────────
  For each batch of 128 paths:

    executor.map(load_image, batch_paths)   ← NUM_WORKERS threads
      │                                        (os.cpu_count() on this machine)
      │   Each thread:
      │     cv2.imread(path)         ← disk read
      │     COLOR_BGR2RGB            ← channel swap
      │     cv2.resize((224, 224))   ← resize
      │     return uint8 RGB array
      │
      ▼
    Filter None results (failed loads removed from batch)
    Stack into np.array([128, 224, 224, 3], dtype=float32)
        │
        ▼
  PHASE 3 — EfficientNet batch inference (single call)
  ─────────────────────────────────────────────────────
    effnet_preprocess(batch_np)           ← normalise in-place
    model.predict(batch, verbose=0)       ← one forward pass for 128 images
    Output: (128, 1280) float32
        │
        ▼
  PHASE 4 — Handcrafted features (per-image, sequential)
  ────────────────────────────────────────────────────────
    For each of the 128 images:
      extract_handcrafted(img)    ← RGB/HSV stats, Canny, Laplacian, histogram
      Output: (32,) float32
        │
        ▼
  PHASE 5 — Concatenate and accumulate
  ─────────────────────────────────────
    feats = np.concatenate([deep_features[i], handcrafted])   (1312,)
    X.append(feats)
        │
        ▼
  Next batch of 128 images → repeat Phases 2–5

        │
        ▼
  After all 100 batches:
    X = np.array(X, dtype=float32)    shape (12691, 1312)
    np.save("Features/X.npy", X)
    np.save("Features/y_veg.npy", ...)
    np.save("Features/y_fresh.npy", ...)
    np.save("Features/image_paths.npy", ...)
```

### What runs in parallel and what does not

| Operation | Parallelism | Reason |
|-----------|------------|--------|
| Image loading (disk read + decode + resize) | `ThreadPoolExecutor`, `os.cpu_count()` threads | I/O bound; threads release GIL during I/O |
| EfficientNet forward pass | TensorFlow internal (oneDNN thread pool) | Compute bound; best handled by one large matrix op |
| Handcrafted feature extraction | Sequential (per image in batch) | Fast OpenCV ops; overhead of parallelising exceeds gain |
| np.concatenate + X.append | Sequential | Memory operation; no benefit from parallelism |

The image loading and EfficientNet inference are **not concurrent** — loading completes first for each batch, then inference runs. This is correct: TensorFlow's internal thread pool fully utilises the CPU during the forward pass, and mixing disk I/O threads with inference threads would cause cache thrashing that degrades both.

### Batch size

```python
BATCH_SIZE = 128   # extract_dataset_features.py
```

128 is the batch size for dataset extraction. This controls how many images are stacked into a single `model.predict()` call. Larger batches give better matrix operation efficiency up to the point where the batch no longer fits in L3 cache or RAM — at 128 images × 224 × 224 × 3 × 4 bytes ≈ 77 MB float32, this is well within the 16 GB RAM available.

For single-image inference in `predict_cli.py`, no batching is used — the augmentation gate runs 6 individual forward passes one at a time, which is correct since each augmented view must be scored independently.

### Actual timing (from the training run)

```
Batches: 100%|████████████████████████| 100/100 [04:49<00:00,  2.90s/it]
Saved feature matrix: (12691, 1312)
```

100 batches × 2.90 seconds per batch = 290 seconds total ≈ 4 minutes 49 seconds. Each batch processes 128 images: 128 parallel image loads + one EfficientNet forward pass on 128 images + 128 sequential handcrafted extractions.

The CUDA error printed at startup (`Failed call to cuInit: UNKNOWN ERROR (303)`) is expected — there is no GPU on this machine. TensorFlow detects this and falls back to CPU automatically. The oneDNN warnings are informational only and do not affect results or timing.

---

## 7. Single-Image Inference Path

For `predict_cli.py`, the extraction path is different from dataset extraction. There is no batching because only one image is processed:

```python
# extract_features.py

def extract_features(path: str) -> np.ndarray:
    img = _read_rgb(path, _DEEP_INPUT_SIZE)          # load + resize

    deep_in = effnet_preprocess(
        np.expand_dims(img.astype(np.float32), 0)    # shape (1, 224, 224, 3)
    )
    deep_feat = _DEEP_MODEL.predict(deep_in, verbose=0)[0]  # shape (1280,)

    handcrafted = extract_handcrafted_from_array(img)       # shape (32,)

    return np.concatenate([deep_feat, handcrafted], axis=0) # shape (1312,)
```

When the augmentation gate is active (`use_augmentation_gate=True`), `augment_and_score()` in `predict_cli.py` runs this same path six additional times — one per augmented view. This is the reason the gate adds significant latency on CPU: six EfficientNetB0 forward passes on a single image each take roughly the same time as one, so the total inference time increases approximately 6×.

---

## 8. Why the Model Weights Are Never Updated

EfficientNetB0 in this pipeline is a **frozen feature extractor**. The weights loaded from ImageNet pretraining are never modified — no fine-tuning, no gradient computation, no weight updates.

This is a deliberate design decision. Fine-tuning would require:
- A much larger vegetable dataset (12,691 images is too small to fine-tune 5.3 million parameters safely without overfitting)
- A GPU for practical training time
- A different training loop, loss function, and optimiser configuration

Instead, the pipeline uses the observation that ImageNet-pretrained features — textures, colour gradients, edge patterns — are general enough that a linear-boundary classifier (SVM) trained on top of them achieves high accuracy without any gradient signal touching EfficientNet at all. The vegetable test accuracy of 99.61% and freshness ROC-AUC of 0.9994 confirm this approach works well for this dataset and task.

The SVM is what learns the freshness and vegetable decision boundaries. EfficientNet simply provides a stable, high-quality representation of each image for the SVM to operate on.

---

## 9. Choosing Between B0 and Larger Variants

If the project were to retrain with a different EfficientNet variant, the decision matrix is:

| Consideration | Use B0 | Use B1/B2 | Use B3 |
|--------------|--------|-----------|--------|
| Inference time constraint | strict (< 5 min CPU) | moderate | relaxed (GPU available) |
| Augmentation gate active | yes (6× per image) | — | — |
| Current accuracy already sufficient | yes (99.61% / 98.94%) | — | — |
| Dataset size for fine-tuning | < 50k images | 50–200k | > 200k |
| Input detail needed | 224px sufficient | 240–260px beneficial | 300px needed |

For this project — 12,691 images, CPU-only inference, augmentation gate requiring repeated forward passes, and accuracy already near ceiling — B0 is the correct choice. Moving to B1 or B2 would increase extraction time and augmentation calibration time substantially with no expected improvement in the SVM's downstream accuracy.

The upgrade case would arise if the dataset grew substantially (say, to 100k+ images with more vegetable classes and more challenging imaging conditions) or if a GPU became available and fine-tuning became practical. In that scenario, B2 or B3 with fine-tuning on the final few blocks would be worth investigating.