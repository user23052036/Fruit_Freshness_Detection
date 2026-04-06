# Autoencoders — Concepts, Comparisons, and Project Notes

This document covers autoencoders from first principles: what they are, how they work, how they compare to PCA, why they were considered for this project, and what the design of an autoencoder-based freshness pipeline would look like — including the key problems it would face and how they would be addressed.

These are notes from an earlier design phase. The current pipeline uses EfficientNetB0 features + dual RBF SVM (achieving 98.94% freshness accuracy and 0.9994 ROC-AUC). This document records the autoencoder approach as an alternative that was explored but not implemented.

---

## 1. What an Autoencoder Is

An autoencoder is a neural network trained to do one thing: take an input, compress it into a small representation, and then reconstruct the original input from that compressed form as accurately as possible.

It has no external labels. There is no "fresh" or "rotten" supervision signal during training. The only training signal is: *can you reconstruct what you were given?*

The network has two parts:

```
              ENCODER                         DECODER
              ───────                         ───────

Input       ┌─────────┐   Bottleneck    ┌─────────┐   Output
[1312] ────►│  Dense  │────►[64 dims]───►│  Dense  │────►[1312]
            │ layers  │   (latent       │ layers  │   (reconstruction)
            └─────────┘    space)       └─────────┘

                              ↑
                    This is the compressed
                    representation of the image
```

The encoder learns to summarise the input into a small fixed-size vector — called the **bottleneck** or **latent vector**. The decoder learns to reconstruct the original input from that compressed form. The gap between the input and the reconstruction is called the **reconstruction error**, typically measured as Mean Squared Error (MSE):

```
reconstruction_error = mean( (input − output)² )
```

After training, you can use the autoencoder in two ways:

1. **Use the bottleneck** as a compressed feature vector and feed it to a downstream classifier (like an SVM).
2. **Use the reconstruction error** directly as a score — high error means the image is unlike what the model learned from.

For freshness grading, option 2 is particularly elegant: train the autoencoder only on *fresh* vegetables. It learns what fresh looks like. When a rotten vegetable is fed in, the model struggles to reconstruct it (because rotten vegetables were never in training), producing high reconstruction error → low freshness score.

---

## 2. A Concrete Example

Suppose the autoencoder is trained only on fresh banana images. During training it learns to reconstruct the specific EfficientNet feature pattern of a fresh banana: smooth yellow skin, firm edges, uniform colour.

```
TRAINING (fresh bananas only):

  Fresh banana features [1312]
       │
  Encoder → compress → [64]
       │
  Decoder → reconstruct → [1312]
       │
  Compare to original → error is small (model reconstructs well)
  → Adjust weights to make reconstruction even better


INFERENCE (fresh vs rotten):

  Fresh banana:
    features [1312] → encoder → [64] → decoder → [1312]
    reconstruction error = 0.03  ← low  → high freshness score

  Rotten banana:
    features [1312] → encoder → [64] → decoder → [1312]
    reconstruction error = 0.41  ← high → low freshness score

  Different vegetable (potato):
    features [1312] → encoder → [64] → decoder → [1312]
    reconstruction error = 0.87  ← very high → rejected / wrong class
```

The freshness score would be computed from the error:

```
score = 100 × exp(−error / τ)
```

where τ (tau) is a temperature parameter that controls how steeply the score drops with reconstruction error. This needs to be calibrated on validation data.

---

## 3. The Architecture in Detail

```
    Input [1312 features — EfficientNet 1280 + handcrafted 32]
          │
          ▼
    ┌─────────────────┐
    │    ENCODER      │
    │                 │
    │  Dense(512)     │  ← 1312 → 512 (compress)
    │  ReLU           │
    │                 │
    │  Dense(256)     │  ← 512 → 256
    │  ReLU           │
    │                 │
    │  Dense(64)      │  ← 256 → 64  (bottleneck)
    └────────┬────────┘
             │
        ┌────▼────┐
        │  LATENT │  64-dimensional vector
        │  SPACE  │  "soul" of the vegetable image
        └────┬────┘
             │
    ┌────────▼────────┐
    │    DECODER      │
    │                 │
    │  Dense(256)     │  ← 64 → 256
    │  ReLU           │
    │                 │
    │  Dense(512)     │  ← 256 → 512
    │  ReLU           │
    │                 │
    │  Dense(1312)    │  ← 512 → 1312 (reconstruct)
    │  Sigmoid        │
    └─────────────────┘
          │
          ▼
    Output [1312 — reconstruction of input]
          │
          ▼
    Loss = MSE(input, output)
    → Backprop → update all Dense layer weights
```

The bottleneck dimension (64 here) is a hyperparameter. Too large and the network can memorise inputs without learning anything meaningful. Too small and it cannot capture enough structure to reconstruct inputs accurately. Typical choices for a 1312-dimensional input: 64–256.

---

## 4. Autoencoders vs PCA — What They Share and Where They Differ

Both PCA and autoencoders are dimensionality reduction methods — they compress a high-dimensional input into a lower-dimensional representation. Understanding the difference matters for choosing which to use.

### What PCA Does

PCA finds the directions of maximum variance in the data and projects the input onto those directions. Each direction is a linear combination of the original features:

```
PCA projection of feature vector x:
  z = W × x
  where W is a matrix of eigenvectors (principal components)

  z[1] = 0.3 × feature_1 + 0.1 × feature_2 + 0.8 × feature_3 + ...
  z[2] = 0.7 × feature_1 − 0.2 × feature_2 + 0.1 × feature_3 + ...
  ...
```

Each component is a weighted sum — a straight-line combination. PCA cannot represent "feature_1 AND feature_2 together produce signal X". It can only represent linear blends.

### What an Autoencoder Does

An autoencoder learns a non-linear mapping from input to bottleneck and back. The Dense + ReLU layers can represent complex, conditional relationships:

```
Autoencoder bottleneck dimension z[3]:
  Might encode something like:
  "High IF brownish colour AND wrinkled texture AND low edge density"
  — not expressible as any linear combination of individual features
```

A single bottleneck neuron in an autoencoder can respond to a combination of input features in a non-linear way. PCA cannot do this.

### Side-by-Side Comparison

```
Property                PCA                         Autoencoder
────────────────────────────────────────────────────────────────
Transformation type     Linear only                 Non-linear (ReLU layers)
Conditional logic       Cannot capture              Can capture
Training                Closed-form (eigendecomp)   Gradient descent (slow)
Interpretability        Components interpretable    Bottleneck is opaque
Reconstruction          Always exact on training    Approximate
Class label required    No                          No
GPU needed              No                          Helpful (not required)
Risk of overfitting     None                        Yes, requires regularisation
Bottleneck dimension    Fixed by eigenvector rank   Tunable hyperparameter
Freshness signal        Finds variance directions   Finds reconstruction-relevant
                        (may not align with         structure (trained to care
                        freshness)                  about the specific domain)
```

### Why This Matters for Freshness Grading

PCA finds the directions of most variance across the dataset. In a vegetable freshness dataset, the direction of most variance is likely *which vegetable it is* (the feature patterns for banana vs potato are very different), not *how fresh it is* (the fresh/rotten variation within a class is smaller). PCA might compress away the freshness signal entirely while preserving vegetable identity.

An autoencoder trained only on fresh images of one class has no such problem — it is optimised purely to reconstruct "what fresh looks like" for that class. Freshness variation becomes the signal it must encode, not a casualty of variance maximisation.

---

## 5. How the Autoencoder Approach Would Fit This Pipeline

In the autoencoder design, the pipeline would change as follows:

```
CURRENT PIPELINE (EfficientNet + dual SVM):

  Image → EfficientNetB0 → [1280] + handcrafted [32] → [1312]
        → VarianceThreshold + StandardScaler → [1304]
        → XGBoost feature selection → [349]
        → Vegetable SVM → vegetable label + confidence
        → Freshness SVM → raw margin → normalized score [0–100]
        → Reliability gates → RELIABLE / TENTATIVE / UNRELIABLE


AUTOENCODER PIPELINE (alternative design):

  Training phase:
    For each vegetable class:
      Filter training set → only FRESH samples of that vegetable
      Train Autoencoder on fresh features → banana_ae, potato_ae, ...

  Inference phase:
    Image → EfficientNetB0 → [1280] + handcrafted [32] → [1312]
          → Vegetable SVM → predicted class (e.g., "banana")
          → Load banana_ae
          → banana_ae.encode(features) → [64]
          → banana_ae.decode([64]) → [1312] (reconstruction)
          → error = MSE(original, reconstruction)
          → score = 100 × exp(−error / τ)
          → Reliability gate: is error unusually high?
```

The vegetable SVM would still be needed — the autoencoder approach still requires knowing which vegetable's expert model to use.

---

## 6. Why One Autoencoder Per Vegetable

A single autoencoder trained on all five vegetable classes would learn a blurry compromise representation — what an "average fresh vegetable" looks like — rather than the specific texture and colour patterns of any particular class.

```
SINGLE AE (bad idea):

  Training data: fresh apple + fresh banana + fresh capsicum + ...
                 │
                 ▼
  AE learns: "average freshness pattern"
             ↑
             This does not exist — fresh apple and fresh banana
             have almost nothing in common in feature space

  Consequence at inference:
    Fresh banana → high error (doesn't match "average")
    Fresh apple  → high error (same reason)
    Both appear "rotten" despite being fresh


PER-VEGETABLE AE (correct approach):

  banana_ae: trained only on fresh bananas
             → knows exactly what fresh banana feature patterns look like

  At inference on fresh banana:
    features → banana_ae → low error → high score ✓

  At inference on rotten banana:
    features → banana_ae → high error (rotten doesn't match what it learned) → low score ✓

  At inference on potato (mislabelled by veg_svm as banana):
    features → banana_ae → very high error (potato ≠ banana) → low score
    → also serves as an indirect signal of veg misclassification
```

The per-vegetable structure maps directly onto the existing pipeline's design: the vegetable SVM already predicts which class the input belongs to. That prediction selects which expert autoencoder to use. This is exactly the same routing logic used for per-vegetable normalization bounds in the current SVM pipeline.

**Training data requirement:** Each per-class autoencoder needs enough fresh training samples. With 12,691 total images and a roughly balanced dataset, each vegetable class has approximately 1,200–2,500 training images. Autoencoders typically require only a few hundred clean samples to train effectively on feature vectors, so this is sufficient for all five classes.

---

## 7. Key Problem — Domain Shift (Lighting and Imaging Conditions)

This is where the autoencoder approach introduces a fragility that the current SVM pipeline does not have.

The autoencoder is trained to reconstruct pixel-level or feature-level patterns exactly. If training images were taken under specific lighting conditions and inference images look different (different camera, different exposure, different colour temperature), the reconstruction error will be high even for fresh produce — not because the vegetable is rotten, but because the imaging conditions changed.

```
TRAINING CONDITIONS:          DEPLOYMENT CONDITIONS:
  Lab with controlled           Phone camera in kitchen
  white-balance lighting  vs.   with warm yellow lighting

  Fresh apple under lab light:  Fresh apple under kitchen light:
    feature vector A              feature vector B
    (specific R/G/B pattern)      (shifted R/G/B — looks yellower)

  apple_ae trained on A         Inference on B:
  → error = 0.04 (low)          → error = 0.38 (HIGH!)
  → score = 96 (fresh)          → score = 24 (appears rotten!)
```

The model is not detecting rot — it is detecting a change in lighting. This is called **domain shift**: the distribution of inference images is different from the distribution of training images.

### The Mitigation: Augmentation During Training

The solution is to intentionally expose the autoencoder to this variation during training so that it learns to reconstruct across lighting conditions:

```
AUGMENTED TRAINING PROCEDURE:

  For each fresh banana image in training:
    Generate multiple augmented versions:
      ├── Brightness × 1.15    (simulate brighter exposure)
      ├── Brightness × 0.85    (simulate darker exposure)
      ├── Hue shift ±10°       (simulate colour temperature change)
      ├── Saturation × 0.8     (simulate desaturated camera)
      └── Gaussian noise       (simulate sensor noise)

    Train the autoencoder to reconstruct the ORIGINAL image
    from each of these augmented versions.

  Effect:
    The bottleneck must encode "fresh banana freshness"
    in a way that is invariant to lighting changes —
    because it has to reconstruct the original from
    many different-looking versions.
```

After augmented training:
- Reconstruction error becomes insensitive to lighting variation (the model learned to ignore it).
- Reconstruction error becomes sensitive to freshness variation (the signal it was not augmented away from).

This is the same principle used in the current pipeline's augmentation stability gate — but applied to training rather than inference-time uncertainty detection.

### Residual Risk

Even with augmentation, domain shift cannot be fully eliminated. The augmentations used during training define what variation the model becomes robust to. If real deployment involves a variation not represented in training augmentations (e.g., extreme shadows, backgrounds with similar colours to the vegetable), the model may still produce misleading errors.

The current SVM pipeline is less susceptible to this because:
- EfficientNet features are already somewhat invariant to lighting (the network was pretrained on 1.2 million varied ImageNet images).
- The freshness SVM operates on margin distance, not pixel-level reconstruction fidelity.
- The Mahalanobis OOD gate provides a distributional alarm when inputs are too far from training conditions.

---

## 8. Using Reconstruction Error as a Freshness Score

The reconstruction error needs to be converted into a [0–100] score comparable to the current SVM-based score. The exponential decay function is the natural choice:

```
score = 100 × exp(−error / τ)

  error = 0.0  →  score = 100   (perfect reconstruction = perfectly fresh)
  error = τ    →  score = 37    (one "characteristic decay length")
  error = 2τ   →  score = 14
  error → ∞    →  score → 0

Where τ (tau) is calibrated on the validation set.
```

Calibrating τ requires finding the value where the fresh/rotten score distributions are best separated. In the current SVM pipeline, this calibration is done via p5/p95 normalization per vegetable. For autoencoders, τ would be fit separately per vegetable expert model.

```
CALIBRATION PROCEDURE:

  On val set, for each vegetable AE:
    1. Compute reconstruction error for all fresh val samples   → errors_fresh
    2. Compute reconstruction error for all rotten val samples  → errors_rotten
    3. Choose τ that maximises separation:
         e.g., τ = mean(errors_fresh) + std(errors_fresh)
         (95th percentile of fresh errors — above this means "rotten behaviour")

  RELIABILITY GATE equivalent:
    If error > (some high threshold):
      → UNRELIABLE (image is far outside what the AE learned,
                    possibly a different vegetable or OOD input)
```

---

## 9. Why the SVM Approach Was Kept Instead

The autoencoder approach was considered but the SVM pipeline was implemented instead. The key reasons:

**The SVM approach requires less data for the freshness model.** The freshness SVM is a discriminative model that directly learns the boundary between fresh and rotten from both classes. An autoencoder is a generative model that only sees fresh training data — it must infer "rottenness" from reconstruction failure. With ~1,200–2,500 images per class, the SVM has a data advantage.

**The SVM's margin is a better-calibrated score.** The raw SVM decision function value has a direct geometric interpretation (signed distance from the hyperplane). Reconstruction error from an autoencoder is harder to calibrate and sensitive to architecture choices (number of layers, bottleneck size, activation functions). The formal threshold selection process (`select_thresholds()`) would need to be redesigned for the reconstruction error domain.

**The current results are already near ceiling.** Freshness test accuracy of 98.94%, ROC-AUC of 0.9994, and a score delta of 70.73 points between fresh and rotten mean scores. There is limited headroom for improvement from a more complex model.

**Domain shift is managed differently.** The Mahalanobis OOD gate explicitly detects when an input is outside the training distribution — flagging it as UNRELIABLE before any score is produced. This is a cleaner solution than baking robustness into the model via augmentation.

---

## 10. Summary

| Property | Current SVM Pipeline | Autoencoder Alternative |
|----------|---------------------|------------------------|
| Feature source | EfficientNetB0 + handcrafted → [1312] | Same |
| Freshness signal | SVM decision margin (discriminative) | Reconstruction error (generative) |
| Training signal | Binary fresh/rotten labels | Fresh-only, no labels needed |
| Normalization | Per-veg p5/p95 bounds | Per-veg τ calibration |
| Domain shift sensitivity | Low (EfficientNet is robust) | High (requires augmented training) |
| OOD handling | Mahalanobis gate | Reconstruction error spike |
| Test accuracy (freshness) | **98.94%** | Not measured |
| ROC-AUC | **0.9994** | Not measured |
| Implementation complexity | Moderate | Higher (one model per vegetable, τ calibration, augmented training) |
| Data requirement | Both fresh and rotten needed | Fresh only (advantageous for new classes) |

The autoencoder approach has one significant advantage the SVM approach does not: it can detect freshness degradation without ever having seen a rotten example during training. For a new vegetable class where only fresh images are available, an autoencoder would still work — the SVM would not. If the project were extended to new vegetable classes with limited rotten training data, the autoencoder approach would be worth revisiting.