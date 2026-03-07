Your current pipeline processes images like this:

```
image1 → EfficientNet → features
image2 → EfficientNet → features
image3 → EfficientNet → features
...
```

This is called **single-sample inference**.

Even if you use threads, the model is still executed **one image at a time**, which wastes CPU vectorization and cache efficiency.

---

# The fast approach (theory)

The correct approach is **vectorized batch inference**.

Instead of sending one image to the network, we send **many images at once**.

```
64 images
↓
EfficientNet
↓
64 feature vectors
```

So the pipeline becomes:

```
disk
↓
parallel image loading
↓
batch of images (size = B)
↓
EfficientNet forward pass
↓
B feature vectors
```

---

# Why batching is faster

Neural networks are implemented as **matrix multiplications**.

Single image:

```
1 × 224 × 224 × 3
```

Batch of 64 images:

```
64 × 224 × 224 × 3
```

Matrix math libraries (MKL / oneDNN) are optimized for **large matrices**.

So they run much faster when operations are grouped.

---

# CPU execution model

On CPU, TensorFlow uses:

```
SIMD vectorization
+
thread pools
+
cache optimization
```

Batching allows all three to work efficiently.

Without batching, CPU spends time:

```
loading weights
switching context
small matrix operations
```

instead of doing large optimized math.

---

# What we parallelize

Efficient pipeline separates two tasks:

### 1. I/O bound

Reading images from disk.

```
load image
resize
decode
```

This is slow because of disk access.

We parallelize it with threads.

---

### 2. Compute bound

Running EfficientNet.

```
convolutions
matrix multiplications
```

TensorFlow already parallelizes this internally.

So we run **one large batch**, not multiple small jobs.

---

# Final optimized architecture

```
Dataset
↓
parallel image loader
↓
queue of images
↓
batch builder (size B)
↓
EfficientNet inference
↓
1280 deep features
+
32 handcrafted features
↓
1312 feature vector
↓
save to X.npy
```

---

# Why this matters for your dataset

You have:

```
36,000 images
```

Single inference:

```
~60–90 minutes
```

Batch inference:

```
~15–25 minutes
```

On CPU.

---

# Why GPU pipelines are even faster

GPU executes batches extremely efficiently.

Example:

```
batch = 256
↓
36k images processed in ~2 minutes
```

But on CPU, batch sizes are smaller.

---

# The key performance variables

Three parameters determine speed:

```
batch_size
number_of_loader_threads
image preprocessing cost
```

Typical optimal values for your machine:

```
batch_size = 64
threads = number_of_cpu_cores
```

---

# Important design rule

Always separate:

```
I/O pipeline
+
model inference pipeline
```

Never mix them.

Your current code mixes both, which reduces throughput.

---
