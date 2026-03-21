You are mixing up **three different jobs**:

1. **train the model**
2. **estimate the score scale**
3. **evaluate the final model**

They are not the same thing.

---

## The core mistake

You are asking:

> “Why not train on all 70%, then use the same training set to find the 5th and 95th percentile margins, then normalize test data?”

Because the margins on the training set are **not a fair estimate** of what the model will output on new data.

That is the whole point.

The model has already been optimized on those same points. So the training margins are usually **too optimistic**.

---

## What “optimistic” means here

Suppose after training, for one vegetable, the decision function on the **training set** looks like this:

```text
training margins:  [-4.8, -3.9, -3.2, -2.7,  2.9,  3.4,  4.1,  4.7]
```

So your percentiles become roughly:

* p5 ≈ -4.5
* p95 ≈ +4.5

Now look at **new unseen data**:

```text
validation/test margins: [-2.0, -1.2, -0.4, 0.6, 1.3, 1.8, 2.1]
```

These are much tighter.

If you normalize using training bounds `[-4.5, +4.5]`, then a real fresh sample with margin `+2.1` will not look very fresh. It will get pushed toward the middle.

That is the compression problem.

---

## Why training margins are larger

Because the model was fitted on those points.

SVM tries to place the boundary so that the **training points** are separated as well as possible. So the training set is not a neutral sample. It is a sample the model already “knows”.

That means the model can make training points look farther from the boundary than future points.

Not because the data magically changed. Because the model was optimized on them.

---

## Simple example

Imagine a one-dimensional case.

### Training data

Fresh samples: `+3, +4, +5`
Rotten samples: `-3, -4, -5`

The SVM boundary may be around `0`, so the training margins are wide.

### New data

Fresh samples: `+1, +1.5`
Rotten samples: `-1, -1.2`

These are still correctly classified, but they sit much closer to the boundary.

So if you take percentiles from training, you are scaling against an exaggerated spread.

Result: test scores look weaker than they should.

---

## Why validation fixes this

Validation data is not used to fit the model.

So its margin distribution is a better approximation of what happens on unseen data.

That is why validation bounds are used for calibration / normalization.

They answer this question:

> “What margin range should I expect from data the model did not train on?”

That is exactly what you need for score scaling.

---

## Why not just use all data for training and skip validation?

Because then you lose the only honest estimate of generalization.

If you use the same data for:

* fitting the model
* setting score bounds
* judging quality

you are double-dipping.

The pipeline looks good on paper and then fails on new images.

---

## Your proposed idea, cleaned up

You said:

> train on the data points, get the hyperplane with maximized margin, then compute p5/p95 of the original distribution and use those to normalize testing

This only works if “original distribution” means **fresh unseen data**.

But you do **not** have that distribution at training time.

You only have the observed dataset. And if you use the training part of it, the distribution is already biased by the model fitting process.

So the percentile bounds from training are not “original distribution bounds”. They are **training-output bounds**.

That is not the same thing.

---

## Proper split logic

For a clean pipeline:

* **Train set (70%)**: fit the SVM
* **Validation set (20%)**: estimate percentiles / tune normalization / choose thresholds
* **Test set (10%)**: final untouched evaluation

This is standard because each part has one job.

---

## If 20% feels like a waste

Then the real alternative is not “use training bounds”.

The real alternative is:

* use cross-validation
* generate out-of-fold decision scores
* compute percentiles from those out-of-fold scores

That gives you a much better estimate without wasting too much data.

---

## The shortest correct answer

You use validation bounds because **training margins are biased by the model having already seen those samples**. Validation gives a more realistic margin spread for unseen data, so your normalization does not compress all future scores toward the middle.

---

Good. Now we’ll remove all ambiguity with a **numerical toy example**. No theory, just numbers.

---

# Setup

We simulate your exact pipeline:

* Binary SVM → outputs `decision_function`
* Then you do:
  [
  \text{score} = \frac{f(x) - p5}{p95 - p5}
  ]

---

# Step 1 — Training margins (what model saw)

```text
TRAIN margins:
[-4.5, -3.8, -3.2, -2.9, 2.7, 3.1, 3.8, 4.4]
```

Compute percentiles:

```text
p5  ≈ -4.3
p95 ≈ +4.2
```

Range ≈ **8.5 (very wide)**

---

# Step 2 — Validation/Test margins (real-world behavior)

```text
VALIDATION margins:
[-2.0, -1.5, -0.8, 0.2, 0.9, 1.4, 1.8, 2.1]
```

Compute percentiles:

```text
p5  ≈ -1.9
p95 ≈ +2.0
```

Range ≈ **3.9 (much tighter)**

---

# Step 3 — Take ONE test sample

Let’s say a new image gives:

```text
f(x) = +2.0
```

This is clearly **fresh and strong**.

---

# Step 4 — Normalize using TRAINING bounds

[
\text{score} = \frac{2.0 - (-4.3)}{4.2 - (-4.3)} = \frac{6.3}{8.5} \approx 0.74
]

👉 Score = **74 / 100**

---

# Step 5 — Normalize using VALIDATION bounds

[
\text{score} = \frac{2.0 - (-1.9)}{2.0 - (-1.9)} = \frac{3.9}{3.9} = 1.0
]

👉 Score = **100 / 100**

---

# 🔴 This is the entire problem

Same model
Same input
Same margin

But:

| Method            | Score                      |
| ----------------- | -------------------------- |
| Training bounds   | 74 (looks mid-ish)         |
| Validation bounds | 100 (correctly very fresh) |

---

# Step 6 — Now check a rotten sample

Take:

```text
f(x) = -2.0
```

---

### Using training bounds:

[
\frac{-2.0 - (-4.3)}{8.5} = \frac{2.3}{8.5} \approx 0.27
]

👉 Score ≈ **27**

---

### Using validation bounds:

[
\frac{-2.0 - (-1.9)}{3.9} \approx 0
]

👉 Score ≈ **0**

---

# 🔴 Again:

| Method            | Score                     |
| ----------------- | ------------------------- |
| Training bounds   | 27 (not very rotten)      |
| Validation bounds | 0 (correctly very rotten) |

---

# Final conclusion (no fluff)

Training bounds:

* too wide
* inflate denominator
* compress everything toward middle

Validation bounds:

* match real margin distribution
* preserve contrast
* give meaningful scores

---

# The exact sentence you should remember

> Training margins are wider because the model was optimized on them, so using them for normalization makes real-world predictions look artificially less confident.

---

# Now I’ll push you harder

Your current system assumes:

> validation distribution ≈ real-world distribution

### What if:

* lighting changes
* camera changes
* background changes

Then:

```text
new margins → [-1.2, +1.2]
```

Your validation bounds become wrong again.

---

# Question for you (don’t skip)

If margins shrink further in production, what happens to your scores?

* Do they inflate?
* Do they collapse?
* Or become unstable?

Answer this, and I’ll tell you the next flaw in your system.
