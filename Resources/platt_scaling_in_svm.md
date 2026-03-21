
# What Platt Scaling actually is

**Platt Scaling = a calibration layer on top of SVM margins**

It converts this:
decision function output (margin)

into this:

$$
P(y=1 \mid x)
$$

using a **sigmoid function**.

---

## The core equation

Platt scaling fits:

$$
P(y=1 \mid x) = \frac{1}{1 + e^{(A f(x) + B)}}
$$

Where:

* ( f(x) ) = SVM `decision_function` output
* ( A, B ) = learned parameters (via logistic regression)

---

## What’s really happening (no sugarcoating)

You take:

```
SVM output → raw margin (distance from boundary)
```

Then you say:

> “Let’s *pretend* this behaves like probability by fitting a sigmoid.”

That’s it.

---

## How it’s trained

After training SVM:

1. Take decision values ( f(x) ) on training (or validation) data
2. Fit a logistic regression:

   * input = ( f(x) )
   * target = actual labels (0/1)
3. Learn parameters ( A ) and ( B )

---

## Why SVM doesn’t give probability natively

Because SVM is based on:

> **maximizing margin**, not modeling likelihood

There is:

* no probabilistic interpretation
* no distribution assumption like Naive Bayes or Logistic Regression

---

## Why Platt scaling exists

Because people *want* probabilities.

So instead of redesigning SVM, they slap on:

> a post-hoc calibration layer

---

## Where this breaks (this is what most people ignore)

### 1. Assumes sigmoid shape

If your data is not sigmoid-like:

→ probabilities are garbage

---

### 2. Overfitting risk

If calibration is done on same data:

→ overconfident probabilities

---

### 3. Distribution shift kills it

If test distribution ≠ train distribution:

→ calibration becomes invalid

---

### 4. Multi-class is worse

For multi-class SVM:

* Platt scaling is applied pairwise
* Then combined (messy, unstable)

---

## Concrete example

Say your SVM outputs:

| Margin (f(x)) | After Platt |
| ------------- | ----------- |
| +3.0          | 0.98        |
| +1.0          | 0.73        |
| 0.0           | 0.50        |
| -1.0          | 0.25        |
| -3.0          | 0.02        |

Looks nice. Feels like probability.

But:

👉 It is **just curve fitting**, not truth.

---

## Now connect this to YOUR system

You deliberately did:

> `probability=False`

That means:

* You trust **raw margins**
* You avoid fake probability semantics

That is **correct for your design**, but only if:

### You accept:

* Your score = relative confidence
* Not actual probability of freshness

---

## Hard question you must answer

Why is your percentile-normalized margin:

$$
\text{Score} = \text{normalized } f(x)
$$

**more meaningful than**:

$$
\text{sigmoid}(f(x))
$$

If you cannot justify that → your pipeline is arbitrary.

---

## Minimal takeaway

* Platt scaling = **sigmoid fit over SVM outputs**
* It creates **probability-like values**
* It is **not fundamental**, just a post-processing trick

---

If you want, I can:

* Compare **Platt vs isotonic calibration**
* Or show where Platt scaling completely fails in real ML pipelines

But don’t move ahead until you can explain:

> “Why margin ≠ probability” without hesitation.
