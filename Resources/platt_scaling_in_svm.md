# Platt Scaling in SVMs

## What Platt Scaling Is

Platt scaling is a post-hoc calibration technique that converts the raw margin output of a trained Support Vector Machine into a probability estimate. Given a sample $x$, the SVM's `decision_function` returns $f(x)$, the signed distance from the decision boundary. Platt scaling fits a sigmoid function over this output:

$$P(y = 1 \mid x) = \frac{1}{1 + e^{Af(x) + B}}$$

where $A$ and $B$ are scalar parameters learned by fitting a logistic regression on the decision values and true labels — typically on a held-out validation set or via cross-validation to prevent overfitting. The result is a value in $(0, 1)$ that behaves superficially like a probability: a margin of $+3.0$ might map to $0.98$, a margin of $0.0$ maps to $0.50$, and a margin of $-3.0$ maps to $0.02$.

---

## Why SVMs Do Not Produce Probabilities Natively

SVMs are not probabilistic models. Training an SVM minimises a hinge loss subject to a margin-maximisation constraint — it makes no assumption about the likelihood of the data, the shape of the class-conditional distributions, or the posterior $P(y \mid x)$. There is no principled connection between the magnitude of $f(x)$ and any probability. The margin tells you how far a sample is from the decision boundary in the transformed feature space; it does not tell you how likely the prediction is to be correct.

Platt scaling exists precisely because practitioners want probability-like outputs from a model that was never designed to produce them. It is a pragmatic post-processing layer, not a theoretically grounded extension of the SVM.

---

## Limitations

**Assumes a sigmoid shape.** Platt scaling is only well-calibrated when the relationship between margin magnitude and prediction correctness is approximately sigmoidal. If the true score-to-probability curve has a different shape — a flat plateau, a bimodal structure, or sharp edges near the boundary — a sigmoid fit will produce systematically miscalibrated outputs.

**Overfitting risk.** If the sigmoid parameters are estimated on the same data the SVM was trained on, the calibrated probabilities will be overconfident. The parameters learn to explain noise in the training margins rather than the true underlying relationship. This is why calibration must always be performed on a held-out set.

**Distribution shift.** The sigmoid parameters are tied to the marginal distribution of $f(x)$ on the calibration set. If the test distribution differs — different lighting conditions, camera angles, or seasonal variation — the mapping learned by Platt scaling becomes invalid even if the SVM itself generalises reasonably well.

**Multi-class instability.** For multi-class SVMs, Platt scaling is applied pairwise to each binary sub-classifier, and the resulting estimates are then combined using a voting or normalisation scheme. This process is both mathematically ad hoc and numerically unstable in regions where the pairwise margins conflict with one another.

---

## Isotonic Calibration as an Alternative

Isotonic regression is a non-parametric alternative to Platt scaling. Rather than assuming the score-to-probability relationship follows a sigmoid, it fits a piecewise-constant monotone function from the data directly. This makes it strictly more expressive: any sigmoid relationship is also a monotone relationship, but not the reverse.

The tradeoff is sample efficiency. Isotonic calibration requires more data to produce a stable estimate than a two-parameter sigmoid fit. On small calibration sets, Platt scaling can actually outperform isotonic regression simply because it has fewer degrees of freedom to overfit.

In this project, the vegetable classifier uses isotonic calibration via `CalibratedClassifierCV` with `FrozenEstimator`. The base SVC weights are frozen — not retrained — and the isotonic layer is fit exclusively on `cal_val`, which is held out from the threshold selection data (`thr_val`). This disjoint split ensures that the calibrated probabilities do not encode information about the downstream gate thresholds.

---

## Why This Project Uses Raw Margins for Freshness Scoring

The freshness SVM is trained with `probability=False`, and its `decision_function` output is used directly as the scoring signal rather than being passed through a calibration layer. This is a deliberate design choice.

The goal of the freshness score is to rank samples within a vegetable class by their relative position in the learned feature space — fresh samples should score higher than rotten ones. What matters is the *ordering* induced by $f(x)$, not its magnitude in probability units. The raw margin already provides a well-defined ordering: the further a sample is from the decision boundary on the fresh side, the more confidently it occupies the fresh region of feature space. Passing the margin through a sigmoid would preserve this ordering but introduce a distortion in the spacing between scores — one calibrated to the class balance and distribution of the calibration set, not to any independently verifiable property of vegetable freshness.

The normalisation applied in this project — mapping the raw margin to a 0–100 score using the per-vegetable $p_5$ and $p_{95}$ percentiles of validation-set decision values — is also a post-processing step, but a deliberately limited one. It rescales the margin to a human-readable range while preserving the original ordering exactly. It does not claim to estimate $P(\text{fresh} \mid x)$; it claims only that a score of 80 places the sample in the upper quintile of that vegetable's validation-set margin distribution. This is a weaker and more defensible claim than a calibrated probability, and it is the appropriate claim given that only binary labels (fresh/rotten) are available and no continuous freshness ground truth exists to calibrate against.

The practical implication is that the score must not be interpreted as a probability. A banana scoring 75 is not "75% likely to be fresh" — it is a banana whose freshness SVM margin falls in the upper half of the range observed across banana validation samples. Whether that constitutes a meaningful freshness signal depends on the quality of the underlying binary labels and the consistency of imaging conditions, both of which are stated explicitly as limitations in the evaluation output.