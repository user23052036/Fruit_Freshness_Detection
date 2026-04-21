
**1. Briefly tell me about your project.**

I built an end-to-end machine learning pipeline for vegetable freshness classification and grading. The system takes a photo of a vegetable and answers two questions: what vegetable is it, and how fresh is it on a 0–100 scale. It uses EfficientNetB0 for deep feature extraction combined with 32 handcrafted visual features, giving a 1312-dimensional feature vector per image. Two separate SVM classifiers handle vegetable identification and freshness scoring respectively. What makes it more than a simple classifier is the formal reliability gating system — every prediction is tagged as RELIABLE, TENTATIVE, or UNRELIABLE based on Mahalanobis OOD detection, centroid consistency checking, and boundary proximity analysis. The final system achieves 99.61% vegetable accuracy and 98.94% freshness accuracy on a held-out test set of 2539 images, with a ROC-AUC of 0.9994.

---

**2. What challenges did you face and how did you overcome them?**

Several significant ones.

The first was **calibration leakage**. I was initially using the same validation set both to calibrate the SVM's probability outputs via isotonic regression and to select the reliability thresholds. This meant the threshold selection was seeing data whose probabilities had already been tuned to it, so thresholds appeared tight in validation but would have failed on genuinely new data. I fixed this by splitting the validation set 50/50 into two disjoint halves — cal_val for probability calibration, thr_val for threshold selection — stratified by freshness class to preserve balance.

The second was **biased feature selection**. Originally I ranked features using only the freshness task label and reused that ranking for the vegetable classifier too. That's scientifically invalid — features that discriminate fresh from rotten don't necessarily help identify which vegetable it is. I redesigned this as a dual XGBoost ranking: one ranking per task, independently. Then I took the union of the top-200 features from each task, giving a 349-feature set where 149 features are freshness-specific, 149 are vegetable-specific, and 51 are shared. This improved combined validation accuracy noticeably.

A third challenge was **normalization bounds stability for small classes**. I'm using per-vegetable p5/p95 percentile bounds to normalize the raw SVM margin into a 0–100 score. Early on I was computing these bounds on half the validation set, which left thin classes like cucumber with fewer than 50 samples — making the percentile estimates unreliable. The fix was to compute normalization bounds on the full validation set. This is safe because a percentile transform is a fixed linear scale with no label-dependent information, so it doesn't constitute leakage.

---

**3. Why did you use two separate SVMs instead of one model?**

Because the two tasks require fundamentally different information from the image. Identifying that something is a banana versus a potato relies heavily on shape, colour distribution, and structural features from EfficientNet that encode object morphology. Determining whether a banana is fresh or rotten relies more on surface texture, edge softness, HSV hue shifts, and Laplacian variance — features that capture spoilage-related degradation. A single multi-output model would either compromise one task for the other or require a very careful loss weighting that's hard to justify. Two separate SVMs, each trained on a task-appropriate feature subset determined by dual XGBoost ranking, gives each task exactly the signal it needs.

---

**4. Why SVM and not a deep learning classifier end-to-end?**

A few reasons. First, the dataset has around 12,000 images across 10 classes — relatively small by deep learning standards. Fine-tuning a full network on this would risk overfitting without significant augmentation infrastructure. Second, SVMs with RBF kernels are very effective in high-dimensional spaces once the features are well-designed, which EfficientNetB0 already handles for the deep representation. Third, SVMs give a natural, interpretable confidence signal — the decision function's signed distance from the boundary — which I use directly as the freshness score. Getting a calibrated, meaningful score out of a neural network's softmax is considerably more involved. Fourth, the entire pipeline trains in a fraction of the time, which matters for iteration speed.

---

**5. What does RELIABLE, TENTATIVE, UNRELIABLE mean and why does it matter?**

Most classifiers just output a prediction and a confidence score. The problem is that a 90% confidence softmax output doesn't tell you whether the model is actually trustworthy on that specific input — it just reflects the model's internal geometry. My system instead applies a formal multi-gate reliability check.

UNRELIABLE means the prediction should not be acted on at all. This fires when the Mahalanobis distance exceeds the P99 of the training distribution — the input is outside the domain the model was trained on.

TENTATIVE means the score is shown but the fresh/rotten label is withheld. This fires when vegetable confidence is low, when the centroid consistency check fails, or when the raw margin is very close to the decision boundary.

RELIABLE means all gates passed and the output — including the freshness label and confidence band — can be trusted.

This matters operationally because a system deployed in a food supply chain cannot just output numbers. It needs to tell the downstream user when to trust the output and when to escalate to human judgment. On my test set, 92.3% of samples reach RELIABLE state, with zero catastrophic silent failures — no cases where the vegetable was misclassified, freshness was wrong, and the system still called it RELIABLE.

---

**6. How did you ensure the test set was not contaminated?**

Strictly, through a one-way data flow enforced in code. The test set is never loaded until evaluate_models.py runs. All calibration — normalization bounds, Mahalanobis thresholds, formal T_boundary and T_instability selection, isotonic probability calibration, centroid ratio thresholds — happens exclusively on the training set or the validation set. The validation set itself is split into cal_val and thr_val before any of that calibration begins. The test set is a genuinely held-out final measurement, not an input to any design decision.

---

**7. How did you validate that the freshness score actually orders fresh above rotten reliably?**

Through inversion rate analysis. I sample random pairs of one fresh and one rotten image and measure what fraction of pairs the fresh image scores higher than the rotten one. The raw margin inversion rate on my test set is 0.07% — meaning in 99.93% of randomly drawn pairs, the fresh item scores above the rotten one. After per-vegetable normalization this stays at 0.07% globally. I also verified this per-vegetable: cucumber and potato are the weakest at around 0.6% inversion, all others are essentially zero. And the score delta between fresh and rotten means — 86.84 versus 16.10 — gives a 70-point separation with zero overlap, meaning no rotten sample scored above the average fresh score.

---

**8. What would you improve if you had more time?**

A few things. Cucumber and potato are consistently the weakest vegetables — they have higher OOD rates, lower RELIABLE percentages, and slightly higher inversion rates than the others. I'd investigate whether more training data or class-specific augmentation strategies help. The OOD gate currently fires on 2.4% of test samples but catches only 1 error while blocking 61 correct predictions — the precision is poor. I'd explore per-class Mahalanobis thresholds rather than a single global threshold, or a lighter-weight anomaly detector. I'd also formally enable and retune the augmentation instability gate, which is currently stored but disabled — it was found to have zero error-catching utility in ablation, but the test was done on in-distribution data only; it may matter more for domain-shifted real-world inputs.
