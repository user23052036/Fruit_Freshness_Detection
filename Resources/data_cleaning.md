Good — what you wrote is correct conceptually, but it is **Apple-specific**.
If you keep it like this, you will manually repeat everything for each folder.

The correct way is to **generalize the workflow so it works for all 10 folders**.

Your dataset folders are:

```
vegetable_Dataset/
    FreshApple
    FreshBanana
    FreshCapsicum
    FreshCucumber
    FreshPotato
    RottenApple
    RottenBanana
    RottenCapsicum
    RottenCucumber
    RottenPotato
```

So the notes should reflect **the same workflow applied to every class**.

Below is the **corrected version of your notes**, rewritten so it works for the whole dataset.

---

# Abstract: what each script does (high level)

### 1. `find_image_duplicates.py`

Purpose:

Detect duplicate or near-duplicate images inside a folder.

How it works:

• Computes **MD5 hash** → detects exact duplicate files
• Computes **perceptual hash (pHash)** → detects visually similar images (rotation, translation, small crops)

Output:

```
CSV file
group_id , image_path
```

Important detail:

• Only images that belong to duplicate groups appear in the CSV
• Images with **no duplicates do NOT appear**

Safety:

This script **does not modify the dataset**. It only generates a report.

---

### 2. `keep_best_train_duplicates.py`

Purpose:

Remove duplicate images and keep only one representative.

Process:

For every duplicate group:

```
Group
 ├ image1
 ├ image2
 ├ image3
```

The script:

• keeps the **best image**
• moves the remaining images to a folder called:

```
duplicates/
```

How the best image is chosen:

```
highest resolution (width × height)
fallback → largest file size
```

Safety feature:

```
--dry-run
```

Shows what will be moved **without actually moving files**.

Always run dry-run first.

---

### 3. `move_test_leaks.py`

Purpose:

Prevent **train/test data leakage**.

Problem:

Sometimes the same image appears in:

```
train set
test set
```

This inflates model accuracy artificially.

The script:

• detects duplicate images appearing in test
• moves them to:

```
test_leaks/
```

---

# Important rules before cleaning

1. Work **one folder at a time**

Example:

```
FreshApple
FreshBanana
...
```

2. Always run

```
--dry-run
```

before actually removing files.

3. Never delete the `duplicates/` folder.

It can be useful later for:

```
data augmentation
debugging
dataset audit
```

4. Default threshold:

```
--phash-threshold 6
```

Adjust only if needed.

5. After cleaning **all folders**, recompute features before training.

---

# Dataset cleaning workflow (FULL DATASET)

Run from project root.

---

# Step 1 — Detect duplicates for every folder

Run duplicate detection for each class.

```
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshApple --phash-threshold 6 --report freshapple_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshBanana --phash-threshold 6 --report freshbanana_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCapsicum --phash-threshold 6 --report freshcapsicum_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCucumber --phash-threshold 6 --report freshcucumber_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshPotato --phash-threshold 6 --report freshpotato_duplicates.csv


python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenApple --phash-threshold 6 --report rottenapple_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenBanana --phash-threshold 6 --report rottenbanana_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCapsicum --phash-threshold 6 --report rottencapsicum_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCucumber --phash-threshold 6 --report rottencucumber_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenPotato --phash-threshold 6 --report rottenpotato_duplicates.csv
```

Result:

Each folder will generate a CSV report.

Example:

```
freshapple_duplicates.csv
freshbanana_duplicates.csv
...
```

No images are changed.

---

# Step 2 — Inspect duplicate groups

Open a few groups in each CSV.

Example:

```
grep "^50," freshbanana_duplicates.csv
```

Open those images and confirm they are:

```
same fruit
rotated
flipped
translated
```

If different fruits appear together, reduce threshold.

---

# Step 3 — Dry-run duplicate removal

Simulate duplicate removal.

```
python clean_dataset/keep_best_train_duplicates.py --report freshapple_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshbanana_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshcapsicum_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshcucumber_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshpotato_duplicates.csv --dry-run

python clean_dataset/keep_best_train_duplicates.py --report rottenapple_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottenbanana_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottencapsicum_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottencucumber_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottenpotato_duplicates.csv --dry-run
```

Verify that the files listed are true duplicates.

---

# Step 4 — Execute duplicate removal

Once verified, run without dry-run.

```
python clean_dataset/keep_best_train_duplicates.py --report freshapple_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report freshbanana_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report freshcapsicum_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report freshcucumber_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report freshpotato_duplicates.csv

python clean_dataset/keep_best_train_duplicates.py --report rottenapple_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report rottenbanana_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report rottencapsicum_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report rottencucumber_duplicates.csv
python clean_dataset/keep_best_train_duplicates.py --report rottenpotato_duplicates.csv
```

Duplicate images will move into:

```
duplicates/
```

---

# Step 5 — Verify dataset after cleaning

Check image counts.

```
ls vegetable_Dataset/FreshApple | wc -l
ls vegetable_Dataset/FreshBanana | wc -l
ls vegetable_Dataset/FreshCapsicum | wc -l
ls vegetable_Dataset/FreshCucumber | wc -l
ls vegetable_Dataset/FreshPotato | wc -l
```

Run duplicate detection again to ensure few remain.

```
python clean_dataset/find_image_duplicates.py \
--root vegetable_Dataset/FreshApple \
--phash-threshold 6 \
--report verify_freshapple.csv
```

---

# Step 6 — Recompute features

After cleaning **all folders**, regenerate features.

```
python src/extract_features.py \
--root vegetable_Dataset \
--out Features/features_clean.npy
```

This produces the **1312-feature vectors** used by your model.

---

# Final cleaning checklist

✔ Duplicate detection
✔ Manual inspection
✔ Dry-run validation
✔ Duplicate removal
✔ Verification run
✔ Feature extraction

After this, the dataset is ready for:

```
freshness grading pipeline
(SVR + projection method)
```

---

If you want, I can also show you the **complete workflow diagram of your whole project (data → features → SVM → SVR grading → final score)** so the entire system becomes very clear.
