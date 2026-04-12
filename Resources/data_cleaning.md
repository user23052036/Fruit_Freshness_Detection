# Dataset Cleaning — Vegetable Freshness Pipeline

This document describes the full cleaning workflow for `vegetable_Dataset/`. It covers
duplicate detection, test-leak removal, and verification before feature re-extraction.
Run the steps in the order shown. Never skip the dry-run.

---

## Dataset Layout

```
vegetable_Dataset/
├── FreshApple/
├── FreshBanana/
├── FreshCapsicum/
├── FreshCucumber/
├── FreshPotato/
├── RottenApple/
├── RottenBanana/
├── RottenCapsicum/
├── RottenCucumber/
└── RottenPotato/
```

All three scripts live in `clean_dataset/`. Run every command from the project root.

---

## What Each Script Does

### `find_image_duplicates.py` — Detection only, no file changes

Scans a folder recursively and groups images that are identical or visually near-identical.
Uses two complementary methods:

| Method | What it catches | How |
|---|---|---|
| MD5 hash | Exact byte-for-byte copies | Identical hash → same group |
| Perceptual hash (pHash) | Near-duplicates (crops, rotations, rescales, minor edits) | Hamming distance ≤ `--phash-threshold` |

Any two images that match on *either* criterion are placed in the same group.

**Output:** A CSV with two columns — `group_id` and `path`. Only images that belong to a
duplicate group appear in the file. Unique images are not listed.

**This script never moves or deletes anything.**

```
Arguments:
  --root             Path to the folder to scan (required)
  --report           Output CSV path (required)
  --phash-threshold  Max Hamming distance for near-duplicate match (default: 6)
```

A threshold of 6 means images must differ by 6 or fewer bits out of 64 to be grouped.
Lower values → stricter matching, fewer groups. Raise only if obvious duplicates are missed.

---

### `keep_best_train_duplicates.py` — Removes intra-folder duplicates

Reads a report from `find_image_duplicates.py`. For every duplicate group, it keeps the
single highest-quality image in place and moves all others to a holding folder.

**Quality scoring:** resolution (width × height). Ties broken by file size.

The moved files go to `--output` (default: `duplicates/`). Nothing is permanently deleted —
files can be recovered from there at any time.

```
Arguments:
  --report    CSV report from find_image_duplicates.py (required)
  --output    Folder to move duplicates into (default: duplicates/)
  --dry-run   Print what would be moved without touching any files
```

Always run with `--dry-run` first and read the output before the real run.

---

### `move_test_leaks.py` — Removes cross-split contamination

Reads a report from `find_image_duplicates.py`. For every duplicate group it checks whether
the group contains images from both a training folder and a test folder. If it does, the test
copies are moved to a holding folder — because a model trained on one copy of an image and
evaluated on another copy of the same image will show inflated accuracy.

Identification is by path substring: any path containing `--test-keyword` (default: `"test"`)
is classified as a test image. Adjust the keyword to match your folder naming.

The moved files go to `--output` (default: `test_leaks/`). Nothing is permanently deleted.

```
Arguments:
  --report        CSV report from find_image_duplicates.py (required)
  --test-keyword  Substring that marks a path as belonging to the test split (default: test)
  --output        Folder to move leaked test images into (default: test_leaks/)
```

This script is only needed if your raw dataset already contains a pre-made train/test split
and you suspect images were shared across both. If you built the split yourself with
`train_split.py` after cleaning, run this before splitting.

---

## Full Cleaning Workflow

### Step 1 — Detect duplicates in every class folder

Run `find_image_duplicates.py` once per folder. Each call produces one CSV report.
No files are changed by any of these commands.

```bash
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshApple    --phash-threshold 6 --report freshapple_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshBanana   --phash-threshold 6 --report freshbanana_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCapsicum --phash-threshold 6 --report freshcapsicum_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCucumber --phash-threshold 6 --report freshcucumber_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshPotato   --phash-threshold 6 --report freshpotato_duplicates.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenApple    --phash-threshold 6 --report rottenapple_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenBanana   --phash-threshold 6 --report rottenbanana_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCapsicum --phash-threshold 6 --report rottencapsicum_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCucumber --phash-threshold 6 --report rottencucumber_duplicates.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenPotato   --phash-threshold 6 --report rottenpotato_duplicates.csv
```

After this step you have 10 CSV files, one per class folder. An empty CSV (header only)
means no duplicates were found in that folder.

---

### Step 2 — Inspect the reports before doing anything

Before moving a single file, open several groups and look at the actual images.

```bash
# Show all paths in group 3 of the banana report
grep "^3," freshbanana_duplicates.csv

# Count how many duplicate groups were found
tail -n +2 freshbanana_duplicates.csv | cut -d',' -f1 | sort -u | wc -l

# Count total images flagged (all reports combined)
tail -n +2 *_duplicates.csv | wc -l
```

Open the listed images and confirm they show the same vegetable from the same source,
not two different images that happen to look similar. If unrelated images appear in
the same group, lower `--phash-threshold` (try 4) and re-run detection for that folder.

---

### Step 3 — Handle test leaks (if applicable)

Run this only if your raw dataset has a pre-existing train/test split and you want to
check for cross-split image contamination. Pass each report to `move_test_leaks.py`.

```bash
python clean_dataset/move_test_leaks.py --report freshapple_duplicates.csv    --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report freshbanana_duplicates.csv   --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report freshcapsicum_duplicates.csv --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report freshcucumber_duplicates.csv --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report freshpotato_duplicates.csv   --test-keyword test --output test_leaks

python clean_dataset/move_test_leaks.py --report rottenapple_duplicates.csv    --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report rottenbanana_duplicates.csv   --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report rottencapsicum_duplicates.csv --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report rottencucumber_duplicates.csv --test-keyword test --output test_leaks
python clean_dataset/move_test_leaks.py --report rottenpotato_duplicates.csv   --test-keyword test --output test_leaks
```

Leaked test images land in `test_leaks/`. The training copies remain untouched in their
original folders. After this step, re-run Step 1 on the affected folders to regenerate
fresh reports before proceeding to Step 4 — the old reports now reference missing paths.

---

### Step 4 — Dry-run duplicate removal

Simulate the removal for every class folder. Read the output carefully.

```bash
python clean_dataset/keep_best_train_duplicates.py --report freshapple_duplicates.csv    --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshbanana_duplicates.csv   --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshcapsicum_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshcucumber_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report freshpotato_duplicates.csv   --dry-run

python clean_dataset/keep_best_train_duplicates.py --report rottenapple_duplicates.csv    --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottenbanana_duplicates.csv   --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottencapsicum_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottencucumber_duplicates.csv --dry-run
python clean_dataset/keep_best_train_duplicates.py --report rottenpotato_duplicates.csv   --dry-run
```

Each line of output has the form `MOVE <source> -> <destination>`. The source is what
will be moved; the destination is where it will go inside `duplicates/`. The image NOT
listed for a given group is the one being kept — verify it is the right choice.

---

### Step 5 — Execute duplicate removal

Once the dry-run output looks correct, run without `--dry-run`.

```bash
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

Removed images are in `duplicates/`. Do not delete this folder — it is useful for auditing
and can serve as an augmentation pool later.

---

### Step 6 — Verify the cleaned dataset

Re-run duplicate detection on each folder and confirm that few or no groups remain.

```bash
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshApple    --phash-threshold 6 --report verify_freshapple.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshBanana   --phash-threshold 6 --report verify_freshbanana.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCapsicum --phash-threshold 6 --report verify_freshcapsicum.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshCucumber --phash-threshold 6 --report verify_freshcucumber.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/FreshPotato   --phash-threshold 6 --report verify_freshpotato.csv

python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenApple    --phash-threshold 6 --report verify_rottenapple.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenBanana   --phash-threshold 6 --report verify_rottenbanana.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCapsicum --phash-threshold 6 --report verify_rottencapsicum.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenCucumber --phash-threshold 6 --report verify_rottencucumber.csv
python clean_dataset/find_image_duplicates.py --root vegetable_Dataset/RottenPotato   --phash-threshold 6 --report verify_rottenpotato.csv
```

Check image counts per folder:

```bash
for f in vegetable_Dataset/*/; do echo "$f: $(ls "$f" | wc -l)"; done
```

A cleaned verification report should be empty (header only) or contain only a small
number of groups — residual near-duplicates that were below the threshold the first time.
If significant groups remain, inspect them and decide whether to lower the threshold or
leave them.

---

### Step 7 — Re-extract features

Feature extraction must be re-run after any change to the dataset. The `X.npy` matrix
built before cleaning contains rows for images that no longer exist in the training folders,
and its index alignment with `y_veg.npy`, `y_fresh.npy`, and `image_paths.npy` is
broken the moment any file is moved.

```bash
python extract_dataset_features.py
```

This regenerates:

```
Features/X.npy              feature matrix aligned to cleaned dataset
Features/y_veg.npy          vegetable labels
Features/y_fresh.npy        freshness labels
Features/image_paths.npy    path per row (required for augmentation in train_svm.py)
```

After this, run the full training pipeline from `train_split.py` forward.

---

## Decision Reference

| Situation | Action |
|---|---|
| pHash groups contain clearly different vegetables | Lower `--phash-threshold` (try 4), re-run detection |
| pHash groups contain slightly different photos of the same item | Threshold is correct, proceed |
| Group has only one image | Not possible — single images are never in the report |
| A moved image turns out to be the wrong one | Retrieve it from `duplicates/` and move it back manually |
| Verification report still has groups after cleaning | Inspect; they are likely genuine near-duplicates below the original threshold |
| Test leaks folder is empty after `move_test_leaks.py` | No cross-split contamination was found — this is the expected outcome if you built the split yourself after cleaning |

---

## File Summary

| File produced | Created by | Purpose |
|---|---|---|
| `*_duplicates.csv` | `find_image_duplicates.py` | Duplicate group report per folder |
| `verify_*.csv` | `find_image_duplicates.py` | Post-cleaning verification report |
| `duplicates/` | `keep_best_train_duplicates.py` | Holds removed intra-folder duplicates |
| `test_leaks/` | `move_test_leaks.py` | Holds test images found in training folders |