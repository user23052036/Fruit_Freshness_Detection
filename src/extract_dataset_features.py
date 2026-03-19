import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from extract_features import model, preprocess_input, extract_handcrafted
from utils import TARGET_VEGETABLES

DATASET_DIR = "vegetable_Dataset"
FEATURE_DIR = "Features"

BATCH_SIZE = 128
NUM_WORKERS = os.cpu_count() or 4


def parse_folder(folder):
    """
    Parse folder name to (vegetable, freshness_int).
    Uses string slicing, not str.replace, to avoid replacing
    occurrences of 'fresh'/'rotten' inside the vegetable name.
    """
    folder = folder.lower()

    if folder.startswith("fresh"):
        return folder[len("fresh"):], 1

    if folder.startswith("rotten"):
        return folder[len("rotten"):], 0

    return None, None


def load_image(path):

    try:

        img = cv2.imread(path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224,224))

        return img

    except:
        return None


def main():

    image_paths = []
    y_veg = []
    y_fresh = []

    print("Scanning dataset...")

    for folder in sorted(os.listdir(DATASET_DIR)):

        veg, fresh = parse_folder(folder)

        if veg not in TARGET_VEGETABLES:
            continue

        folder_path = os.path.join(DATASET_DIR, folder)

        for img in os.listdir(folder_path):

            if not img.lower().endswith((".jpg",".jpeg",".png")):
                continue

            image_paths.append(os.path.join(folder_path,img))
            y_veg.append(veg)
            y_fresh.append(fresh)

    print("Total images:", len(image_paths))

    os.makedirs(FEATURE_DIR, exist_ok=True)

    X = []
    final_y_veg   = []
    final_y_fresh = []
    final_image_paths = []   # paths that survived None-image filter

    print("Extracting features...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:

        for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Batches"):

            batch_paths = image_paths[start:start+BATCH_SIZE]
            batch_yveg = y_veg[start:start+BATCH_SIZE]
            batch_yfresh = y_fresh[start:start+BATCH_SIZE]

            imgs = list(executor.map(load_image, batch_paths))

            valid_imgs    = []
            valid_yveg    = []
            valid_yfresh  = []
            valid_paths   = []

            for img, veg, fresh, path in zip(imgs, batch_yveg, batch_yfresh, batch_paths):

                if img is None:
                    continue

                valid_imgs.append(img)
                valid_yveg.append(veg)
                valid_yfresh.append(fresh)
                valid_paths.append(path)

            if len(valid_imgs) == 0:
                continue

            batch_np = np.array(valid_imgs, dtype=np.float32)

            deep = preprocess_input(batch_np)

            deep_features = model.predict(deep, verbose=0)

            for i, img in enumerate(valid_imgs):

                handcrafted = extract_handcrafted(img)

                feats = np.concatenate([deep_features[i], handcrafted])

                X.append(feats)
                final_y_veg.append(valid_yveg[i])
                final_y_fresh.append(valid_yfresh[i])
                final_image_paths.append(valid_paths[i])

    X = np.array(X, dtype=np.float32)

    np.save(os.path.join(FEATURE_DIR, "X.npy"),           X)
    np.save(os.path.join(FEATURE_DIR, "y_veg.npy"),       np.array(final_y_veg))
    np.save(os.path.join(FEATURE_DIR, "y_fresh.npy"),     np.array(final_y_fresh))
    # Save per-image paths (aligned to X rows) so train_split.py
    # can propagate val paths to train_svm.py for real augmentation calibration
    np.save(os.path.join(FEATURE_DIR, "image_paths.npy"), np.array(final_image_paths))

    print("Saved feature matrix:", X.shape)


if __name__ == "__main__":
    main()