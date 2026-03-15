import os
import argparse
import hashlib
from PIL import Image
import imagehash
import pandas as pd
from tqdm import tqdm


VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def list_images(root):
    images = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(VALID_EXT):
                images.append(os.path.join(dirpath, f))
    return images


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path):
    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default="duplicate_report.csv")
    parser.add_argument("--phash-threshold", type=int, default=6)

    args = parser.parse_args()

    images = list_images(args.root)

    print(f"Found {len(images)} images")

    rows = []

    for path in tqdm(images):

        md5 = file_hash(path)
        ph = compute_phash(path)

        rows.append({
            "path": path,
            "md5": md5,
            "phash": ph
        })

    df = pd.DataFrame(rows)

    groups = []

    group_id = 0

    used = set()

    for i in range(len(df)):

        if i in used:
            continue

        row_i = df.iloc[i]

        group = [i]

        for j in range(i + 1, len(df)):

            if j in used:
                continue

            row_j = df.iloc[j]

            if row_i["md5"] == row_j["md5"]:
                group.append(j)
                used.add(j)

            elif row_i["phash"] and row_j["phash"]:

                dist = imagehash.hex_to_hash(row_i["phash"]) - imagehash.hex_to_hash(row_j["phash"])

                if dist <= args.phash_threshold:
                    group.append(j)
                    used.add(j)

        if len(group) > 1:

            for idx in group:
                groups.append({
                    "group_id": group_id,
                    "path": df.iloc[idx]["path"]
                })

            group_id += 1

    report = pd.DataFrame(groups)

    report.to_csv(args.report, index=False)

    print(f"Duplicate groups saved to {args.report}")


if __name__ == "__main__":
    main()