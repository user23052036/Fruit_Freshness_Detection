import os
import argparse
import shutil
import pandas as pd
from PIL import Image


def image_score(path):

    try:
        img = Image.open(path)
        w, h = img.size
        return w * h
    except Exception:
        return 0


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="duplicates")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.report)

    os.makedirs(args.output, exist_ok=True)

    for group_id, group in df.groupby("group_id"):

        paths = list(group["path"])

        best = max(paths, key=image_score)

        for p in paths:

            if p == best:
                continue

            dst = os.path.join(args.output, os.path.basename(p))

            if args.dry_run:
                print("MOVE", p, "->", dst)
            else:
                shutil.move(p, dst)


if __name__ == "__main__":
    main()