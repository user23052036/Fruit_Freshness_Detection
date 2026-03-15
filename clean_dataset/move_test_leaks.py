import os
import argparse
import shutil
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--report", required=True)
    parser.add_argument("--test-keyword", default="test")
    parser.add_argument("--output", default="test_leaks")

    args = parser.parse_args()

    df = pd.read_csv(args.report)

    os.makedirs(args.output, exist_ok=True)

    for group_id, group in df.groupby("group_id"):

        paths = list(group["path"])

        train = []
        test = []

        for p in paths:

            if args.test_keyword in p.lower():
                test.append(p)
            else:
                train.append(p)

        if train and test:

            for p in test:

                dst = os.path.join(args.output, os.path.basename(p))

                print("Leak:", p)

                shutil.move(p, dst)


if __name__ == "__main__":
    main()