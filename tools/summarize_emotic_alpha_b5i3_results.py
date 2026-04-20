#!/usr/bin/env python

import argparse
import csv
import glob
import os

import numpy as np

METRICS = [
    "map",
    "hamming_loss",
    "avg_precision",
    "one_error",
    "ranking_loss",
    "coverage",
    "macrof1",
    "microf1",
]

METHOD_FILES = {
    "finetune": "finetune.csv",
    "lwf": "lwf.csv",
    "ewc": "ewc.csv",
    "replay": "replay_prs.csv",
    "agcn": "agcn.csv",
}


def resolve_result_file(result_dir, method):
    if method == "clif":
        matches = sorted(glob.glob(os.path.join(result_dir, "clif_*.csv")))
        return matches[-1] if matches else None
    filename = METHOD_FILES[method]
    direct_path = os.path.join(result_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    stem, suffix = os.path.splitext(filename)
    pattern = os.path.join(result_dir, f"{stem}_*{suffix}")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def load_summary(path):
    data = np.loadtxt(path, delimiter=",")
    return {metric: float(data[idx, -1]) for idx, metric in enumerate(METRICS)}


def main():
    parser = argparse.ArgumentParser(description="Summarize EMOTIC alpha B5-I3 experiment results.")
    parser.add_argument("--results-root", default="./result")
    parser.add_argument("--split", default="B5I3")
    parser.add_argument("--order", default="alphabetical")
    parser.add_argument("--protocol", default="original")
    parser.add_argument("--loss-type", default="softmargin")
    parser.add_argument("--scale", default="formal")
    parser.add_argument(
        "--include-resnet",
        action="store_true",
        help="Include ResNet18 comparison runs in the summary output.",
    )
    args = parser.parse_args()

    subjects = [
        ("vit", os.path.join("vit", args.order, args.protocol, args.split, args.loss_type, args.scale)),
    ]
    if args.include_resnet:
        subjects.append(
            ("resnet18", os.path.join("resnet18", args.order, args.protocol, args.split, args.loss_type, args.scale))
        )
    methods = ["finetune", "lwf", "ewc", "replay", "agcn", "clif"]

    writer = csv.writer(os.sys.stdout)
    writer.writerow(["backbone", "method", *METRICS, "result_file"])

    for backbone, relative_dir in subjects:
        result_dir = os.path.join(args.results_root, relative_dir)
        for method in methods:
            result_file = resolve_result_file(result_dir, method)
            if result_file is None:
                writer.writerow([backbone, method, *([""] * len(METRICS)), ""])
                continue
            summary = load_summary(result_file)
            writer.writerow(
                [
                    backbone,
                    method,
                    *[summary[metric] for metric in METRICS],
                    result_file,
                ]
            )


if __name__ == "__main__":
    main()
