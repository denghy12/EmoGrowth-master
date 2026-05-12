#!/usr/bin/env python

import argparse
import csv
import glob
import os
from pathlib import Path

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
    "finetune": "finetune",
    "lwf": "lwf",
    "ewc": "ewc",
    "replay": "replay_prs",
    "agcn": "agcn",
    "clif": "clif",
}


def resolve_result_file(result_dir, method):
    prefix = METHOD_FILES[method]
    matches = sorted(glob.glob(os.path.join(result_dir, f"{prefix}*.csv")))
    return matches[-1] if matches else None


def load_summary(path):
    data = np.loadtxt(path, delimiter=",")
    return {metric: float(data[idx, -1]) for idx, metric in enumerate(METRICS)}


def main():
    parser = argparse.ArgumentParser(description="Summarize EMOTIC results from the unified result/ directory.")
    parser.add_argument("--results-root", default="./result")
    parser.add_argument("--extractor", default="vit")
    parser.add_argument("--order", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--loss-type", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument(
        "--methods",
        default="finetune,lwf,ewc,replay,agcn,clif",
        help="Comma-separated method list.",
    )
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    result_dir = os.path.join(
        args.results_root,
        args.extractor,
        args.order,
        args.protocol,
        args.split,
        args.loss_type,
        args.scale,
    )

    writer = csv.writer(os.sys.stdout)
    writer.writerow(["method", *METRICS, "result_file"])
    for method in methods:
        result_file = resolve_result_file(result_dir, method)
        if result_file is None:
            writer.writerow([method, *([""] * len(METRICS)), ""])
            continue
        summary = load_summary(result_file)
        writer.writerow([method, *[summary[metric] for metric in METRICS], result_file])


if __name__ == "__main__":
    main()
