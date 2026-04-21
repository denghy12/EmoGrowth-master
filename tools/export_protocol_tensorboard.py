import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


METRIC_NAMES = [
    "map",
    "hamming_loss",
    "avg_precision",
    "one_error",
    "ranking_loss",
    "coverage",
    "macrof1",
    "microf1",
]
CLASS_METRIC_NAMES = [
    "ap",
    "f1",
    "precision",
    "recall",
    "support",
    "predicted_positive",
    "tp",
    "fp",
    "fn",
    "tn",
]
PROTOCOLS = ["agcnpp_il", "strict", "agcnpp_cl"]
MODELS = ["agcn", "clif"]
PROTOCOL_LABELS = {
    "agcnpp_il": "AGCN++ IL",
    "strict": "Strict Seen",
    "agcnpp_cl": "AGCN++ CL",
}
MODEL_LABELS = {
    "agcn": "AGCN",
    "clif": "CLIF",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export EMOTIC protocol results to a TensorBoard logdir."
    )
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--detail-root", default="./result_detail")
    parser.add_argument("--logdir", default="./tensorboard_runs/protocol_vit_alpha_b5i3_softmargin_formal")
    parser.add_argument("--extractor", default="vit")
    parser.add_argument("--order", default="alphabetical")
    parser.add_argument("--split", default="B5I3")
    parser.add_argument("--loss", default="softmargin")
    parser.add_argument("--scale", default="formal")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default="6006")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start tensorboard after writing event files.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only export interactive scalar curves; skip task matrix image summaries.",
    )
    return parser.parse_args()


def require_tensorboard_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard export requires PyTorch with torch.utils.tensorboard. "
            "On the server, try: pip install tensorboard"
        ) from exc
    return SummaryWriter


def result_dir(args, protocol):
    return (
        Path(args.result_root)
        / args.extractor
        / args.order
        / protocol
        / args.split
        / args.loss
        / args.scale
    )


def detail_dir(args, protocol):
    return (
        Path(args.detail_root)
        / args.extractor
        / args.order
        / protocol
        / args.split
        / args.loss
        / args.scale
    )


def read_result_matrix(path):
    rows = []
    with path.open(newline="") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append([float(value) for value in line.split(",")])
    return np.asarray(rows, dtype=float)


def total_result_path(args, protocol, model):
    root = result_dir(args, protocol)
    if model == "agcn":
        return root / "agcn.csv"
    matches = sorted(
        root.glob("clif_lamda_le_*_lamda_kd_relation_aff_*_lamda_kd_relation_data_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No CLIF result CSV found under {root}")
    return matches[0]


def detail_result_path(args, protocol, model, kind):
    root = detail_dir(args, protocol)
    matches = list(root.glob(f"*_{kind}_metrics.csv"))
    matches = [
        path
        for path in matches
        if (("_clif_" in path.name) if model == "clif" else ("_clif_" not in path.name))
    ]
    if not matches:
        raise FileNotFoundError(f"No {kind} detail CSV found for {protocol}/{model} under {root}")
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def experiment_label(protocol, model):
    return f"{PROTOCOL_LABELS[protocol]} {MODEL_LABELS[model]}"


def safe_tag(text):
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )


def class_tag(record):
    return f"{safe_tag(record['class_name'])}_task{record['class_task']}"


def run_name(protocol, model):
    return f"{safe_tag(PROTOCOL_LABELS[protocol])}/{MODEL_LABELS[model]}"


def collect_total_results(args):
    results = {}
    for protocol in PROTOCOLS:
        for model in MODELS:
            matrix = read_result_matrix(total_result_path(args, protocol, model))
            results[(protocol, model)] = matrix
    return results


def collect_per_task(args):
    rows = {}
    for protocol in PROTOCOLS:
        for model in MODELS:
            path = detail_result_path(args, protocol, model, "per_task")
            with path.open(newline="") as handle:
                records = []
                for row in csv.DictReader(handle):
                    record = {}
                    for key, value in row.items():
                        if key in {
                            "trained_task",
                            "trained_seen_classes",
                            "eval_task",
                            "class_start",
                            "class_end_exclusive",
                            "num_classes",
                            "num_samples",
                        }:
                            record[key] = int(value)
                        elif key in METRIC_NAMES:
                            record[key] = float(value)
                        else:
                            record[key] = value
                    records.append(record)
            rows[(protocol, model)] = records
    return rows


def collect_per_class(args):
    rows = {}
    for protocol in PROTOCOLS:
        for model in MODELS:
            path = detail_result_path(args, protocol, model, "per_class")
            with path.open(newline="") as handle:
                records = []
                for row in csv.DictReader(handle):
                    record = {}
                    for key, value in row.items():
                        if key in {
                            "trained_task",
                            "trained_seen_classes",
                            "class_id",
                            "class_task",
                            "support",
                            "predicted_positive",
                            "tp",
                            "fp",
                            "fn",
                            "tn",
                        }:
                            record[key] = int(value)
                        elif key in {"ap", "f1", "precision", "recall"}:
                            record[key] = float(value)
                        else:
                            record[key] = value
                    records.append(record)
            rows[(protocol, model)] = records
    return rows


def task_metric_matrix(records, metric):
    max_trained = max(record["trained_task"] for record in records)
    max_eval = max(record["eval_task"] for record in records)
    matrix = np.full((max_eval + 1, max_trained + 1), np.nan, dtype=float)
    for record in records:
        matrix[record["eval_task"], record["trained_task"]] = record[metric]
    return matrix


def add_scalar_series(writer, tag, values_by_step, num_steps):
    for step in range(num_steps):
        value = values_by_step.get(step, np.nan)
        writer.add_scalar(tag, float(value), step)


def add_per_task_scalars(writer, label, records, num_steps):
    eval_tasks = sorted({record["eval_task"] for record in records})
    for eval_task in eval_tasks:
        task_records = [
            record for record in records if record["eval_task"] == eval_task
        ]
        for metric in METRIC_NAMES:
            values_by_step = {
                record["trained_task"]: record[metric]
                for record in task_records
            }
            add_scalar_series(writer, f"task{eval_task}/{metric}", values_by_step, num_steps)
            add_scalar_series(writer, f"{metric}/task{eval_task}", values_by_step, num_steps)


def add_per_class_scalars(writer, label, records, num_steps):
    records_by_class = {}
    for record in records:
        records_by_class.setdefault(class_tag(record), []).append(record)

    for tag in sorted(records_by_class):
        class_records = records_by_class[tag]
        for metric in CLASS_METRIC_NAMES:
            values_by_step = {
                record["trained_task"]: record[metric]
                for record in class_records
            }
            add_scalar_series(writer, f"{tag}/{metric}", values_by_step, num_steps)
            add_scalar_series(writer, f"{metric}/{tag}", values_by_step, num_steps)


def add_dashboard_readme(writer):
    writer.add_text(
        "dashboard_guide",
        "\n".join(
            [
                "# EMOTIC continual multi-label dashboard",
                "",
                "Run selector:",
                "- Each run is one method/model pair, for example AGCNplusplus_IL/CLIF.",
                "- Use TensorBoard's Runs selector to toggle or overlay methods.",
                "- Every scalar card is written at steps 0 through 7. Not-yet-visible classes/tasks use NaN before their first valid task.",
                "",
                "First-level scalar groups:",
                "- task0 ... task7: per-task metrics from per_task_metrics.csv.",
                "- Affection_task0 ... Yearning_task7: per-class metrics from per_class_metrics.csv.",
                "- map, microf1, f1, predicted_positive, etc.: metric-first views.",
                "",
                "Task group view:",
                "- task0/map",
                "- task0/microf1",
                "- task0/macrof1",
                "",
                "Class view:",
                "- Affection_task0/ap",
                "- Affection_task0/f1",
                "- Affection_task0/predicted_positive",
                "",
                "Variable view:",
                "- f1/Affection_task0",
                "- predicted_positive/Affection_task0",
                "- microf1/task0",
            ]
        ),
        global_step=0,
    )


def make_heatmap_figure(matrix, title, metric):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 5.2))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        mask=np.isnan(matrix),
        cbar=True,
    )
    ax.set_title(title)
    ax.set_xlabel("trained task")
    ax.set_ylabel("eval task")
    fig.tight_layout()
    return fig


def add_matrix_figures(writer, label, records):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; skipping task_matrix image summaries. "
            "Scalar dashboards were still exported."
        )
        return

    for metric in ["map", "macrof1", "microf1"]:
        matrix = task_metric_matrix(records, metric)
        fig = make_heatmap_figure(matrix, f"{label}: {metric}", metric)
        writer.add_figure(f"{metric}/task_matrix", fig, global_step=0)
        plt.close(fig)


def export_tensorboard(args):
    SummaryWriter = require_tensorboard_writer()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    total_results = collect_total_results(args)
    per_task_results = collect_per_task(args)
    per_class_results = collect_per_class(args)

    for protocol in PROTOCOLS:
        for model in MODELS:
            label = experiment_label(protocol, model)
            writer = SummaryWriter(log_dir=str(logdir / run_name(protocol, model)))
            add_dashboard_readme(writer)
            total_matrix = total_results[(protocol, model)]
            task_records = per_task_results[(protocol, model)]
            class_records = per_class_results[(protocol, model)]
            num_steps = total_matrix.shape[1] - 1
            add_per_task_scalars(writer, label, task_records, num_steps)
            add_per_class_scalars(writer, label, class_records, num_steps)
            if not args.no_figures:
                add_matrix_figures(writer, label, task_records)
            writer.flush()
            writer.close()
    return logdir


def launch_tensorboard(args, logdir):
    command = [
        "tensorboard",
        "--logdir",
        str(logdir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("Launching:", " ".join(command))
    subprocess.run(command, check=False)


def main():
    args = parse_args()
    logdir = export_tensorboard(args)
    print(f"Wrote TensorBoard events to {logdir}")
    print("Open with:")
    print(f"tensorboard --logdir {logdir} --host {args.host} --port {args.port}")
    if args.launch:
        launch_tensorboard(args, logdir)


if __name__ == "__main__":
    main()
