import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

import h5py
import numpy as np
import pandas as pd


DEFAULT_FEATURE_FILENAME = "feature_vit_b_16_bbox.npy"
DIMENSION_FILENAME = "affective_dimension.npy"
METADATA_FILENAME = "metadata.csv"
CLASS_FILENAME = "class_order.json"
MANIFEST_FILENAME = "prepare_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a more balanced EMOTIC class order and label_session."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="./data/emotic_processed",
        help="Existing EMOTIC processed directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./data/emotic_processed_balanced",
        help="Directory that will contain the balanced split files.",
    )
    parser.add_argument("--init-cls", type=int, default=8)
    parser.add_argument("--increment", type=int, default=6)
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated evaluation splits.",
    )
    parser.add_argument(
        "--feature-filename",
        type=str,
        default=DEFAULT_FEATURE_FILENAME,
        help="Feature file to link into the balanced split directory.",
    )
    return parser.parse_args()


def build_task_sizes(total_class, init_cls, increment):
    task_sizes = [init_cls]
    while sum(task_sizes) + increment <= total_class:
        task_sizes.append(increment)
    remainder = total_class - sum(task_sizes)
    if remainder > 0:
        task_sizes.append(remainder)
    return task_sizes


def load_metadata(source_root):
    metadata_path = os.path.join(source_root, METADATA_FILENAME)
    metadata = pd.read_csv(metadata_path)
    metadata["categories"] = metadata["categories"].fillna("").map(
        lambda value: [item for item in str(value).split("|") if item]
    )
    return metadata


def build_class_to_train_samples(metadata):
    class_to_samples = defaultdict(set)
    train_metadata = metadata[metadata["split"] == "train"].reset_index(drop=True)
    for train_idx, categories in enumerate(train_metadata["categories"]):
        for category in categories:
            class_to_samples[category].add(train_idx)
    return class_to_samples


def task_train_sizes(groups, class_to_samples):
    sizes = []
    for group in groups:
        sample_union = set()
        for category in group:
            sample_union |= class_to_samples[category]
        sizes.append(len(sample_union))
    return sizes


def score_groups(groups, class_to_samples):
    sizes = task_train_sizes(groups, class_to_samples)
    mean = sum(sizes) / len(sizes)
    variance = sum((size - mean) ** 2 for size in sizes)
    return variance, sizes


def initial_groups(classes, task_sizes_cfg, class_to_samples):
    capacities = list(task_sizes_cfg)
    groups = [[] for _ in task_sizes_cfg]
    covered = [set() for _ in task_sizes_cfg]

    for category in classes:
        candidates = [index for index, capacity in enumerate(capacities) if capacity > 0]
        best_group = min(
            candidates,
            key=lambda index: (
                len(covered[index] | class_to_samples[category]),
                len(groups[index]),
            ),
        )
        groups[best_group].append(category)
        capacities[best_group] -= 1
        covered[best_group] |= class_to_samples[category]

    return groups


def optimize_groups(class_to_samples, task_sizes_cfg):
    classes = sorted(
        class_to_samples,
        key=lambda category: len(class_to_samples[category]),
        reverse=True,
    )
    best_groups = initial_groups(classes, task_sizes_cfg, class_to_samples)
    best_score, best_sizes = score_groups(best_groups, class_to_samples)

    improved = True
    while improved:
        improved = False
        for left in range(len(best_groups)):
            for right in range(left + 1, len(best_groups)):
                for left_idx in range(len(best_groups[left])):
                    for right_idx in range(len(best_groups[right])):
                        candidate = [group[:] for group in best_groups]
                        candidate[left][left_idx], candidate[right][right_idx] = (
                            candidate[right][right_idx],
                            candidate[left][left_idx],
                        )
                        candidate_score, candidate_sizes = score_groups(
                            candidate, class_to_samples
                        )
                        if candidate_score < best_score:
                            best_groups = candidate
                            best_score = candidate_score
                            best_sizes = candidate_sizes
                            improved = True

    return best_groups, best_sizes


def build_labels(metadata, class_order):
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}
    labels = np.zeros((len(metadata), len(class_order)), dtype=np.float32)

    for sample_idx, categories in enumerate(metadata["categories"]):
        for category in categories:
            labels[sample_idx, class_to_idx[category]] = 1.0

    return labels


def create_label_session(labels, split_names, eval_splits, class_order, init_cls, increment, output_path):
    task_sizes_cfg = build_task_sizes(len(class_order), init_cls, increment)
    refs = np.empty((4, len(task_sizes_cfg)), dtype=h5py.ref_dtype)

    offset = 0
    with h5py.File(output_path, "w") as h5_file:
        for task_id, task_size in enumerate(task_sizes_cfg):
            current_class_ids = np.arange(offset, offset + task_size)
            seen_class_ids = np.arange(0, offset + task_size)

            train_indices = np.where(
                (split_names == "train")
                & (labels[:, current_class_ids].sum(axis=1) > 0)
            )[0]
            test_indices = np.where(
                np.isin(split_names, eval_splits)
                & (labels[:, seen_class_ids].sum(axis=1) > 0)
            )[0]

            train_labels = labels[train_indices][:, current_class_ids]
            test_labels = labels[test_indices][:, seen_class_ids]

            train_label_ds = h5_file.create_dataset(
                f"train_label_{task_id}",
                data=train_labels.T.astype(np.float32),
                compression="gzip",
            )
            train_index_ds = h5_file.create_dataset(
                f"train_index_{task_id}",
                data=(train_indices.astype(np.int64) + 1).reshape(1, -1),
                compression="gzip",
            )
            test_label_ds = h5_file.create_dataset(
                f"test_label_{task_id}",
                data=test_labels.T.astype(np.float32),
                compression="gzip",
            )
            test_index_ds = h5_file.create_dataset(
                f"test_index_{task_id}",
                data=(test_indices.astype(np.int64) + 1).reshape(1, -1),
                compression="gzip",
            )

            refs[0, task_id] = train_label_ds.ref
            refs[1, task_id] = train_index_ds.ref
            refs[2, task_id] = test_label_ds.ref
            refs[3, task_id] = test_index_ds.ref

            offset += task_size

        h5_file.create_dataset("label_session", data=refs)


def ensure_link_or_copy(source_path, output_path):
    if os.path.lexists(output_path):
        if os.path.islink(output_path) and os.path.realpath(output_path) == os.path.realpath(source_path):
            return
        os.remove(output_path)

    try:
        os.symlink(os.path.abspath(source_path), output_path)
    except OSError:
        shutil.copy2(source_path, output_path)


def compute_task_stats(labels, split_names, class_order, init_cls, increment, eval_splits):
    task_sizes_cfg = build_task_sizes(len(class_order), init_cls, increment)
    stats = []
    seen_train = np.zeros(len(split_names), dtype=bool)
    offset = 0
    for task_id, task_size in enumerate(task_sizes_cfg):
        current_class_ids = np.arange(offset, offset + task_size)
        seen_class_ids = np.arange(0, offset + task_size)
        train_mask = (split_names == "train") & (labels[:, current_class_ids].sum(axis=1) > 0)
        test_mask = np.isin(split_names, eval_splits) & (labels[:, seen_class_ids].sum(axis=1) > 0)
        seen_train |= train_mask
        stats.append(
            {
                "task": task_id,
                "classes": class_order[offset : offset + task_size],
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "cum_unique_train_samples": int(seen_train.sum()),
            }
        )
        offset += task_size
    return stats


def main():
    args = parse_args()
    source_root = os.path.abspath(args.source_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    metadata = load_metadata(source_root)
    class_to_samples = build_class_to_train_samples(metadata)
    task_sizes_cfg = build_task_sizes(len(class_to_samples), args.init_cls, args.increment)
    groups, balanced_train_sizes = optimize_groups(class_to_samples, task_sizes_cfg)
    class_order = [category for group in groups for category in group]

    labels = build_labels(metadata, class_order)
    split_names = metadata["split"].to_numpy()
    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]

    label_session_name = f"label_session_b{args.init_cls}i{args.increment}.mat"
    create_label_session(
        labels=labels,
        split_names=split_names,
        eval_splits=eval_splits,
        class_order=class_order,
        init_cls=args.init_cls,
        increment=args.increment,
        output_path=os.path.join(output_root, label_session_name),
    )

    with open(os.path.join(output_root, CLASS_FILENAME), "w") as class_file:
        json.dump(class_order, class_file, indent=2)

    for filename in [args.feature_filename, DIMENSION_FILENAME, METADATA_FILENAME, "CVPR17_Annotations.mat"]:
        source_path = os.path.join(source_root, filename)
        if os.path.exists(source_path):
            ensure_link_or_copy(source_path, os.path.join(output_root, filename))

    class_frequency = Counter()
    for categories in metadata[metadata["split"] == "train"]["categories"]:
        class_frequency.update(categories)

    manifest = {
        "source_root": source_root,
        "output_root": output_root,
        "num_samples": int(len(metadata)),
        "num_classes": int(len(class_order)),
        "task_sizes": task_sizes_cfg,
        "eval_splits": eval_splits,
        "feature_filename": args.feature_filename,
        "task_train_samples_balanced": balanced_train_sizes,
        "task_stats": compute_task_stats(
            labels=labels,
            split_names=split_names,
            class_order=class_order,
            init_cls=args.init_cls,
            increment=args.increment,
            eval_splits=eval_splits,
        ),
        "class_frequency_train": dict(class_frequency),
    }
    with open(os.path.join(output_root, MANIFEST_FILENAME), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
