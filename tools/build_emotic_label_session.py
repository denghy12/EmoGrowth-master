import argparse
import shutil
import json
import os
import glob

import h5py
import numpy as np
import pandas as pd


METADATA_FILENAME = "metadata.csv"
CLASS_FILENAME = "class_order.json"
DIMENSION_FILENAME = "affective_dimension.npy"
MANIFEST_FILENAME = "label_session_manifest.json"
FEATURE_GLOB = "feature_*_bbox.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an EMOTIC label_session file from an existing metadata/class order pair."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory that contains metadata.csv and class_order.json.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output directory. Defaults to data-root.",
    )
    parser.add_argument("--init-cls", type=int, required=True)
    parser.add_argument("--increment", type=int, required=True)
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated evaluation splits.",
    )
    parser.add_argument(
        "--class-order-mode",
        type=str,
        default="existing",
        choices=["existing", "alphabetical"],
        help="How to derive the class order used by the generated label_session.",
    )
    parser.add_argument(
        "--train-assignment-mode",
        type=str,
        default="repeat_current",
        choices=["repeat_current", "strict_owner"],
        help="How training samples are assigned to tasks.",
    )
    parser.add_argument(
        "--train-label-mode",
        type=str,
        default="auto",
        choices=["auto", "current", "seen"],
        help="Which label slice is exposed to the training loader.",
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


def load_metadata(data_root):
    metadata = pd.read_csv(os.path.join(data_root, METADATA_FILENAME))
    metadata["categories"] = metadata["categories"].fillna("").map(
        lambda value: [item for item in str(value).split("|") if item]
    )
    return metadata


def build_labels(metadata, class_order):
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}
    labels = np.zeros((len(metadata), len(class_order)), dtype=np.float32)

    for sample_idx, categories in enumerate(metadata["categories"]):
        for category in categories:
            labels[sample_idx, class_to_idx[category]] = 1.0

    return labels


def resolve_class_order(data_root, mode):
    class_order = json.load(open(os.path.join(data_root, CLASS_FILENAME)))
    if mode == "alphabetical":
        return sorted(class_order)
    return class_order


def build_task_groups(class_order, init_cls, increment):
    task_sizes_cfg = build_task_sizes(len(class_order), init_cls, increment)
    groups = []
    offset = 0
    for task_id, task_size in enumerate(task_sizes_cfg):
        groups.append(
            {
                "task_id": task_id,
                "classes": class_order[offset : offset + task_size],
            }
        )
        offset += task_size
    return groups


def resolve_train_label_mode(train_assignment_mode, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    if train_assignment_mode == "strict_owner":
        return "seen"
    return "current"


def build_class_task_ids(total_class, init_cls, increment):
    task_sizes_cfg = build_task_sizes(total_class, init_cls, increment)
    class_task_ids = np.empty(total_class, dtype=np.int64)
    offset = 0
    for task_id, task_size in enumerate(task_sizes_cfg):
        class_task_ids[offset : offset + task_size] = task_id
        offset += task_size
    return class_task_ids, task_sizes_cfg


def compute_owner_tasks(labels, class_task_ids):
    owner_tasks = np.full(labels.shape[0], -1, dtype=np.int64)
    for sample_idx in range(labels.shape[0]):
        positive_ids = np.where(labels[sample_idx] > 0)[0]
        if positive_ids.size > 0:
            owner_tasks[sample_idx] = int(class_task_ids[positive_ids].max())
    return owner_tasks


def select_train_mask(split_names, labels, current_class_ids, task_id, owner_tasks, assignment_mode):
    train_split = split_names == "train"
    if assignment_mode == "strict_owner":
        return train_split & (owner_tasks == task_id)
    return train_split & (labels[:, current_class_ids].sum(axis=1) > 0)


def create_label_session(
    labels,
    split_names,
    eval_splits,
    class_order,
    init_cls,
    increment,
    output_path,
    train_assignment_mode,
    train_label_mode,
):
    task_sizes_cfg = build_task_sizes(len(class_order), init_cls, increment)
    refs = np.empty((4, len(task_sizes_cfg)), dtype=h5py.ref_dtype)
    class_task_ids, _ = build_class_task_ids(len(class_order), init_cls, increment)
    owner_tasks = compute_owner_tasks(labels, class_task_ids)
    offset = 0

    with h5py.File(output_path, "w") as h5_file:
        for task_id, task_size in enumerate(task_sizes_cfg):
            current_class_ids = np.arange(offset, offset + task_size)
            seen_class_ids = np.arange(0, offset + task_size)

            train_mask = select_train_mask(
                split_names=split_names,
                labels=labels,
                current_class_ids=current_class_ids,
                task_id=task_id,
                owner_tasks=owner_tasks,
                assignment_mode=train_assignment_mode,
            )
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(
                np.isin(split_names, eval_splits)
                & (labels[:, seen_class_ids].sum(axis=1) > 0)
            )[0]

            if train_label_mode == "seen":
                train_label_ids = seen_class_ids
            else:
                train_label_ids = current_class_ids

            train_labels = labels[train_indices][:, train_label_ids]
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


def ensure_link_or_copy(source_path, target_path):
    if os.path.lexists(target_path):
        if os.path.islink(target_path) and os.path.realpath(target_path) == os.path.realpath(source_path):
            return
        os.remove(target_path)

    try:
        os.symlink(os.path.abspath(source_path), target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def copy_if_exists(source_path, target_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)


def compute_task_stats(
    labels,
    split_names,
    class_order,
    init_cls,
    increment,
    eval_splits,
    train_assignment_mode,
    train_label_mode,
):
    class_task_ids, task_sizes_cfg = build_class_task_ids(len(class_order), init_cls, increment)
    owner_tasks = compute_owner_tasks(labels, class_task_ids)
    offset = 0
    seen_train = np.zeros(len(split_names), dtype=bool)
    stats = []
    total_occurrences = 0
    for task_id, task_size in enumerate(task_sizes_cfg):
        current_class_ids = np.arange(offset, offset + task_size)
        seen_class_ids = np.arange(0, offset + task_size)
        train_mask = select_train_mask(
            split_names=split_names,
            labels=labels,
            current_class_ids=current_class_ids,
            task_id=task_id,
            owner_tasks=owner_tasks,
            assignment_mode=train_assignment_mode,
        )
        test_mask = np.isin(split_names, eval_splits) & (labels[:, seen_class_ids].sum(axis=1) > 0)
        seen_train |= train_mask
        train_label_dim = len(seen_class_ids) if train_label_mode == "seen" else len(current_class_ids)
        train_samples = int(train_mask.sum())
        total_occurrences += train_samples
        stats.append(
            {
                "task": task_id,
                "classes": class_order[offset : offset + task_size],
                "train_samples": train_samples,
                "train_label_dim": train_label_dim,
                "test_samples": int(test_mask.sum()),
                "cum_unique_train_samples": int(seen_train.sum()),
            }
        )
        offset += task_size
    unique_train = int(((split_names == "train") & (labels.sum(axis=1) > 0)).sum())
    repeat_ratio = float(total_occurrences) / float(unique_train) if unique_train else 0.0
    return stats, total_occurrences, unique_train, repeat_ratio


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root or data_root)
    os.makedirs(output_root, exist_ok=True)

    class_order = resolve_class_order(data_root, args.class_order_mode)
    metadata = load_metadata(data_root)
    labels = build_labels(metadata, class_order)
    split_names = metadata["split"].to_numpy()
    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    train_label_mode = resolve_train_label_mode(
        args.train_assignment_mode, args.train_label_mode
    )
    output_name = f"label_session_b{args.init_cls}i{args.increment}.mat"
    output_path = os.path.join(output_root, output_name)

    metadata_path = os.path.join(data_root, METADATA_FILENAME)
    if os.path.abspath(metadata_path) != os.path.join(output_root, METADATA_FILENAME):
        shutil.copy2(metadata_path, os.path.join(output_root, METADATA_FILENAME))
    copy_if_exists(
        os.path.join(data_root, DIMENSION_FILENAME),
        os.path.join(output_root, DIMENSION_FILENAME),
    )
    for feature_path in sorted(glob.glob(os.path.join(data_root, FEATURE_GLOB))):
        ensure_link_or_copy(feature_path, os.path.join(output_root, os.path.basename(feature_path)))

    with open(os.path.join(output_root, CLASS_FILENAME), "w") as class_file:
        json.dump(class_order, class_file, indent=2)

    create_label_session(
        labels=labels,
        split_names=split_names,
        eval_splits=eval_splits,
        class_order=class_order,
        init_cls=args.init_cls,
        increment=args.increment,
        output_path=output_path,
        train_assignment_mode=args.train_assignment_mode,
        train_label_mode=train_label_mode,
    )

    task_stats, total_occurrences, unique_train, repeat_ratio = compute_task_stats(
        labels=labels,
        split_names=split_names,
        class_order=class_order,
        init_cls=args.init_cls,
        increment=args.increment,
        eval_splits=eval_splits,
        train_assignment_mode=args.train_assignment_mode,
        train_label_mode=train_label_mode,
    )

    manifest = {
        "source_root": data_root,
        "output_root": output_root,
        "class_order_mode": args.class_order_mode,
        "train_assignment_mode": args.train_assignment_mode,
        "train_label_mode": train_label_mode,
        "init_cls": args.init_cls,
        "increment": args.increment,
        "eval_splits": eval_splits,
        "task_groups": build_task_groups(class_order, args.init_cls, args.increment),
        "task_stats": task_stats,
        "train_total_occurrences": total_occurrences,
        "train_unique_samples": unique_train,
        "train_repeat_ratio": repeat_ratio,
    }
    with open(os.path.join(output_root, MANIFEST_FILENAME), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
