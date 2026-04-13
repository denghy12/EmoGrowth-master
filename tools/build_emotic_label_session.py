import argparse
import shutil
import json
import os

import h5py
import numpy as np
import pandas as pd


METADATA_FILENAME = "metadata.csv"
CLASS_FILENAME = "class_order.json"
DIMENSION_FILENAME = "affective_dimension.npy"
MANIFEST_FILENAME = "label_session_manifest.json"


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


def copy_if_exists(source_path, target_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)


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
    output_name = f"label_session_b{args.init_cls}i{args.increment}.mat"
    output_path = os.path.join(output_root, output_name)

    metadata_path = os.path.join(data_root, METADATA_FILENAME)
    if os.path.abspath(metadata_path) != os.path.join(output_root, METADATA_FILENAME):
        shutil.copy2(metadata_path, os.path.join(output_root, METADATA_FILENAME))
    copy_if_exists(
        os.path.join(data_root, DIMENSION_FILENAME),
        os.path.join(output_root, DIMENSION_FILENAME),
    )

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
    )

    manifest = {
        "source_root": data_root,
        "output_root": output_root,
        "class_order_mode": args.class_order_mode,
        "init_cls": args.init_cls,
        "increment": args.increment,
        "eval_splits": eval_splits,
        "task_groups": build_task_groups(class_order, args.init_cls, args.increment),
    }
    with open(os.path.join(output_root, MANIFEST_FILENAME), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
