import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd


METADATA_FILENAME = "metadata.csv"
CLASS_FILENAME = "class_order.json"


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
    parser.add_argument("--init-cls", type=int, required=True)
    parser.add_argument("--increment", type=int, required=True)
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated evaluation splits.",
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


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    class_order = json.load(open(os.path.join(data_root, CLASS_FILENAME)))
    metadata = load_metadata(data_root)
    labels = build_labels(metadata, class_order)
    split_names = metadata["split"].to_numpy()
    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    output_name = f"label_session_b{args.init_cls}i{args.increment}.mat"
    output_path = os.path.join(data_root, output_name)

    create_label_session(
        labels=labels,
        split_names=split_names,
        eval_splits=eval_splits,
        class_order=class_order,
        init_cls=args.init_cls,
        increment=args.increment,
        output_path=output_path,
    )

    print(output_path)


if __name__ == "__main__":
    main()
