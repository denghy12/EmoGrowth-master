import argparse
import os
import pickle
from io import BytesIO

import numpy as np
from PIL import Image
from tfrecord.reader import tfrecord_loader


def load_label_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def convert_split(tfrecord_path):
    data = []
    fine_labels = []
    coarse_labels = []
    filenames = []

    for record in tfrecord_loader(tfrecord_path, None):
        image = Image.open(BytesIO(record["image"])).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        # Match CIFAR original storage: channel-first flattened rows.
        data.append(arr.transpose(2, 0, 1).reshape(-1))
        fine_labels.append(int(record["label"][0]))
        coarse_labels.append(int(record["coarse_label"][0]))
        filenames.append(record["id"].decode("utf-8") + ".png")

    return {
        b"batch_label": os.path.basename(tfrecord_path).encode("utf-8"),
        b"data": np.stack(data, axis=0),
        b"fine_labels": fine_labels,
        b"coarse_labels": coarse_labels,
        b"filenames": filenames,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert TFDS CIFAR100 to cifar-100-python format.")
    parser.add_argument("--src", required=True, help="Source TFDS CIFAR100 version directory.")
    parser.add_argument("--dst", required=True, help="Destination cifar-100-python directory.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    train_path = os.path.join(args.src, "cifar100-train.tfrecord-00000-of-00001")
    test_path = os.path.join(args.src, "cifar100-test.tfrecord-00000-of-00001")
    fine_label_path = os.path.join(args.src, "label.labels.txt")
    coarse_label_path = os.path.join(args.src, "coarse_label.labels.txt")

    fine_label_names = load_label_names(fine_label_path)
    coarse_label_names = load_label_names(coarse_label_path)

    train_dict = convert_split(train_path)
    test_dict = convert_split(test_path)
    meta_dict = {
        b"fine_label_names": fine_label_names,
        b"coarse_label_names": coarse_label_names,
    }

    with open(os.path.join(args.dst, "train"), "wb") as f:
        pickle.dump(train_dict, f, protocol=2)
    with open(os.path.join(args.dst, "test"), "wb") as f:
        pickle.dump(test_dict, f, protocol=2)
    with open(os.path.join(args.dst, "meta"), "wb") as f:
        pickle.dump(meta_dict, f, protocol=2)

    print(f"Converted CIFAR100 TFDS data into {args.dst}")


if __name__ == "__main__":
    main()
