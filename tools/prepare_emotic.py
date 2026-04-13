import argparse
import csv
import json
import os
import shutil
from collections import Counter

import h5py
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16
from tqdm import tqdm

try:
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None

try:
    from torchvision.models import ViT_B_16_Weights
except ImportError:
    ViT_B_16_Weights = None


DEFAULT_EXTRACTOR = "vit_b_16"
DIMENSION_FILENAME = "affective_dimension.npy"
METADATA_FILENAME = "metadata.csv"
CLASS_FILENAME = "class_order.json"
MANIFEST_FILENAME = "prepare_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare EMOTIC features and label_session files for EmoGrowth."
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="/mnt/share/dataset/emotion/EMOTIC/CVPR17_Annotations.mat",
        help="Path to CVPR17_Annotations.mat",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/mnt/share/dataset/emotion/EMOTIC/cvpr_emotic",
        help="Root directory that contains EMOTIC images.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./data/emotic_processed",
        help="Directory used for all local derived files.",
    )
    parser.add_argument("--init-cls", type=int, default=8)
    parser.add_argument("--increment", type=int, default=6)
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated evaluation splits. Example: val,test",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.15,
        help="Relative expansion applied around the person bounding box.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for feature extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild features even if output files already exist.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=DEFAULT_EXTRACTOR,
        choices=["resnet18", "vit_b_16"],
        help="Vision backbone used for offline EMOTIC feature extraction.",
    )
    parser.add_argument(
        "--feature-filename",
        type=str,
        default=None,
        help="Optional override for the generated feature filename.",
    )
    return parser.parse_args()


def normalize_categories(categories):
    if categories is None:
        return []
    if isinstance(categories, np.ndarray):
        return [str(item) for item in categories.tolist()]
    return [str(categories)]


def resolve_categories(person):
    combined = getattr(person, "combined_categories", None)
    if combined is not None:
        categories = normalize_categories(combined)
        if categories:
            return categories

    raw = getattr(person, "annotations_categories", None)
    if isinstance(raw, np.ndarray):
        categories = []
        for annotator in raw:
            annotator_categories = normalize_categories(
                getattr(annotator, "categories", None)
            )
            categories.extend(annotator_categories)
        return sorted(set(categories))

    return normalize_categories(getattr(raw, "categories", None))


def resolve_continuous(person):
    combined = getattr(person, "combined_continuous", None)
    if combined is not None and hasattr(combined, "valence"):
        return (
            float(combined.valence),
            float(combined.arousal),
            float(combined.dominance),
        )

    raw = getattr(person, "annotations_continuous", None)
    if isinstance(raw, np.ndarray):
        values = np.array(
            [
                [
                    float(annotator.valence),
                    float(annotator.arousal),
                    float(annotator.dominance),
                ]
                for annotator in raw
            ],
            dtype=np.float32,
        )
        return tuple(values.mean(axis=0).tolist())

    return (
        float(raw.valence),
        float(raw.arousal),
        float(raw.dominance),
    )


def load_emotic_records(annotation_path, image_root):
    mat = sio.loadmat(annotation_path, squeeze_me=True, struct_as_record=False)
    records = []
    category_counter = Counter()

    for split in ["train", "val", "test"]:
        split_items = mat[split]
        if not isinstance(split_items, np.ndarray):
            split_items = np.array([split_items], dtype=object)

        for item in split_items:
            persons = item.person if isinstance(item.person, np.ndarray) else [item.person]
            image_path = os.path.join(image_root, item.folder, item.filename)
            rel_path = os.path.join(item.folder, item.filename)
            image_size = (int(item.image_size.n_col), int(item.image_size.n_row))

            for person in persons:
                categories = resolve_categories(person)
                if not categories:
                    continue

                category_counter.update(categories)
                bbox = np.asarray(person.body_bbox, dtype=np.float32).tolist()
                valence, arousal, dominance = resolve_continuous(person)
                records.append(
                    {
                        "split": split,
                        "image_path": image_path,
                        "rel_path": rel_path,
                        "image_size": image_size,
                        "bbox": bbox,
                        "categories": categories,
                        "valence": valence,
                        "arousal": arousal,
                        "dominance": dominance,
                        "gender": str(person.gender),
                        "age": str(person.age),
                    }
                )

    class_order = [
        name for name, _ in sorted(category_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return records, class_order, category_counter


def build_multi_hot(records, class_order):
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}
    labels = np.zeros((len(records), len(class_order)), dtype=np.float32)
    dimensions = np.zeros((len(records), 3), dtype=np.float32)

    for idx, record in enumerate(records):
        for category in record["categories"]:
            labels[idx, class_to_idx[category]] = 1.0
        dimensions[idx] = np.array(
            [record["valence"], record["arousal"], record["dominance"]],
            dtype=np.float32,
        )

    return labels, dimensions


class EmoticCropDataset(Dataset):
    def __init__(self, records, transform, bbox_expand_ratio):
        self.records = records
        self.transform = transform
        self.bbox_expand_ratio = bbox_expand_ratio

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            crop = crop_person(image, record["bbox"], self.bbox_expand_ratio)
        return self.transform(crop)


def crop_person(image, bbox, expand_ratio):
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0.0, min(x1, width - 1))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height - 1))
    y2 = max(0.0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return image.copy()

    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * expand_ratio
    pad_h = box_h * expand_ratio

    crop_box = (
        int(max(0.0, x1 - pad_w)),
        int(max(0.0, y1 - pad_h)),
        int(min(width, x2 + pad_w)),
        int(min(height, y2 + pad_h)),
    )
    return image.crop(crop_box)


def get_feature_filename(extractor_name, custom_filename=None):
    if custom_filename:
        return custom_filename
    return "feature_{}_bbox.npy".format(extractor_name.lower())


def _imagenet_fallback_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def create_extractor(extractor_name):
    extractor_name = extractor_name.lower()
    if extractor_name == "resnet18":
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1
            try:
                model = resnet18(weights=weights)
                transform = weights.transforms()
            except Exception:
                model = resnet18(weights=None)
                transform = _imagenet_fallback_transform()
        else:
            model = resnet18(pretrained=True)
            transform = _imagenet_fallback_transform()
        model.fc = torch.nn.Identity()
    elif extractor_name == "vit_b_16":
        if ViT_B_16_Weights is None:
            raise RuntimeError(
                "torchvision in this environment does not provide ViT_B_16_Weights."
            )
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        try:
            model = vit_b_16(weights=weights)
            transform = weights.transforms()
        except Exception:
            model = vit_b_16(weights=None)
            transform = _imagenet_fallback_transform()
        model.heads = torch.nn.Identity()
    else:
        raise ValueError("Unsupported extractor: {}".format(extractor_name))

    model.eval()
    return model, transform


def extract_features(
    records,
    output_path,
    batch_size,
    num_workers,
    device_name,
    bbox_expand_ratio,
    extractor_name,
):
    if os.path.exists(output_path):
        return np.load(output_path).astype(np.float32)

    model, transform = create_extractor(extractor_name)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = EmoticCropDataset(records, transform, bbox_expand_ratio)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting EMOTIC features"):
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch)
            outputs = outputs.reshape(outputs.shape[0], -1).cpu().numpy().astype(np.float32)
            features.append(outputs)

    feature_array = np.concatenate(features, axis=0)
    np.save(output_path, feature_array)
    return feature_array


def build_task_sizes(total_class, init_cls, increment):
    task_sizes = [init_cls]
    while sum(task_sizes) + increment <= total_class:
        task_sizes.append(increment)
    remainder = total_class - sum(task_sizes)
    if remainder > 0:
        task_sizes.append(remainder)
    return task_sizes


def write_metadata(records, output_path):
    fieldnames = [
        "sample_id",
        "split",
        "rel_path",
        "image_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "valence",
        "arousal",
        "dominance",
        "gender",
        "age",
        "categories",
    ]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, record in enumerate(records):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": record["split"],
                    "rel_path": record["rel_path"],
                    "image_path": record["image_path"],
                    "bbox_x1": record["bbox"][0],
                    "bbox_y1": record["bbox"][1],
                    "bbox_x2": record["bbox"][2],
                    "bbox_y2": record["bbox"][3],
                    "valence": record["valence"],
                    "arousal": record["arousal"],
                    "dominance": record["dominance"],
                    "gender": record["gender"],
                    "age": record["age"],
                    "categories": "|".join(record["categories"]),
                }
            )


def create_label_session(
    labels,
    split_names,
    eval_splits,
    class_order,
    init_cls,
    increment,
    output_path,
):
    task_sizes = build_task_sizes(len(class_order), init_cls, increment)
    class_starts = []
    offset = 0
    for size in task_sizes:
        class_starts.append((offset, offset + size))
        offset += size

    refs = np.empty((4, len(task_sizes)), dtype=h5py.ref_dtype)
    with h5py.File(output_path, "w") as h5_file:
        for task_id, (start, end) in enumerate(class_starts):
            current_class_ids = np.arange(start, end)
            seen_class_ids = np.arange(0, end)

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
                "train_label_{}".format(task_id),
                data=train_labels.T.astype(np.float32),
                compression="gzip",
            )
            train_index_ds = h5_file.create_dataset(
                "train_index_{}".format(task_id),
                data=(train_indices.astype(np.int64) + 1).reshape(1, -1),
                compression="gzip",
            )
            test_label_ds = h5_file.create_dataset(
                "test_label_{}".format(task_id),
                data=test_labels.T.astype(np.float32),
                compression="gzip",
            )
            test_index_ds = h5_file.create_dataset(
                "test_index_{}".format(task_id),
                data=(test_indices.astype(np.int64) + 1).reshape(1, -1),
                compression="gzip",
            )

            refs[0, task_id] = train_label_ds.ref
            refs[1, task_id] = train_index_ds.ref
            refs[2, task_id] = test_label_ds.ref
            refs[3, task_id] = test_index_ds.ref

        h5_file.create_dataset("label_session", data=refs)


def maybe_copy_annotation(annotation_path, output_root):
    copied_path = os.path.join(output_root, os.path.basename(annotation_path))
    if not os.path.exists(copied_path):
        shutil.copy2(annotation_path, copied_path)
    return copied_path


def main():
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    copied_annotation = maybe_copy_annotation(args.annotation_path, output_root)
    records, class_order, category_counter = load_emotic_records(
        copied_annotation, args.image_root
    )
    labels, dimensions = build_multi_hot(records, class_order)

    feature_path = os.path.join(
        output_root, get_feature_filename(args.extractor, args.feature_filename)
    )
    if args.overwrite and os.path.exists(feature_path):
        os.remove(feature_path)

    features = extract_features(
        records=records,
        output_path=feature_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        bbox_expand_ratio=args.bbox_expand_ratio,
        extractor_name=args.extractor,
    )
    np.save(os.path.join(output_root, DIMENSION_FILENAME), dimensions.astype(np.float32))

    write_metadata(records, os.path.join(output_root, METADATA_FILENAME))
    with open(os.path.join(output_root, CLASS_FILENAME), "w") as class_file:
        json.dump(class_order, class_file, indent=2)

    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    label_session_name = "label_session_b{}i{}.mat".format(args.init_cls, args.increment)
    create_label_session(
        labels=labels,
        split_names=np.array([record["split"] for record in records]),
        eval_splits=eval_splits,
        class_order=class_order,
        init_cls=args.init_cls,
        increment=args.increment,
        output_path=os.path.join(output_root, label_session_name),
    )

    manifest = {
        "annotation_path": os.path.abspath(args.annotation_path),
        "copied_annotation_path": copied_annotation,
        "image_root": os.path.abspath(args.image_root),
        "output_root": output_root,
        "num_samples": int(len(records)),
        "num_classes": int(len(class_order)),
        "task_sizes": build_task_sizes(len(class_order), args.init_cls, args.increment),
        "extractor": args.extractor,
        "feature_path": feature_path,
        "feature_shape": list(features.shape),
        "eval_splits": eval_splits,
        "class_frequency": dict(category_counter),
    }
    with open(os.path.join(output_root, MANIFEST_FILENAME), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
