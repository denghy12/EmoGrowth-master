#!/usr/bin/env python

import hashlib
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = REPO_ROOT / "result"


def iter_legacy_roots():
    for path in sorted(REPO_ROOT.glob("results*")):
        if path.is_dir():
            yield path
    compare_root = REPO_ROOT / "result_vit_vs_resnet18"
    if compare_root.exists():
        for path in sorted(compare_root.glob("results*")):
            if path.is_dir():
                yield path


def infer_extractor(subject):
    subject = subject.lower()
    return "resnet18" if "resnet18" in subject else "vit"


def infer_order(root_name, subject):
    joined = f"{root_name} {subject}".lower()
    if "alpha" in joined:
        return "alphabetical"
    if "balanced" in joined:
        return "balanced"
    return "frequency"


def infer_protocol(root_name, subject):
    joined = f"{root_name} {subject}".lower()
    if "strict_owner" in joined or "strict" in joined:
        return "strict"
    return "original"


def infer_scale(root_name, subject, filename):
    joined = f"{root_name} {subject} {filename}".lower()
    return "smoke" if "smoke" in joined else "formal"


def infer_loss(root_name, subject, filename):
    joined = f"{root_name} {subject} {filename}".lower()
    for loss_name in ["bce_pos_weight", "cb_bce", "asl", "db"]:
        if loss_name in joined:
            return loss_name
    return "softmargin"


def digest(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as handle:
        hasher.update(handle.read())
    return hasher.hexdigest()


def resolve_destination_name(source_file, subject, loss_name):
    filename = source_file.name
    stem = source_file.stem
    suffix = source_file.suffix

    if filename.startswith("clif_") and not filename.startswith(f"clif_{loss_name}_"):
        return Path(f"clif_{loss_name}_{filename[len('clif_'):]}")

    if filename.endswith(".csv") and filename in {
        "finetune.csv",
        "lwf.csv",
        "ewc.csv",
        "agcn.csv",
    }:
        return Path(f"{stem}_{loss_name}{suffix}")

    if filename == "replay_prs.csv":
        return Path(f"replay_prs_{loss_name}{suffix}")

    return source_file.name


def copy_result(source_file, target_dir, target_name):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / target_name
    if target_path.exists():
        if digest(source_file) == digest(target_path):
            return target_path, False
        alt_name = f"{target_path.stem}__{source_file.parent.parent.name}{target_path.suffix}"
        target_path = target_dir / alt_name
    shutil.copy2(source_file, target_path)
    return target_path, True


def main():
    copied = 0
    skipped = 0
    for legacy_root in iter_legacy_roots():
        for source_file in sorted(legacy_root.rglob("*.csv")):
            parts = source_file.parts
            if len(parts) < 3:
                continue
            split_name = source_file.parent.name
            subject = source_file.parent.parent.name
            root_name = legacy_root.name
            extractor = infer_extractor(subject)
            order_name = infer_order(root_name, subject)
            protocol_name = infer_protocol(root_name, subject)
            scale_name = infer_scale(root_name, subject, source_file.name)
            loss_name = infer_loss(root_name, subject, source_file.name)
            target_dir = TARGET_ROOT / extractor / order_name / protocol_name / split_name / loss_name / scale_name
            target_name = resolve_destination_name(source_file, subject, loss_name)
            _, was_copied = copy_result(source_file, target_dir, target_name)
            if was_copied:
                copied += 1
            else:
                skipped += 1
    print(f"Copied {copied} files, skipped {skipped} identical files.")


if __name__ == "__main__":
    main()
