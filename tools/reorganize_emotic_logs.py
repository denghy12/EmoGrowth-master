#!/usr/bin/env python

import hashlib
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs"
ARCHIVE_LOG_ROOT = REPO_ROOT / "archive" / "legacy_logs"
LEGACY_METHOD_DIRS = ["agcn", "clif", "ewc", "finetune", "lwf", "replay"]


def digest(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as handle:
        hasher.update(handle.read())
    return hasher.hexdigest()


def infer_extractor(text):
    return "resnet18" if "resnet18" in text else "vit"


def infer_order(text):
    if "alpha" in text:
        return "alphabetical"
    if "balanced" in text:
        return "balanced"
    return "frequency"


def infer_protocol(text):
    if "strict_owner" in text or "strict" in text:
        return "strict"
    return "original"


def infer_split(path):
    parts = path.parts
    if len(parts) >= 2 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"B{parts[-3]}I{parts[-2]}"
    text = " ".join(parts).lower()
    if "b5i3" in text:
        return "B5I3"
    if "b8i6" in text:
        return "B8I6"
    return "unknown"


def infer_loss(text):
    for loss_name in ["bce_pos_weight", "cb_bce", "asl", "db"]:
        if loss_name in text:
            return loss_name
    return "softmargin"


def infer_scale(text):
    return "smoke" if "smoke" in text else "formal"


def target_dir_for_log(log_file, source_root):
    rel_path = log_file.relative_to(source_root)
    text = " ".join(rel_path.parts).lower()
    extractor = infer_extractor(text)
    order_name = infer_order(text)
    protocol_name = infer_protocol(text)
    split_name = infer_split(rel_path)
    loss_name = infer_loss(text)
    scale_name = infer_scale(text)
    method_name = rel_path.parts[0]
    return LOG_ROOT / extractor / order_name / protocol_name / split_name / loss_name / scale_name / method_name


def copy_log(source_file, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_file.name
    if target_path.exists() and digest(source_file) == digest(target_path):
        return False
    if target_path.exists():
        target_path = target_dir / f"{target_path.stem}__legacy{target_path.suffix}"
    shutil.copy2(source_file, target_path)
    return True


def main():
    copied = 0
    skipped = 0
    for source_root in [ARCHIVE_LOG_ROOT, LOG_ROOT]:
        if not source_root.exists():
            continue
        for method_dir in LEGACY_METHOD_DIRS:
            root = source_root / method_dir
            if not root.exists():
                continue
            for log_file in sorted(root.rglob("*.log")):
                if not log_file.exists():
                    continue
                target_dir = target_dir_for_log(log_file, source_root)
                if copy_log(log_file, target_dir):
                    copied += 1
                else:
                    skipped += 1
    print(f"Copied {copied} log files, skipped {skipped} identical logs.")


if __name__ == "__main__":
    main()
