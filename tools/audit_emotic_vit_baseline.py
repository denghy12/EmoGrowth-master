#!/usr/bin/env python

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_issue(issues, path, message):
    issues.append(f"{path}: {message}")


def read_text(path):
    return path.read_text(encoding="utf-8")


def audit_defaults(issues):
    trainer_path = REPO_ROOT / "trainer_ml.py"
    trainer_text = read_text(trainer_path)
    if 'os.path.join(data_root, "feature_resnet18_bbox.npy")' in trainer_text:
        add_issue(issues, trainer_path, "EMOTIC default feature path still points to ResNet18.")

    prepare_path = REPO_ROOT / "tools" / "prepare_emotic.py"
    prepare_text = read_text(prepare_path)
    if 'DEFAULT_EXTRACTOR = "resnet18"' in prepare_text:
        add_issue(issues, prepare_path, "EMOTIC default extractor still points to ResNet18.")


def audit_exps(issues):
    exp_root = REPO_ROOT / "exps"
    for path in sorted(exp_root.glob("emotic_*.json")):
        text = read_text(path)
        if "resnet18" in text:
            add_issue(issues, path, "Root EMOTIC config still contains ResNet18 references.")

    compare_root = exp_root / "compare"
    for path in sorted(compare_root.glob("*.json")):
        if "compare" not in path.name and "legacy" not in path.name:
            add_issue(issues, path, "Compare config filename should include compare or legacy.")


def audit_scripts(issues):
    tools_root = REPO_ROOT / "tools"
    for path in sorted(tools_root.glob("run_emotic_*.sh")):
        if "compare" in path.name:
            continue
        text = read_text(path)
        if "resnet18" in text:
            add_issue(issues, path, "Default EMOTIC run script still references ResNet18.")


def main():
    issues = []
    audit_defaults(issues)
    audit_exps(issues)
    audit_scripts(issues)

    if issues:
        print("EMOTIC ViT baseline audit failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("EMOTIC ViT baseline audit passed.")


if __name__ == "__main__":
    main()
