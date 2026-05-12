#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WAIT_FOR_RESULT="${1:-}"
if [[ -n "${WAIT_FOR_RESULT}" ]]; then
  echo "Waiting for prerequisite result: ${WAIT_FOR_RESULT}"
  while [[ ! -f "${WAIT_FOR_RESULT}" ]]; do
    sleep 60
  done
fi

CONFIGS=(
  "./exps/emotic_finetune_formal_vit_b16_alpha_b5i3.json"
  "./exps/emotic_lwf_formal_vit_b16_alpha_b5i3.json"
  "./exps/emotic_ewc_formal_vit_b16_alpha_b5i3.json"
  "./exps/emotic_replay_formal_vit_b16_alpha_b5i3.json"
  "./exps/emotic_agcn_formal_vit_b16_alpha_b5i3.json"
  "./exps/emotic_clif_formal_vit_b16_alpha_b5i3.json"
  "./exps/compare/emotic_finetune_formal_resnet18_compare_alpha_b5i3.json"
  "./exps/compare/emotic_lwf_formal_resnet18_compare_alpha_b5i3.json"
  "./exps/compare/emotic_ewc_formal_resnet18_compare_alpha_b5i3.json"
  "./exps/compare/emotic_replay_formal_resnet18_compare_alpha_b5i3.json"
  "./exps/compare/emotic_agcn_formal_resnet18_compare_alpha_b5i3.json"
  "./exps/compare/emotic_clif_formal_resnet18_compare_alpha_b5i3.json"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python main_ml.py --config="${config}"
done
