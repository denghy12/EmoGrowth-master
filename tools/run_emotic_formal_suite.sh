#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

source /opt/conda/etc/profile.d/conda.sh
conda activate EmoGrowth

configs=(
  "./exps/emotic_ewc_formal.json"
  "./exps/emotic_lwf_formal.json"
  "./exps/emotic_replay_formal.json"
  "./exps/emotic_agcn_formal.json"
  "./exps/emotic_clif_formal.json"
)

for config in "${configs[@]}"; do
  printf '[%s] Running %s\n' "$(date '+%F %T')" "$config"
  python main_ml.py --config="$config"
done
