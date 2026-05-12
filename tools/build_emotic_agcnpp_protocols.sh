#!/usr/bin/env bash
set -euo pipefail

python ./tools/build_emotic_label_session.py \
  --data-root ./data/emotic_processed \
  --output-root ./data/emotic_processed_alpha_agcnpp_il_b5i3 \
  --init-cls 5 \
  --increment 3 \
  --class-order-mode alphabetical \
  --train-assignment-mode repeat_current \
  --train-label-mode current

python ./tools/build_emotic_label_session.py \
  --data-root ./data/emotic_processed \
  --output-root ./data/emotic_processed_alpha_agcnpp_cl_b5i3 \
  --init-cls 5 \
  --increment 3 \
  --class-order-mode alphabetical \
  --train-assignment-mode repeat_current \
  --train-label-mode seen
