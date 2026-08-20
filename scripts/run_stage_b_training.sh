#!/bin/zsh
# Long-running CPU training entrypoint for the local job manager.
cd "$(dirname "$0")/.." || exit 1
export MPLCONFIGDIR=/private/tmp/matting_mpl
mkdir -p "$MPLCONFIGDIR"
exec .venv/bin/python -u scripts/train_matting.py \
  --epochs 40 \
  --batch-size 4 \
  --base-channels 8 \
  --output-dir matting_models/stage_b_alpha_v1 \
  >> matting_models/stage_b_alpha_v1/training.log 2>&1
