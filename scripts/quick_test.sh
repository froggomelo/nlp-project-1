#!/bin/bash
# Quick test of the report's main finding, in about 15 minutes on a 16-core CPU.
#   Runs three base-model conditions of the report over 4 generations (generation 0
#   on real data, then 3 retrained ones), at the tuned learning rate and training
#   length and the first seed, side by side:
#     quick_base_real_only        real data only
#     quick_base_T3_replace100    full replacement, labels sampled at T = 3
#     quick_base_T3_accumulate    accumulation, labels sampled at T = 3
#   The runs repeat the first four generations of the report's runs of the same
#   conditions. Then draws visuals/figures/quick_test.{pdf,png} and prints the
#   results next to the report's values.
#   Takes no parameters; the switches of scripts/common.sh can be set in the
#   environment, e.g. JOBS=1 MIN_FREE_GB=3 bash scripts/quick_test.sh with less memory.
#   A run that already finished is skipped; SKIP_FINISHED=0 retrains it.
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/common.sh

GENERATIONS=4
LR=1e-4                  # the tuned base-model settings (docs/experiments.md#1-tuning)
ITERS=1000000

QUEUE=()
add QUEUE quick_base_real_only --model_variant base --lr "$LR" --train_iters "$ITERS" \
  --generations "$GENERATIONS" --init_seed 5 --train_seed 0 --gen_replace_frac 0
add QUEUE quick_base_T3_replace100 --model_variant base --lr "$LR" --train_iters "$ITERS" \
  --generations "$GENERATIONS" --init_seed 5 --train_seed 0 --gen_temperature 3 --gen_replace_frac 1.0
add QUEUE quick_base_T3_accumulate --model_variant base --lr "$LR" --train_iters "$ITERS" \
  --generations "$GENERATIONS" --init_seed 5 --train_seed 0 --gen_temperature 3 --gen_mode accumulate
echo "=== Quick test: ${#QUEUE[@]} runs to train, $JOBS at once ==="
run_queue "${QUEUE[@]}"
write_summary "$SAVE_FOLDER/${STAMP}_quick_test_summary.log" || exit 1

python -I visuals/quick_test_figure.py
