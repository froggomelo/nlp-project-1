#!/bin/bash
# Tunes the learning rate and training length of both models.
#   Trains generation 0 of each model on the real pool for 3,000,000 sequences at
#   learning rates 1e-5, 3e-5, 1e-4 and 3e-4, with a fifth evaluator on the 1,473
#   validation classes. scripts/select_hparams.py then picks, per model, the
#   learning rate and training length (1, 2 or 3 million sequences) with the best
#   validation accuracy, preferring cheaper settings within 0.01. It writes the
#   table and the choice, as BASE_LR=..., to results/<stamp>_tuning.log, which
#   scripts/nlp_project_run.sh reads.
#   Takes no parameters; JOBS and the other switches of scripts/common.sh can be set
#   in the environment, e.g. JOBS=4 bash scripts/tune_run.sh
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/common.sh

TUNE_LRS=(1e-5 3e-5 1e-4 3e-4)
TUNE_LENGTHS=(1000000 2000000 3000000)  # candidate training lengths, all read from the longest run

QUEUE=()
for variant in base extended; do
  for lr in "${TUNE_LRS[@]}"; do
    # The validation-class evaluator scores each setting, so the test classes are
    # never used for selection
    add QUEUE "tune_${variant}_lr${lr}" --model_variant "$variant" --generations 1 \
      --lr "$lr" --train_iters "${TUNE_LENGTHS[-1]}" \
      --pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class fsl_val_class \
      --pe_classes train train train test val \
      --pe_exemplars train train val train train \
      --pe_fs_relabel_scheme train val train train train \
      --pe_burstiness 1 1 1 1 1
  done
done
echo "=== Tuning: ${#QUEUE[@]} runs to train, $JOBS at once ==="
run_queue "${QUEUE[@]}"
write_summary "$SAVE_FOLDER/${STAMP}_tuning_summary.log" || exit 1

TUNE_LOG="$SAVE_FOLDER/${STAMP}_tuning.log"
if CHOSEN=$(python scripts/select_hparams.py --results "$SAVE_FOLDER" \
    --lrs "${TUNE_LRS[@]}" --lengths "${TUNE_LENGTHS[@]}" 2>"$TUNE_LOG"); then
  printf '%s\n' "$CHOSEN" >> "$TUNE_LOG"
  cat "$TUNE_LOG"
else
  cat "$TUNE_LOG"
  echo "Tuning selection failed; see $TUNE_LOG" >&2
  exit 1
fi
