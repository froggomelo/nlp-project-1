#!/bin/bash
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

### Overarching params
SAVE_FOLDER="./results"
INIT_SEED=5

mkdir -p "$SAVE_FOLDER" || exit 1
LOG_FILE="$SAVE_FOLDER/$(date +%Y%m%d%H%M%S)_nlp_project_run_$$.log"
echo "Saving terminal output to $LOG_FILE"

### Params for Main Paper Section 3
MAIN_RUN_ITERS=1000000

python -u -m src.main \
--data_file data/omniglot_resnet18_randomized_order_s0.h5 \
--mixing_coeffs 1.0 \
--pt_burstiness 1 \
--train_context_len 2 \
--fs_relabel 5 \
--fs_relabel_split 8 2 0 \
--exemplar_split 1 4 0 \
--pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class \
--pe_classes train train train test \
--pe_exemplars train train val train \
--pe_fs_relabel_scheme train val train train \
--pe_burstiness 1 1 1 1 \
--train_iters $MAIN_RUN_ITERS \
--train_bs 32 \
--eval_every 5000 \
--eval_iters 1000 \
--lr 0.00001 \
--d_model 64 \
--class_split 50 1473 100 \
--run omniglot50_rl5 \
--ckpt_every 1000 \
--init_seed $INIT_SEED \
--base_folder "$SAVE_FOLDER" 2>&1 | tee "$LOG_FILE"
