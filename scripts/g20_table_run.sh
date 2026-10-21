#!/bin/bash
# Extends the remaining 5-generation rows of the results table to 20 generations,
# matching the 20-generation real_only, replace100 and accumulate runs:
#   L8B1:     L = 8, B = 1, full replacement, T = 1
#   L8B4:     L = 8, B = 4, full replacement, T = 1
#   T3_50:    T = 3, 50% replaced
#   T3_90:    T = 3, 90% replaced
#   T3_100:   T = 3, full replacement
#   T3_ACC:   T = 3, accumulation
#   VARIANT=base (default) or extended picks the model. The base T3_100 and T3_ACC
#   runs already come from scripts/followup_run.sh, so for the base model RUN=all
#   runs only the first four; for the extended model it runs all six.
#   RUN=<one of the names above> runs one experiment, so several can run side by
#   side (cap the threads, e.g. OMP_NUM_THREADS=4). The learning rate and training
#   length come from the newest results/<stamp>_tuning.log of
#   scripts/nlp_project_run.sh, unless set in the environment as BASE_LR and
#   BASE_ITERS, or EXTENDED_LR and EXTENDED_ITERS.
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

RUN=${RUN:-all}              # L8B1, L8B4, T3_50, T3_90, T3_100, T3_ACC or all
VARIANT=${VARIANT:-base}     # base or extended
SAVE_FOLDER="./results"
INIT_SEED=5
GENERATIONS=20
GEN_DATASET_SIZE=20000       # pool size N
CKPT_EVERY=50000             # sparse checkpoints, as in the sweeps
BURST_CONTEXT_LEN=8

case "$VARIANT" in base|extended) ;; *)
  echo "Unknown VARIANT=$VARIANT: use base or extended" >&2
  exit 1 ;;
esac
PREFIX=$(echo "$VARIANT" | tr '[:lower:]' '[:upper:]')   # BASE or EXTENDED

### Model settings: the environment, else the newest tuning log
TUNING_LOG=$(ls -t "$SAVE_FOLDER"/*_tuning.log 2>/dev/null | head -1)
for var in ${PREFIX}_LR ${PREFIX}_ITERS; do
  if [ -z "${!var}" ] && [ -n "$TUNING_LOG" ]; then
    value=$(grep -E "^${var}=[0-9.e+-]+$" "$TUNING_LOG" | tail -1 | cut -d= -f2)
    printf -v "$var" '%s' "$value"
  fi
  if [ -z "${!var}" ]; then
    echo "No value for $var: set it in the environment or run the tuning stage first" >&2
    exit 1
  fi
done

mkdir -p "$SAVE_FOLDER" || exit 1
STAMP=$(date +%Y%m%d%H%M%S)
SUMMARY_FILE="$SAVE_FOLDER/${STAMP}_g20_${VARIANT}_${RUN}_summary.log"
LR_VAR=${PREFIX}_LR
ITERS_VAR=${PREFIX}_ITERS

COMMON_ARGS=(
  --data_file data/omniglot_resnet18_randomized_order_s0.h5
  --mixing_coeffs 1.0
  --pt_burstiness 1
  --train_context_len 2
  --fs_relabel 5
  --fs_relabel_split 8 2 0
  --exemplar_split 1 4 0
  --pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class
  --pe_classes train train train test
  --pe_exemplars train train val train
  --pe_fs_relabel_scheme train val train train
  --pe_burstiness 1 1 1 1
  --train_bs 32
  --eval_every 5000
  --eval_iters 1000
  --d_model 64
  --class_split 50 1473 100
  --ckpt_every "$CKPT_EVERY"
  --init_seed "$INIT_SEED"
  --base_folder "$SAVE_FOLDER"
  --circuit-discovery
  --model_variant "$VARIANT" --lr "${!LR_VAR}" --train_iters "${!ITERS_VAR}"
  --generations "$GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE"
)

SUMMARY=()
FAILED=0

# run_experiment <name> <extra src.main args...>
# Saves terminal output to results/<stamp>_<name>.log and continues on failure.
run_experiment() {
  local name=$1
  shift
  local log="$SAVE_FOLDER/${STAMP}_${name}.log"
  echo "=== Running $name (output saved to $log) ==="
  python -u -m src.main "${COMMON_ARGS[@]}" --run "$name" "$@" 2>&1 | tee "$log"
  local status=$?
  if [ $status -eq 0 ]; then
    SUMMARY+=("OK      $name  $log")
  else
    SUMMARY+=("FAILED  $name  $log (exit $status)")
    FAILED=1
  fi
}

# burst_args <B>: a longer context with burstiness B, as in the burstiness sweep
# of scripts/nlp_project_run.sh; the evaluators use the same setting.
burst_args() {
  echo --train_context_len "$BURST_CONTEXT_LEN" --pt_burstiness "$1" --pt_unique_rest 1 \
    --pe_burstiness "$1" "$1" "$1" "$1" --pe_unique_rest 1 1 1 1
}

echo "=== Run $RUN, $VARIANT model. Settings: lr ${!LR_VAR}, ${!ITERS_VAR} sequences (environment, else ${TUNING_LOG:-no tuning log}) ==="

case "$RUN" in L8B1|all)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_L${BURST_CONTEXT_LEN}B1_replace100" $(burst_args 1) --gen_replace_frac 1.0 ;;
esac
case "$RUN" in L8B4|all)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_L${BURST_CONTEXT_LEN}B4_replace100" $(burst_args 4) --gen_replace_frac 1.0 ;;
esac
case "$RUN" in T3_50|all)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_T3_replace50" --gen_temperature 3 --gen_replace_frac 0.5 ;;
esac
case "$RUN" in T3_90|all)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_T3_replace90" --gen_temperature 3 --gen_replace_frac 0.9 ;;
esac
# The base model's versions of these two already come from scripts/followup_run.sh
ALL_LONG=$([ "$VARIANT" = extended ] && echo all)
case "$RUN" in T3_100|$ALL_LONG)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_T3_replace100" --gen_temperature 3 --gen_replace_frac 1.0 ;;
esac
case "$RUN" in T3_ACC|$ALL_LONG)
  run_experiment "collapse_${VARIANT}_G${GENERATIONS}_T3_accumulate" --gen_temperature 3 --gen_mode accumulate ;;
esac

if [ ${#SUMMARY[@]} -eq 0 ]; then
  echo "Unknown RUN=$RUN: use L8B1, L8B4, T3_50, T3_90, T3_100, T3_ACC or all" >&2
  exit 1
fi

echo "=== Summary ===" | tee "$SUMMARY_FILE"
printf '%s\n' "${SUMMARY[@]}" | tee -a "$SUMMARY_FILE"
exit $FAILED
