#!/bin/bash
# Forced collapse: reruns the collapse conditions with the synthetic tokens
# sampled at a raised temperature, so the well-trained generation 0 model writes
# errors into the pool. At T = 1 the tuned base model regenerates almost every
# label correctly and nothing collapses; T = 3 and T = 5 give roughly 1% and 8%
# label errors. real_only does not sample, so the T = 1 real_only runs of
# scripts/nlp_project_run.sh serve as its control.
#   Runs one experiment at a time. Takes no parameters; edit the constants below.
#   The learning rate and training length per variant come from the newest
#   results/<stamp>_tuning.log of scripts/nlp_project_run.sh, unless set in the
#   environment, e.g. BASE_LR=1e-4 EXTENDED_LR=3e-4 bash scripts/forced_collapse_run.sh
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

SAVE_FOLDER="./results"
INIT_SEED=5
COLLAPSE_GENERATIONS=6       # generation 0 on real data, then 5 synthetic generations
GEN_DATASET_SIZE=20000       # pool size N
CKPT_EVERY=50000             # sparse checkpoints, as in the sweeps
TEMPERATURES=(3 5)
CONDITIONS=(replace50 replace90 replace100 accumulate)
CONDITION_ARGS=("--gen_replace_frac 0.5" "--gen_replace_frac 0.9" "--gen_replace_frac 1.0" "--gen_mode accumulate")

### Per-variant settings: the environment, else the newest tuning log
TUNING_LOG=$(ls -t "$SAVE_FOLDER"/*_tuning.log 2>/dev/null | head -1)
for var in BASE_LR BASE_ITERS EXTENDED_LR EXTENDED_ITERS; do
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
SUMMARY_FILE="$SAVE_FOLDER/${STAMP}_forced_summary.log"

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

echo "=== Settings: base lr $BASE_LR, $BASE_ITERS sequences; extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences (environment, else ${TUNING_LOG:-no tuning log}) ==="

for variant in base extended; do
  if [ "$variant" = base ]; then
    VARIANT=(--model_variant base --lr "$BASE_LR" --train_iters "$BASE_ITERS")
  else
    VARIANT=(--model_variant extended --lr "$EXTENDED_LR" --train_iters "$EXTENDED_ITERS")
  fi
  for t in "${TEMPERATURES[@]}"; do
    for i in "${!CONDITIONS[@]}"; do
      # CONDITION_ARGS entries are split into flag and value on purpose.
      run_experiment "collapse_${variant}_T${t}_${CONDITIONS[$i]}" "${VARIANT[@]}" \
        --generations "$COLLAPSE_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE" \
        --gen_temperature "$t" ${CONDITION_ARGS[$i]}
    done
  done
done

echo "=== Summary ===" | tee "$SUMMARY_FILE"
printf '%s\n' "${SUMMARY[@]}" | tee -a "$SUMMARY_FILE"
exit $FAILED
