#!/bin/bash
# Follow-up collapse experiments, in two groups that can run side by side:
#   lowtemp: the extended model sampled at T = 0.5, which sharpens the next
#     symbol distribution, in the two settings where that distribution has a
#     tail to lose: the next symbol drawn from all 50 training classes (global)
#     and Zipf alpha = 1.0 classes (context). Tests whether rare symbols vanish.
#   long: the base model at T = 3 over 20 generations, full replacement against
#     accumulation. Tests whether accumulation keeps the label error bounded
#     (Gerstgrasser et al., 2024) once its own samples are mostly wrong.
# real_only does not sample, so the T = 1 real_only runs of
# scripts/nlp_project_run.sh serve as its control and are not rerun.
#   GROUP=lowtemp or GROUP=long runs one group; the default runs both in order.
#   The learning rate and training length per variant come from the newest
#   results/<stamp>_tuning.log of scripts/nlp_project_run.sh, unless set in the
#   environment, e.g. BASE_LR=1e-4 EXTENDED_LR=3e-4 GROUP=long bash scripts/followup_run.sh
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

GROUP=${GROUP:-all}          # lowtemp, long or all
SAVE_FOLDER="./results"
INIT_SEED=5
GEN_DATASET_SIZE=20000       # pool size N
CKPT_EVERY=50000             # sparse checkpoints, as in the sweeps
LOW_TEMPERATURE=0.5
LOW_GENERATIONS=6
LONG_TEMPERATURE=3
LONG_GENERATIONS=20
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
SUMMARY_FILE="$SAVE_FOLDER/${STAMP}_followup_${GROUP}_summary.log"

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
BASE=(--model_variant base --lr "$BASE_LR" --train_iters "$BASE_ITERS")
EXTENDED=(--model_variant extended --lr "$EXTENDED_LR" --train_iters "$EXTENDED_ITERS")

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

echo "=== Group $GROUP. Settings: base lr $BASE_LR, $BASE_ITERS sequences; extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences (environment, else ${TUNING_LOG:-no tuning log}) ==="

if [ "$GROUP" = lowtemp ] || [ "$GROUP" = all ]; then
  for i in "${!CONDITIONS[@]}"; do
    # CONDITION_ARGS entries are split into flag and value on purpose.
    run_experiment "collapse_extended_global_T${LOW_TEMPERATURE}_${CONDITIONS[$i]}" "${EXTENDED[@]}" \
      --next_symbol_source global --generations "$LOW_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE" \
      --gen_temperature "$LOW_TEMPERATURE" ${CONDITION_ARGS[$i]}
  done
  for i in "${!CONDITIONS[@]}"; do
    run_experiment "collapse_extended_zipf1.0_T${LOW_TEMPERATURE}_${CONDITIONS[$i]}" "${EXTENDED[@]}" \
      --zipf_alpha 1.0 --generations "$LOW_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE" \
      --gen_temperature "$LOW_TEMPERATURE" ${CONDITION_ARGS[$i]}
  done
fi

if [ "$GROUP" = long ] || [ "$GROUP" = all ]; then
  LONG=(--generations "$LONG_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE" --gen_temperature "$LONG_TEMPERATURE")
  run_experiment "collapse_base_G${LONG_GENERATIONS}_T${LONG_TEMPERATURE}_replace100" "${BASE[@]}" "${LONG[@]}" --gen_replace_frac 1.0
  run_experiment "collapse_base_G${LONG_GENERATIONS}_T${LONG_TEMPERATURE}_accumulate" "${BASE[@]}" "${LONG[@]}" --gen_mode accumulate
fi

echo "=== Summary ===" | tee "$SUMMARY_FILE"
printf '%s\n' "${SUMMARY[@]}" | tee -a "$SUMMARY_FILE"
exit $FAILED
