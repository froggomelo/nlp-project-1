#!/bin/bash
# Runs every NLP project experiment one after the other:
#   1. base model, 2. extended model, 3. the five model collapse conditions
#   for both variants, 4. the pool size, Zipf and burstiness collapse sweeps.
#   Takes no parameters; edit the constants below instead.
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

### Overarching params
SAVE_FOLDER="./results"
INIT_SEED=5
MAIN_RUN_ITERS=1000000       # training sequences per run (per generation for collapse runs)

### Model collapse params
COLLAPSE_GENERATIONS=6       # generation 0 on real data, then 5 synthetic generations
GEN_DATASET_SIZE=20000       # pool size N

### Collapse sweeps: each runs every replace fraction below for both variants
REPLACE_FRACS=(0 0.5 0.9 1.0)
REPLACE_NAMES=(real_only replace50 replace90 replace100)
SWEEP_CKPT_EVERY=50000       # sparser checkpoints than the main runs (~17 MB instead of ~0.8 GB per generation)
SMALL_POOL_SWEEP=1           # 1: rerun with a smaller pool, 0: off
SMALL_POOL_SIZE=5000
ZIPF_SWEEP=1                 # 1: rerun with Zipfian train class distributions, 0: off
ZIPF_ALPHAS=(0.5 1.0 1.5)
BURST_SWEEP=1                # 1: rerun with a longer context and varying burstiness, 0: off
BURST_CONTEXT_LEN=8
BURST_VALUES=(1 2 4)         # each must be <= BURST_CONTEXT_LEN / 2

### Circuit discovery params
CIRCUIT_DISCOVERY=1          # 1: discover circuits after every generation (needs requirements-circuit.txt), 0: off

mkdir -p "$SAVE_FOLDER" || exit 1
STAMP=$(date +%Y%m%d%H%M%S)
SUMMARY_FILE="$SAVE_FOLDER/${STAMP}_summary.log"

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
  --train_iters "$MAIN_RUN_ITERS"
  --train_bs 32
  --eval_every 5000
  --eval_iters 1000
  --lr 0.00001
  --d_model 64
  --class_split 50 1473 100
  --ckpt_every 1000
  --init_seed "$INIT_SEED"
  --base_folder "$SAVE_FOLDER"
)

if [ "$CIRCUIT_DISCOVERY" -eq 1 ]; then
  COMMON_ARGS+=(--circuit-discovery)
fi

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

### Experiments
run_experiment base     --model_variant base
run_experiment extended --model_variant extended

for variant in base extended; do
  COLLAPSE=(--model_variant "$variant" --generations "$COLLAPSE_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE")
  run_experiment "collapse_${variant}_real_only"  "${COLLAPSE[@]}" --gen_replace_frac 0
  run_experiment "collapse_${variant}_replace50"  "${COLLAPSE[@]}" --gen_replace_frac 0.5
  run_experiment "collapse_${variant}_replace90"  "${COLLAPSE[@]}" --gen_replace_frac 0.9
  run_experiment "collapse_${variant}_replace100" "${COLLAPSE[@]}" --gen_replace_frac 1.0
  run_experiment "collapse_${variant}_accumulate" "${COLLAPSE[@]}" --gen_mode accumulate
done

# run_collapse_sweep <name suffix> <extra src.main args...>
# Runs collapse_<variant>_<suffix>_<replace name> for both variants and every
# replace fraction. The extra args come last, so they override COMMON_ARGS.
run_collapse_sweep() {
  local suffix=$1
  shift
  local variant i
  for variant in base extended; do
    for i in "${!REPLACE_FRACS[@]}"; do
      run_experiment "collapse_${variant}_${suffix}_${REPLACE_NAMES[$i]}" \
        --model_variant "$variant" --generations "$COLLAPSE_GENERATIONS" \
        --gen_dataset_size "$GEN_DATASET_SIZE" --gen_replace_frac "${REPLACE_FRACS[$i]}" \
        --ckpt_every "$SWEEP_CKPT_EVERY" "$@"
    done
  done
}

if [ "$SMALL_POOL_SWEEP" -eq 1 ]; then
  run_collapse_sweep "N${SMALL_POOL_SIZE}" --gen_dataset_size "$SMALL_POOL_SIZE"
fi

if [ "$ZIPF_SWEEP" -eq 1 ]; then
  for alpha in "${ZIPF_ALPHAS[@]}"; do
    run_collapse_sweep "zipf${alpha}" --zipf_alpha "$alpha"
  done
fi

if [ "$BURST_SWEEP" -eq 1 ]; then
  # Context items beyond the burst and distractor are distinct classes (unique_rest);
  # the evaluators use the same context length and burstiness as training.
  for b in "${BURST_VALUES[@]}"; do
    run_collapse_sweep "L${BURST_CONTEXT_LEN}B${b}" --train_context_len "$BURST_CONTEXT_LEN" \
      --pt_burstiness "$b" --pt_unique_rest 1 --pe_burstiness "$b" "$b" "$b" "$b" --pe_unique_rest 1 1 1 1
  done
fi

echo "=== Summary ===" | tee "$SUMMARY_FILE"
printf '%s\n' "${SUMMARY[@]}" | tee -a "$SUMMARY_FILE"
exit $FAILED
