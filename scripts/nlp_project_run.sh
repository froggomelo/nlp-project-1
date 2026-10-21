#!/bin/bash
# Runs every NLP project experiment in order:
#   1. learning rate and training length tuning on generation 0 for both variants,
#   2. base model, 3. extended model, 4. the five model collapse conditions for
#   both variants, 5. the extended collapse conditions with the next symbol drawn
#   from all training classes, 6. three collapse conditions over 20 generations,
#   7. full replacement with argmax and with temperature 2 sampling, 8. the pool
#   size, Zipf and burstiness collapse sweeps.
#   Takes no parameters; edit the constants below instead. TUNE_RUNS, MAIN_RUNS,
#   JOBS, the per-variant settings and the sweep switches can also be set in the
#   environment, e.g. the main runs only:
#     SMALL_POOL_SWEEP=0 ZIPF_SWEEP=0 BURST_SWEEP=0 GLOBAL_SWEEP=0 LONG_SWEEP=0 SAMPLING_SWEEP=0 bash scripts/nlp_project_run.sh
#   sweeps only: TUNE_RUNS=0 MAIN_RUNS=0 bash scripts/nlp_project_run.sh
#   six experiments at once: JOBS=6 bash scripts/nlp_project_run.sh
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

### Overarching params
SAVE_FOLDER="./results"
INIT_SEED=5
MAIN_RUN_ITERS=1000000       # training sequences per run (per generation for collapse runs) before tuning
JOBS=${JOBS:-1}              # experiments run at once; each uses about 1.5 CPU cores and 2 GB of memory
                             # (up to about 7 GB for the 20-generation accumulate runs)

### Tuning: one generation 0 run per variant and learning rate, on the collapse pool
TUNE_RUNS=${TUNE_RUNS:-1}    # 1: tune and use the chosen settings below, 0: skip and use the constants below
TUNE_LRS=(1e-5 3e-5 1e-4 3e-4)
TUNE_LENGTHS=(1000000 2000000 3000000)  # candidate training lengths, all read from the longest run

### Per-variant settings. With TUNE_RUNS=1, scripts/select_hparams.py overwrites them.
BASE_LR=${BASE_LR:-1e-5}
BASE_ITERS=${BASE_ITERS:-$MAIN_RUN_ITERS}
EXTENDED_LR=${EXTENDED_LR:-1e-5}
EXTENDED_ITERS=${EXTENDED_ITERS:-$MAIN_RUN_ITERS}

### Model collapse params
COLLAPSE_GENERATIONS=6       # generation 0 on real data, then 5 synthetic generations
GEN_DATASET_SIZE=20000       # pool size N

MAIN_RUNS=${MAIN_RUNS:-1}    # 1: run base, extended and the main collapse runs, 0: skip them (sweeps only)

### Collapse sweeps: each runs every replace fraction below for both variants
REPLACE_FRACS=(0 0.5 0.9 1.0)
REPLACE_NAMES=(real_only replace50 replace90 replace100)
SWEEP_CKPT_EVERY=50000       # sparser checkpoints than the main runs (~17 MB instead of ~0.8 GB per generation)
GLOBAL_SWEEP=${GLOBAL_SWEEP:-1}          # 1: rerun the extended collapse conditions with --next_symbol_source global, 0: off
LONG_SWEEP=${LONG_SWEEP:-1}              # 1: rerun real_only, replace100 and accumulate over LONG_GENERATIONS, 0: off
LONG_GENERATIONS=20
SAMPLING_SWEEP=${SAMPLING_SWEEP:-1}      # 1: rerun replace100 with argmax and with SAMPLING_TEMPERATURE sampling, 0: off
SAMPLING_TEMPERATURE=2
SMALL_POOL_SWEEP=${SMALL_POOL_SWEEP:-1}  # 1: rerun with a smaller pool, 0: off (or set in the environment)
SMALL_POOL_SIZE=5000
ZIPF_SWEEP=${ZIPF_SWEEP:-1}              # 1: rerun with Zipfian train class distributions, 0: off
ZIPF_ALPHAS=(0.5 1.0 1.5)
BURST_SWEEP=${BURST_SWEEP:-1}            # 1: rerun with a longer context and varying burstiness, 0: off
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

# Each experiment writes its summary line to a numbered file here, so background
# experiments can report back and the summary keeps the run order.
STATUS_DIR="$SAVE_FOLDER/.${STAMP}_status"
mkdir -p "$STATUS_DIR" || exit 1
N_STATUS=0

# new_status_file: sets STATUS_FILE to the next numbered summary line file.
new_status_file() {
  STATUS_FILE=$(printf '%s/%04d' "$STATUS_DIR" "$N_STATUS")
  N_STATUS=$((N_STATUS + 1))
}

# record_status <exit status> <name> <log> <status file>
record_status() {
  if [ "$1" -eq 0 ]; then
    echo "OK      $2  $3" > "$4"
  else
    echo "FAILED  $2  $3 (exit $1)" > "$4"
  fi
}

# run_experiment <name> <extra src.main args...>
# Saves terminal output to results/<stamp>_<name>.log and continues on failure.
# With JOBS > 1 it waits for a free slot and runs the experiment in the background,
# and the output goes to the log only.
run_experiment() {
  local name=$1
  shift
  local log="$SAVE_FOLDER/${STAMP}_${name}.log"
  new_status_file
  if [ "$JOBS" -gt 1 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
      wait -n
    done
    echo "=== Starting $name (output saved to $log) ==="
    (
      python -u -m src.main "${COMMON_ARGS[@]}" --run "$name" "$@" > "$log" 2>&1
      status=$?
      record_status $status "$name" "$log" "$STATUS_FILE"
      echo "=== Finished $name (exit $status) ==="
    ) &
  else
    echo "=== Running $name (output saved to $log) ==="
    python -u -m src.main "${COMMON_ARGS[@]}" --run "$name" "$@" 2>&1 | tee "$log"
    record_status $? "$name" "$log" "$STATUS_FILE"
  fi
}

# variant_args <variant>: the variant's learning rate and training length, which
# come after COMMON_ARGS and override it.
variant_args() {
  if [ "$1" = base ]; then
    echo --lr "$BASE_LR" --train_iters "$BASE_ITERS"
  else
    echo --lr "$EXTENDED_LR" --train_iters "$EXTENDED_ITERS"
  fi
}

### Experiments
if [ "$TUNE_RUNS" -eq 1 ]; then
  # Generation 0 of a collapse run, trained for the longest candidate length. A
  # fifth evaluator on the validation classes scores each setting, so the test
  # classes are never used for selection.
  for variant in base extended; do
    for lr in "${TUNE_LRS[@]}"; do
      run_experiment "tune_${variant}_lr${lr}" \
        --model_variant "$variant" --generations 1 --gen_dataset_size "$GEN_DATASET_SIZE" \
        --lr "$lr" --train_iters "${TUNE_LENGTHS[-1]}" --ckpt_every "$SWEEP_CKPT_EVERY" \
        --pe_names fsl_train fsl_val_rl fsl_train_valex fsl_test_class fsl_val_class \
        --pe_classes train train train test val \
        --pe_exemplars train train val train train \
        --pe_fs_relabel_scheme train val train train train \
        --pe_burstiness 1 1 1 1 1
    done
  done

  # Selection needs every tuning run to have finished.
  wait

  TUNE_LOG="$SAVE_FOLDER/${STAMP}_tuning.log"
  CHOSEN=$(python scripts/select_hparams.py --results "$SAVE_FOLDER" \
    --lrs "${TUNE_LRS[@]}" --lengths "${TUNE_LENGTHS[@]}" 2>"$TUNE_LOG")
  if [ $? -ne 0 ]; then
    new_status_file
    record_status 1 "tuning selection" "$TUNE_LOG" "$STATUS_FILE"
  fi
  eval "$CHOSEN"
  printf '%s\n' "$CHOSEN" >> "$TUNE_LOG"
  cat "$TUNE_LOG"
fi
echo "=== Settings: base lr $BASE_LR, $BASE_ITERS sequences; extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences ==="

if [ "$MAIN_RUNS" -eq 1 ]; then
  run_experiment base     --model_variant base     $(variant_args base)
  run_experiment extended --model_variant extended $(variant_args extended)

  for variant in base extended; do
    COLLAPSE=(--model_variant "$variant" --generations "$COLLAPSE_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE" $(variant_args "$variant"))
    run_experiment "collapse_${variant}_real_only"  "${COLLAPSE[@]}" --gen_replace_frac 0
    run_experiment "collapse_${variant}_replace50"  "${COLLAPSE[@]}" --gen_replace_frac 0.5
    run_experiment "collapse_${variant}_replace90"  "${COLLAPSE[@]}" --gen_replace_frac 0.9
    run_experiment "collapse_${variant}_replace100" "${COLLAPSE[@]}" --gen_replace_frac 1.0
    run_experiment "collapse_${variant}_accumulate" "${COLLAPSE[@]}" --gen_mode accumulate
  done
fi

if [ "$GLOBAL_SWEEP" -eq 1 ]; then
  # The next symbol comes from the training class distribution instead of the
  # context, the setting in which the brief expects the symbol distribution to skew.
  GLOBAL=(--model_variant extended --next_symbol_source global --generations "$COLLAPSE_GENERATIONS"
          --gen_dataset_size "$GEN_DATASET_SIZE" --ckpt_every "$SWEEP_CKPT_EVERY" $(variant_args extended))
  for i in "${!REPLACE_FRACS[@]}"; do
    run_experiment "collapse_extended_global_${REPLACE_NAMES[$i]}" "${GLOBAL[@]}" --gen_replace_frac "${REPLACE_FRACS[$i]}"
  done
  run_experiment "collapse_extended_global_accumulate" "${GLOBAL[@]}" --gen_mode accumulate
fi

if [ "$LONG_SWEEP" -eq 1 ]; then
  # More generations, as the brief expects the base model's few mistakes to matter
  # only after many of them. Full replacement and accumulation are the two settings
  # Gerstgrasser et al. compare; real_only is the control.
  for variant in base extended; do
    LONG=(--model_variant "$variant" --generations "$LONG_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE"
          --ckpt_every "$SWEEP_CKPT_EVERY" $(variant_args "$variant"))
    run_experiment "collapse_${variant}_G${LONG_GENERATIONS}_real_only"  "${LONG[@]}" --gen_replace_frac 0
    run_experiment "collapse_${variant}_G${LONG_GENERATIONS}_replace100" "${LONG[@]}" --gen_replace_frac 1.0
    run_experiment "collapse_${variant}_G${LONG_GENERATIONS}_accumulate" "${LONG[@]}" --gen_mode accumulate
  done
fi

if [ "$SAMPLING_SWEEP" -eq 1 ]; then
  # Full replacement with no sampling noise (argmax) and with more of it (higher
  # temperature). The main replace100 runs sample at temperature 1.
  for variant in base extended; do
    SAMPLING=(--model_variant "$variant" --generations "$COLLAPSE_GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE"
              --gen_replace_frac 1.0 --ckpt_every "$SWEEP_CKPT_EVERY" $(variant_args "$variant"))
    run_experiment "collapse_${variant}_argmax_replace100" "${SAMPLING[@]}" --gen_sampling argmax
    run_experiment "collapse_${variant}_T${SAMPLING_TEMPERATURE}_replace100" "${SAMPLING[@]}" --gen_temperature "$SAMPLING_TEMPERATURE"
  done
fi

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
        --ckpt_every "$SWEEP_CKPT_EVERY" $(variant_args "$variant") "$@"
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

wait
echo "=== Summary ===" | tee "$SUMMARY_FILE"
find "$STATUS_DIR" -type f | sort | xargs -r cat | tee -a "$SUMMARY_FILE"
FAILED=0
if grep -q '^FAILED' "$SUMMARY_FILE"; then
  FAILED=1
fi
rm -rf "$STATUS_DIR"
exit $FAILED
