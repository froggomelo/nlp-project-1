#!/bin/bash
# Repeats the 14 runs of the report (Table 1: base and extended, 20 generations,
# real only, full replacement and accumulation at T = 1, and 50%, 90% and 100%
# replacement and accumulation at T = 3) for more seeds.
#   Each seed pair INIT:TRAIN sets --init_seed (the weights) and --train_seed (the
#   pool, the batch order, which items are replaced and the sampled labels). The
#   evaluation and circuit-discovery sequences keep their default seeds, so every
#   seed is measured on the same sequences. The original runs use 5:0.
#   Runs up to JOBS experiments at once, at most MAX_ACCUMULATE of them
#   accumulation runs (each reaches about 7 GB), and starts a new one only while
#   at least MIN_FREE_GB of memory is available. Cap the threads per run with
#   OMP_NUM_THREADS. Run names end in _seed<INIT>, e.g.
#   collapse_base_G20_T3_replace50_seed15.
#   The learning rate and training length per variant come from the newest
#   results/<stamp>_tuning.log of scripts/nlp_project_run.sh, unless set in the
#   environment as BASE_LR, BASE_ITERS, EXTENDED_LR and EXTENDED_ITERS.
#     SEEDS="15:10 25:20" JOBS=5 OMP_NUM_THREADS=2 bash scripts/seed_runs.sh
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

SEEDS=${SEEDS:-"15:10 25:20"}   # INIT:TRAIN pairs; seeds must differ from the eval seed (1)
JOBS=${JOBS:-5}
MAX_ACCUMULATE=${MAX_ACCUMULATE:-2}
MIN_FREE_GB=${MIN_FREE_GB:-8}
POLL_SECONDS=${POLL_SECONDS:-30}         # between checks for a free slot
START_WAIT_SECONDS=${START_WAIT_SECONDS:-60}  # after a start, so its memory shows up
SAVE_FOLDER="./results"
GENERATIONS=20
GEN_DATASET_SIZE=20000       # pool size N
CKPT_EVERY=50000             # sparse checkpoints, as in the sweeps

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
SUMMARY_FILE="$SAVE_FOLDER/${STAMP}_seeds_summary.log"
STATUS_DIR="$SAVE_FOLDER/.${STAMP}_seeds_status"
mkdir -p "$STATUS_DIR" || exit 1

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
  --base_folder "$SAVE_FOLDER"
  --circuit-discovery
  --generations "$GENERATIONS" --gen_dataset_size "$GEN_DATASET_SIZE"
)

# The 14 conditions: name suffix, then the extra src.main args
CONDITIONS=(
  "G20_real_only|--gen_replace_frac 0"
  "G20_replace100|--gen_replace_frac 1.0"
  "G20_accumulate|--gen_mode accumulate"
  "G20_T3_replace50|--gen_temperature 3 --gen_replace_frac 0.5"
  "G20_T3_replace90|--gen_temperature 3 --gen_replace_frac 0.9"
  "G20_T3_replace100|--gen_temperature 3 --gen_replace_frac 1.0"
  "G20_T3_accumulate|--gen_temperature 3 --gen_mode accumulate"
)

# Queue: accumulation runs first, as they are the longest and the most limited,
# then the extended runs before the base runs, as they take longer
QUEUE=()
for kind in accumulate other; do
  for variant in extended base; do
    for pair in $SEEDS; do
      for c in "${CONDITIONS[@]}"; do
        suffix=${c%%|*}
        case "$suffix" in *accumulate) is_acc=accumulate ;; *) is_acc=other ;; esac
        [ "$is_acc" = "$kind" ] && QUEUE+=("$variant|$pair|$c")
      done
    done
  done
done

# start_job <queue entry> <status file>: runs one experiment in the background
start_job() {
  local variant pair suffix args init train lr iters name log
  IFS='|' read -r variant pair suffix args <<< "$1"
  init=${pair%%:*}
  train=${pair##*:}
  if [ "$variant" = base ]; then lr=$BASE_LR; iters=$BASE_ITERS; else lr=$EXTENDED_LR; iters=$EXTENDED_ITERS; fi
  name="collapse_${variant}_${suffix}_seed${init}"
  log="$SAVE_FOLDER/${STAMP}_${name}.log"
  echo "=== Starting $name (init seed $init, train seed $train; output saved to $log) ==="
  (
    # args holds flag-value pairs, split on purpose
    python -u -m src.main "${COMMON_ARGS[@]}" --model_variant "$variant" --lr "$lr" --train_iters "$iters" \
      --init_seed "$init" --train_seed "$train" --run "$name" $args > "$log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
      echo "OK      $name  $log" > "$2"
    else
      echo "FAILED  $name  $log (exit $status)" > "$2"
    fi
    echo "=== Finished $name (exit $status) ==="
  ) &
  RUNNING_PIDS+=($!)
  RUNNING_ACC+=($([[ "$suffix" == *accumulate ]] && echo 1 || echo 0))
}

# prune: drops finished jobs from the running lists
prune() {
  local pids=() accs=() i
  for i in "${!RUNNING_PIDS[@]}"; do
    if kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      pids+=("${RUNNING_PIDS[$i]}")
      accs+=("${RUNNING_ACC[$i]}")
    fi
  done
  RUNNING_PIDS=("${pids[@]}")
  RUNNING_ACC=("${accs[@]}")
}

echo "=== Seeds $SEEDS: ${#QUEUE[@]} runs, $JOBS at once, at most $MAX_ACCUMULATE accumulating. Settings: base lr $BASE_LR, $BASE_ITERS sequences; extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences (environment, else ${TUNING_LOG:-no tuning log}) ==="

RUNNING_PIDS=()
RUNNING_ACC=()
PENDING=("${QUEUE[@]}")
N_STARTED=0
while [ ${#PENDING[@]} -gt 0 ]; do
  prune
  free_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  n_acc=0
  for a in "${RUNNING_ACC[@]}"; do n_acc=$((n_acc + a)); done
  started=0
  if [ ${#RUNNING_PIDS[@]} -lt "$JOBS" ] && [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    # Start the first pending run that the limits allow
    for i in "${!PENDING[@]}"; do
      if [[ "${PENDING[$i]}" == *accumulate* ]] && [ "$n_acc" -ge "$MAX_ACCUMULATE" ]; then
        continue
      fi
      start_job "${PENDING[$i]}" "$(printf '%s/%04d' "$STATUS_DIR" "$N_STARTED")"
      N_STARTED=$((N_STARTED + 1))
      unset 'PENDING[i]'
      PENDING=("${PENDING[@]}")
      started=1
      break
    done
  fi
  # Give a new run time to allocate its memory before the next check
  if [ $started -eq 1 ]; then sleep "$START_WAIT_SECONDS"; else sleep "$POLL_SECONDS"; fi
done
wait

echo "=== Summary ===" | tee "$SUMMARY_FILE"
cat "$STATUS_DIR"/* | tee -a "$SUMMARY_FILE"
grep -q '^FAILED' "$SUMMARY_FILE" && exit 1
exit 0
