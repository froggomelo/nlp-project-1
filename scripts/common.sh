# Shared settings and helpers for scripts/tune_run.sh and scripts/nlp_project_run.sh.
# Sourced from the project root; not run on its own.
#   Runs up to JOBS experiments at once, at most MAX_ACCUMULATE of them accumulation
#   runs, and starts a run only while MIN_FREE_GB of memory are available. A run
#   whose results folder already holds every generation with the same settings is
#   skipped (SKIP_FINISHED=1), so the scripts can be restarted after an interruption.

### Switches, also settable in the environment
JOBS=${JOBS:-5}                    # experiments at once
MAX_ACCUMULATE=${MAX_ACCUMULATE:-2}  # accumulation runs at once (each reaches about 7 GB)
MIN_FREE_GB=${MIN_FREE_GB:-8}      # start a new run only while this much memory is available
SKIP_FINISHED=${SKIP_FINISHED:-1}  # 1: skip runs that already finished with the same settings
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}  # PyTorch threads per run; more threads than cores
                                   # across runs slows circuit discovery down sharply
POLL_SECONDS=${POLL_SECONDS:-30}
START_WAIT_SECONDS=${START_WAIT_SECONDS:-60}  # after a start, so its memory shows up

### Fixed settings
SAVE_FOLDER="./results"
GEN_DATASET_SIZE=20000             # pool size N
CKPT_EVERY=50000                   # checkpoint every 50,000 sequences

mkdir -p "$SAVE_FOLDER" || exit 1
STAMP=$(date +%Y%m%d%H%M%S)
STATUS_DIR="$SAVE_FOLDER/.${STAMP}_status"
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
  --gen_dataset_size "$GEN_DATASET_SIZE"
  --base_folder "$SAVE_FOLDER"
  --circuit-discovery
)

# finished <name> <key=value...>: whether the newest results/<stamp>_<name> folder has
# circuit results for every generation and a config matching the given settings
# (several values of one flag are joined with commas)
finished() {
  local name=$1
  shift
  [ "$SKIP_FINISHED" -eq 1 ] || return 1
  local dir
  dir=$(ls -d "$SAVE_FOLDER"/[0-9]*_"$name" 2>/dev/null | grep -E "/[0-9]{14}_${name}$" | sort | tail -1)
  [ -n "$dir" ] || return 1
  python - "$dir" "$@" <<'EOF'
import json, os, sys
import h5py
run, settings = sys.argv[1], dict(kv.split('=', 1) for kv in sys.argv[2:])
with open(os.path.join(run, 'config.json')) as f:
  config = json.load(f)
def same(have, want):
  try:
    return float(have) == float(want)
  except (TypeError, ValueError):
    return str(have) == want
for key, want in settings.items():
  have, wants = config.get(key), want.split(',')
  if isinstance(have, list):
    ok = len(have) == len(wants) and all(same(h, w) for h, w in zip(have, wants))
  else:
    ok = len(wants) == 1 and same(have, wants[0])
  if not ok:
    sys.exit(1)
try:
  with h5py.File(os.path.join(run, 'head_circuit.h5'), 'r') as h:
    sys.exit(0 if len(h['gen']) == config['generations'] else 1)
except (OSError, KeyError):
  sys.exit(1)
EOF
}

N_STATUS=0
RUNNING_PIDS=()
RUNNING_ACC=()

# next_status_file: sets STATUS_FILE to the next numbered summary line file
next_status_file() {
  STATUS_FILE=$(printf '%s/%04d' "$STATUS_DIR" "$N_STATUS")
  N_STATUS=$((N_STATUS + 1))
}

# add <queue array name> <run name> <src.main args...>: queues a run unless it already
# finished with the same settings (the args' --key value pairs are checked in its config)
add() {
  local queue=$1 name=$2
  shift 2
  local settings=() acc=0 key values i j
  local args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in --*)
      key=${args[$i]#--}
      values=""
      for ((j = i + 1; j < ${#args[@]}; j++)); do
        [[ "${args[$j]}" == --* ]] && break
        values+="${values:+,}${args[$j]}"
      done
      [ -n "$values" ] && settings+=("$key=$values") ;;
    esac
  done
  [[ "$name" == *accumulate* ]] && acc=1
  if finished "$name" "${settings[@]}"; then
    next_status_file
    echo "SKIPPED  $name  (finished earlier)" > "$STATUS_FILE"
    echo "=== Skipping $name: finished earlier with the same settings ==="
    return
  fi
  eval "$queue+=(\"\$name|\$acc|\${args[*]}\")"
}

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

# run_queue <entries...>: each entry is "<name>|<accumulate 0 or 1>|<src.main args>",
# the args split on spaces. Returns when every run of the queue has finished.
run_queue() {
  local pending=("$@") name acc args free_gb n_acc started i a log
  while [ ${#pending[@]} -gt 0 ]; do
    prune
    free_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    n_acc=0
    for a in "${RUNNING_ACC[@]}"; do n_acc=$((n_acc + a)); done
    started=0
    if [ ${#RUNNING_PIDS[@]} -lt "$JOBS" ] && [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
      for i in "${!pending[@]}"; do
        IFS='|' read -r name acc args <<< "${pending[$i]}"
        [ "$acc" -eq 1 ] && [ "$n_acc" -ge "$MAX_ACCUMULATE" ] && continue
        next_status_file
        log="$SAVE_FOLDER/${STAMP}_${name}.log"
        echo "=== Starting $name (output saved to $log) ==="
        (
          # args holds flag-value pairs, split on purpose
          python -u -m src.main "${COMMON_ARGS[@]}" --run "$name" $args > "$log" 2>&1
          status=$?
          if [ $status -eq 0 ]; then
            echo "OK       $name  $log" > "$STATUS_FILE"
          else
            echo "FAILED   $name  $log (exit $status)" > "$STATUS_FILE"
          fi
          echo "=== Finished $name (exit $status) ==="
        ) &
        RUNNING_PIDS+=($!)
        RUNNING_ACC+=("$acc")
        unset 'pending[i]'
        pending=("${pending[@]}")
        started=1
        break
      done
    fi
    if [ $started -eq 1 ]; then sleep "$START_WAIT_SECONDS"; else sleep "$POLL_SECONDS"; fi
  done
  wait
  RUNNING_PIDS=()
  RUNNING_ACC=()
}

# write_summary <file>: lists every run of this invocation as OK, FAILED or SKIPPED;
# returns 1 if any run failed
write_summary() {
  echo "=== Summary ===" | tee "$1"
  cat "$STATUS_DIR"/* 2>/dev/null | tee -a "$1"
  ! grep -q '^FAILED' "$1"
}
