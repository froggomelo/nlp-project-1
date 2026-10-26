#!/bin/bash
# Runs every experiment behind the report, for three seeds:
#   1. Tuning: scripts/tune_run.sh picks the learning rate and training length per
#      model on the validation classes, and every later run uses them.
#   2. Table 1: base and extended models over 20 generations, with real data only,
#      full replacement and accumulation at T = 1, and 50%, 90% and 100%
#      replacement and accumulation at T = 3.
#   3. Margin formula: 50%, 90% and 100% replacement and accumulation at T = 3 and
#      T = 5 over 6 generations, for both models.
#   Takes no parameters. The switches below and in scripts/common.sh can also be set
#   in the environment, e.g. JOBS=3 SEEDS="5:0" bash scripts/nlp_project_run.sh
#   A run whose results folder already holds every generation with the same
#   settings is skipped, so the script can be restarted after an interruption.
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

### Switches
SEEDS=${SEEDS:-"5:0 15:10 25:20"}  # INIT:TRAIN seed pairs (--init_seed sets the weights, --train_seed
                                   # the pool, batch order, replaced items and sampled labels)
TUNE_RUNS=${TUNE_RUNS:-1}          # 1: run scripts/tune_run.sh first, 0: use the newest tuning log
TABLE_RUNS=${TABLE_RUNS:-1}        # 1: run the Table 1 experiments, 0: skip them
FORCED_RUNS=${FORCED_RUNS:-1}      # 1: run the margin-formula experiments, 0: skip them

### 1. Tuning
if [ "$TUNE_RUNS" -eq 1 ]; then
  bash scripts/tune_run.sh || { echo "Tuning failed" >&2; exit 1; }
fi
source scripts/common.sh

# The tuned settings: the environment, else the newest results/<stamp>_tuning.log
TUNING_LOG=$(ls -t "$SAVE_FOLDER"/*_tuning.log 2>/dev/null | head -1)
for var in BASE_LR BASE_ITERS EXTENDED_LR EXTENDED_ITERS; do
  if [ -z "${!var}" ] && [ -n "$TUNING_LOG" ]; then
    value=$(grep -E "^${var}=[0-9.e+-]+$" "$TUNING_LOG" | tail -1 | cut -d= -f2)
    printf -v "$var" '%s' "$value"
  fi
  if [ -z "${!var}" ]; then
    echo "No value for $var: run scripts/tune_run.sh or set it in the environment" >&2
    exit 1
  fi
done
echo "=== Settings: base lr $BASE_LR, $BASE_ITERS sequences; extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences (environment, else ${TUNING_LOG:-no tuning log}) ==="

### 2-3. Collapse runs
# Condition name, generations, then the condition's src.main args
TABLE_CONDITIONS=(
  "G20_real_only|20|--gen_replace_frac 0"
  "G20_replace100|20|--gen_replace_frac 1.0"
  "G20_accumulate|20|--gen_mode accumulate"
  "G20_T3_replace50|20|--gen_temperature 3 --gen_replace_frac 0.5"
  "G20_T3_replace90|20|--gen_temperature 3 --gen_replace_frac 0.9"
  "G20_T3_replace100|20|--gen_temperature 3 --gen_replace_frac 1.0"
  "G20_T3_accumulate|20|--gen_temperature 3 --gen_mode accumulate"
)
FORCED_CONDITIONS=()
for t in 3 5; do
  FORCED_CONDITIONS+=(
    "T${t}_replace50|6|--gen_temperature $t --gen_replace_frac 0.5"
    "T${t}_replace90|6|--gen_temperature $t --gen_replace_frac 0.9"
    "T${t}_replace100|6|--gen_temperature $t --gen_replace_frac 1.0"
    "T${t}_accumulate|6|--gen_temperature $t --gen_mode accumulate"
  )
done
CONDITIONS=()
[ "$TABLE_RUNS" -eq 1 ] && CONDITIONS+=("${TABLE_CONDITIONS[@]}")
[ "$FORCED_RUNS" -eq 1 ] && CONDITIONS+=("${FORCED_CONDITIONS[@]}")

# Accumulation and extended runs first, as they take longest
QUEUE=()
for kind in accumulate other; do
  for variant in extended base; do
    for pair in $SEEDS; do
      init=${pair%%:*}
      train=${pair##*:}
      # The first seed pair of the report keeps the plain run names
      suffix=$([ "$pair" = "5:0" ] || echo "_seed${init}")
      if [ "$variant" = base ]; then lr=$BASE_LR; iters=$BASE_ITERS; else lr=$EXTENDED_LR; iters=$EXTENDED_ITERS; fi
      for c in "${CONDITIONS[@]}"; do
        IFS='|' read -r cond gens cargs <<< "$c"
        case "$cond" in *accumulate) is=accumulate ;; *) is=other ;; esac
        [ "$is" = "$kind" ] || continue
        # cargs holds flag-value pairs, split on purpose
        add QUEUE "collapse_${variant}_${cond}${suffix}" --model_variant "$variant" \
          --lr "$lr" --train_iters "$iters" --generations "$gens" \
          --init_seed "$init" --train_seed "$train" $cargs
      done
    done
  done
done
echo "=== Collapse runs: ${#QUEUE[@]} to train, $JOBS at once (seeds $SEEDS) ==="
run_queue "${QUEUE[@]}"
write_summary "$SAVE_FOLDER/${STAMP}_summary.log"
