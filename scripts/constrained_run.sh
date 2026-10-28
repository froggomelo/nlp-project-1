#!/bin/bash
# Ablation: constrained next-symbol generation for the extended model.
#   Repeats the extended model's 20-generation T = 3 runs of Table 1 with
#   --gen_symbol_constraint context, which masks the symbol head to the classes in
#   the context before sampling, so generated next symbols always occur in context.
#   Conditions: 50%, 90% and 100% replacement and accumulation at T = 3, and full
#   replacement and accumulation at T = 1, for the three seeds of the report, which
#   cover every extended-model row of Table 1. 0% replacement samples
#   nothing, so the existing collapse_extended_G20_real_only runs are its control.
#   Each run reuses the generation 0 model of the matching unconstrained run
#   (--gen0_from), so both arms start from identical checkpoints.
#   Runs are named collapse_extended_G20_T<T>ctx_<condition>[_seed<init>] and go to
#   results/constrained/. Takes no parameters; the switches of scripts/common.sh
#   can be set in the environment, e.g. JOBS=4 bash scripts/constrained_run.sh, and
#   ONLY="T1ctx_accumulate" runs just the named conditions.
set -o pipefail

# Resolve all data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/common.sh

SEEDS=${SEEDS:-"5:0 15:10 25:20"}
OUT_FOLDER="$SAVE_FOLDER/constrained"
mkdir -p "$OUT_FOLDER" || exit 1

# The tuned extended-model settings: the environment, else the newest tuning log
TUNING_LOG=$(ls -t "$SAVE_FOLDER"/*_tuning.log 2>/dev/null | head -1)
for var in EXTENDED_LR EXTENDED_ITERS; do
  if [ -z "${!var}" ] && [ -n "$TUNING_LOG" ]; then
    printf -v "$var" '%s' "$(grep -E "^${var}=[0-9.e+-]+$" "$TUNING_LOG" | tail -1 | cut -d= -f2)"
  fi
  [ -n "${!var}" ] || { echo "No value for $var: run scripts/tune_run.sh first" >&2; exit 1; }
done

# Condition name, temperature, then the condition's src.main args
CONDITIONS=(
  "T3ctx_accumulate|3|--gen_mode accumulate"
  "T3ctx_replace50|3|--gen_replace_frac 0.5"
  "T3ctx_replace90|3|--gen_replace_frac 0.9"
  "T3ctx_replace100|3|--gen_replace_frac 1.0"
  "T1ctx_replace100|1|--gen_replace_frac 1.0"
  "T1ctx_accumulate|1|--gen_mode accumulate"
)
if [ -n "$ONLY" ]; then
  KEEP=()
  for c in "${CONDITIONS[@]}"; do
    [[ " $ONLY " == *" ${c%%|*} "* ]] && KEEP+=("$c")
  done
  CONDITIONS=("${KEEP[@]}")
fi

# Runs, logs and the finished-run check use the constrained folder from here on
SAVE_FOLDER="$OUT_FOLDER"
QUEUE=()
for c in "${CONDITIONS[@]}"; do
  IFS='|' read -r cond t cargs <<< "$c"
  for pair in $SEEDS; do
    init=${pair%%:*}
    train=${pair##*:}
    suffix=$([ "$pair" = "5:0" ] || echo "_seed${init}")
    # Generation 0 does not depend on the condition, so any matching run supplies it
    gen0=$(ls -d ./results/[0-9]*_collapse_extended_G20_T3_replace100"$suffix" 2>/dev/null \
           | grep -E "_G20_T3_replace100${suffix}$" | sort | tail -1)
    [ -n "$gen0" ] || { echo "No unconstrained run for seed $init to reuse generation 0 from" >&2; exit 1; }
    # cargs holds flag-value pairs, split on purpose
    add QUEUE "collapse_extended_G20_${cond}${suffix}" --model_variant extended \
      --lr "$EXTENDED_LR" --train_iters "$EXTENDED_ITERS" --generations 20 \
      --init_seed "$init" --train_seed "$train" --gen_temperature "$t" \
      --gen_symbol_constraint context --base_folder "$OUT_FOLDER" --gen0_from "$gen0" $cargs
  done
done
echo "=== Constrained runs: ${#QUEUE[@]} to train, $JOBS at once (extended lr $EXTENDED_LR, $EXTENDED_ITERS sequences) ==="
run_queue "${QUEUE[@]}"
write_summary "$OUT_FOLDER/${STAMP}_summary.log"
