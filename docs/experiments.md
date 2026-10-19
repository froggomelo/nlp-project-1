# Experiments

`scripts/nlp_project_run.sh` runs 91 experiments and trains 580 models in total. This page lists every experiment, the settings it changes, and its run name. [results.md](results.md) reports the results. [model_collapse.md](model_collapse.md) describes the collapse pipeline and its pool metrics. [circuit_discovery.md](circuit_discovery.md) describes the circuit metrics that every run records.

## Shared Settings

Every experiment starts from the settings below. Later tables name only the settings an experiment changes.

| Setting | Value | Flag |
|---|---|---|
| Training sequences per run or generation | tuned per variant, 1,000,000 before tuning | `--train_iters` |
| Batch size | 32 | `--train_bs` |
| Learning rate | tuned per variant, 1e-5 before tuning | `--lr` |
| Model | 2 layers, 8 heads, width 64, attention only | `--depth`, `--num_heads`, `--d_model` |
| Context length L | 2 | `--train_context_len` |
| Burstiness B | 1 | `--pt_burstiness` |
| Value labels | 5, reassigned in every sequence | `--fs_relabel 5` |
| Train classes | 50 | `--class_split 50 1473 100` |
| Train class distribution | uniform | `--zipf_alpha 0` |
| Init seed | 5 | `--init_seed` |
| Checkpoint interval | 1,000 sequences | `--ckpt_every` |
| Circuit discovery | on | `--circuit-discovery` |

Collapse experiments add the settings below.

| Setting | Value | Flag |
|---|---|---|
| Generations | 6, generation 0 on real data | `--generations` |
| Pool size N | 20,000 | `--gen_dataset_size` |
| Next symbol source, extended model | context | `--next_symbol_source` |
| Token sampling | softmax at T = 1 | `--gen_sampling`, `--gen_temperature` |
| Initialisation per generation | fresh | `--gen_init` |

## Experiment Grid

The script runs six stages in order.

| Stage | Experiments | Trained models | Switch |
|---|---|---|---|
| 1. Tuning | 8 | 8 | `TUNE_RUNS` |
| 2. Single and main collapse runs | 12 | 62 | `MAIN_RUNS` |
| 3. Next symbol drawn globally | 5 | 30 | `GLOBAL_SWEEP` |
| 4. 20 generations | 6 | 120 | `LONG_SWEEP` |
| 5. Argmax and temperature 2 sampling | 4 | 24 | `SAMPLING_SWEEP` |
| 6. Pool size, Zipf and burstiness sweeps | 56 | 336 | `SMALL_POOL_SWEEP`, `ZIPF_SWEEP`, `BURST_SWEEP` |

### Stage 1: Tuning

Each ✓ is one run named `tune_<variant>_lr<lr>`. It trains generation 0 of a collapse run, on a real pool of N = 20,000, for 3,000,000 sequences.

| Variant | lr 1e-5 | lr 3e-5 | lr 1e-4 | lr 3e-4 |
|---|:-:|:-:|:-:|:-:|
| base | ✓ | ✓ | ✓ | ✓ |
| extended | ✓ | ✓ | ✓ | ✓ |

`scripts/select_hparams.py` then picks a learning rate and a training length of 1,000,000, 2,000,000 or 3,000,000 sequences per variant. Every later stage uses the chosen pair for that variant. See [Tuning](#tuning).

### Stages 2 to 6: Single and Collapse Runs

The two single runs train one generation on fresh data at every step.

| Experiment | Variant |
|---|---|
| `base` | base |
| `extended` | extended |

Each ✓ in the grid below is one collapse experiment, run once for every variant in its row. Its run name is `collapse_<variant>_<tag>_<condition>`, for example `collapse_extended_zipf1.0_replace90`. The Default row has no tag, as in `collapse_base_replace50`.

| Setting | Tag | Changed flags | Variants | `real_only` | `replace50` | `replace90` | `replace100` | `accumulate` |
|---|---|---|---|:-:|:-:|:-:|:-:|:-:|
| Default | | | base, extended | ✓ | ✓ | ✓ | ✓ | ✓ |
| Global next symbol | `global` | `--next_symbol_source global` | extended | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 generations | `G20` | `--generations 20` | base, extended | ✓ | | | ✓ | ✓ |
| Argmax sampling | `argmax` | `--gen_sampling argmax` | base, extended | | | | ✓ | |
| Temperature 2 | `T2` | `--gen_temperature 2` | base, extended | | | | ✓ | |
| Small pool | `N5000` | `--gen_dataset_size 5000` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| Zipf 0.5 | `zipf0.5` | `--zipf_alpha 0.5` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| Zipf 1.0 | `zipf1.0` | `--zipf_alpha 1.0` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| Zipf 1.5 | `zipf1.5` | `--zipf_alpha 1.5` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 1 | `L8B1` | `--train_context_len 8 --pt_burstiness 1` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 2 | `L8B2` | `--train_context_len 8 --pt_burstiness 2` | base, extended | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 4 | `L8B4` | `--train_context_len 8 --pt_burstiness 4` | base, extended | ✓ | ✓ | ✓ | ✓ | |

The burstiness rows also pass `--pt_unique_rest 1`, `--pe_burstiness B B B B`, and `--pe_unique_rest 1 1 1 1`. Every row below Default also passes `--ckpt_every 50000`. The five collapse conditions are defined in the table below.

| Condition | Flags | Real share of the pool from generation 1 |
|---|---|---|
| `real_only` | `--gen_replace_frac 0` | 100% |
| `replace50` | `--gen_replace_frac 0.5` | 50% |
| `replace90` | `--gen_replace_frac 0.9` | 10% |
| `replace100` | `--gen_replace_frac 1.0` | 0% |
| `accumulate` | `--gen_mode accumulate` | 1 / (g + 1) at generation g |

The grid holds 38 experiments for the base model and 43 for the extended model. With 75 collapse experiments of 6 generations, 6 of 20 generations, 2 single runs and 8 tuning runs, the script trains 580 models.

## Tuning

The brief asks for tuned hyperparameters on a train-validation-test split. The first runs at lr 1e-5 and 1,000,000 sequences left the extended model with a weak induction circuit at generation 0: an induction score of 0.31 and held-out accuracy of 0.50, against 0.82 and 0.93 for the base model. Collapse can only change in-context learning that formed in the first place.

**Data.** Tuning trains on the same pool of N = 20,000 real sequences as generation 0 of a collapse run, so the chosen settings fit the setting they are used in.

**Validation evaluator.** The tuning runs add a fifth evaluator, `fsl_val_class`, on the 1,473 validation classes. Like `fsl_test_class`, it needs in-context learning, as the model never trained on these classes. Selection uses only this evaluator, so the 100 test classes stay unseen until the final results.

**Score.** The score is query accuracy, the in-context learning measure shared by both variants. It is averaged over the evaluations in the last 50,000 sequences before each candidate length, which smooths the noise of single evaluations.

**Training length from one run.** Adam runs at a constant learning rate, and every run uses the same seeds. The first 1,000,000 sequences of a 3,000,000 sequence run are therefore the 1,000,000 sequence run. One run per learning rate covers all three lengths.

**Preferring cheaper settings.** Settings are compared in order of cost, shorter training first and then smaller learning rate. A costlier setting is chosen only when it beats the best cheaper one by more than 0.01, about one standard error of an evaluation. Every collapse run trains for the chosen length in each generation, so a longer length multiplies the cost of every later stage.

**Chosen settings.** The tuned run selects lr 1e-4 for the base model and lr 3e-4 for the extended model, both at 1,000,000 sequences. Their validation accuracies are 0.989 and 0.970. [results.md](results.md#tuning) gives the full table.

The script writes the table and the choice to `results/<stamp>_tuning.log`. With `TUNE_RUNS=0`, the script skips tuning and uses `BASE_LR`, `BASE_ITERS`, `EXTENDED_LR` and `EXTENDED_ITERS` at its top, so the chosen values can be copied there for later runs.

## Sweep Settings

### Global Next Symbol

With `--next_symbol_source context`, the next symbol is copied from the context. Skew can then only come from the model preferring some context symbols over others. The default runs show no skew: the next symbol's KL divergence to generation 0 stays at or below 0.02, and its entropy and tail mass stay flat. With `global`, the next symbol is drawn from all training classes, as described in [model_collapse.md](model_collapse.md#next-symbol-source). The model must then learn an unconditional symbol distribution, the setting of Shumailov et al. (2024), in which the brief expects the extended model to favour some symbols and skew the distribution. The sweep runs the extended model only, under all five conditions, because the base model has no next symbol.

### Generations

The default runs stop after 5 synthetic generations. The brief expects the base model's few mistakes to influence collapse only after sufficiently many generations, and at generation 5 the induction score and logit difference of the base model are still falling. The `G20` runs extend `real_only`, `replace100` and `accumulate` to 19 synthetic generations. Full replacement and accumulation are the two settings Gerstgrasser et al. compare: under replacement the error should keep growing, under accumulation it should level off. `real_only` is the control. At generation g the accumulate pool holds 20,000 × (g + 1) items, 420,000 at generation 20, and that run needs up to about 7 GB of memory.

### Sampling

Generated tokens are sampled from softmax(logits / T) at T = 1 in every other run. The sampling sweep repeats full replacement with `--gen_sampling argmax`, which removes the sampling noise, and with T = 2, which adds to it. If the base model's logit difference stops falling under argmax, the sampled label errors account for that decline.

### Forced Collapse

At T = 1 the tuned base model regenerates all but about one label in 10,000 correctly, and neither model collapses. `scripts/forced_collapse_run.sh` therefore repeats `replace50`, `replace90`, `replace100`, and `accumulate` at T = 3 and T = 5 for both variants. The raised temperature makes the generation 0 model write label errors into the pool. The script runs 16 experiments of 6 generations, 96 models in total, one at a time. It reads the tuned learning rate and training length from the newest `results/<stamp>_tuning.log`, unless `BASE_LR`, `BASE_ITERS`, `EXTENDED_LR`, and `EXTENDED_ITERS` are set in the environment. `real_only` samples nothing. The T = 1 `real_only` runs of the main script serve as its control. The runs are named `collapse_<variant>_T<T>_<condition>`, save checkpoints every 50,000 sequences, and write `results/<stamp>_forced_summary.log`.

### Pool Size

The small pool sweep sets N to 5,000. Training sequences per generation stay at 1,000,000. Each pool item is therefore seen about 200 times per generation instead of 50. The Zipf and burstiness sweeps keep N = 20,000.

### Zipf Class Distribution

`--zipf_alpha` draws training classes with probability proportional to 1 / rank^α over the 50 train classes. The table gives the resulting skew.

| α | Most frequent class | Top 5 classes | Rarest 25 classes |
|---|---|---|---|
| 0 | 2.0% | 10.0% | 50.0% |
| 0.5 | 7.8% | 25.3% | 32.3% |
| 1.0 | 22.2% | 50.7% | 15.2% |
| 1.5 | 42.9% | 75.5% | 4.9% |

Values are reassigned in every sequence. Zipf therefore changes how often each symbol appears, not which value it carries. The base model sees the skew only through symbol frequency. The extended model draws its next symbol from the context, and the skew shapes the next symbol distribution that `generations.h5` tracks. The real pool at generation 0 already carries the skew, and `kl_to_gen0` measures drift from that skewed reference.

### Burstiness

Burstiness B is the number of context items that share the query's class. Each sequence also holds B distractor items of one other class. The rest of the context holds other items, each from a distinct class.

| Setting | Bursty items | Distractor items | Other items |
|---|---|---|---|
| L = 2, B = 1 | 1 | 1 | 0 |
| L = 8, B = 1 | 1 | 1 | 6 |
| L = 8, B = 2 | 2 | 2 | 4 |
| L = 8, B = 4 | 4 | 4 | 0 |

B cannot exceed L / 2. The sampler asserts this.

## Design Decisions

**Labels of other items.** Before this change, other items kept their class index as their label. Class indices reach 49, while the model has 5 value labels. Labels of 5 and above embed as zero vectors. Circuit discovery could then pick such a label as the wrong answer and index a logit that does not exist. `fewshot_relabel` in `src/samplers.py` now gives each other item a random label from the three labels the burst and distractor pair does not use. Only the bursty items carry the query's value. A check over 2,000 sequences per setting confirmed every label lies in 0 to 4 and the query's label appears only on bursty items. Runs at L = 2 have no other items and are unchanged.

**Distinct other items.** `--pt_unique_rest 1` draws the other items from distinct classes that differ from the query and distractor classes. Without it, the query class could reappear as an other item with a different label. An other class could also appear twice with two labels, which would give the extended model two values to copy for the next symbol.

**Evaluators follow training.** Burstiness sweep evaluators use the training context length and burstiness. Each model is measured on its own training distribution. A comparison across B on one shared distribution needs an extra evaluator set through `--pe_context_len` and `--pe_burstiness`.

**Replace fractions only.** The sweeps cover the four replacement conditions. Accumulation runs only at the default setting and in the global next symbol sweep.

**Sparse checkpoints.** Each training at the default interval writes 1,001 checkpoints, about 0.8 GB. The sweeps would write about 270 GB at that interval. Sweep runs save every 50,000 sequences instead, 21 checkpoints per generation. The single and default collapse runs keep the dense interval that the plotting notebook expects. The full script needs about 56 GB of disk.

## Running

Circuit discovery needs Python 3.10 or newer. Run the script with `.venv-circuit` first on the path, as set up in [circuit_discovery.md](circuit_discovery.md#running).

```bash
PATH="$PWD/.venv-circuit/bin:$PATH" nohup caffeinate -i bash scripts/nlp_project_run.sh > results/full_run_nohup.log 2>&1 &
```

`caffeinate -i` keeps macOS awake until the script exits. `GLOBAL_SWEEP`, `LONG_SWEEP`, `SAMPLING_SWEEP`, `SMALL_POOL_SWEEP`, `ZIPF_SWEEP`, and `BURST_SWEEP` at the top of the script switch each sweep on or off. `TUNE_RUNS=0` skips tuning, and `MAIN_RUNS=0` skips the two single runs and the ten main collapse runs. All eight can also be set in the environment, for example `TUNE_RUNS=0 MAIN_RUNS=0 bash scripts/nlp_project_run.sh` to run only the sweeps with the settings at the top of the script. `SMALL_POOL_SIZE`, `ZIPF_ALPHAS`, `BURST_CONTEXT_LEN`, `BURST_VALUES`, and `REPLACE_FRACS` set the swept values. `REPLACE_NAMES` must list one name per entry of `REPLACE_FRACS`. `LONG_GENERATIONS` and `SAMPLING_TEMPERATURE` set the generations of the `G20` runs and the temperature of the `T2` runs.

`JOBS=N` runs N experiments at once. Each waits for a free slot and writes its output only to its log. Tuning selection waits for every tuning run, and the summary keeps the run order. A run uses about 1.5 CPU cores and 2 GB of memory, so the free cores and memory set N. `BASE_LR`, `BASE_ITERS`, `EXTENDED_LR`, and `EXTENDED_ITERS` can also be set in the environment, so that a run with `TUNE_RUNS=0` reuses the tuned values.

A one-million-sequence training at L = 2 took 2.2 minutes on the project Mac. Training at L = 8 processes about three times as many tokens per sequence and runs slower.

## Outputs

Every experiment writes `results/<timestamp>_<name>/` with the layout described in [model_collapse.md](model_collapse.md#outputs), including `generations.h5`, `circuits.h5`, and `gen_<g>/circuit/`. Its terminal output goes to `results/<timestamp>_<name>.log`, and `results/<timestamp>_summary.log` lists each experiment as OK or FAILED.
