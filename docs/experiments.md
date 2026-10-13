# Experiments

`scripts/nlp_project_run.sh` runs 68 experiments one after the other and trains 398 models in total. This page lists every experiment, the settings it changes, and its run name. [model_collapse.md](model_collapse.md) describes the collapse pipeline and its pool metrics. [circuit_discovery.md](circuit_discovery.md) describes the circuit metrics that every run records.

## Shared Settings

Every experiment starts from the settings below. Later tables name only the settings an experiment changes.

| Setting | Value | Flag |
|---|---|---|
| Training sequences per run or generation | 1,000,000 | `--train_iters` |
| Batch size | 32 | `--train_bs` |
| Learning rate | 1e-5 | `--lr` |
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

The two single runs train one generation on fresh data at every step.

| Experiment | Variant |
|---|---|
| `base` | base |
| `extended` | extended |

Each ✓ in the grid below is one collapse experiment, run once for the base and once for the extended model. Its run name is `collapse_<variant>_<tag>_<condition>`, for example `collapse_extended_zipf1.0_replace90`. The Default row has no tag, as in `collapse_base_replace50`.

| Setting | Tag | Changed flags | `real_only` | `replace50` | `replace90` | `replace100` | `accumulate` |
|---|---|---|:-:|:-:|:-:|:-:|:-:|
| Default | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Small pool | `N5000` | `--gen_dataset_size 5000` | ✓ | ✓ | ✓ | ✓ | |
| Zipf 0.5 | `zipf0.5` | `--zipf_alpha 0.5` | ✓ | ✓ | ✓ | ✓ | |
| Zipf 1.0 | `zipf1.0` | `--zipf_alpha 1.0` | ✓ | ✓ | ✓ | ✓ | |
| Zipf 1.5 | `zipf1.5` | `--zipf_alpha 1.5` | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 1 | `L8B1` | `--train_context_len 8 --pt_burstiness 1` | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 2 | `L8B2` | `--train_context_len 8 --pt_burstiness 2` | ✓ | ✓ | ✓ | ✓ | |
| L = 8, B = 4 | `L8B4` | `--train_context_len 8 --pt_burstiness 4` | ✓ | ✓ | ✓ | ✓ | |

The burstiness rows also pass `--pt_unique_rest 1`, `--pe_burstiness B B B B`, and `--pe_unique_rest 1 1 1 1`. Every row below Default also passes `--ckpt_every 50000`. The five collapse conditions are defined in the table below.

| Condition | Flags | Real share of the pool from generation 1 |
|---|---|---|
| `real_only` | `--gen_replace_frac 0` | 100% |
| `replace50` | `--gen_replace_frac 0.5` | 50% |
| `replace90` | `--gen_replace_frac 0.9` | 10% |
| `replace100` | `--gen_replace_frac 1.0` | 0% |
| `accumulate` | `--gen_mode accumulate` | 1 / (g + 1) at generation g |

The grid holds 33 experiments per variant. With 66 collapse experiments of 6 generations and 2 single runs, the script trains 398 models.

## Sweep Settings

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

**Replace fractions only.** The sweeps cover the four replacement conditions. Accumulation runs only at the default setting.

**Sparse checkpoints.** Each training at the default interval writes 1,001 checkpoints, about 0.8 GB. The sweeps would write about 270 GB at that interval. Sweep runs save every 50,000 sequences instead, 21 checkpoints per generation. The single and default collapse runs keep the dense interval that the plotting notebook expects. The full script needs about 56 GB of disk.

## Running

Circuit discovery needs Python 3.10 or newer. Run the script with `.venv-circuit` first on the path, as set up in [circuit_discovery.md](circuit_discovery.md#running).

```bash
PATH="$PWD/.venv-circuit/bin:$PATH" nohup caffeinate -i bash scripts/nlp_project_run.sh > results/full_run_nohup.log 2>&1 &
```

`caffeinate -i` keeps macOS awake until the script exits. `SMALL_POOL_SWEEP`, `ZIPF_SWEEP`, and `BURST_SWEEP` at the top of the script switch each sweep on or off. `SMALL_POOL_SIZE`, `ZIPF_ALPHAS`, `BURST_CONTEXT_LEN`, `BURST_VALUES`, and `REPLACE_FRACS` set the swept values. `REPLACE_NAMES` must list one name per entry of `REPLACE_FRACS`.

A one-million-sequence training at L = 2 took 2.2 minutes on the project Mac. Training at L = 8 processes about three times as many tokens per sequence and runs slower.

## Outputs

Every experiment writes `results/<timestamp>_<name>/` with the layout described in [model_collapse.md](model_collapse.md#outputs), including `generations.h5`, `circuits.h5`, and `gen_<g>/circuit/`. Its terminal output goes to `results/<timestamp>_<name>.log`, and `results/<timestamp>_summary.log` lists each experiment as OK or FAILED.
