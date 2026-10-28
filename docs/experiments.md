# Experiments

`scripts/nlp_project_run.sh` runs every experiment behind the report: 8 tuning runs and 90 collapse runs, the latter for three seeds. This page lists each experiment, the settings it changes, and its run name. [results.md](results.md) reports the results. [model_collapse.md](model_collapse.md) describes the collapse pipeline and its pool metrics, and [circuit_discovery.md](circuit_discovery.md) and [head_circuit.md](head_circuit.md) the circuit measures that every run records.

## Shared Settings

Every experiment starts from the settings below. Later tables name only the settings an experiment changes.

| Setting | Value | Flag |
|---|---|---|
| Model | 2 layers, 8 heads, width 64, attention only | `--depth`, `--num_heads`, `--d_model` |
| Context | 2 exemplar-label pairs and a query | `--train_context_len 2` |
| Value labels | 5, reassigned in every sequence | `--fs_relabel 5` |
| Classes | 50 train, 1,473 validation, 100 test | `--class_split 50 1473 100` |
| Batch size | 32 | `--train_bs` |
| Learning rate | tuned: 1e-4 base, 3e-4 extended | `--lr` |
| Training sequences per generation | tuned: 1,000,000 | `--train_iters` |
| Pool size N | 20,000 | `--gen_dataset_size` |
| Next symbol source, extended model | context | `--next_symbol_source` |
| Initialisation per generation | fresh | `--gen_init` |
| Evaluation | every 5,000 sequences on 1,000 sequences per evaluator | `--eval_every`, `--eval_iters` |
| Checkpoint interval | 50,000 sequences | `--ckpt_every` |
| Circuit discovery | on, after every generation | `--circuit-discovery` |

## Stages

| Stage | Experiments | Run names |
|---|---|---|
| 1. Tuning | 8 | `tune_<variant>_lr<lr>` |
| 2. Table 1 | 14 conditions × 3 seeds = 42 | `collapse_<variant>_G20_<condition>` |
| 3. Margin formula | 16 conditions × 3 seeds = 48 | `collapse_<variant>_T<T>_<condition>` |

`<variant>` is `base` or `extended`. Runs of the second and third seed end in `_seed15` and `_seed25`.

### 1. Tuning

`scripts/tune_run.sh` trains generation 0 of each model on the real pool for 3,000,000 sequences at learning rates 1e-5, 3e-5, 1e-4, and 3e-4. `scripts/select_hparams.py` then picks one learning rate and one training length per model, and every later run uses them. `scripts/nlp_project_run.sh` calls the tuning script first; with `TUNE_RUNS=0` it reads the newest tuning log instead. The chosen values are lr 1e-4 for the base model and lr 3e-4 for the extended model, both at 1,000,000 sequences. Tuning runs once, for the first seed.

**Validation evaluator.** The tuning runs add a fifth evaluator, `fsl_val_class`, on the 1,473 validation classes. Like `fsl_test_class`, it needs in-context learning, as the model never trains on these classes. Selection uses only this evaluator, so the 100 test classes stay unseen until the final results.

**Score.** The score is query accuracy, averaged over the evaluations in the last 50,000 sequences before each candidate length of 1,000,000, 2,000,000, or 3,000,000 sequences.

**Training length from one run.** Adam runs at a constant learning rate and every run uses the same seeds. The first 1,000,000 sequences of a 3,000,000 sequence run are therefore the 1,000,000 sequence run, and one run per learning rate covers all three lengths.

**Preferring cheaper settings.** Settings are compared in order of cost, shorter training first and then smaller learning rate. A costlier setting is chosen only when it beats the best cheaper one by more than 0.01, about one standard error of an evaluation. Every collapse run trains for the chosen length in each generation, so a longer length multiplies the cost of every later stage.

The script writes the table and the choice to `results/<stamp>_tuning.log`.

### 2. Table 1

Both models train for 20 generations, generation 0 on real data. Each ✓ is one experiment per model and seed.

| Condition | Flags | Real share from generation 1 | T = 1 | T = 3 |
|---|---|---|:-:|:-:|
| `real_only` | `--gen_replace_frac 0` | 100% | ✓ | |
| `replace50` | `--gen_replace_frac 0.5` | 50% | | ✓ |
| `replace90` | `--gen_replace_frac 0.9` | 10% | | ✓ |
| `replace100` | `--gen_replace_frac 1.0` | 0% | ✓ | ✓ |
| `accumulate` | `--gen_mode accumulate` | 1 / (g + 1) at generation g | ✓ | ✓ |

The T = 3 runs add `--gen_temperature 3` and carry `T3` in their name, as in `collapse_base_G20_T3_replace50`. `real_only` samples nothing, so it has no T = 3 version.

### 3. Margin Formula

`replace50`, `replace90`, `replace100`, and `accumulate` at T = 3 and T = 5, for both models, over 6 generations. The report's margin formula predicts each generation's label error from the previous generation's margin, and these runs supply the pairs at two temperatures. Their first six generations at T = 3 match the 20-generation T = 3 runs of the same seed exactly.

## Ablation: Constrained Next Symbols

`scripts/constrained_run.sh` repeats the extended model's 20-generation runs with `--gen_symbol_constraint context` ([model_collapse.md](model_collapse.md#constrained-next-symbols)), which restricts generated next symbols to the classes in the context. It asks whether recursive training still degrades the model when it can no longer generate invalid next symbols.

| Condition | T | Run name |
|---|---|---|
| 50%, 90%, and 100% replacement, accumulation | 3 | `collapse_extended_G20_T3ctx_<condition>` |
| Full replacement, accumulation | 1 | `collapse_extended_G20_T1ctx_<condition>` |

Each condition runs for the three seeds, 18 runs in all, in `results/constrained/`, which covers every extended-model row of Table 1. `ONLY="<condition> ..."` runs just the named conditions. Each run reuses the generation 0 model of the matching unconstrained run through `--gen0_from`, so both arms start from the same checkpoint. `real_only` samples nothing, so the unconstrained `collapse_extended_G20_real_only` runs serve as its control.

`scripts/symbol_distribution.py` then measures every generation's next-symbol distribution on 5,000 held-out real prompts: the unmasked probability on the context classes, and the distribution renormalised over them with the true query label supplied (its entropy, its KL divergence from the uniform distribution over the distinct context classes, and its preference by position and for the query's class), as well as fully autoregressive generation with and without the constraint. It writes `results/constrained/analysis/<run>.tsv`. `python -I visuals/constrained_figures.py` prints the generation 19 comparison and the LaTeX rows of Table 1 with the extended model's runs constrained (the base model has no next symbol, so its columns are unchanged), and draws the trends into `visuals/figures/constrained_trends.pdf`.

[constrained_ablation.md](constrained_ablation.md) reports the results. The KL divergence from uniform falls when the symbol head stops using the context, because flat logits are trivially uniform over the two context classes. It is therefore read together with the unmasked context mass.

## Design Decisions

**Forced errors.** At T = 1 the tuned base model regenerates all but about one label in 10,000 correctly over 5 generations, and about 0.6% after 20 generations, so the replacement levels barely differ. A raised temperature makes the same well-trained generation 0 model write label errors into the pool, so any later degradation comes from the errors rather than from a weaker model. T = 3 and T = 5 give a range of error rates.

**Twenty generations.** At generation 5 several conditions are still changing: 50% replacement at T = 3 is still losing accuracy, and accumulation only starts to decline after generation 14. Twenty generations show where each condition settles.

**Seeds.** Each seed pair `init:train` sets `--init_seed`, which fixes the initial weights, and `--train_seed`, which fixes the pool, the batch order, the replaced items, and the sampled labels. The pairs are 5:0, 15:10, and 25:20. The evaluation and circuit-discovery sequences keep their default seeds, so every seed is measured on the same sequences. The learning rates are tuned once and shared by all seeds.

**Parallel runs.** The script runs `JOBS` experiments at once, five by default. Each run's PyTorch threads are capped at `OMP_NUM_THREADS=2`. With the default of eight threads per run, six parallel runs put 48 threads on 16 cores, and circuit discovery, which runs many small operations, slowed from under a minute to over an hour per model. Training in JAX is unaffected. At most two accumulation runs run at once, as each reaches about 7 GB at generation 20, and a run starts only while 8 GB of memory are free. Running in parallel does not change results beyond small numerical differences: a 20-generation run reproduced the first six generations of a 6-generation run that ran alone, and a rerun of a constrained accumulation run matched its first eight generations exactly and stayed within 0.02 query accuracy afterwards.

**Restarting.** The two run scripts share their flags, scheduler, and checks through `scripts/common.sh`. Before queueing a run, they look for a finished results folder of the same name whose `config.json` matches the run's flags and whose `head_circuit.h5` holds every generation, and skip the run if they find one.

**Sparse checkpoints.** Runs save a checkpoint every 50,000 sequences, 21 per generation. A 20-generation run then takes about 1 GB for the base model and 1.9 GB for the extended model, and a 6-generation run about 0.3 GB and 0.6 GB.

## Running

```bash
nohup bash scripts/nlp_project_run.sh > results/run.log 2>&1 &
```

From scratch, the script takes about 20 hours with five runs at once and needs about 85 GB of disk. The [README](../README.md#running-the-experiments) lists the environment variables that change seeds, parallelism, and stages.

## Quick Test

`scripts/quick_test.sh` checks in about 15 minutes that the code reproduces the report's main finding. It runs three base-model conditions of the report for 4 generations, at the tuned settings and the first seed, so the runs repeat the first four generations of the report's runs:

| Run | Flags | Report run it repeats |
|---|---|---|
| `quick_base_real_only` | `--gen_replace_frac 0` | `collapse_base_G20_real_only` |
| `quick_base_T3_replace100` | `--gen_temperature 3 --gen_replace_frac 1.0` | `collapse_base_T3_replace100` |
| `quick_base_T3_accumulate` | `--gen_temperature 3 --gen_mode accumulate` | `collapse_base_T3_accumulate` |

`python -I visuals/quick_test_figure.py` then draws test accuracy, margin, and ACDC circuit size per generation into `visuals/figures/quick_test.{pdf,png}` and prints them next to the values of the report's runs:

| Run | Measure | Gen. 0 | Gen. 1 | Gen. 2 | Gen. 3 |
|---|---|---|---|---|---|
| real only | accuracy | 0.98 | 0.98 | 0.98 | 0.99 |
| | margin | 19.9 | 17.2 | 19.4 | 19.8 |
| | ACDC edges | 17 | 25 | 19 | 19 |
| 100% replaced, T = 3 | accuracy | 0.98 | 0.99 | 0.88 | 0.70 |
| | margin | 19.9 | 10.8 | 5.3 | 2.2 |
| | ACDC edges | 17 | 55 | 188 | 209 |
| accumulate, T = 3 | accuracy | 0.98 | 0.99 | 0.99 | 0.99 |
| | margin | 19.9 | 9.5 | 4.6 | 3.1 |
| | ACDC edges | 17 | 38 | 51 | 44 |

Under full replacement the margin halves and the circuit triples at generation 1 while accuracy holds, and accuracy falls only from generation 2. Under accumulation accuracy holds although the margin falls. The same seeds give the same runs, but small numerical differences between machines can shift the later generations a little.

## Runs Behind the Report

Every number and figure in the report comes from the runs below. Their `config.json` files hold exactly the flags that the current `scripts/nlp_project_run.sh` passes, apart from the settings `src.main` fills in itself.

| Report | Runs | Produced by |
|---|---|---|
| Learning rates and training length | `tune_<variant>_lr<lr>` | `scripts/tune_run.sh` |
| Table 1, Figures 2 and 3, Figures S2 and S3 | `collapse_<variant>_G20_<condition>`, with `_seed15` and `_seed25` | `scripts/nlp_project_run.sh`, stage 2 |
| Margin formula (r = 0.994) | `collapse_<variant>_T<T>_<condition>`, first seed | `scripts/nlp_project_run.sh`, stage 3 |
| Context share of the extended model's next symbols | `collapse_extended_G20_T3_replace100` | `scripts/offcontext_check.py` |
| Figures and the rows of Table 1 | all of the above | `python -I visuals/results_figures.py` |

The report states the margin formula's fit for the first seed only, so stage 3 of the script also runs the other two seeds, which the report does not use.

**History.** The runs were first made in stages by separate scripts: the tuning and the T = 1 runs of Table 1 by an earlier `scripts/nlp_project_run.sh`, the margin-formula runs by `scripts/forced_collapse_run.sh`, the T = 3 runs of Table 1 by `scripts/followup_run.sh` and `scripts/g20_table_run.sh`, and the second and third seeds by `scripts/seed_runs.sh`. These scripts passed the same flags and were merged into the current two scripts in commit `7310e49`. They also ran exploratory sweeps that the report does not use (the next symbol drawn from all training classes, Zipfian classes, a smaller pool, burstiness, argmax and T = 2 sampling). The tag `paper-original-scripts` keeps them:

```bash
git show paper-original-scripts:scripts/seed_runs.sh
```

## Outputs

Every experiment writes `results/<stamp>_<name>/` with the layout described in [model_collapse.md](model_collapse.md#outputs), including `generations.h5`, `circuits.h5`, `head_circuit.h5`, and `gen_<g>/circuit/`. Its terminal output goes to `results/<stamp>_<name>.log`, and `results/<stamp>_summary.log` lists each experiment as OK, FAILED, or SKIPPED.
