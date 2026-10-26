# NLP project: model collapse and the induction circuit

This code trains two-layer transformers on the in-context learning task of Singh et al. (2024), retrains them over generations on data that the previous model relabels, and traces how the induction circuit changes. `src/` holds the training and circuit code, `scripts/` the run scripts, `visuals/` the report figures, `report/` the report, and `data/` the dataset. Results go to `results/`.

## Setup

Run all commands from the project folder, with Python 3.10–3.12, on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-circuit.txt
python -m pip install --no-deps auto-circuit==1.0.1
```

Activate the environment again with `source .venv/bin/activate` in every new terminal.

## Running the experiments

```bash
nohup bash scripts/nlp_project_run.sh > results/run.log 2>&1 &
```

This runs every experiment in the report for three seeds:

| Stage | Runs | What it gives |
|---|---|---|
| 1. Tuning | 8 | `scripts/tune_run.sh` trains generation 0 of each model at learning rates 1e-5, 3e-5, 1e-4 and 3e-4, and `scripts/select_hparams.py` picks the learning rate and training length on the validation classes. Every later run uses them. |
| 2. Table 1 | 42 | Base and extended models over 20 generations: real data only, full replacement and accumulation at T = 1, and 50%, 90% and 100% replacement and accumulation at T = 3. |
| 3. Margin formula | 48 | 50%, 90% and 100% replacement and accumulation at T = 3 and T = 5 over 6 generations. |

From scratch it takes about 20 hours with five runs at once and needs about 85 GB of disk and 30 GB of memory. A run that already finished with the same settings is skipped, so the script can be restarted after an interruption and only trains what is missing. Each run's output goes to `results/<timestamp>_<name>.log`, and `results/<timestamp>_summary.log` lists every run as OK, FAILED or SKIPPED.

Tuning can also run on its own, which writes the chosen settings to `results/<timestamp>_tuning.log`:

```bash
bash scripts/tune_run.sh
```

Settings can be changed in the environment, for example `JOBS=3 SEEDS="5:0" bash scripts/nlp_project_run.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `"5:0 15:10 25:20"` | `init:train` seed pairs. The first pair keeps the plain run names; the others add `_seed<init>`. |
| `JOBS` | `5` | Runs at once. |
| `OMP_NUM_THREADS` | `2` | PyTorch threads per run. Keep `JOBS × OMP_NUM_THREADS` at or below the number of cores, or circuit discovery slows down sharply. |
| `MAX_ACCUMULATE`, `MIN_FREE_GB` | `2`, `8` | Accumulation runs at once (about 7 GB each), and the free memory needed to start a run. |
| `TUNE_RUNS` | `1` | `0` skips tuning and reads the settings from the newest tuning log, or from `BASE_LR`, `BASE_ITERS`, `EXTENDED_LR` and `EXTENDED_ITERS` when set. |
| `TABLE_RUNS`, `FORCED_RUNS` | `1` | `0` skips the Table 1 or margin-formula experiments. |
| `SKIP_FINISHED` | `1` | `0` retrains runs that already finished. |

## Outputs and figures

Each run writes `results/<timestamp>_<name>/` with one `gen_<g>/` folder per generation (metrics in `log.h5`, checkpoints, the pool, and the discovered circuits) and three per-generation summaries: `generations.h5` (pool label error and real share), `circuits.h5` (ACDC circuit size and recovery) and `head_circuit.h5` (margin, induction-head scores and patching tests).

```bash
python -I visuals/results_figures.py
```

draws the report figures into `visuals/figures/` and prints the rows of Table 1.

## Further documentation

[model_collapse.md](docs/model_collapse.md) describes the collapse pipeline, [circuit_discovery.md](docs/circuit_discovery.md) the circuit discovery, and [head_circuit.md](docs/head_circuit.md) the induction-head tests. [Paper_README.md](docs/Paper_README.md) documents the original code of Singh et al. (2024).
