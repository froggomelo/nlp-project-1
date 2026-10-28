# NLP Project 1: Induction Heads and Model Collapse: How the Real-to-Synthetic Data Ratio Shapes In-Context Learning Circuits

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

## Quick test

```bash
bash scripts/quick_test.sh
```

This checks the main finding of the report in about 15 minutes on a 16-core CPU. It retrains the base model over 4 generations under real data only, full replacement at T = 3, and accumulation at T = 3, three runs at once. It then draws `visuals/figures/quick_test.png` (test accuracy, margin, and ACDC circuit size per generation) and prints each value next to the report's. Under full replacement, the margin halves and the circuit grows at generation 1 while accuracy holds, and accuracy falls from generation 2. A run starts only while 8 GB of memory are free, so on a smaller machine run `JOBS=1 MIN_FREE_GB=3 bash scripts/quick_test.sh`, one run at a time (about 2 GB each, 30 minutes in all). [experiments.md](docs/experiments.md#quick-test) lists the expected values.

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

[experiments.md](docs/experiments.md#settings) lists the environment variables that change the seeds, parallelism, memory limits and stages.

## Outputs and figures

Each run writes `results/<timestamp>_<name>/` with one `gen_<g>/` folder per generation (metrics in `log.h5`, checkpoints, the pool, and the discovered circuits) and three per-generation summaries: `generations.h5` (pool label error and real share), `circuits.h5` (ACDC circuit size and recovery) and `head_circuit.h5` (margin, induction-head scores and patching tests).

```bash
python -I visuals/results_figures.py
```

draws the report figures into `visuals/figures/` and prints the rows of Table 1. [experiments.md](docs/experiments.md#runs-behind-the-report) maps each table and figure of the report to its runs, and names the earlier scripts that first produced them (kept under the git tag `paper-original-scripts`).

## Further documentation

This project:

- [experiments.md](docs/experiments.md): every experiment, its settings and run names, the quick test, the environment settings, and which runs lie behind each table and figure of the report.
- [results.md](docs/results.md): the results of those experiments.
- [model_collapse.md](docs/model_collapse.md): the collapse pipeline, its pool metrics and its outputs.
- [circuit_discovery.md](docs/circuit_discovery.md): circuit discovery with attribution and ACDC.
- [head_circuit.md](docs/head_circuit.md): the head-to-head tests of the induction circuit.
- [attention_plots.md](docs/attention_plots.md): the attention-head plots of `plotting/nlp_project.ipynb`.
- [constrained_ablation.md](docs/constrained_ablation.md): the ablation that restricts the extended model's generated next symbols to the context (run but not analysed in the report); [experiments.md](docs/experiments.md#ablation-constrained-next-symbols) describes how to run it.

The original code of Singh et al. (2024):

- [Paper_README.md](docs/Paper_README.md): the original paper documentation and run scripts.
- [setup.md](docs/setup.md): the original GPU installation guide.
- [artificial_optogenetics_guide.md](docs/artificial_optogenetics_guide.md): the model's activation access and manipulation.
- [example_cache_explanation.md](docs/example_cache_explanation.md): the layout of the activation cache.
