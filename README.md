# NLP project

`src/` contains training code, `scripts/` run scripts, `plotting/` notebooks and plotting tools, `docs/` background documentation, and `data/` the dataset. Results stay in `results/`.

Run all commands from the project folder. Use Python 3.9–3.12; the commands below use Python 3.10.

1. **Set up the environment** (CPU):

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m pip install jupyterlab ipywidgets
   ```

   In a new terminal, activate the environment again with `source .venv/bin/activate`.

2. **Run training:**

   ```bash
   bash scripts/nlp_project_run.sh
   ```

   The script uses the included `data/omniglot_resnet18_randomized_order_s0.h5` dataset. Edit `MAIN_RUN_ITERS` in the script for a shorter run (default: 1,000,000 sequences, or 31,250 batches).

3. **Find the outputs:** each run creates `results/<timestamp>_omniglot50_rl5/`, containing:

   - `config.json` — run settings.
   - `log.h5` — training and evaluation metrics.
   - `checkpoints/*.eqx` — saved model checkpoints.

4. **Plot the results:**

   ```bash
   mkdir -p results/figures
   jupyter lab plotting/ih_paper_plots.ipynb
   ```

   The notebook setup cell locates the project root automatically. Use the `.venv` Python kernel and keep `base_folder = './results'`. In the notebook, replace the exact path segment `/omniglot50_rl5/` with `/<timestamp>_omniglot50_rl5/` for your chosen run.

   Run the opening setup cells, then **Figure 3a** for loss curves. Continue through **Figure 3b** and **Figure 4a** for attention and ablation plots. Stop before **Figure 4b**: that section and later plots need additional training runs that this script does not produce.

   Plots appear inline and are saved as PDFs in `results/figures/` (including `figure_3a.pdf`, `figure_3b.pdf`, and `figure_4a.pdf`). Replotting replaces PDFs with the same names.

See [Paper_README.md](docs/Paper_README.md) for the original paper documentation.
