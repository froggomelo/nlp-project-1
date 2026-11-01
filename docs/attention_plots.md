# Attention Head Plots

`plotting/nlp_project.ipynb` plots how the attention heads change across model collapse generations: which heads act as previous-token and induction heads in each generation, which of them the discovered circuit uses, and how strong the induction head stays under each condition. It reads the collapse runs of `scripts/nlp_project_run.sh`.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Runs | every collapse run found in `results/`, and the newest run of each 20-generation condition at T = 3 | the notebook runs unchanged after each new run of the script |
| Sequences | 1,024 real training sequences drawn with seed 7 | the circuit discovery prompts use the same seed, so attention and circuits describe the same inputs |
| Attention source | the JAX model's own attention scores | the plots need no PyTorch |
| Head scores | previous-token and induction scores | they identify the two roles of the induction circuit (Singh et al., 2024) |
| Collapse view | the final model of every generation, scored on the same real sequences | score changes then reflect the model, not the data |
| Missing runs | sections without their runs are skipped | the notebook works while the script is still running |

## Head Scores

A sequence with context length 2 reads `x1 y1 x2 y2 xq`. Each `xi` is an exemplar, each `yi` its label, and `xq` the query whose label the model predicts. The extended model's sequences continue with the query label `yq` and the next symbol `xn`.

The previous-token score of a head is the attention each label position gives to its own exemplar, averaged over the sequences. The induction score is the attention the query gives to the label of the matching exemplar minus the attention it gives to the other label. Only sequences with one matching and one non-matching context item count towards it. A score of 1 means the head always looks up the correct label.

## Figures

The notebook saves every figure as PNG and PDF in `results/figures/nlp_project/`.

| File | Contents |
|---|---|
| `collapse_<...>_heads_per_generation` | the layer 0 previous-token and last-layer induction score of every head per generation, and the last-layer heads in each generation's ACDC circuit when circuit discovery ran |
| `collapse_<variant>_induction_across_generations` | the strongest induction score per generation for each condition of the 20-generation runs at T = 3 (`collapse_<variant>_G20_real_only` and `collapse_<variant>_G20_T3_<condition>`, first seed) |

## Validation on the Report's Runs

The notebook was run on the runs behind the report. Under full replacement at T = 3, generation 0 of the base model has three layer 0 previous-token heads (scores 0.88 to 0.97) and last-layer induction scores up to 0.96, and its ACDC circuit uses three last-layer heads. The induction scores fall to about 0 by generation 5 and the previous-token scores to 0.2 to 0.6, while the circuit grows to use all eight last-layer heads. Across conditions, the strongest induction score stays above 0.75 under real data only, accumulation and 50% replacement, and falls to about 0.25 or less by generation 5 under 90% and 100% replacement.

## Running

```bash
python -m pip install jupyterlab
jupyter lab plotting/nlp_project.ipynb
```

Use the `.venv` kernel and run all cells. Each collapse run adds one model load per generation; the 64 collapse runs of the report's `results/` take under 3 minutes on a 16-core CPU. `N_SEQ` and `SEQ_SEED` in the second code cell change the sequences.

## Limitations

The scores average over sequences and ignore positions outside their definitions, so a head can matter through positions the scores do not cover. The induction score assumes one matching context item and is defined only for sequences with a non-matching one.

## References

Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., and Saxe, A. M. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. ICML, 2024.
