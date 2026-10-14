# Attention Head Plots

`plotting/nlp_project.ipynb` plots the attention heads of the trained models. It shows where each head attends, which heads act as previous-token and induction heads, how those heads form during training, how much the model depends on each one, and how they change across model collapse generations. The notebook reads the outputs of `scripts/nlp_project_run.sh` and needs no extra training.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Runs | the newest `<timestamp>_base` and `<timestamp>_extended` runs, and every collapse run found | the notebook runs unchanged after each new run of the script |
| Sequences | 1,024 real training sequences drawn with seed 7 | the circuit discovery prompts use the same seed, so attention and circuits describe the same inputs |
| Attention source | the JAX model's own attention scores | the plots need no PyTorch and run in either environment |
| Head scores | previous-token and induction scores | they identify the two roles of the induction circuit (Singh et al., 2024) |
| Head ablation | one head's values set to zero with the existing `--opto_ablate_heads` mechanism | it reuses the paper's ablation code and measures how much accuracy depends on each head |
| Collapse view | the final model of every generation, scored on the same real sequences | score changes then reflect the model, not the data |
| Missing runs | sections without their runs are skipped | the notebook works while the script is still running |

## Head Scores

A sequence with context length 2 reads `x1 y1 x2 y2 xq`. Each `xi` is an exemplar, each `yi` its label, and `xq` the query whose label the model predicts. The extended model's sequences continue with the query label `yq` and the next symbol `xn`.

The previous-token score of a head is the attention each label position gives to its own exemplar, averaged over the sequences. The induction score is the attention the query gives to the label of the matching exemplar minus the attention it gives to the other label. Only sequences with one matching and one non-matching context item count towards it. A score of 1 means the head always looks up the correct label.

## Figures

The notebook saves every figure as PNG and PDF in `results/figures/nlp_project/`.

| File | Contents |
|---|---|
| `<variant>_attention_maps` | mean attention map of every head, one row per layer |
| `<variant>_head_scores` | previous-token and induction score of every head |
| `<variant>_single_sequences` | the strongest previous-token and induction heads on three sequences |
| `<variant>_scores_over_training` | training loss and both scores at 40 checkpoints across training |
| `<variant>_head_ablations` | accuracy drop when each head is ablated, ordered by head score |
| `extended_attention_from_yq` | attention at the extended model's next-symbol prediction |
| `collapse_<...>_heads_per_generation` | both scores per generation, and the last-layer heads in each generation's ACDC circuit when circuit discovery ran |
| `collapse_<variant>_induction_across_generations` | the strongest induction score per generation for each main collapse condition |

## Validation on the Trained Base Model

The notebook was run on the base and extended models trained by the run script. In the base model, head A0.2 has the highest previous-token score (0.93) and head A1.3 the highest induction score (0.84). The circuit discovery picks out the same two heads, A0.2 feeding the key input of A1.3. Both scores rise together between 150,000 and 400,000 sequences, the same interval in which the training loss drops below the random-label level. Ablating A0.2 lowers accuracy by 2.6 percentage points, more than any other single head. The collapse sections were tested on a two-generation run.

## Running

```bash
jupyter lab plotting/nlp_project.ipynb
```

Use the `.venv` or `.venv-circuit` kernel and run all cells. The full notebook takes under a minute for the base and extended runs on a laptop CPU. Each collapse run adds one model load per generation. `N_SEQ`, `SEQ_SEED`, and `N_CKPTS_OVER_TRAINING` in the second code cell change the sequences and the number of checkpoints sampled.

## Limitations

The scores average over sequences and ignore positions outside their definitions. A head can matter through positions the scores do not cover, which the ablation plot shows. Ablating one head at a time misses redundant heads that can stand in for each other. The induction score assumes one matching context item and is defined only for sequences with a non-matching one.

## References

Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., and Saxe, A. M. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. ICML, 2024.
