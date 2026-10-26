# Results

This page reports the results of the experiments in [experiments.md](experiments.md). Table 1 and the claims below use seeds 5, 15, and 25; the margin-formula runs at T = 3 and T = 5 so far exist for seed 5 only, and `scripts/nlp_project_run.sh` adds the other two seeds. `python -I visuals/results_figures.py` redraws the report figures and prints Table 1 from the same runs.

Four measures carry the comparison. **Accuracy** is the query accuracy of the `fsl_test_class` evaluator, averaged over the last five evaluations of a generation; its 100 test classes never enter training, so a correct answer needs in-context learning. **Label error** is `query_label_err` in `generations.h5`, the share of pool items whose query value differs from the value the query symbol carries in context. The **margin** is `logit_diff_clean` of the exemplar-swap prompts in `head_circuit.h5`, the logit lead of the correct label over the other label in the context. **Edges** is the size of the ACDC circuit at τ = 0.001 in `circuits.h5`, out of 257.

## Tuning

Tuning selects lr 1e-4 for the base model and lr 3e-4 for the extended model, both at 1,000,000 sequences. The table gives validation accuracy on the 1,473 validation classes.

| Variant | lr | 1,000,000 | 2,000,000 | 3,000,000 |
|---|---|---|---|---|
| base | 1e-5 | 0.958 | 0.973 | 0.983 |
| base | 3e-5 | 0.976 | 0.984 | 0.989 |
| base | **1e-4** | **0.989** | 0.993 | 0.998 |
| base | 3e-4 | 0.994 | 0.997 | 0.993 |
| extended | 1e-5 | 0.746 | 0.865 | 0.865 |
| extended | 3e-5 | 0.898 | 0.899 | 0.900 |
| extended | 1e-4 | 0.946 | 0.954 | 0.951 |
| extended | **3e-4** | **0.970** | 0.967 | 0.960 |

No costlier setting beats the chosen one by more than 0.01. Tuning lifts the extended model from 0.75 to 0.97 at 1,000,000 sequences, so every collapse run starts from a model that has learned the task. At generation 0 the base and extended models reach test accuracies of 0.97–0.98 and 0.96–0.97 across seeds.

## Table 1

Generation 19 of every 20-generation run, as mean ± standard deviation over the three seeds.

| Condition | Real share | Base accuracy | Base margin | Base edges | Extended margin | Extended edges |
|---|---|---|---|---|---|---|
| *T = 1* | | | | | | |
| `real_only` | 100% | 0.98 ± 0.01 | 17.5 ± 2.2 | 23 ± 5 | 15.7 ± 0.3 | 42 ± 7 |
| `replace100` | 0% | 0.98 ± 0.00 | 12.5 ± 0.3 | 35 ± 7 | 12.8 ± 0.9 | 92 ± 36 |
| `accumulate` | 5% | 0.99 ± 0.00 | 9.8 ± 1.0 | 20 ± 3 | 6.3 ± 0.1 | 34 ± 5 |
| *T = 3* | | | | | | |
| `replace50` | 50% | 0.79 ± 0.01 | 2.6 ± 0.1 | 194 ± 11 | 3.8 ± 0.3 | 89 ± 19 |
| `replace90` | 10% | 0.33 ± 0.01 | 0.4 ± 0.1 | 207 ± 3 | 0.1 ± 0.0 | 180 ± 4 |
| `replace100` | 0% | 0.19 ± 0.02 | 0.1 ± 0.0 | 207 ± 7 | −0.1 ± 0.1 | 163 ± 7 |
| `accumulate` | 5% | 0.92 ± 0.02 | 0.7 ± 0.0 | 17 ± 5 | 0.6 ± 0.0 | 11 ± 2 |

Accuracy and margin agree closely across seeds. Circuit size varies most, above all for the extended model at T = 1, and edge counts below a margin of 1 are less reliable.

## Findings Across Seeds

| Finding | Seed 5 | Seed 15 | Seed 25 |
|---|---|---|---|
| T = 1, full replacement: label error at generation 19 | 0.61% | 0.97% | 0.25% |
| T = 3, full replacement: margin at generation 1 relative to generation 0 | 0.54 | 0.54 | 0.56 |
| …edges at generation 0 → 1, accuracy at generation 1 | 17 → 55, 0.99 | 30 → 117, 0.95 | 19 → 51, 0.98 |
| …edges and accuracy at generation 3 | 209, 0.70 | 215, 0.61 | 210, 0.67 |
| Induction head, full replacement, generation 2 relative to 0: attention to the match vs direct effect | 0.92 vs 0.54 | 0.84 vs 0.27 | 0.90 vs 0.41 |
| …attention below half of generation 0 | gen. 4 | gen. 3 | gen. 4 |
| Accumulation at T = 3: attention to the match over generations 0–15 | 0.90–0.98 | 0.90–0.97 | 0.89–0.97 |
| …fall in direct effect by generation 15 | 84% | 82% | 91% |
| …lowest accuracy over generations 0–15 | 0.97 | 0.96 | 0.96 |
| …fall in margin by generation 19; first generation with margin below 1 | 96%, gen. 16 | 95%, gen. 15 | 96%, gen. 16 |
| Final accuracy at T = 3 with 50%, 10%, and 0% real data | 0.78, 0.31, 0.19 | 0.81, 0.33, 0.21 | 0.79, 0.33, 0.18 |

Every finding of the report holds in all three seeds. At T = 1 the tuned models write almost no errors and accuracy holds, while the margin falls and the circuit grows. Once errors are forced at T = 3, the margin halves and the circuit spreads before accuracy falls. Within the induction head, the copy (direct effect) weakens before the match (attention). The real share sets the accuracy at which each condition settles, and accumulation delays the decline without stopping the margin's fall.

## Margin Formula

If the correct label leads the four wrong labels by a margin D, sampling at temperature T makes a share 4e^(−D/T) / (1 + 4e^(−D/T)) of the labels wrong. Across both models, T = 3 and T = 5, and the four conditions of the 6-generation runs of seed 5, this predicts the next generation's label error with r = 0.994 and a mean absolute error of 2.1 percentage points. On the 20-generation T = 3 runs, it gives r = 0.971, 0.977, and 0.984 for the three seeds, with mean absolute errors of 2.6 to 3.2 points.

Accuracy at generation 5 of the 6-generation runs, seed 5:

| Variant | T | `replace50` | `replace90` | `replace100` | `accumulate` |
|---|---|---|---|---|---|
| base | 3 | 0.802 | 0.383 | 0.250 | 0.985 |
| base | 5 | 0.799 | 0.327 | 0.212 | 0.983 |
| extended | 3 | 0.832 | 0.399 | 0.209 | 0.969 |
| extended | 5 | 0.870 | 0.357 | 0.187 | 0.968 |

## Exploratory Experiments

Earlier one-seed experiments are no longer part of the pipeline, and their run folders have been removed; their terminal logs remain in `results/`. They found:

* **Six generations at T = 1.** The base model's 50%, 90%, and 100% replacement runs trained identical models, as it regenerated all but about one label in 10,000 correctly.
* **Argmax sampling.** With full replacement, the base model wrote no errors and its margin did not fall, so the margin's decline at T = 1 comes from sampling noise. At T = 2, label error grew faster than linearly, to 17% for the base model and 61% for the extended model by generation 5.
* **Next symbol drawn from all 50 classes** (`--next_symbol_source global`). The extended model's next symbol distribution drifted but lost no class within five generations at T = 1. At T = 0.5 under full replacement it narrowed further, with the top 10 symbols rising from 21% to 53% of the mass, while accuracy held.
* **Pool size, Zipf class distributions, and burstiness.** A pool of 5,000 and Zipfian classes did not cause collapse at T = 1. With a longer context of eight items and one bursty item, the generation 0 models were weaker (0.85 accuracy), and full replacement raised label error to 14% over 20 generations. With four bursty items, no errors arose.

## Limitations

Three seeds bound the seed-to-seed variation but not its tails. Circuit size is the noisiest measure, and the extended model's single-head scores vary between generations even without synthetic data, so its accuracy and margin are the more reliable readings. Errors are forced through the sampling temperature rather than through a weaker model.

## References

Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J., Korbak, T., Agrawal, R., Pai, D., Gromov, A., Roberts, D. A., Yang, D., Donoho, D. L., and Koyejo, S. Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. arXiv:2404.01413, 2024.

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., and Gal, Y. AI models collapse when trained on recursively generated data. Nature, 631(8022), 755–759, 2024.
