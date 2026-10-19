# Results

This page reports the results of every experiment set. The numbers come from the newest run of each experiment. These are the tuned runs of `scripts/nlp_project_run.sh` started at `20261015142800` and the forced collapse runs of `scripts/forced_collapse_run.sh` started at `20261017023157`. [experiments.md](experiments.md) defines the experiments and their flags. Every run uses one seed.

Four measures carry the comparison. Accuracy is the query accuracy of the `fsl_test_class` evaluator, averaged over the last five evaluations of a generation. Its 100 test classes never enter training. A correct answer therefore needs in-context learning. Label error is `query_label_err` in `generations.h5`, the share of pool items whose query value differs from the value the query symbol carries in context. The logit difference (LD) is the full model's logit difference on the label swap prompts of circuit discovery, the margin of the correct value over the swapped one. The next symbol KL is `next_symbol_kl_to_gen0`, the Kullback–Leibler divergence of the pool's next symbol distribution from generation 0, and exists for the extended model only.

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

No costlier setting beats the chosen one by more than 0.01. The base model at lr 3e-4 gains only 0.005 at 1,000,000 sequences, and the smaller learning rate is kept. Tuning lifts the extended model from 0.75 to 0.97 at 1,000,000 sequences. Every later stage starts from a model that has learned the task.

## Single Runs

The base model reaches 0.988 accuracy with an LD of 19.9. Its strongest induction head A1.3 has an induction score of 0.97. The extended model reaches 0.974 query accuracy and 0.976 next value accuracy with an LD of 13.7. [head_circuit.md](head_circuit.md#findings) describes the mechanism of each model.

## Main Collapse Runs

The main runs use a pool of N = 20,000 and six generations. The table gives accuracy and LD at generation 0 and generation 5.

| Variant | Condition | Accuracy, gen 0 | Accuracy, gen 5 | LD, gen 0 | LD, gen 5 | Label error, gen 5 |
|---|---|---|---|---|---|---|
| base | `real_only` | 0.982 | 0.985 | 19.9 | 20.0 | 0 |
| base | `replace50` | 0.982 | 0.994 | 19.9 | 14.5 | 0.0001 |
| base | `replace90` | 0.982 | 0.994 | 19.9 | 14.5 | 0.0001 |
| base | `replace100` | 0.982 | 0.994 | 19.9 | 14.5 | 0.0001 |
| base | `accumulate` | 0.982 | 0.993 | 19.9 | 11.1 | 0 |
| extended | `real_only` | 0.968 | 0.975 | 15.3 | 16.0 | 0 |
| extended | `replace50` | 0.968 | 0.979 | 15.3 | 15.1 | 0.001 |
| extended | `replace90` | 0.968 | 0.974 | 15.3 | 12.0 | 0.004 |
| extended | `replace100` | 0.968 | 0.970 | 15.3 | 13.1 | 0.003 |
| extended | `accumulate` | 0.968 | 0.973 | 15.3 | 8.3 | 0.001 |

Neither model collapses. Accuracy stays within 0.012 of generation 0 in every condition. The tuned base model regenerates all but about one label in 10,000 correctly. Its three replacement conditions produce the same pool in every generation and train identical models. Synthetic data still lowers the LD by about a quarter under replacement and by almost half under accumulation. The models stay correct but answer with a smaller margin. The extended model's next symbol distribution barely moves. Its KL stays at or below 0.016, its entropy falls from 3.911 to 3.893 nats, and all 50 classes remain.

## Sweeps

### Global Next Symbol

Drawing the next symbol from all training classes gives the only collapse at T = 1 in the default setting. Accuracy falls from 0.937 to 0.874 under `replace90` and to 0.879 under `replace100`, while `real_only` and `accumulate` rise to 0.962 and 0.970. Label error under full replacement grows by about 0.01 per generation and reaches 0.057. The next symbol distribution concentrates slowly. Its top 10 mass rises from 0.214 to 0.28 and its KL reaches 0.035, but no class disappears. The rare class loss reported by Shumailov et al. (2024) does not appear within five generations. The value circuit is weak in this setting, with an LD of 1.2 at generation 0. Recovery ratios of near-zero gaps are unstable and range from −4.3 to 1.1 here.

### Twenty Generations

The base model's label error under full replacement settles at about 0.006 from generation 7. Accuracy stays between 0.977 and 0.998, and the LD levels off near 12.5. The extended model drifts without levelling off. Its label error grows steadily to 0.015 and its KL to 0.033 by generation 19, and its share of next symbols absent from their context reaches 0.10. Its accuracy still ends at 0.968. Under accumulation the extended model's label error stays at or below 0.003 and its KL at 0.006. The contrast matches the prediction of Gerstgrasser et al. (2024) on a small scale. Errors keep growing under replacement and stay bounded under accumulation. The extended model's LD under accumulation falls steadily from 15.3 to 6.3, the largest LD loss at T = 1 in the main settings, with no change in accuracy.

### Sampling

Argmax sampling with full replacement reproduces the base model's `real_only` run exactly. The base model then writes no errors, and its LD decline at T = 1 is entirely the result of sampling noise. The extended model still writes a few deterministic errors under argmax. Its label error reaches 0.001 and its KL 0.020, and its accuracy ends at 0.980. Temperature 2 gives a real collapse.

| Variant | Generation | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| base | accuracy | 0.982 | 0.994 | 0.979 | 0.957 | 0.909 | 0.854 |
| base | label error | 0 | 0 | 0.005 | 0.026 | 0.075 | 0.172 |
| extended | accuracy | 0.968 | 0.978 | 0.964 | 0.911 | 0.796 | 0.543 |
| extended | label error | 0 | 0.012 | 0.063 | 0.215 | 0.435 | 0.610 |

Label error grows faster than linearly in both models. The extended model collapses faster, as the brief predicts, with next value accuracy falling from 0.97 to 0.43. The next symbol distribution stays flat even here, with KL at or below 0.009.

### Forced Collapse

`scripts/forced_collapse_run.sh` repeats the replacement and accumulation conditions at T = 3 and T = 5. The table gives accuracy at generation 5. Generation 0 is 0.982 for the base model and 0.968 for the extended model.

| Variant | T | `replace50` | `replace90` | `replace100` | `accumulate` |
|---|---|---|---|---|---|
| base | 3 | 0.802 | 0.383 | 0.250 | 0.985 |
| base | 5 | 0.799 | 0.327 | 0.212 | 0.983 |
| extended | 3 | 0.832 | 0.399 | 0.209 | 0.969 |
| extended | 5 | 0.870 | 0.357 | 0.187 | 0.968 |

Full replacement drives both models to chance, which is 0.2 for five values. Label error reaches 0.75 to 0.80 and the LD falls to about 0. Under `replace90` and `replace100` the ACDC circuits grow from 42 to 65 edges at generation 0 to 125 to 190 edges. No small circuit explains a model that has lost its mechanism. The real share of the pool sets how far accuracy falls. A 50% real share keeps accuracy at 0.80 to 0.87. Accumulation keeps accuracy within 0.003 of generation 0, although its pools reach label errors of 0.31 to 0.49. Every real item stays in the pool and is trained on in every generation. The LD under accumulation still falls from 19.9 to 1.4 to 2.1 in the base model and to 2.5 to 3.1 in the extended model.

### Pool Size

At N = 5,000 neither model collapses. Accuracy at generation 5 lies between 0.948 and 0.974, about 0.02 below the N = 20,000 runs. The base model's four conditions again train identical models.

### Zipf Class Distribution

The base model is unaffected by class skew. Its accuracy at generation 5 lies between 0.952 and 0.991 at every α. The extended model weakens as α grows. At α = 1.5 its accuracy falls from 0.949 to 0.918 to 0.926 under replacement against 0.956 under `real_only`. At α = 1.0 the fall is smaller, to 0.949 to 0.963 against 0.979. The next symbol KL stays at or below 0.024 at every α. The skewed reference already holds 0.83 of its mass in the top 10 classes at α = 1.5, and the generated pools keep that shape.

### Burstiness

L = 8 with B = 1 is the hardest setting. Accuracy at generation 0 is only 0.853 for the base model and 0.832 for the extended model. Under full replacement the base model falls to 0.746 with a label error of 0.060. Its induction score drops from 0.88 to between 0.24 and 0.45 in every condition, `real_only` included. Retraining on the fixed pool causes that drop, not the synthetic data. The extended model gains accuracy over generations, to 0.854 to 0.886, but its next symbol pool drifts the most of any T = 1 run. The KL reaches 0.120, the top 10 mass rises from 0.215 to 0.348, and 24% of generated next symbols fall outside their context. At B = 2 and B = 4 both models stay between 0.927 and 0.996 in every condition.

## Discussion

At the default temperature neither model collapses within five or twenty generations. The tuned models make too few errors to compound. Collapse appears once the sampled labels carry errors, at T = 2 and above, and its depth follows the real share of the pool. Accumulation prevents it in every setting tested. The extended model collapses faster than the base model at T = 2, as the brief predicts, but the gap closes at T = 3 and T = 5. The predicted skew of the next symbol distribution stays small. It is largest at L = 8 with B = 1 and in the `global` setting, and it never removes a class.

The LD is the most sensitive measure. It falls under every synthetic condition long before accuracy moves. Accuracy alone would miss the loss of margin that precedes collapse.

## Limitations

Every number comes from one seed. Differences below about 0.01 in accuracy are within the noise of a single evaluation. The base model's replacement conditions give identical models at T = 1 and do not test the replace fraction. The extended model switches between two value mechanisms across generations, described in [head_circuit.md](head_circuit.md#findings). Its induction score is therefore 0.13 at generation 0 of the main runs and about 0.7 in the Zipf runs, and single readings of it should not be compared across runs. The results page in `results/site/` was built before the tuned runs finished and needs `python -m plotting.results_page` to show these numbers.

## References

Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J., Korbak, T., Agrawal, R., Pai, D., Gromov, A., Roberts, D. A., Yang, D., Donoho, D. L., and Koyejo, S. Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. arXiv:2404.01413, 2024.

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., and Gal, Y. AI models collapse when trained on recursively generated data. Nature, 631(8022), 755–759, 2024.
