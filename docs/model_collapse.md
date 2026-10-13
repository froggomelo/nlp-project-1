# Model Collapse Extension

This extension adds the model collapse pipeline required by Project 1 of COMS4054A/COMS7062A/COMS7066A. Model collapse is the loss of information about the real data distribution when each generation of a model is trained on data produced by the previous generation. The brief asks for two versions of the in-context learning task from Singh et al. (2024). The base model predicts the value paired with the query symbol. The extended model also predicts the next symbol in the sequence and that symbol's value. The extended model is expected to collapse faster, as it may favour individual symbols and skew the symbol distribution over generations.

Both versions run through the same entrypoint, `python -m src.main`, selected with `--model_variant`. The default `--model_variant base` with `--generations 1` and no `--gen_dataset_size` reproduces the original training exactly. A short regression run matched all 41 logged datasets of the unmodified code bit for bit.

## Sequence Format

Symbols are Omniglot characters, represented as ResNet18 feature vectors. Values are integer labels, reassigned at random in every sequence when `--fs_relabel` is set. The run script uses a context of two symbol-value pairs.

The base sequence is `[x1, l1, x2, l2, xq]`. The model predicts the query value `lq` at the query position.

The extended sequence appends the query value and a next pair, `[x1, l1, x2, l2, xq, lq, xn, ln]`. Every token except the final `ln` is embedded. Training uses teacher forcing, which feeds the true earlier tokens rather than the model's own predictions. The model makes three predictions.

| Position | Output head | Target |
|---|---|---|
| `xq` | value head | `lq`, the query's value |
| `lq` | symbol head | the class of `xn`, the next symbol |
| `xn` | value head | `ln`, the next symbol's value |

The symbol head is a second output layer over all 1623 Omniglot classes. The training loss is the mean of the three cross-entropies. The model must stay causal. A non-causal model would let the query position see its own value, and the code rejects that combination.

## Next Symbol Source

The brief does not state where the next symbol comes from. `--next_symbol_source` offers two conditions. The `context` default is a conservative baseline.

With `context`, the next symbol is drawn uniformly from the distinct symbols already in context, as a fresh exemplar of that class. Its value is the value that symbol carries in context. Every target can therefore be copied by an induction head, and the full three-token task remains in-context learning. Skew can only arise from the model favouring some context symbols over others. That preference still compounds across the whole pool, since context symbols follow the Zipfian training distribution.

With `global`, an experimental condition, the next symbol is drawn from the training class distribution over all training classes. A symbol that appears in context takes its context value. An unseen symbol takes a value not used in the context, chosen at random. That value cannot be learned. This condition resembles the setting of Shumailov et al. (2024), where the model must learn an unconditional distribution and the rare classes are expected to disappear first.

## Collapse Pipeline

Setting `--gen_dataset_size N` switches training to a fixed pool of N sequences. Generation 0 draws N real sequences once and trains on them. Before each later generation, the previous generation's model generates new data. `--gen_mode` sets how that data enters the pool.

With `replace` (the default), a fixed set of floor(pN) items is chosen once, with p set by `--gen_replace_frac` (default 0.1). Before every later generation, the previous generation's model regenerates the targets of exactly those items, and they are written back in place. The pool size stays at N, and its real share stays at 1 − p from generation 1 onwards. Synthetic errors still compound, because each generation's synthetic items come from a model trained partly on the previous generation's synthetic items. `--gen_replace_frac 0` gives a real-data control and `--gen_replace_frac 1` full replacement.

With `accumulate`, nothing is removed. Before each later generation, the previous generation's model generates one new sequence from the prompt of each of the N real items, and these are appended to the pool. At generation g the pool holds N × (g + 1) items, and the real share is 1 / (g + 1).

Each generated item keeps a real context and query as the prompt. Only the model outputs drift, which keeps the prompt distribution comparable across generations. Each generation trains for the same `--train_iters`.

Generation proceeds one token at a time. The base model regenerates `lq` only and serves as the control. The extended model samples `lq`, feeds it back, samples the next symbol's class, attaches a random training exemplar of that class, feeds it back, and samples `ln`. Every token is sampled from softmax(logits / T), with T set by `--gen_temperature` (default 1.0). `--gen_sampling argmax` replaces sampling with the most likely token and serves as an ablation. Generated symbols are restricted to training classes. Validation and test classes never enter the training pool as a result.

Each generation starts from the same initialisation by default. `--gen_init continue` trains the previous generation's weights further instead.

## Running

`scripts/nlp_project_run.sh` takes no parameters and runs every experiment in order: the base model, the extended model, and then five collapse conditions for each variant, each over 6 generations with N = 20,000.

| Condition | Flags | Real share from generation 1 |
|---|---|---|
| `real_only` | `--gen_replace_frac 0` | 100% |
| `replace50` | `--gen_replace_frac 0.5` | 50% |
| `replace90` | `--gen_replace_frac 0.9` | 10% |
| `replace100` | `--gen_replace_frac 1.0` | 0% |
| `accumulate` | `--gen_mode accumulate` | 1 / (g + 1) |

To run one experiment by hand, call `src.main` directly, for example with `--model_variant extended --generations 6 --gen_dataset_size 20000 --gen_replace_frac 0.5`, plus the shared flags in the script's `COMMON_ARGS`.

| Flag | Default | Meaning |
|---|---|---|
| `--model_variant` | `base` | `base` or `extended` |
| `--next_symbol_source` | `context` | `context` or `global`, extended model only |
| `--generations` | `1` | number of generations, values above 1 need `--gen_dataset_size` |
| `--gen_dataset_size` | unset | pool size N, unset keeps fresh sampling every step |
| `--gen_mode` | `replace` | `replace` or `accumulate` |
| `--gen_replace_frac` | `0.1` | replace mode: fraction p of the pool that is synthetic |
| `--gen_sampling` | `sample` | `sample` or `argmax` |
| `--gen_temperature` | `1.0` | sampling temperature T |
| `--gen_init` | `fresh` | `fresh` or `continue` |

## Outputs

A pooled run writes one folder per generation. `gen_<g>/log.h5` and `gen_<g>/checkpoints/` follow the original format. `gen_<g>/pool_meta.npz` stores the pool's labels, symbols, and `origin_gen`, the generation that produced each item, where 0 marks real data.

`generations.h5` in the run folder holds one row per generation. It is computed on the pool before that generation trains. Every metric appears twice, once over the whole pool and once with a `_new` suffix over the items regenerated in that generation. At generation 0 the `_new` items are the real data.

| Metric | Meaning |
|---|---|
| `real_frac` | share of the pool that is still real data |
| `query_label_err` | share of items whose query value differs from the query symbol's value in context |
| `query_label_hist` | histogram of query values |
| `next_symbol_hist` | histogram of next symbols over training classes |
| `next_symbol_entropy` | entropy of the next symbol distribution, in nats |
| `next_symbol_kl_to_gen0` | Kullback–Leibler (KL) divergence from the generation 0 distribution, which measures how far the pool has drifted |
| `next_symbol_n_unique` | number of distinct next symbols |
| `next_symbol_top10_mass` | probability mass on the ten most frequent next symbols |
| `next_symbol_tail_mass` | mass on the rarer half of the training classes, ranked by generation 0 frequency |
| `next_offcontext_rate` | share of next symbols absent from their context |
| `next_value_err` | share of in-context next symbols whose value differs from the context value |

The `next_*` metrics exist for the extended model only. In `context` mode, `next_offcontext_rate` counts pure model errors, as real next symbols always appear in context.

The extended model adds these evaluation metrics to each evaluator in `log.h5`. The existing metrics such as `acc` and `in_context_acc` stay at the query position, which keeps them comparable with the base model.

| Metric | Meaning |
|---|---|
| `loss_symbol`, `loss_next_value`, `loss_total` | cross-entropy of the next symbol, of its value, and the mean of all three |
| `symbol_acc`, `next_value_acc` | accuracy of the next symbol and of its value |
| `seq_acc` | share of sequences with all three predictions correct |
| `symbol_pred_in_ctx` | share of predicted next symbols that appear in context |
| `next_value_acc_in_ctx`, `next_value_acc_unseen` | next value accuracy split by whether the next symbol appears in context, NaN when a split is empty |

## Implementation

`src/generations.py` holds the pool, the replace and accumulate steps, and the pool metrics. `src/main.py` contains the three-part loss, the extended evaluation, and the generation loop. The original training loop now sits in `train_generation`, which runs once per generation. `src/samplers.py` appends the next symbol and assigns its value. `src/models.py` adds the symbol head. `src/main_utils.py` defines the flags and checks invalid combinations.

## Limitations

Training logs only the combined loss. The three separate losses appear at evaluation. The optogenetic clamps in `src/opto.py` assume the base sequence layout, and the extended model rejects them. Pool batches are drawn with replacement. Equal passes per generation therefore hold in expectation rather than exactly. In accumulation, every generation trains for the same number of iterations on a growing pool, so each item is seen fewer times as the pool grows. In `global` mode the value of an unseen next symbol is random, which caps `next_value_acc` below 1. Evaluation sets loaded with `--load_eval_data` carry no next symbols and cannot be used with the extended model.

## References

Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., and Saxe, A. M. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. ICML, 2024.

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., and Gal, Y. AI models collapse when trained on recursively generated data. Nature, 631(8022), 755–759, 2024.
