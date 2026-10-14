# Circuit Discovery

This extension finds the circuit each generation's model uses for the in-context learning task. A circuit is the small set of connections between model components that reproduces a behaviour on its own. Tracking it across generations shows whether model collapse changes the mechanism, not only the accuracy. Discovery uses auto-circuit (Miller et al., 2024), the library recommended by the project paper. It is off by default and enabled with `--circuit-discovery`.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model analysed | each generation's trained model, right after it trains | covers generation 0 and every model that produces data for the next generation |
| Prompts | the same real sequences in every generation | circuit changes then reflect the model alone, not drift in the data |
| Value circuits | two per generation, one per corruption | each corruption exposes a different part of the induction circuit |
| Extended model | value circuits plus a next-symbol circuit | the symbol head is where the extended model is expected to collapse |
| Toggle | `--circuit-discovery` flag, `CIRCUIT_DISCOVERY` in the run script | runs without the flag need no PyTorch and train exactly as before |

## Model Conversion

auto-circuit only works on PyTorch models built with TransformerLens, a library that exposes every internal activation of a transformer through named hooks. This project trains Equinox models in JAX. Each trained model is therefore copied into an equivalent TransformerLens `HookedTransformer` before discovery.

The copy reproduces the attention-only architecture, the pre-LayerNorm blocks, and the rotary position embedding, which rotates adjacent dimension pairs. The fused query, key, and value weights are split per head. The model has no token vocabulary. It embeds image features and labels itself. The JAX model therefore computes the interleaved input embedding, and the copy starts from that residual stream. The value head and the extended model's symbol head each give a separate copy with their own unembedding.

Every conversion is checked before use. The copy's logits must match the JAX logits within 1e-4, or the run stops with an error. On the trained base model the largest difference is 3.9e-6 on logits of magnitude 10. A 0.01 shift in one key weight matrix raises it to 1.8e-3, which the check rejects.

## Prompt Pairs

Discovery compares clean and corrupt versions of the same sequence. The corrupt version changes the answer the model should give. An edge matters when replacing its clean activation with its corrupt activation changes the logit difference, the correct answer's logit minus the wrong answer's logit.

The fixed prompt set holds `--circuit_n_prompts` real training sequences drawn once with `--circuit_seed`. Half score the edges. The other half measure faithfulness.

| Target | Clean prediction | Corruption | Wrong answer |
|---|---|---|---|
| `value_exemplar_swap` | value at the final position | the exemplars of the target symbol and the other context symbol trade places at every position but the last, labels unchanged | the other context value |
| `value_label_swap` | value at the final position | the two context value labels trade places at every position | the other context value |
| `next_symbol` | next symbol at `lq`, extended model only | every symbol is replaced by an exemplar of an unused training class, labels unchanged | the replacement classes |

The two value corruptions bind the target symbol to the other context value. Copying in-context therefore yields the wrong answer in both. The exemplar swap moves the matching symbol to the other position. It exposes where the induction head attends, carried by the previous-token head into the induction head's key input. The label swap keeps every position's symbol and changes the value stored there. It exposes what the induction head copies through its value input.

The next symbol is drawn uniformly from the context symbols. Its answers are all context symbols, and the logit difference averages over them. The sequence is cut after `lq` to make that position the last one, which auto-circuit reads. The model is causal and the cut leaves its activations at `lq` unchanged.

## Discovery and Faithfulness

Edges are scored by mask-gradient attribution with `mask_val=0`, which auto-circuit documents as equivalent to edge attribution patching. Each score is the gradient of the logit difference with respect to a mask that interpolates one edge between its clean and corrupt activations. Corrupt activations come from resample ablation.

The graph is factorised. Every attention head output connects to the query, key, and value inputs of every later head and to the final residual stream, alongside the input embedding. The two-layer model with eight heads per layer has 257 edges. All sequence positions of an edge are ablated together.

Faithfulness keeps the top k edges clean and ablates the rest on the held-out half, for k between 0 and 257. Recovery is the share of the full model's logit difference the circuit restores, measured from the fully ablated model. `n_edges_to_recover` is the smallest tested k with recovery of at least 80%. auto-circuit picks the tested values. In the validation run below they are 0 to 10, every 10 up to 100, 200, and 257. The count is coarse above 10 as a result.

## Validation on a Trained Model

The pipeline was tested on the fully trained base model in `results/20260917152241_omniglot50_rl5` (1,000,000 training sequences) with 512 prompts. The full model's logit difference is 9.18. Both corruptions reverse it to about −9.1.

| Edges kept | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| `value_exemplar_swap` recovery | 22% | 64% | 88% | 94% | 98% | 100% |
| `value_label_swap` recovery | 11% | 43% | 51% | 63% | 79% | 99% |

The exemplar swap finds a compact induction circuit. Its strongest edges run from the input embedding into head A0.2's value input, from A0.2 into the key inputs of heads A1.3 and A1.1, and from the layer 1 heads to the output. Attention patterns confirm these roles. A0.2 places 0.93 of each label position's attention on the preceding exemplar, the highest previous-token score in layer 0. A1.3 places 0.90 of the query's attention on the matching label, the highest induction score in layer 1.

The label swap circuit is less sparse and runs through value inputs. Its strongest internal edge, A0.6 into the value input of A1.3, carries label content to the same induction head. Mean ablation with the label swap was also tried. It recovers only 24% at 50 edges and was dropped.

## Running

auto-circuit needs Python 3.10 or newer. The original `.venv` uses Python 3.9 and stays usable for runs without the flag. A second environment holds both stacks.

```bash
uv venv .venv-circuit --python 3.11
source .venv-circuit/bin/activate
uv pip install -r requirements.txt -r requirements-circuit.txt
uv pip install --no-deps auto-circuit==1.0.1
```

`python -m venv` with any Python 3.10 or newer works in place of `uv`. auto-circuit is installed without its dependencies. Its full list pulls in packages for its built-in tasks that can move JAX off the pinned 0.4.26.

`scripts/nlp_project_run.sh` enables discovery for every experiment with `CIRCUIT_DISCOVERY=1`. Set it to 0 to train without PyTorch. A single run takes the flag directly, for example `python -m src.main <COMMON_ARGS> --model_variant base --generations 6 --gen_dataset_size 20000 --circuit-discovery`.

A saved checkpoint can be analysed without training.

```bash
python -m src.circuits --checkpoint results/<run>/checkpoints/<iter>.eqx \
  --run_config results/<run>/config.json --out_folder results/<run>/circuit
```

Each circuit takes under a minute on a laptop CPU. A run with `--circuit-discovery` but without the packages stops before training with an install hint. Training is unaffected by the flag. A two-generation test run produced identical training losses and accuracies with and without it.

| Flag | Default | Meaning |
|---|---|---|
| `--circuit-discovery` | off | discover circuits after each generation trains |
| `--circuit_n_prompts` | `512` | number of fixed real sequences, split evenly between scoring and faithfulness |
| `--circuit_bs` | `64` | batch size for discovery |
| `--circuit_seed` | `7` | seed for the fixed sequences and the next-symbol replacement classes |

## Outputs

Each generation writes the raw circuit plus three visual views per target to `gen_<g>/circuit/`. Single runs without a pool write them to `circuit/` in the run folder.

| File | Contents |
|---|---|
| `<target>_edges.json` | every edge with its source, destination, and score, sorted by absolute score |
| `<target>_faithfulness.json` | the summary scalars and the recovery curve over edge counts |
| `<target>_graph.html` | auto-circuit's interactive graph, capped at 40 edges |
| `<target>_component.{png,pdf}` | compact ACDC-style component graph; arrow style shows the destination Q/K/V input |
| `<target>_sankey.{png,pdf}` | left-to-right flow graph; ribbon width is the absolute edge score |
| `<target>_prune_scores.pt` | raw auto-circuit edge scores, loadable with `torch.load` |

Both static views contain exactly the same top edges. Positive attribution is blue, negative attribution is red, and line/ribbon width is proportional to the absolute score. The recovering circuit can be larger than the 40-edge readability cap; in that case the subtitle states, for example, “top 40 shown | first tested >= 80% recovery: 200 edges.”

To add both views to an existing run without repeating circuit discovery, run:

```bash
python -m plotting.circuit_plots results/<run>/circuit
```

Pass a single `*_edges.json` file instead of a directory to render only one circuit, or use `--max-edges N` to change the readability cap.

## Reading the Graphs

`A0.2` means attention head 2 in layer 0; `A1.3` means attention head 3 in layer 1. `Embed` is the residual stream entering the transformer, and `Output` is the final residual stream read by the unembedding. In the component view, dotted, dashed, and solid arrows enter a head's query (Q), key (K), and value (V) inputs respectively. An arrow ending at `Output` is also solid.

The two pictures encode edge strength in different ways. In the component view, thicker arrows have larger absolute attribution scores. In the Sankey view, the same quantity controls ribbon width. Blue means the clean activation on that edge locally supports the correct-minus-wrong logit difference; red means it locally opposes it. A negative score does not by itself mean that a head is “bad” or inhibitory: scores are local mask-gradient attributions and interacting edges can change one another's effects.

Read complete routes rather than isolated thick edges. For example, `Embed -> A0.2.V -> A1.3.K -> Output` says that input content enters the value channel of `A0.2`; that head's output helps form the keys used by `A1.3`; and `A1.3` directly affects the prediction. Several individually strong edges may recover nothing until the final edge that completes their path is retained.

The two value targets answer different mechanistic questions:

| Target | What strong routes usually identify |
|---|---|
| `value_exemplar_swap` | where the query finds the matching symbol, especially previous-token-to-induction Q/K routes |
| `value_label_swap` | what label/value content is copied, especially routes entering later heads through V |

Always interpret the picture together with `<target>_faithfulness.json`. `logit_diff_full` should be clearly separated from `logit_diff_ablated`; otherwise recovery is ill-conditioned. `n_edges_to_recover` is the smallest tested top-k circuit reaching 80% recovery on held-out prompts. If it exceeds the 40-edge display cap, the image is only a readable excerpt of the faithful circuit, not the entire circuit.

`circuits.h5` in the run folder holds one row per generation for each `<target>/<metric>`.

| Metric | Meaning |
|---|---|
| `logit_diff_full`, `logit_diff_ablated` | logit difference of the full model and of the fully ablated model |
| `n_edges_to_recover` | smallest tested circuit size that restores 80% of the logit difference |
| `n_edges` | number of edges in the graph |
| `conversion_max_diff` | largest logit difference between the JAX model and its conversion |
| `n_prompts` | number of prompt pairs kept after filtering |

## Implementation

`src/circuits.py` holds the conversion, the prompt pairs, discovery, and the checkpoint command. It is the only module that imports PyTorch. `src/main.py` builds the fixed prompts once, calls discovery after each generation trains, and moves the real training sampler into `make_train_data_sampler` for reuse. `src/main_utils.py` defines the flags and checks the dependencies at startup.

## Limitations

Recovery is unstable when the full and ablated logit differences are close. Early or collapsed models can show this, and their `n_edges_to_recover` should be read alongside `logit_diff_full`. Edges ablate every position at once, which hides which position each connection acts on. The conversion supports MLP blocks, other position embeddings, and models without normalisation, but only the attention-only rotary configuration of the run script has been tested. The circuit describes the unclamped weights when optogenetic clamps are active. The next-symbol circuit averages over the context symbols, which treats them as equally correct.

## References

Miller, J., Chughtai, B., and Saunders, W. Transformer circuit faithfulness metrics are not robust. COLM, 2024. arXiv:2407.08734.

Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., and Saxe, A. M. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. ICML, 2024.
