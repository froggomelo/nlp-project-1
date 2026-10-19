# Head-to-Head Circuit Tests

Circuit discovery ([circuit_discovery.md](circuit_discovery.md)) ranks edges by attribution and keeps the ACDC circuit. Its figures suggest that the previous-token head A0.2 feeds the induction head A1.3 through A1.3's key input. This extension tests that claim causally. It asks which layer-0 heads talk to which layer-1 heads, through which input (query, key or value), and whether the resulting circuit reproduces the model's behaviour. It runs after circuit discovery on every generation's model, so the same tests cover the base and extended single runs and every collapse generation.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model, prompts and metric | the converted model, prompt pairs and logit difference D of circuit discovery | the results are directly comparable with the reported 9.18 and the 80% recovery target |
| Patching | auto-circuit's edge masks with resample ablation from the corrupt prompt | one edge into one head's Q, K or V input can be patched on its own, exactly as ACDC does |
| Candidate heads | P: the layer-0 head with the highest previous-token score; I: the layer-1 head with the highest induction score | heads are found from the data in every model and generation, not taken from a fixed list |
| Edge selection and testing | the edge matrix on the discovery half of the prompts, all other tests on the held-out half | circuits are not selected and scored on the same prompts |
| Head ablation | zero, mean and resample ablation of every edge leaving a head | zero and mean ablation remove a head; resample ablation replaces its output with the corrupt prompt's, which also tests what it carries |
| In-context accuracy | the `fsl_test_class` evaluator: held-out classes, so the answer must come from the context | accuracy can stay high while D falls, so both are reported |
| Restoring an edge | on the corrupt prompt, the edge alone carries the sender's clean output | in a corrupt run the sender's own output is corrupt, so leaving the edge unpatched restores nothing |
| Conditioning | ratios are stored as NaN when D_clean − D_corrupt is below 1 | untrained or collapsed models give ratios of near-zero gaps |

## Measures

D is the correct answer's logit minus the wrong answer's logit at the final position, averaged over prompts. The two value corruptions come from circuit discovery. `value_exemplar_swap` moves the matching exemplar, which tests where the query looks (Q and K routes). `value_label_swap` swaps the two labels, which tests what is copied (V routes). The extended model adds `next_symbol`.

* **Share of the gap lost** by an intervention: (D_clean − D) / (D_clean − D_corrupt). 0 means no effect and 1 means the full corrupt behaviour.
* **Recovery** R of a circuit: (D_circuit − D_corrupt) / (D_clean − D_corrupt), with the circuit's edges clean and every other edge patched from the corrupt prompt.
* **Retrieval** of a layer-1 head: the attention from the final position to the label after the matching exemplar, minus its attention to the other labels. It is the induction score of the notebook, measured per intervention. A value near 1 means the head looks up the correct label.
* **Direct effect** of a layer-1 head: its output at the final position through the final LayerNorm and unembedding, as correct minus wrong logit.

## Experiments

1. **Head ablation.** Every head is ablated three ways, reporting the share of the gap lost, accuracy on the prompts, and accuracy on held-out classes.
2. **Edge matrix.** Each edge from the input embedding or a layer-0 head into a layer-1 head's Q, K or V input carries its corrupt activation, one at a time (9 × 8 × 3 edges). The figure shows the share of the gap lost. Weight-based Q-, K- and V-composition scores (Elhage et al., 2021) are shown beside it, with the layer-1 LayerNorm weight folded in.
3. **The P → I connection, per input channel.**
   * *Disrupt*: only P → I.X carries its corrupt activation.
   * *Corrupt all inputs*: every edge into I.X is corrupt.
   * *Restore in input*: as before, but P → I.X stays clean. The share of the damage it undoes is the restore score.
   * *Restore alone*: on the corrupt prompt, only P → I.X carries P's clean output.

   Controls repeat the disruption for the other senders into I.X and for P into the other layer-1 heads. The strongest head-to-head edge in the matrix is tested the same way when it is not P → I.
4. **Circuit verification.** Each circuit keeps its edges clean and patches the rest, reporting R and accuracy on the prompts and on held-out classes.
   * `single_path`: P reads the input, P → I.K, I reads its query and value from the input and writes to the output.
   * `induction_heads`: the same pattern for every layer-0 head with a previous-token score of at least 0.5 and every layer-1 head with an induction score of at least 0.5.
   * `induction_heads_plus`: `induction_heads` plus every edge whose single-edge noising costs at least 2% of the gap, with their input and output edges.
   * `acdc`: the ACDC circuit at τ = 0.001.
   * `full`: every edge, the check that R = 1.

## Outputs

Each model writes `head_circuit/` next to its `circuit/` folder (`gen_<g>/head_circuit/` in a collapse run).

| File | Contents |
|---|---|
| `<target>_head_circuit.json` | all numbers: head scores and roles, head ablations, edge matrix, composition scores, the connection tests with controls, and every circuit with its edges |
| `<target>_edge_matrix.{png,pdf}` | single-edge noising per Q/K/V input, above the composition scores |
| `<target>_head_ablation.{png,pdf}` | share of the gap lost and held-out accuracy drop per head and ablation |
| `<target>_connection_<P>_<I>.{png,pdf}` | D, accuracy and retrieval of the receiver for every intervention, per channel |
| `<target>_circuits.{png,pdf}` | R and accuracy of every circuit |

`head_circuit.h5` in the run folder holds one row per generation. Each `<target>/<metric>` row holds:
* P and I and their scores;
* `PI_<X>_disrupt`, `PI_<X>_restore` and `PI_<X>_restore_alone` for each channel X;
* `<circuit>_recovery`, `<circuit>_accuracy` and `<circuit>_eval_accuracy`.

Collapse runs also draw `head_circuit_<target>_generations.{png,pdf}`. Two commands redraw the figures:

```bash
python -m plotting.head_circuit_plots results/<run>/head_circuit
python -m plotting.head_circuit_plots --conditions results
```

The first redraws one model's figures from its JSON files. The second compares the main collapse conditions in `results/figures/head_circuit/`.

## Running

The tests run with `--circuit-discovery` after circuit discovery, on the same prompts; `--no_head_circuit` switches them off. They take about 20 seconds per model on a laptop CPU. A saved checkpoint can be analysed on its own:

```bash
python -m src.head_circuit --checkpoint results/<run>/checkpoints/<iter>.eqx \
  --run_config results/<run>/config.json --out_folder results/<run>/head_circuit
```

`scripts/nlp_project_run.sh` reads its sweep switches and `MAIN_RUNS` from the environment. The main configurations run alone with:

```bash
SMALL_POOL_SWEEP=0 ZIPF_SWEEP=0 BURST_SWEEP=0 bash scripts/nlp_project_run.sh
```

The sweeps then run on their own with:

```bash
MAIN_RUNS=0 bash scripts/nlp_project_run.sh
```

## Validation

On the base model retrained by the run script, D_clean is 9.18 and the ACDC circuit recovers 89%, as in circuit discovery. With every edge kept R is 1. Zero-ablating A0.2, A1.3 or A0.1 gives held-out accuracies of 88.7%, 90.7% and 93.6%, identical to the JAX model's `--opto_ablate_heads`.

## Findings

The findings below come from the tuned runs of `scripts/nlp_project_run.sh` started at `20261015142800`. These are the two single runs and the ten main collapse runs at N = 20,000 with six generations each. The sweeps write the same rows to their own `head_circuit.h5`. The gap is D_clean − D_corrupt for the exemplar swap, and the P → I.K numbers use that target unless stated otherwise. [results.md](results.md) reports accuracy and pool metrics for every experiment.

### Base Model

The previous-token head A0.2 talks to the induction head A1.3 through A1.3's key input. A0.2 has a previous-token score of 0.97 and A1.3 an induction score of 0.97, the highest in their layers. A0.2 → A1.3.K is the strongest edge in the matrix. Corrupting it alone costs 22% of the gap and drops A1.3's retrieval from 0.97 to −0.02. The same edge into A1.3's query or value input costs 0.0% and 0.1%. Corrupting every key input of A1.3 while keeping this edge clean undoes 57% of the damage. On the corrupt prompt, giving A1.3 only A0.2's clean output restores 21% of the gap and lifts its retrieval from −0.97 to 0.02.

The connection is one strand of a redundant circuit. A0.2 also feeds the keys of A1.1 (7%), A1.2 (3%), and A1.5 (3%). The other previous-token heads, A0.1 and A0.5 with scores of 0.93 and 0.94, feed A1.3's key at 3% and 4%. The induction-head circuit joins the three previous-token heads to the seven layer-1 heads with induction scores of at least 0.5.

| Base model, exemplar swap | Edges | R | Accuracy | Held-out classes |
|---|---|---|---|---|
| A0.2 → A1.3 path only | 8 | 21% | 9% | 10% |
| Previous-token heads A0.1, A0.2, A0.5 → induction heads A1.0 to A1.5, A1.7 | 52 | 97% | 100% | 98.5% |
| ACDC, τ = 0.001 | 19 | 81% | 100% | 97% |
| Full model | 257 | 100% | 100% | 98.9% |

The induction-head circuit recovers 97% of the gap and stays within 0.4 percentage points of the full model on held-out classes.

**Head ablation.** Zero or mean ablation costs 4% of the gap for A0.2 and 13% for A1.3. Held-out accuracy falls from 98.9% to 95.5% and 97.9%. Resample ablation costs 47% and 45% and drops held-out accuracy to 51% and 58%. Other heads compensate when a head is silenced, but the model follows a head that carries wrong information.

**The value route.** For the label swap, the induction-head circuit recovers 0.2%. Label content reaches A1.3's value input through layer-0 heads, mainly A0.6 (15% of the gap), A0.0 (6%), and A0.3 (6%). Restoring A0.6 → A1.3.V alone undoes 48% of the damage from corrupting A1.3's value input. Adding the edges above 2% lifts recovery to 74% with 85 edges. ACDC reaches 73% with 37 edges, below the 80% target at τ = 0.001.

**Across collapse generations.** At T = 1 the mechanism survives every condition almost unchanged. P = A0.2 and I = A1.3 in every generation of every condition except generation 5 of `accumulate`, which moves to A0.1 and A1.1. The three replacement conditions train identical models, described in [results.md](results.md#main-collapse-runs), and share one row in the table.

| Base model | I's induction score, gen 0 → gen 5 | Gap, gen 0 → gen 5 | P → I.K disrupt | P → I.K restore | Induction-head circuit R |
|---|---|---|---|---|---|
| `real_only` | 0.96 → 0.97 | 39.7 → 40.0 | 17–26% | 62–81% | 0.96–0.97 |
| `replace50`, `replace90`, `replace100` | 0.96 → 0.94 | 39.7 → 29.0 | 14–20% | 53–74% | 0.96 |
| `accumulate` | 0.96 → 0.90 | 39.7 → 22.1 | 17–30% | 62–80% | 0.95–0.97 |

The disrupt and restore ranges for `accumulate` exclude generation 5, where P and I change. The induction-head circuit keeps its held-out accuracy at 98–100% throughout. ACDC recovers 79–88% for the exemplar swap and 75–86% for the label swap. Synthetic data shrinks the gap by up to 44%, while the circuit that carries it stays in place.

### Extended Model

The extended model solves the value task with a different mechanism. A single layer-1 head, A1.0, carries the answer. Its induction score is only 0.28, and no other layer-1 head scores above 0. The best previous-token head, A0.7, scores 0.41. Resample ablation of A1.0 costs 92% of the gap and drops held-out accuracy from 97.9% to 3%. Zero ablation costs 48%. The strongest edge for the exemplar swap runs from the input embedding straight into A1.0's key input (29% of the gap). Layer-0 heads add smaller key edges from A0.4 (9%), A0.1 (8%), and A0.7 (4%). The A0.7 → A1.0 path recovers only 4%, while the circuit of edges above 2% recovers 93% with 27 edges and ACDC 96% with 29 edges. A1.0 therefore matches the query mainly against the raw exemplar embeddings, not through previous-token composition.

For the label swap, values reach A1.0's value input through A0.0 (29% of the gap), A0.2 (16%), A0.3 (14%), and A0.7 (11%). ACDC recovers 94% with 45 edges.

The next symbol has no dominant edge. The strongest, from the input embedding into A1.1's value input, costs 4% of the gap. Resample ablation of a single head costs at most 11% (A0.4). ACDC needs 174 edges to recover 98%.

**Across collapse generations.** Two mechanisms alternate. Some generations use the A1.0 mechanism, with an induction score of 0.09 to 0.31. Others form a classic induction circuit, A0.2 or A0.5 into A1.3 or A1.4, with an induction score of 0.47 to 0.72. In those generations P → I.K costs 8–89% of the gap and the induction-head circuit recovers 0.22–0.96.

| Extended model | Mechanism by generation 0–5 | Gap, gen 0 → gen 5 |
|---|---|---|
| `real_only` | A1.0, induction, induction, A1.0, A1.0, induction | 30.6 → 31.6 |
| `replace50` | A1.0, induction, induction, A1.0, A1.0, induction | 30.6 → 29.6 |
| `replace90` | A1.0, induction, induction, induction, A1.0, induction | 30.6 → 24.3 |
| `replace100` | A1.0, induction, induction, induction, induction, induction | 30.6 → 26.7 |
| `accumulate` | A1.0, induction, induction, A1.0, induction, A1.0 | 30.6 → 16.7 |

Retraining alone switches the mechanism, as `real_only` shows. Under full replacement the model settles on A0.2 → A1.3 from generation 1. From generation 2 its induction score is 0.69–0.71, P → I.K disruption costs 49–66% of the gap, and restoration undoes 74–88%. ACDC recovers 90–99% in every generation and condition, and clean accuracy stays at 98–100%. The induction score of the extended model therefore reports which mechanism a generation formed, not how well it learned the task.

## Limitations

Edges are patched at every position at once, as in circuit discovery, so a connection cannot be pinned to the label or the query position. Single-edge noising understates edges that are redundant with others. Several previous-token heads feed several induction heads, so the role-based circuit is needed to see their joint effect. The role thresholds (0.5) and the edge threshold (2%) are fixed choices. Composition scores ignore the LayerNorm's centring and input-dependent scale. Retrieval is defined only when the final position is an exemplar, so the next-symbol target has none.

## References

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. NeurIPS, 2023.

Elhage, N., Nanda, N., Olsson, C., et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. ICLR, 2023.
