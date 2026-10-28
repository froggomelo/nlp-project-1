# Ablation: Constrained Next-Symbol Generation

**Question.** Does recursive training still degrade the extended model's induction circuit, symbol-selection distribution, and next-label accuracy when generated next symbols are constrained to the symbols already in the context?

**Design.** `--gen_symbol_constraint context` masks the symbol head's logits outside the distinct context classes before sampling, keeping the model's preferences among them and the temperature ([model_collapse.md](model_collapse.md#constrained-next-symbols)). Only the next-symbol choice changes; the query and next labels are still sampled from the value head. Each constrained run reuses the generation 0 checkpoint of the matching unconstrained run (seeds 5, 15, 25), so both arms start from the same model and differ only in the synthetic data they generate. `real_only` samples nothing and is shared. [experiments.md](experiments.md#ablation-constrained-next-symbols) lists the runs. All numbers below are generation 19, mean ± sd over three seeds, unless stated.

**Validation.** Before the runs: constrained sampling never left the context (0 of 20,000 for a sound and a collapsed model at T = 1 and 3), probabilities sum to 1 over the context classes with the model's preference ratio and the temperature applied after masking, empirical sampling matched the renormalised probabilities within 4 SE, duplicate context symbols were handled, an untrained model produced chance-level labels while staying in context (the labels come from the model, not the ground truth), and the default mode reproduced an existing run bit for bit. In all 18 constrained runs the generated off-context rate is exactly 0 in every generation. The T = 3 accumulation runs of seeds 5 and 15 first stopped on a plotting error when ACDC found no edges; after the fix both were retrained from generation 0, and the tables use these retrained runs. Seed 15 reproduced its first run exactly; seed 5 matched through generation 7 and then differed slightly (generation 19 query accuracy 0.558 against 0.563), so repeated runs are close but not always bit-identical.

**Measures.** Query and next-label accuracy are clean accuracies on the 100 held-out test classes. The margin and ACDC circuit size come from circuit discovery on the exemplar-swap prompts. The symbol measures come from 5,000 held-out real prompts with the true query label supplied (`scripts/symbol_distribution.py`): the *unmasked context mass* is the probability the symbol head gives the context classes, and *KL to uniform* compares the distribution renormalised over the context classes with the true 50/50 rule. Zero off-context generation in the constrained arm is enforced by the mask; whether the model has *learned* the context rule is read from the unmasked context mass.

## Generation 19

| Condition | Arm | Query acc. | Next-label acc. | Margin | ACDC edges | Context mass | KL to uniform | Pool next-label error |
|---|---|---|---|---|---|---|---|---|
| Generation 0 (shared) | | 0.96 ± 0.00 | 0.96 ± 0.01 | 14.8 ± 0.8 | 49 ± 3 | 0.984 ± 0.002 | 0.247 ± 0.005 | 0 |
| Real only (shared) | | 0.95 ± 0.01 | 0.94 ± 0.01 | 15.7 ± 0.3 | 42 ± 7 | 0.985 ± 0.001 | 0.247 ± 0.010 | 0 |
| *T = 3* | | | | | | | | |
| 50% replaced | unconstrained | 0.84 ± 0.03 | 0.93 ± 0.03 | 3.8 ± 0.3 | 89 ± 19 | 0.657 ± 0.010 | 0.217 ± 0.009 | 0.05 ± 0.00 |
| | constrained | 0.87 ± 0.02 | 0.88 ± 0.01 | 2.2 ± 0.2 | 130 ± 17 | **0.983 ± 0.001** | 0.264 ± 0.005 | 0.35 ± 0.00 |
| 90% replaced | unconstrained | 0.34 ± 0.01 | 0.38 ± 0.01 | 0.1 ± 0.0 | 180 ± 4 | 0.150 ± 0.004 | 0.205 ± 0.006 | 0.27 ± 0.00 |
| | constrained | 0.35 ± 0.03 | 0.35 ± 0.03 | 0.0 ± 0.0 | 148 ± 9 | **0.987 ± 0.001** | 0.289 ± 0.010 | 0.71 ± 0.00 |
| 100% replaced | unconstrained | 0.20 ± 0.01 | 0.20 ± 0.01 | −0.1 ± 0.1 | 163 ± 7 | 0.040 ± 0.001 | 0.160 ± 0.007 | 0.79 ± 0.03 |
| | constrained | 0.20 ± 0.02 | 0.19 ± 0.01 | −0.0 ± 0.1 | 132 ± 4 | **0.988 ± 0.000** | 0.278 ± 0.010 | 0.80 ± 0.00 |
| Accumulate | unconstrained | 0.93 ± 0.01 | 0.92 ± 0.01 | 0.6 ± 0.0 | 11 ± 2 | 0.125 ± 0.009 | 0.024 ± 0.004 | 0.31 ± 0.01 |
| | constrained | **0.57 ± 0.06** | **0.56 ± 0.05** | 0.0 ± 0.0 | 1 ± 2 | **0.985 ± 0.001** | 0.057 ± 0.007 | 0.63 ± 0.00 |
| *T = 1* | | | | | | | | |
| 100% replaced | unconstrained | 0.95 ± 0.02 | 0.93 ± 0.02 | 12.8 ± 0.9 | 92 ± 36 | 0.883 ± 0.006 | 0.558 ± 0.008 | 0.03 ± 0.01 |
| | constrained | 0.94 ± 0.02 | 0.92 ± 0.01 | 11.9 ± 1.2 | 107 ± 27 | 0.991 ± 0.001 | 0.545 ± 0.015 | 0.03 ± 0.01 |
| Accumulate | unconstrained | 0.98 ± 0.01 | 0.97 ± 0.02 | 6.3 ± 0.1 | 34 ± 5 | 0.950 ± 0.006 | 0.126 ± 0.002 | 0.01 ± 0.00 |
| | constrained | 0.98 ± 0.00 | 0.97 ± 0.00 | 7.4 ± 0.7 | 31 ± 9 | 0.985 ± 0.001 | 0.135 ± 0.008 | 0.01 ± 0.00 |

The pool next-label error counts only pool items whose next symbol occurs in the context. In the unconstrained arm at T = 3, about 90–96% of generated next symbols are off-context and excluded, so the two arms' pool errors are not directly comparable; the error of newly generated labels is about the same in both arms (0.77 ± 0.01 in each under accumulation). `python -I visuals/constrained_figures.py` prints the full table, including the per-prompt bias |q − ½|, and the LaTeX rows of Table 1 with the constrained extended runs.

![Trends](../visuals/figures/constrained_trends.png)

*Constrained (solid) and unconstrained (dashed) runs at T = 3, mean over three seeds. Real only is the shared control.*

## Findings

**Does collapse still occur when next symbols are constrained?** Yes, for the query and next labels. Under replacement at T = 3 the two arms are almost indistinguishable: query accuracy settles at 0.87 vs 0.84 (50%), 0.35 vs 0.34 (90%) and 0.20 vs 0.20 (100%), with next-label accuracy, margins and the generation at which accuracy first drops below 0.9 (generations 2–5) matching. The query-value collapse is therefore not driven by invalid symbols but by the label errors the raised temperature writes into the pool. The symbol distribution does not collapse: the unmasked context mass stays at 0.98–0.99 in every constrained run, against 0.04–0.66 unconstrained.

**Does the model develop preferences among context symbols?** No systematic preference: the renormalised probability of the first context position stays at 0.49–0.53 and that of the query's class at 0.50–0.56 in every condition. The per-prompt bias is already present at generation 0 (KL to uniform 0.25; the fresh model favours one of the two symbols per prompt without a consistent direction). At T = 3 constrained replacement raises it only slightly (0.26–0.29). At T = 1, full replacement roughly doubles it, to 0.55 in both arms, rising from generation 0 to generation 5 and then levelling off: sampling the model's own preferences at T = 1 reinforces them, whereas T = 3 flattens them before they enter the pool. Under unconstrained replacement the falling KL (to 0.16) reflects a symbol head that has stopped using the context, since flat logits are trivially 50/50, not a better-calibrated rule.

**Does the circuit deteriorate before query accuracy?** Yes, in both arms. At T = 3 the margin falls to 0.38–0.56 of its generation 0 value after one generation in every condition, while accuracy first drops below 0.9 only at generation 2 or later. Under constrained accumulation the margin falls below 1 at generations 9–11 and accuracy below 0.9 at generations 12–17. The extended model's induction-head match and copy scores vary too much across seeds to time this precisely (generation 0 direct effect 5.9 ± 3.5), although the copy falls to about 0 under every T = 3 condition while the match keeps 0.19–0.70 of its generation 0 value.

**Does accumulation protect the contextual distribution, or only query accuracy?** Unconstrained, accumulation protects query accuracy (0.93) but not the contextual distribution (context mass 0.13). Constrained, the contextual distribution is kept (0.985) but accumulation no longer protects query accuracy: it falls to 0.57 ± 0.06, with the margin at 0 and the ACDC circuit empty or nearly so (1 ± 2 edges). Accumulation's protection of query accuracy in the main experiment therefore depended on the generated data containing mostly off-context next symbols.

**How much of the previous degradation was associated with invalid symbols?** All of the symbol-distribution degradation: the unmasked context mass, which fell to 0.04 under unconstrained full replacement, stays at 0.98–0.99 once generated symbols are valid, and even at T = 1 it declines unconstrained (to 0.88 under full replacement and 0.95 under accumulation) but not constrained (0.99). None of the query-value degradation under replacement, which is unchanged. Under accumulation the association reverses: constraining the symbols coincides with a deeper query collapse.

**Does the constraint prevent, delay, or change the failure mode?** It prevents the collapse of the symbol distribution, leaves the collapse of the query and next labels under replacement unchanged in size and timing, and changes the failure mode under accumulation from a slow loss of margin with accuracy intact into an accuracy collapse.

## Interpretation and Limits

The constraint changes the synthetic data, not only an evaluation, so the arms differ in what the next generation trains on. A plausible account of the accumulation result, consistent with the measurements but not tested by them, is that a wrong next label attached to an off-context symbol does not contradict anything in the context, whereas a wrong label attached to an in-context symbol directly contradicts the copy rule. The constraint moves every generated label error onto in-context symbols: the share of in-context items with wrong labels in the accumulated pool rises to 0.63, against 0.31 unconstrained, while the error rate of newly generated labels is the same in both arms (0.77). Testing this would need a run that keeps off-context symbols but removes their label errors, or one that relabels in-context symbols at a matched rate. Other limits: three seeds; one model size and task; errors forced through the temperature rather than a weaker model; circuit sizes below a margin of 1 are unreliable.

## Summary for the Extended Abstract

Constraining generated next symbols to those in the context isolates two parts of the extended model's collapse. It prevents the symbol head's loss of the context rule entirely (the probability on context symbols stays at 0.98–0.99 against 0.04–0.66 unconstrained) and introduces no systematic preference between context symbols. It leaves the collapse of the query and next labels under replacement unchanged (accuracy 0.20 with no real data, 0.35 with 10%, 0.87 with 50%), and the margin again halves within one generation, before accuracy falls. Under accumulation, however, it removes the protection of query accuracy (0.57 against 0.93), apparently because every generated label error now falls on a symbol present in the context. Collapse in this task is therefore driven by label errors; the off-context symbols of the unconstrained runs were a separate failure of the symbol head that, under accumulation, also diluted those label errors.
