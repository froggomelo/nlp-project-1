# Labelling guide

There are two labelling jobs. Both feed the method directly, so consistency matters more than speed.

---

## Job 1: translated pairs (`labels/pairs_to_label.csv`)

Save the finished copy as `labels/pairs_labelled.csv`.

Each row is one isiXhosa chunk and one English chunk from the **same issue**. Chunks are ~60-word
windows cut without regard to article boundaries, so a real translation may only overlap partly.

| Label | Meaning | Example |
|---|---|---|
| **2** | Same content: one is a translation or close summary of the other for most of the chunk | Same obituary, same notice, same report in both languages |
| **1** | Partial: clearly the same item/event, but only part of the chunks overlap, or one is a much shorter summary | English summary of a long isiXhosa meeting report |
| **0** | Different content, even if the topic or names are similar | Two different adverts; two different letters that both mention Parliament |

Leave `label` empty if you genuinely cannot tell, and say why in a `note` column.

**Before you start, hide the `source` and `sim` columns.** `sim` is the model's score and `source`
says how the pair was found; we use your labels to test both, so they must not influence you.

How to judge without reading isiXhosa fluently:
- Use the `shared` column: names, places and numbers in both chunks. Several specific shared names or
  dates usually mean the same item. Generic ones ("King", "Town") do not.
- Check whether dates, amounts and people appear in the same order.

Consistency:
- Add a `labeller` column with your initials.
- If two people label, both label the **same first 50 rows**; the second person writes theirs in a
  `label_2` column. `pairs.py tune` then reports Cohen's kappa.

---

## Job 2: spelling rewrites (`labels/normaliser_to_label.csv`)

Save the finished copy as `labels/normaliser_labelled.csv`. **This job needs someone who reads isiXhosa.**

Each row is a historical word type, the rewrite the normaliser proposes (`proposed`), and one example
of the word in context.

| Column | What to write |
|---|---|
| `correct` | **1** if `proposed` is the right modern spelling of this word in this context; **0** if not |
| `gold` | If the proposal is wrong, or `proposed` is empty, write the correct modern spelling. Leave empty if the original is already correct modern isiXhosa |
| `note` | Anything unusual: a name, an English or Dutch word, an OCR error, a dialect form |

Rows with an empty `proposed` are words the normaliser did not change; they measure what it misses.
Mark names of people, places, peoples and God in `note` (e.g. "name"): rewriting these is the ethically
sensitive case discussed in the paper.

Record in the write-up how much isiXhosa each labeller reads; it belongs in the limitations.
