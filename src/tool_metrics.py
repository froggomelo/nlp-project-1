"""How do modern NLP tools see 1880s isiXhosa, before and after modernising the spelling?

For each variant's usable isiXhosa chunks (and two reference points: Imvo's English chunks and modern
isiXhosa from Vukuzenzele) this reports:
  fertility   subword tokens per word, for the LaBSE and XLM-R tokenisers (higher = text is more
              fragmented, i.e. further from what the tokeniser was built on). For every Imvo variant
              the denominator is the RAW word count of the same chunks, because joining prefixes
              changes the number of whitespace words without changing the content.
  oov_tokens  share of word tokens absent from the modern word list (note: partly circular for the
              aspiration rule, which uses the same list; the prefix-only variant is not)
Broken down per year for raw and modern.

Usage:
    python src/tool_metrics.py --variants raw modern modern_asp modern_prefix
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import load_usable

TOKENISERS = {"LaBSE": "sentence-transformers/LaBSE", "XLM-R": "xlm-roberta-base"}
WORD_RE = re.compile(r"[A-Za-z]+")


def fertility(tok, texts, n_words=None):
    n_sub = sum(len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"])
    n_words = n_words if n_words is not None else sum(len(t.split()) for t in texts)
    return n_sub / max(1, n_words)


def oov(texts, lex):
    words = [w.lower() for t in texts for w in WORD_RE.findall(t)]
    return sum(w not in lex for w in words) / max(1, len(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["raw", "modern", "modern_asp", "modern_prefix"])
    args = ap.parse_args()
    lex = set(pd.read_parquet("data/modern/lexicon.parquet")["word"])
    toks = {name: AutoTokenizer.from_pretrained(path) for name, path in TOKENISERS.items()}

    raw = load_usable()
    raw_words = dict(zip(raw["chunk_id"], raw["text"].str.split().str.len()))
    rows, per_year = [], []
    for v in args.variants:
        path = "data/processed/chunks.parquet" if v == "raw" else f"data/processed/chunks_{v}.parquet"
        df = load_usable(path)
        xh = df[df["lang"] == "xh"]
        row = dict(text=f"Imvo isiXhosa ({v})", oov_tokens=oov(xh["text"], lex))
        nw = sum(raw_words[c] for c in xh["chunk_id"])
        row.update({f"fertility_{n}": fertility(t, xh["text"], nw) for n, t in toks.items()})
        rows.append(row)
        if v in ("raw", "modern"):
            for y, g in xh.groupby("year"):
                nwy = sum(raw_words[c] for c in g["chunk_id"])
                per_year.append(dict(variant=v, year=y, oov_tokens=oov(g["text"], lex),
                                     **{f"fertility_{n}": fertility(t, g["text"], nwy) for n, t in toks.items()}))
        if v == "raw":
            en = df[df["lang"] == "en"]["text"]
            rows.append(dict(text="Imvo English (reference)", oov_tokens=float("nan"),
                             **{f"fertility_{n}": fertility(t, en) for n, t in toks.items()}))

    vuk = pd.concat(pd.read_parquet(p) for p in Path("data/modern/vukuzenzele").rglob("*.parquet"))["text"]
    vuk_chunks = [" ".join(w) for t in vuk for w in [t.split()] if w]
    # OOV is not reported for Vukuzenzele: it is one of the word list's own sources (0 by construction)
    rows.append(dict(text="Modern isiXhosa, Vukuzenzele (reference)", oov_tokens=float("nan"),
                     **{f"fertility_{n}": fertility(t, vuk_chunks) for n, t in toks.items()}))

    out = Path("results/tool_metrics")
    out.mkdir(parents=True, exist_ok=True)
    table, years = pd.DataFrame(rows), pd.DataFrame(per_year)
    table.to_csv(out / "tool_metrics.csv", index=False)
    years.to_csv(out / "tool_metrics_by_year.csv", index=False)
    (out / "tool_metrics.json").write_text(json.dumps(dict(table=table.to_dict("records"),
                                                           by_year=years.to_dict("records")), indent=1))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
