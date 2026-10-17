"""Stage 2a: build a modern isiXhosa word-frequency list from three modern corpora.

Sources (downloaded into data/modern/ by this script if missing):
  vukuzenzele  government newspaper, isiXhosa edition   (dsfsi/vukuzenzele-monolingual, xho)
  wikipedia    isiXhosa Wikipedia, 2023-11-01 dump        (wikimedia/wikipedia, 20231101.xh)
  opus100      isiXhosa side of OPUS-100 en-xh             (Helsinki-NLP/opus-100, en-xh)

Counts are kept per source so the register bias of the list can be inspected (e.g. a word attested
only in OPUS web/subtitle text). Output: data/modern/lexicon.parquet with columns
word, vukuzenzele, wikipedia, opus100, total, n_sources.

Usage:
    python src/build_lexicon.py
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SOURCES = {
    "vukuzenzele": ("dsfsi/vukuzenzele-monolingual", ["xho/train-00000-of-00001.parquet",
                                                      "xho/eval-00000-of-00001.parquet",
                                                      "xho/test-00000-of-00001.parquet"]),
    "wikipedia": ("wikimedia/wikipedia", ["20231101.xh/train-00000-of-00001.parquet"]),
    "opus100": ("Helsinki-NLP/opus-100", ["en-xh/train-00000-of-00001.parquet",
                                          "en-xh/validation-00000-of-00001.parquet",
                                          "en-xh/test-00000-of-00001.parquet"]),
}
WORD_RE = re.compile(r"[a-z]+")


def texts(name, root):
    from huggingface_hub import hf_hub_download
    repo, files = SOURCES[name]
    for f in files:
        path = Path(root) / name / f
        if not path.exists():
            hf_hub_download(repo, f, repo_type="dataset", local_dir=str(Path(root) / name))
        df = pd.read_parquet(path)
        if name == "opus100":
            yield from (t["xh"] for t in df["translation"])
        else:
            yield from df["text"]


def main(root="data/modern"):
    counts = {}
    for name in SOURCES:
        c = Counter()
        for t in texts(name, root):
            c.update(WORD_RE.findall(str(t).lower()))
        counts[name] = c
        print(f"{name}: {sum(c.values()):,} tokens, {len(c):,} types")
    lex = pd.DataFrame(counts).fillna(0).astype(int)
    lex["total"] = lex.sum(axis=1)
    lex["n_sources"] = (lex[list(SOURCES)] > 0).sum(axis=1)
    lex = lex.rename_axis("word").reset_index().sort_values("total", ascending=False)
    out = Path(root) / "lexicon.parquet"
    lex.to_parquet(out, index=False)
    print(f"wrote {len(lex):,} word types to {out}")


if __name__ == "__main__":
    main()
