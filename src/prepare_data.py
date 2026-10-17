"""Stage 0: clean, chunk and split the Imvo Zabantsundu language-split text files.

Reads the per-page isiXhosa (XhTxt) and English (EnTxt) files, cleans OCR artefacts,
cuts each page into fixed-size word windows, flags boilerplate that repeats across
issues (adverts, subscription notices, mastheads), keeps only issues with enough text
in both languages, and assigns every issue to train/val/test (stratified by year).

Outputs (in --out):
    chunks.parquet   one row per chunk, with issue/date/page/lang/text/boilerplate flag
    splits.json      issue date -> split, plus the settings used to make it
    data_card.md     corpus statistics per year and language

Usage:
    python src/prepare_data.py --raw data/raw/TxtLangFiles1870-1894_TB --out data/processed
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

LANG_DIRS = {"xh": "XhTxt", "en": "EnTxt"}
FILE_RE = re.compile(r"IMVO_(\d{4}-\d{2}-\d{2})_p(\d{3})\.txt$")


def clean(text: str) -> str:
    """Undo OCR/layout artefacts without touching spelling."""
    text = text.replace("\r", "")
    text = text.replace("—", " — ").replace("­", "")  # spaced dashes; soft hyphens
    text = text.replace("⁽⁽", '"').replace("„", ",")  # stray OCR quote/comma glyphs
    # word split across a line break: "perni-\ncious" -> "pernicious"
    text = re.sub(r"(\w)[-~¬]\s*\n\s*(\w)", r"\1\2", text)
    # word split mid-line by a stray hyphen + space: "uku- bona" -> "ukubona"
    # (only when a lowercase letter follows, so "Mr. X - Y" style dashes survive)
    text = re.sub(r"(\w)- ([a-z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(words, size):
    """Fixed windows of `size` words; a short tail (< size/2) is merged into the previous chunk."""
    chunks = [words[i:i + size] for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1]) < size // 2:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + tail
    return [" ".join(c) for c in chunks if c]


def shingles(text, n=5):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def flag_boilerplate(df, min_issues, frac):
    """A chunk is boilerplate if >= `frac` of its 5-gram shingles occur in >= `min_issues` issues."""
    sh = [shingles(t) for t in df["text"]]
    seen = defaultdict(set)
    for s, issue in zip(sh, df["issue"]):
        for g in s:
            seen[g].add(issue)
    flags, rep = [], []
    for s in sh:
        r = sum(len(seen[g]) >= min_issues for g in s) / len(s) if s else 0.0
        rep.append(round(r, 3))
        flags.append(r >= frac)
    return flags, rep


EN_FUNCTION_WORDS = set(
    "the of and to in a is that for it was on with as be by at this have from or an are not but his he which "
    "they were their has had been will we our its would who all there more one so if them".split())


def english_ratio(text):
    """Share of tokens that are common English function words (~0.45 for English prose, ~0 for isiXhosa)."""
    toks = re.findall(r"[a-z]+", text.lower())
    return sum(t in EN_FUNCTION_WORDS for t in toks) / max(1, len(toks))


def split_issues(issues, seed, fracs=(0.70, 0.15, 0.15)):
    """Assign whole issues to splits, stratified by year."""
    by_year = defaultdict(list)
    for d in sorted(issues):
        by_year[d[:4]].append(d)
    rng = random.Random(seed)
    out = {}
    for year, ds in sorted(by_year.items()):
        rng.shuffle(ds)
        n_tr = round(len(ds) * fracs[0])
        n_va = round(len(ds) * fracs[1])
        for i, d in enumerate(ds):
            out[d] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return out


def data_card(df, issues_all, kept, args):
    lines = ["# Data card: Imvo Zabantsundu, language-split chunks", "",
             f"Settings: chunk_words={args.chunk_words}, min_words_per_lang={args.min_words}, "
             f"boilerplate = >= {args.bp_frac:.0%} of 5-gram shingles seen in >= {args.bp_min_issues} issues, "
             f"seed={args.seed}", "",
             f"Issues found: {len(issues_all)}; kept (>= {args.min_words} words in both languages): {len(kept)}", ""]
    g = df.groupby(["year", "lang"]).agg(chunks=("text", "size"), words=("n_words", "sum"),
                                         boilerplate=("boilerplate", "mean")).unstack("lang")
    lines += ["| Year | xh chunks | en chunks | xh words | en words | xh boilerplate | en boilerplate |",
              "|---|---|---|---|---|---|---|"]
    for y, r in g.iterrows():
        n = {k: int(r[k]) for k in [("chunks", "xh"), ("chunks", "en"), ("words", "xh"), ("words", "en")]}
        lines.append(f"| {y} | {n[('chunks', 'xh')]:,} | {n[('chunks', 'en')]:,} | {n[('words', 'xh')]:,} | "
                     f"{n[('words', 'en')]:,} | {r[('boilerplate', 'xh')]:.1%} | {r[('boilerplate', 'en')]:.1%} |")
    tot = df.groupby("lang").agg(chunks=("text", "size"), words=("n_words", "sum"), bp=("boilerplate", "mean"),
                                 mm=("lang_mismatch", "mean"))
    lines += ["", "Totals: " + "; ".join(f"{l}: {int(r.chunks):,} chunks, {int(r.words):,} words, "
                                         f"{r.bp:.1%} boilerplate, {r.mm:.1%} language mismatch"
                                         for l, r in tot.iterrows())]
    sp = df.drop_duplicates("issue")["split"].value_counts()
    lines += ["", "Issues per split: " + ", ".join(f"{k}={v}" for k, v in sp.items()),
              "", "Usable chunks (not boilerplate, no language mismatch) per split and language:", ""]
    nb = df[df["usable"]].groupby(["split", "lang"]).size().unstack()
    lines += ["| Split | xh | en |", "|---|---|---|"]
    lines += [f"| {s} | {r['xh']:,} | {r['en']:,} |" for s, r in nb.iterrows()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/TxtLangFiles1870-1894_TB")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--chunk-words", type=int, default=60)
    ap.add_argument("--min-words", type=int, default=1000, help="min words per language for an issue to be kept")
    ap.add_argument("--bp-min-issues", type=int, default=5)
    ap.add_argument("--bp-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang, sub in LANG_DIRS.items():
        for f in sorted((raw / sub).glob("IMVO_*.txt")):
            m = FILE_RE.search(f.name)
            if not m:
                continue
            date, page = m.group(1), int(m.group(2))
            words = clean(f.read_text(encoding="utf-8", errors="ignore")).split()
            for k, text in enumerate(chunk_words(words, args.chunk_words)):
                rows.append(dict(issue=date, year=int(date[:4]), page=page, lang=lang,
                                 chunk_id=f"{date}_p{page:03d}_{lang}_{k:03d}", text=text,
                                 n_words=len(text.split())))
    df = pd.DataFrame(rows)

    issues_all = set(df["issue"])
    words = df.groupby(["issue", "lang"])["n_words"].sum().unstack(fill_value=0)
    kept = set(words[(words["xh"] >= args.min_words) & (words["en"] >= args.min_words)].index)
    df = df[df["issue"].isin(kept)].reset_index(drop=True)

    df["boilerplate"], df["repeat_frac"] = flag_boilerplate(df, args.bp_min_issues, args.bp_frac)
    df["en_ratio"] = df["text"].map(english_ratio).round(3)
    # chunks whose content does not match their section's language (mostly price lists and tables)
    df["lang_mismatch"] = ((df["lang"] == "xh") & (df["en_ratio"] > 0.25)) |                           ((df["lang"] == "en") & (df["en_ratio"] < 0.08))
    df["usable"] = ~df["boilerplate"] & ~df["lang_mismatch"]
    splits = split_issues(kept, args.seed)
    df["split"] = df["issue"].map(splits)

    df.to_parquet(out / "chunks.parquet", index=False)
    (out / "splits.json").write_text(json.dumps({"settings": vars(args), "splits": splits}, indent=1))
    (out / "data_card.md").write_text(data_card(df, issues_all, kept, args), encoding="utf-8")
    print(f"wrote {len(df):,} chunks from {len(kept)} issues to {out}")


if __name__ == "__main__":
    main()
