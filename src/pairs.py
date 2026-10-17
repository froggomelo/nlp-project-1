"""Stage 1a: find candidate translated pairs between the isiXhosa and English sections of each issue.

A candidate pair is a mutual nearest neighbour within one issue: the English chunk is the
isiXhosa chunk's most similar English chunk, and vice versa. Each candidate gets its LaBSE
cosine similarity and a model-free signal (shared capitalised names and numbers) that helps
human labellers and lets us check the encoder independently.

Modes:
    python src/pairs.py candidates            # all mutual-NN candidates -> results/<variant>/pair_candidates.parquet
    python src/pairs.py sheet --n 300         # labelling sheet from val/test issues -> labels/pairs_to_label.csv

The labelling sheet mixes two sources so the labels can serve two purposes:
  raw     mutual-NN candidates stratified by similarity -> tunes the similarity threshold
  anchor  pairs found WITHOUT embeddings (mutual best overlap of names/numbers within an issue)
          -> an encoder-independent set of true pairs, so alignment can be compared across spelling
             variants without favouring the variant the candidates were selected in
    python src/pairs.py tune                  # pick threshold from labelled val pairs, report test P/R
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_embeddings, load_usable, run_dir

NAME_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]{3,}|\d{2,})")


def anchors(text):
    """Capitalised names and multi-digit numbers: shared across languages without translation."""
    return {m.lower() for m in NAME_RE.findall(text)}


def candidates(df, emb):
    rows = []
    for issue, g in df.groupby("issue"):
        xi = g.index[g["lang"] == "xh"].to_numpy()
        ei = g.index[g["lang"] == "en"].to_numpy()
        if len(xi) == 0 or len(ei) == 0:
            continue
        S = emb[xi] @ emb[ei].T
        best_e = S.argmax(1)  # best English for each isiXhosa chunk
        best_x = S.argmax(0)  # best isiXhosa for each English chunk
        for a, b in enumerate(best_e):
            if best_x[b] != a:
                continue
            s_sorted = np.sort(S[a])[::-1]
            xa, eb = df.loc[xi[a]], df.loc[ei[b]]
            shared = anchors(xa["text"]) & anchors(eb["text"])
            rows.append(dict(issue=issue, split=xa["split"], year=xa["year"], xh_id=xa["chunk_id"],
                             en_id=eb["chunk_id"], sim=float(S[a, b]),
                             margin=float(s_sorted[0] - s_sorted[1]) if len(s_sorted) > 1 else np.nan,
                             n_shared=len(shared), shared=" ".join(sorted(shared)[:8]),
                             xh_text=xa["text"], en_text=eb["text"]))
    return pd.DataFrame(rows)


def anchor_pairs(df, min_shared=3, max_issue_share=0.2):
    """Embedding-free pairs: mutual best overlap of names/numbers within an issue.

    Anchors that occur in more than `max_issue_share` of an issue's chunks (place of publication,
    the year, the paper's name) are ignored, so a pair needs specific shared names or numbers.
    """
    rows = []
    for issue, g in df.groupby("issue"):
        anc = {i: anchors(t) for i, t in zip(g.index, g["text"])}
        freq = pd.Series([a for s in anc.values() for a in s]).value_counts()
        common = set(freq[freq > max_issue_share * len(g)].index)
        anc = {i: s - common for i, s in anc.items()}
        xi = g.index[g["lang"] == "xh"].to_numpy()
        ei = g.index[g["lang"] == "en"].to_numpy()
        if len(xi) == 0 or len(ei) == 0:
            continue
        M = np.array([[len(anc[x] & anc[e]) for e in ei] for x in xi])
        best_e, best_x = M.argmax(1), M.argmax(0)
        for a, b in enumerate(best_e):
            if best_x[b] == a and M[a, b] >= min_shared:
                xa, eb = df.loc[xi[a]], df.loc[ei[b]]
                rows.append(dict(issue=issue, split=xa["split"], xh_id=xa["chunk_id"], en_id=eb["chunk_id"],
                                 n_shared=int(M[a, b]), shared=" ".join(sorted(anc[xi[a]] & anc[ei[b]])[:8]),
                                 xh_text=xa["text"], en_text=eb["text"]))
    return pd.DataFrame(rows)


def make_sheet(cand, anch, df, emb, n, seed):
    """Half raw candidates stratified by similarity decile, half embedding-free anchor pairs."""
    c = cand[cand["split"].isin(["val", "test"])].copy()
    c["bin"] = pd.qcut(c["sim"], 10, labels=False, duplicates="drop")
    per_bin = n // (4 * c["bin"].nunique())
    parts = []
    for split in ["val", "test"]:
        for _, g in c[c["split"] == split].groupby("bin"):
            parts.append(g.sample(min(per_bin, len(g)), random_state=seed).assign(source="raw"))
        a = anch[anch["split"] == split]
        parts.append(a.sample(min(n // 4, len(a)), random_state=seed).assign(source="anchor"))
    sheet = pd.concat(parts).drop_duplicates(["xh_id", "en_id"])
    pos = pd.Series(df.index, index=df["chunk_id"])  # similarity in the raw space, for threshold tuning
    sheet["sim"] = np.einsum("ij,ij->i", emb[pos[sheet["xh_id"]].to_numpy()], emb[pos[sheet["en_id"]].to_numpy()])
    sheet = sheet.sample(frac=1, random_state=seed)  # shuffle so labellers can't see the source or bin
    sheet = sheet[["split", "source", "issue", "xh_id", "en_id", "xh_text", "en_text", "shared", "sim"]]
    sheet.insert(0, "label", "")
    return sheet


def agreement(labels_path):
    """Cohen's kappa if a second labeller filled a `label_2` column on some rows (binary: >=1 vs 0)."""
    lab = pd.read_csv(labels_path)
    if "label_2" not in lab:
        return None
    both = lab[lab["label"].astype(str).str.strip().isin(["0", "1", "2"]) &
               lab["label_2"].astype(str).str.strip().isin(["0", "1", "2"])]
    if len(both) < 10:
        return None
    from sklearn.metrics import cohen_kappa_score
    a, b = both["label"].astype(int) >= 1, both["label_2"].astype(int) >= 1
    return {"n": int(len(both)), "kappa": float(cohen_kappa_score(a, b))}


def tune(cand_path, labels_path, min_precision, df, emb, model_tag):
    lab = pd.read_csv(labels_path)
    lab = lab[lab["label"].astype(str).str.strip().isin(["0", "1", "2"])].copy()
    if "source" in lab and model_tag == "LaBSE":  # the threshold is a property of the embedding candidates
        lab = lab[lab["source"] == "raw"]
    # similarity under THIS encoder (the sheet's `sim` column is LaBSE's)
    pos = pd.Series(df.index, index=df["chunk_id"])
    lab["sim"] = np.einsum("ij,ij->i", emb[pos[lab["xh_id"]].to_numpy()], emb[pos[lab["en_id"]].to_numpy()])
    lab["y"] = lab["label"].astype(int) >= 1  # 2 = full translation, 1 = partial overlap, 0 = unrelated
    val, test = lab[lab["split"] == "val"], lab[lab["split"] == "test"]

    def pr(d, t):
        pred = d["sim"] >= t
        tp = (pred & d["y"]).sum()
        p = tp / pred.sum() if pred.sum() else float("nan")
        r = tp / d["y"].sum() if d["y"].sum() else float("nan")
        return float(p), float(r)

    grid = np.round(np.arange(0.30, 0.86, 0.01), 2)
    ok = [t for t in grid if pr(val, t)[0] >= min_precision]
    if ok:
        thr, rule = float(min(ok)), f"lowest threshold with val precision >= {min_precision}"
    else:  # target unreachable: fall back to the best F1 on validation and say so
        def f1(t):
            p, r = pr(val, t)
            return 0.0 if not p or not r or np.isnan(p) or np.isnan(r) else 2 * p * r / (p + r)
        thr = float(max(grid, key=f1))
        rule = f"precision target {min_precision} NOT reached on val; fell back to max-F1 threshold"
    res = {"threshold": thr, "kappa": agreement(labels_path), "rule": rule,
           "n_val": len(val), "n_test": len(test),
           "val_precision_recall": pr(val, thr), "test_precision_recall": pr(test, thr),
           "curve_val": {str(t): pr(val, t) for t in grid}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["candidates", "sheet", "tune"])
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--labels", default="labels/pairs_labelled.csv")
    ap.add_argument("--min-precision", type=float, default=0.9)
    args = ap.parse_args()
    res_dir = run_dir(args.variant, args.model_tag)
    res_dir.mkdir(parents=True, exist_ok=True)
    cand_path = res_dir / "pair_candidates.parquet"

    if args.mode == "candidates":
        chunks = "data/processed/chunks.parquet" if args.variant == "raw" else             f"data/processed/chunks_{args.variant}.parquet"
        df, emb = load_embeddings(load_usable(chunks), args.variant, args.model_tag)
        cand = candidates(df, emb)
        cand.to_parquet(cand_path, index=False)
        print(f"{len(cand):,} mutual-NN candidates from {df['issue'].nunique()} issues "
              f"(splits embedded: {sorted(df['split'].unique())})")
        print(cand.groupby("split")["sim"].describe().round(3))
    elif args.mode == "sheet":
        Path("labels").mkdir(exist_ok=True)
        df, emb = load_embeddings(load_usable(), "raw", args.model_tag)
        anch = anchor_pairs(df[df["split"].isin(["val", "test"])])
        sheet = make_sheet(pd.read_parquet(cand_path), anch, df, emb, args.n, args.seed)
        print(f"{len(anch)} embedding-free anchor pairs available in val/test issues")
        out = Path("labels/pairs_to_label.csv")
        sheet.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"wrote {len(sheet)} pairs to {out} ({sheet['source'].value_counts().to_dict()})")
    else:
        df, emb = load_embeddings(load_usable(), "raw", args.model_tag)
        res = tune(cand_path, args.labels, args.min_precision, df, emb, args.model_tag)
        (res_dir / "pair_threshold.json").write_text(json.dumps(res, indent=1))
        print({k: v for k, v in res.items() if k != "curve_val"})


if __name__ == "__main__":
    main()
