"""Stage 2: modernise the spelling of 1880s isiXhosa with lexicon-gated rules.

Two rule families, each switchable for the ablation:
  asp     restore unwritten aspiration: t->th, p->ph, k->kh, r->rh and the clicks c->ch, q->qh, x->xh
          (e.g. Tixo->Thixo, apa->apha, tetayo->thethayo). Every combination of eligible positions is
          generated; the most frequent variant in the modern word list wins.
  prefix  join prefixes written as separate words: before a capitalised word always
          (e Palamente -> ePalamente, u Klass -> uKlass; English titles such as Mr. are left alone),
          before a lowercase word only if the joined form is attested in the modern word list.

Lexicon gate: a word is rewritten only if the original is (nearly) unattested in modern text
(frequency <= tau_o) and the rewrite is attested (frequency >= tau_v). tau_o and tau_v are tuned
on hand-checked validation words, favouring precision (F0.5) because over-correcting an author's
own spelling is the costly error.

Modes:
    python src/normalise.py candidates              # per-word-type rewrite candidates
    python src/normalise.py sheet --n 240           # labels/normaliser_to_label.csv for hand-checking
    python src/normalise.py tune                    # thresholds from labels/normaliser_labelled.csv
    python src/normalise.py apply --rules all       # data/processed/chunks_modern.parquet
    python src/normalise.py apply --rules asp       # ablation: data/processed/chunks_modern_asp.parquet
    python src/normalise.py apply --rules prefix    # ablation: data/processed/chunks_modern_prefix.parquet
"""
import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ASP = {"t": "th", "p": "ph", "k": "kh", "r": "rh", "c": "ch", "q": "qh", "x": "xh"}
MAX_POSITIONS = 8  # 2^8 = 256 variants per word at most
# prefixes seen written apart before capitalised words in the archive (counted on the corpus)
PREFIX_PROPER = ["e", "u", "i", "o", "no", "ka", "ngu", "wase", "ku", "yase", "ngo", "kwe", "ye", "ne", "kwi",
                 "we", "lase", "kwa", "yi", "le", "nge", "ama", "ase", "base", "nase", "ko", "lika", "sase",
                 "yama", "se", "ze", "lo", "yo", "zase", "nama", "kuka", "be"]
PREFIX_COMMON = ["e", "u", "i", "ku", "kwi", "nge", "ngo", "nga", "ne", "no", "na", "se", "kwa", "ka"]
TITLES = r"(?:Mr|Mrs|Messrs|Rev|Dr|Sir|Miss|Capt|Col|Hon)\b"
WORD_RE = re.compile(r"[A-Za-z]+")
_PROPER_ALT = sorted(set(PREFIX_PROPER) | {p.capitalize() for p in PREFIX_PROPER}, key=len, reverse=True)
PROPER_RE = re.compile(r"\b(" + "|".join(_PROPER_ALT) + r")\s+(?!" + TITLES + r")(?=[A-Z][a-z])")
COMMON_RE = re.compile(r"\b(" + "|".join(sorted(PREFIX_COMMON, key=len, reverse=True)) + r")\s+([a-z]{3,})\b")
CAND_PATH = Path("data/processed/normaliser_candidates.parquet")
THR_PATH = Path("results/normaliser/threshold.json")
PROVISIONAL = {"tau_o": 2, "tau_v": 5}


def load_lexicon(path="data/modern/lexicon.parquet"):
    lex = pd.read_parquet(path)
    return dict(zip(lex["word"], lex["total"]))


def xh_texts(chunks="data/processed/chunks.parquet"):
    df = pd.read_parquet(chunks)
    return df, df[df["lang"] == "xh"]["text"]


def aspiration_variants(word):
    pos = [i for i, ch in enumerate(word) if ch in ASP and not (i + 1 < len(word) and word[i + 1] == "h")]
    if not pos or len(pos) > MAX_POSITIONS:
        return []
    out = []
    for r in range(1, len(pos) + 1):
        for subset in itertools.combinations(pos, r):
            s = set(subset)
            out.append("".join(ASP[ch] if i in s else ch for i, ch in enumerate(word)))
    return out


def build_candidates(texts, lex):
    hist = Counter()
    proper, common = Counter(), Counter()
    for t in texts:
        hist.update(w.lower() for w in WORD_RE.findall(t))
        proper.update((p.lower(), "") for p in PROPER_RE.findall(t))
        common.update((p, w) for p, w in COMMON_RE.findall(t))
    rows = []
    for w, n in hist.items():
        of = lex.get(w, 0)
        if of > 5:  # clearly modern already; never rewritten at any threshold in the grid
            continue
        vs = aspiration_variants(w)
        best = max(vs, key=lambda v: lex.get(v, 0)) if vs else ""
        rows.append(dict(rule="asp", word=w, proposed=best, hist_count=n, orig_freq=of,
                         prop_freq=lex.get(best, 0) if best else 0))
    for (p, w), n in common.items():
        joined = p + w
        rows.append(dict(rule="prefix_common", word=f"{p} {w}", proposed=joined, hist_count=n,
                         orig_freq=lex.get(w, 0), prop_freq=lex.get(joined, 0)))
    for (p, _), n in proper.items():
        rows.append(dict(rule="prefix_proper", word=f"{p} <Name>", proposed=f"{p}<Name>", hist_count=n,
                         orig_freq=0, prop_freq=0))
    return pd.DataFrame(rows)


def accepted(cand, tau_o, tau_v):
    asp = (cand["rule"] == "asp") & (cand["orig_freq"] <= tau_o) & (cand["prop_freq"] >= tau_v) & \
          (cand["proposed"] != "")
    com = (cand["rule"] == "prefix_common") & (cand["prop_freq"] >= tau_v) & \
          (cand["prop_freq"] > 5 * cand["orig_freq"])
    return asp | com | (cand["rule"] == "prefix_proper")


def make_maps(cand, tau_o, tau_v):
    acc = cand[accepted(cand, tau_o, tau_v)]
    asp_map = dict(zip(acc.loc[acc["rule"] == "asp", "word"], acc.loc[acc["rule"] == "asp", "proposed"]))
    common_set = {tuple(w.split(" ")) for w in acc.loc[acc["rule"] == "prefix_common", "word"]}
    return asp_map, common_set


def restore_case(orig, new):
    if len(orig) > 1 and orig.isupper():
        return new.upper()
    if orig[0].isupper():
        return new[0].upper() + new[1:]
    return new


def normalise_text(text, rules, asp_map, common_set, log):
    """Aspiration first (on separate words), then prefix joining. `log` collects (rule, old, new)."""
    if "asp" in rules:
        def asp_sub(m):
            w = m.group(0)
            new = asp_map.get(w.lower())
            if new is None:
                return w
            new = restore_case(w, new)
            log[("asp", w, new)] += 1
            return new
        text = WORD_RE.sub(asp_sub, text)
    if "prefix" in rules:
        def proper_sub(m):
            log[("prefix_proper", m.group(1), "")] += 1
            return m.group(1)
        text = PROPER_RE.sub(proper_sub, text)

        def common_sub(m):
            p, w = m.group(1), m.group(2)
            if (p, w) in common_set:
                log[("prefix_common", f"{p} {w}", p + w)] += 1
                return p + w
            return m.group(0)
        text = COMMON_RE.sub(common_sub, text)
    return text


def capitalised_share(texts):
    """For each lowercase word type, the share of its occurrences that start with a capital letter."""
    tot, cap = Counter(), Counter()
    for t in texts:
        for w in WORD_RE.findall(t):
            lw = w.lower()
            tot[lw] += 1
            if w[0].isupper() and not w.isupper():
                cap[lw] += 1
    return {w: cap[w] / n for w, n in tot.items()}


def context(texts, pattern, width=60):
    rx = re.compile(pattern, re.IGNORECASE)
    for t in texts:
        m = rx.search(t)
        if m:
            return "..." + t[max(0, m.start() - width):m.end() + width] + "..."
    return ""


def make_sheet(cand, texts, n, seed):
    """Stratified sample of word types for hand-checking, split into val/test halves."""
    rng = random.Random(seed)
    texts = list(texts.sample(frac=1, random_state=seed))[:20000]
    asp = cand[(cand["rule"] == "asp") & (cand["prop_freq"] >= 1)].copy()
    asp["bin"] = pd.cut(asp["prop_freq"], [0, 1, 4, 19, 99, np.inf], labels=False)
    parts = [g.sample(min(len(g), int(n * 0.65) // asp["bin"].nunique()), random_state=seed)
             for _, g in asp.groupby("bin")]
    nonprop = cand[(cand["rule"] == "asp") & (cand["prop_freq"] == 0) & (cand["orig_freq"] == 0) &
                   (cand["hist_count"] >= 3)]
    parts.append(nonprop.sample(min(len(nonprop), int(n * 0.2)), random_state=seed).assign(proposed=""))
    com = cand[(cand["rule"] == "prefix_common") & (cand["prop_freq"] >= 1)]
    parts.append(com.sample(min(len(com), int(n * 0.15)), random_state=seed))
    sheet = pd.concat(parts).drop(columns=["bin"], errors="ignore").reset_index(drop=True)
    sheet["context"] = [context(texts, r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b") for w in sheet["word"]]
    idx = list(range(len(sheet)))
    rng.shuffle(idx)
    sheet = sheet.iloc[idx].reset_index(drop=True)
    sheet.insert(0, "split", ["val" if i % 2 == 0 else "test" for i in range(len(sheet))])
    for col in ["correct", "gold", "note"]:
        sheet[col] = ""
    return sheet[["split", "rule", "word", "proposed", "context", "correct", "gold", "note",
                  "hist_count", "orig_freq", "prop_freq"]]


def evaluate(lab, tau_o, tau_v):
    """Type-level precision/recall of the rewrites accepted at (tau_o, tau_v) on labelled words.

    A word 'needs change' if its proposal was judged correct, or the labeller gave a gold form different
    from the original. A rewrite counts as correct only if the proposal was judged correct.
    """
    acc = accepted(lab, tau_o, tau_v) & (lab["proposed"] != "")
    correct = lab["correct"] == 1
    needs = correct | ((lab["gold"].fillna("") != "") & (lab["gold"].fillna("") != lab["word"]))
    tp = (acc & correct).sum()
    p = tp / acc.sum() if acc.sum() else float("nan")
    r = tp / needs.sum() if needs.sum() else float("nan")
    f05 = (1.25 * p * r / (0.25 * p + r)) if p and r and not np.isnan(p) and not np.isnan(r) else 0.0
    return dict(precision=float(p), recall=float(r), f05=float(f05), n_accepted=int(acc.sum()),
                n_needs_change=int(needs.sum()), n=int(len(lab)))


def tune(labels_path):
    lab = pd.read_csv(labels_path)
    lab = lab[lab["correct"].astype(str).str.strip().isin(["0", "1"]) | lab["gold"].notna()].copy()
    lab["correct"] = pd.to_numeric(lab["correct"], errors="coerce").fillna(0).astype(int)
    lab = lab[lab["rule"] != "prefix_proper"]
    val, test = lab[lab["split"] == "val"], lab[lab["split"] == "test"]
    grid = [(o, v) for o in [0, 1, 2, 5] for v in [1, 2, 5, 10, 20, 50]]
    scores = {f"{o},{v}": evaluate(val, o, v) for o, v in grid}
    best = max(grid, key=lambda g: (scores[f"{g[0]},{g[1]}"]["f05"], -g[1]))
    return {"tau_o": best[0], "tau_v": best[1], "criterion": "max F0.5 on validation words",
            "val": scores[f"{best[0]},{best[1]}"], "test": evaluate(test, *best), "grid_val": scores}


def thresholds():
    if THR_PATH.exists():
        t = json.loads(THR_PATH.read_text())
        return t["tau_o"], t["tau_v"], str(THR_PATH)
    return PROVISIONAL["tau_o"], PROVISIONAL["tau_v"], "provisional (no labelled words yet)"


def apply(rules, variant):
    tau_o, tau_v, src = thresholds()
    cand = pd.read_parquet(CAND_PATH)
    asp_map, common_set = make_maps(cand, tau_o, tau_v)
    df = pd.read_parquet("data/processed/chunks.parquet")
    log = Counter()
    xh = df["lang"] == "xh"
    new = [normalise_text(t, rules, asp_map, common_set, log) for t in df.loc[xh, "text"]]
    df["text_raw"] = df["text"]
    df.loc[xh, "text"] = new
    df["changed"] = df["text"] != df["text_raw"]
    out = Path(f"data/processed/chunks_{variant}.parquet")
    df.drop(columns=["text_raw"]).to_parquet(out, index=False)

    changes = pd.DataFrame([dict(rule=r, old=o, new=n, count=c) for (r, o, n), c in log.items()])
    cap_share = capitalised_share(df.loc[xh, "text_raw"])
    res = Path("results/normaliser")
    res.mkdir(parents=True, exist_ok=True)
    changes.sort_values("count", ascending=False).to_parquet(res / f"{variant}_changes.parquet", index=False)
    n_tok = sum(len(WORD_RE.findall(t)) for t in df.loc[xh, "text_raw"])
    by_rule = changes.groupby("rule")["count"].sum().to_dict() if len(changes) else {}
    asp = changes[changes["rule"] == "asp"] if len(changes) else changes
    # words capitalised in >= 90% of their occurrences are likely proper names (God, peoples, places, people):
    # rewriting them is the identity-laden over-correction risk discussed in the ethics section
    if len(asp):
        asp = asp.assign(old_l=asp["old"].str.lower())
        asp = asp.assign(cap_share=asp["old_l"].map(cap_share).fillna(0))
        cap = asp[asp["cap_share"] >= 0.9].groupby("old_l", as_index=False).agg(
            old=("old", "first"), new=("new", "first"), count=("count", "sum"))
    else:
        cap = asp
    summary = dict(variant=variant, rules=sorted(rules), tau_o=tau_o, tau_v=tau_v, threshold_source=src,
                   xh_tokens=n_tok, tokens_changed_by_rule=by_rule,
                   share_tokens_changed=float(changes["count"].sum() / n_tok) if len(changes) else 0.0,
                   share_chunks_changed=float(df.loc[xh, "changed"].mean()),
                   likely_name_rewrites=int(cap["count"].sum()) if len(cap) else 0,
                   likely_name_types=int(len(cap)),
                   top_asp=asp.nlargest(25, "count")[["old", "new", "count"]].values.tolist() if len(asp) else [],
                   top_likely_names=cap.nlargest(25, "count")[["old", "new", "count"]].values.tolist() if len(cap) else [])
    (res / f"{variant}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{variant}: {summary['share_tokens_changed']:.1%} of isiXhosa tokens changed "
          f"({by_rule}); {summary['share_chunks_changed']:.1%} of chunks; thresholds {tau_o},{tau_v} [{src}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["candidates", "sheet", "tune", "apply"])
    ap.add_argument("--rules", choices=["all", "asp", "prefix"], default="all")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--labels", default="labels/normaliser_labelled.csv")
    args = ap.parse_args()

    if args.mode == "candidates":
        _, texts = xh_texts()
        cand = build_candidates(texts, load_lexicon())
        cand.to_parquet(CAND_PATH, index=False)
        print(cand.groupby("rule").size().to_dict(), "candidate rows ->", CAND_PATH)
    elif args.mode == "sheet":
        _, texts = xh_texts()
        sheet = make_sheet(pd.read_parquet(CAND_PATH), texts, args.n, args.seed)
        Path("labels").mkdir(exist_ok=True)
        sheet.to_csv("labels/normaliser_to_label.csv", index=False, encoding="utf-8-sig")
        print(f"wrote {len(sheet)} words to labels/normaliser_to_label.csv", sheet["rule"].value_counts().to_dict())
    elif args.mode == "tune":
        res = tune(args.labels)
        THR_PATH.parent.mkdir(parents=True, exist_ok=True)
        THR_PATH.write_text(json.dumps(res, indent=1))
        print({k: v for k, v in res.items() if k != "grid_val"})
    else:
        rules = {"all": {"asp", "prefix"}, "asp": {"asp"}, "prefix": {"prefix"}}[args.rules]
        variant = "modern" if args.rules == "all" else f"modern_{args.rules}"
        apply(rules, variant)


if __name__ == "__main__":
    main()
