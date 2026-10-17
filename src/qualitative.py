"""What do the two audiences read? Topic clusters in the language-removed embedding space.

1. Remove the language subspace estimated in Stage 1 (results/<variant>/language_basis.npy).
2. Cluster all chunks with k-means fitted on train issues; the number of clusters is chosen by
   silhouette score on a validation sample.
3. For each cluster: isiXhosa share relative to the corpus-wide share (over-representation ratio),
   share of chunks that belong to a reference translated pair, distinctive words per language
   (log-odds against the rest of the corpus), and example isiXhosa chunks nearest the centroid.
4. Per issue: Jensen-Shannon divergence between the isiXhosa and English cluster distributions,
   against a permutation baseline that shuffles the section labels within the issue.
5. Genre check for the translated pairs: which pages and clusters they come from.

Usage:
    python src/qualitative.py --variant raw
"""
import argparse
import json
import re
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from classify import project_out
from common import load_embeddings, load_usable, run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K_GRID = [8, 12, 16, 24, 32]
WORD_RE = re.compile(r"[a-z]{3,}")


def distinctive_words(texts_in, texts_out, n=8, min_count=10):
    a, b = Counter(), Counter()
    for t in texts_in:
        a.update(WORD_RE.findall(t.lower()))
    for t in texts_out:
        b.update(WORD_RE.findall(t.lower()))
    na, nb = sum(a.values()) + 1, sum(b.values()) + 1
    vocab = [w for w, c in a.items() if c >= min_count]
    score = {w: np.log((a[w] + 1) / na) - np.log((b[w] + 1) / nb) for w in vocab}
    return [w for w, _ in sorted(score.items(), key=lambda x: -x[1])[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rd = run_dir(args.variant, args.model_tag)
    chunks = "data/processed/chunks.parquet" if args.variant == "raw" else \
        f"data/processed/chunks_{args.variant}.parquet"
    df, emb = load_embeddings(load_usable(chunks), args.variant, args.model_tag)
    X = project_out(emb, np.load(rd / "language_basis.npy"))
    s1 = json.loads((rd / "stage1.json").read_text())
    pairs = pd.read_parquet(run_dir(s1["pairs_from"], args.model_tag) / "pair_candidates.parquet")
    pairs = pairs[pairs["sim"] >= s1["threshold"]]
    df["in_pair"] = df["chunk_id"].isin(set(pairs["xh_id"]) | set(pairs["en_id"]))
    rng = np.random.default_rng(args.seed)

    # choose the number of clusters on validation
    tr, va = df.index[df["split"] == "train"], df.index[df["split"] == "val"]
    va_s = rng.choice(va, min(5000, len(va)), replace=False)
    sil = {}
    for k in K_GRID:
        km = MiniBatchKMeans(k, random_state=args.seed, n_init=3, batch_size=4096).fit(X[tr])
        sil[k] = float(silhouette_score(X[va_s], km.predict(X[va_s])))
    k_best = max(sil, key=sil.get)
    km = MiniBatchKMeans(k_best, random_state=args.seed, n_init=5, batch_size=4096).fit(X[tr])
    df["cluster"] = km.predict(X)

    xh_share_all = (df["lang"] == "xh").mean()
    clusters = []
    for c, g in df.groupby("cluster"):
        rest = df[df["cluster"] != c]
        xh = g[g["lang"] == "xh"]
        d = np.linalg.norm(X[xh.index] - km.cluster_centers_[c], axis=1)
        clusters.append(dict(
            cluster=int(c), n=len(g), xh_share=float((g["lang"] == "xh").mean()),
            xh_overrep=float((g["lang"] == "xh").mean() / xh_share_all), pair_share=float(g["in_pair"].mean()),
            top_pages=g["page"].value_counts(normalize=True).round(2).head(3).to_dict(),
            words_xh=distinctive_words(xh["text"], rest[rest["lang"] == "xh"]["text"]),
            words_en=distinctive_words(g[g["lang"] == "en"]["text"], rest[rest["lang"] == "en"]["text"]),
            examples_xh=[t[:220] for t in xh.iloc[np.argsort(d)[:3]]["text"]]))
    clusters.sort(key=lambda r: -r["xh_overrep"])

    # per-issue divergence between the two sections' cluster mixes vs a within-issue permutation baseline
    obs, perm, exceed = [], [], []
    for _, g in df[df["split"] == "test"].groupby("issue"):
        lab, cl = (g["lang"] == "xh").to_numpy(), g["cluster"].to_numpy()

        def jsd(l):
            px = np.bincount(cl[l], minlength=k_best) + 0.5
            pe = np.bincount(cl[~l], minlength=k_best) + 0.5
            return jensenshannon(px / px.sum(), pe / pe.sum()) ** 2
        o = jsd(lab)
        ps = np.array([jsd(rng.permutation(lab)) for _ in range(args.n_perm)])
        obs.append(o)
        perm.append(ps.mean())
        exceed.append(o > np.percentile(ps, 95))
    obs, perm = np.array(obs), np.array(perm)
    diff = obs - perm
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(1000)]

    pages_pairs = df[df["in_pair"] & (df["lang"] == "xh")]["page"].value_counts(normalize=True).round(3)
    pages_all = df[df["lang"] == "xh"]["page"].value_counts(normalize=True).round(3)
    res = dict(variant=args.variant, k=k_best, silhouette_val=sil,
               jsd=dict(observed_mean=float(obs.mean()), permuted_mean=float(perm.mean()),
                        diff_mean=float(diff.mean()), diff_ci=[float(np.percentile(boot, 2.5)),
                                                               float(np.percentile(boot, 97.5))],
                        share_issues_above_95pct=float(np.mean(exceed)), n_issues=len(obs)),
               pair_pages=pages_pairs.to_dict(), all_xh_pages=pages_all.to_dict(), clusters=clusters)
    (rd / "qualitative.json").write_text(json.dumps(res, indent=1, default=float))

    with open(rd / "qualitative.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Topic clusters ({args.variant}, k={k_best} by validation silhouette)\n\n")
        j = res["jsd"]
        fh.write(f"Per-issue JSD (test issues, n={j['n_issues']}): observed {j['observed_mean']:.4f} vs "
                 f"permuted {j['permuted_mean']:.4f}; difference {j['diff_mean']:+.4f} "
                 f"[{j['diff_ci'][0]:+.4f}, {j['diff_ci'][1]:+.4f}]; "
                 f"{j['share_issues_above_95pct']:.0%} of issues above their 95th permutation percentile.\n\n")
        fh.write(f"Pages of isiXhosa chunks in translated pairs: {res['pair_pages']}\n"
                 f"Pages of all isiXhosa chunks: {res['all_xh_pages']}\n\n")
        for c in clusters:
            fh.write(f"## Cluster {c['cluster']}: n={c['n']}, isiXhosa share {c['xh_share']:.2f} "
                     f"(x{c['xh_overrep']:.2f}), in pairs {c['pair_share']:.2f}, pages {c['top_pages']}\n")
            fh.write(f"- isiXhosa words: {', '.join(c['words_xh'])}\n- English words: {', '.join(c['words_en'])}\n")
            for e in c["examples_xh"]:
                fh.write(f"- > {e}\n")
            fh.write("\n")

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    cs = pd.DataFrame(clusters)
    ax.bar(range(len(cs)), cs["xh_overrep"], color=["tab:red" if v > 1 else "tab:blue" for v in cs["xh_overrep"]])
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xticks(range(len(cs)), cs["cluster"], fontsize=7)
    ax.set_xlabel("topic cluster")
    ax.set_ylabel("isiXhosa over-representation")
    ax.set_title("Which topics belong to which audience?", fontsize=9)
    fig.tight_layout()
    fig.savefig(rd / "fig_clusters.png", dpi=200)
    print(open(rd / "qualitative.md", encoding="utf-8").read()[:1500])


if __name__ == "__main__":
    main()
