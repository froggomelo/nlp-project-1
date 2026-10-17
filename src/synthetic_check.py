"""Validate the language-removal method on synthetic data where the true content gap is known.

Each synthetic 'issue' has isiXhosa and English chunks. A chunk's embedding is
    content (a topic centroid + noise)  +  language (a direction inside a 3-d language subspace, +/-)
A fraction of English chunks are translations: they copy the content of an isiXhosa chunk.
The audience effect delta shifts the topic distribution of isiXhosa chunks away from English ones.

For each delta we compare the gap (all-chunk accuracy minus translated-pair floor) found by:
  none          no language removal                         (expected: floor ~1, gap ~0: trap 1)
  mean-centre   subtract each language's mean               (expected: erases real content gaps)
  pairs         remove the top-k pair-difference directions (ours; k chosen on validation)
against the oracle gap from a classifier that sees the content vectors only.

Usage:
    python src/synthetic_check.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from classify import fit_best_c, language_basis, project_out

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D, TOPICS, LANG_DIM = 64, 20, 3
N_ISSUES = {"train": 140, "val": 30, "test": 30}
N_XH, N_EN, TRANSLATED = 60, 30, 0.3
DELTAS = [0.0, 0.1, 0.2, 0.4]
SEEDS = [0, 1, 2, 3, 4]
K_GRID = [1, 2, 4, 8]


def make_world(delta, seed):
    rng = np.random.default_rng(seed)
    cent = rng.standard_normal((TOPICS, D))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    lang_basis = np.linalg.qr(rng.standard_normal((D, LANG_DIM)))[0].T  # orthonormal 3 x D
    p_en = np.full(TOPICS, 1 / TOPICS)
    p_xh = (1 - delta) * p_en + delta * np.r_[np.full(TOPICS // 4, 4 / TOPICS), np.zeros(TOPICS - TOPICS // 4)]
    data = {}
    for split, n in N_ISSUES.items():
        C, L, Y, P = [], [], [], []  # content, language part, label (1 = xh), pair id (-1 = none)
        pid = 0
        for _ in range(n):
            cx = cent[rng.choice(TOPICS, N_XH, p=p_xh)] + 0.6 * rng.standard_normal((N_XH, D)) / np.sqrt(D)
            ce = cent[rng.choice(TOPICS, N_EN, p=p_en)] + 0.6 * rng.standard_normal((N_EN, D)) / np.sqrt(D)
            n_tr = int(TRANSLATED * N_EN)
            src = rng.choice(N_XH, n_tr, replace=False)
            ce[:n_tr] = cx[src] + 0.05 * rng.standard_normal((n_tr, D)) / np.sqrt(D)
            px = np.full(N_XH, -1)
            pe = np.full(N_EN, -1)
            px[src] = np.arange(pid, pid + n_tr)
            pe[:n_tr] = np.arange(pid, pid + n_tr)
            pid += n_tr
            for c, sign, p in [(cx, 1, px), (ce, -1, pe)]:
                mix = np.abs(rng.normal(1, 0.3, (len(c), LANG_DIM)))
                C.append(c)
                L.append(sign * 0.8 * (mix @ lang_basis) / np.sqrt(LANG_DIM))
                Y.append(np.full(len(c), 1 if sign == 1 else 0))
                P.append(p)
        data[split] = dict(C=np.vstack(C), L=np.vstack(L), y=np.concatenate(Y), pair=np.concatenate(P))
    return data


def gap(model, X, y, pair):
    a = balanced_accuracy_score(y, model.predict(X))
    m = pair >= 0
    f = balanced_accuracy_score(y[m], model.predict(X[m]))
    return a, f, a - f


def pair_index(d):
    xh = {p: i for i, p in enumerate(d["pair"]) if p >= 0 and d["y"][i] == 1}
    en = {p: i for i, p in enumerate(d["pair"]) if p >= 0 and d["y"][i] == 0}
    keys = sorted(set(xh) & set(en))
    return np.array([xh[k] for k in keys]), np.array([en[k] for k in keys])


def run(delta, seed):
    d = make_world(delta, seed)
    X = {s: d[s]["C"] + d[s]["L"] for s in d}
    out = {}
    # oracle: content only
    _, _, m = fit_best_c(d["train"]["C"], d["train"]["y"], d["val"]["C"], d["val"]["y"], seed)
    out["oracle"] = gap(m, d["test"]["C"], d["test"]["y"], d["test"]["pair"])
    _, _, m = fit_best_c(X["train"], d["train"]["y"], X["val"], d["val"]["y"], seed)
    out["none"] = gap(m, X["test"], d["test"]["y"], d["test"]["pair"])
    mu = {lab: X["train"][d["train"]["y"] == lab].mean(0) for lab in (0, 1)}
    Xm = {s: X[s] - np.where(d[s]["y"][:, None] == 1, mu[1], mu[0]) for s in X}
    _, _, m = fit_best_c(Xm["train"], d["train"]["y"], Xm["val"], d["val"]["y"], seed)
    out["mean-centre"] = gap(m, Xm["test"], d["test"]["y"], d["test"]["pair"])
    ix, ie = pair_index(d["train"])
    basis = language_basis(X["train"], ix, ie, max(K_GRID))
    chosen = None
    for k in K_GRID:
        Xp = {s: project_out(X[s], basis[:k]) for s in X}
        _, _, m = fit_best_c(Xp["train"], d["train"]["y"], Xp["val"], d["val"]["y"], seed)
        _, floor_val, _ = gap(m, Xp["val"], d["val"]["y"], d["val"]["pair"])
        if floor_val <= 0.55 or k == K_GRID[-1]:
            chosen = (k, m, Xp)
            break
    k, m, Xp = chosen
    out["pairs"] = gap(m, Xp["test"], d["test"]["y"], d["test"]["pair"])
    out["pairs_k"] = k
    return out


def main():
    res = []
    for delta in DELTAS:
        for seed in SEEDS:
            r = run(delta, seed)
            res.append(dict(delta=delta, seed=seed, k=r.pop("pairs_k"),
                            **{f"{m}_{n}": v for m, vals in r.items() for n, v in zip(["all", "floor", "gap"], vals)}))
            print(f"delta={delta} seed={seed} " + " ".join(f"{m}: gap={r[m][2]:+.3f} floor={r[m][1]:.2f}"
                                                          for m in r), flush=True)
    out = Path("results/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic.json").write_text(json.dumps(res, indent=1, default=float))

    import pandas as pd
    t = pd.DataFrame(res)
    agg = t.groupby("delta").agg(["mean", "std"])
    methods = ["oracle", "none", "mean-centre", "pairs"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for m, style in zip(methods, ["k-", "x:", "s--", "o-"]):
        ax.errorbar(agg.index, agg[(f"{m}_gap", "mean")], yerr=agg[(f"{m}_gap", "std")], fmt=style, capsize=3,
                    label=m)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xlabel(r"planted audience effect $\delta$")
    ax.set_ylabel("test gap (all - pair floor)")
    ax.set_title("Synthetic check: which method recovers the true gap?", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_synthetic.png", dpi=200)
    summary = agg[[(f"{m}_gap", "mean") for m in methods] + [(f"{m}_floor", "mean") for m in methods]].round(3)
    summary.to_csv(out / "synthetic_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
