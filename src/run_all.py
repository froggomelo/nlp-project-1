"""Run the pipeline stages in order. Embedding (GPU-heavy) is a separate step; see README.txt.

Usage:
    python src/run_all.py prepare        # Stage 0 + Stage 2 text side: data, word list, normaliser variants
    python src/run_all.py analyse        # after embeddings exist: pairs, classifier, comparison, analyses
    python src/run_all.py analyse --model-tag multilingual-e5-base     # second encoder
Options:
    --threshold X   provisional pair-similarity threshold if labels/pairs_labelled.csv is not ready yet
"""
import argparse
import subprocess
import sys
from pathlib import Path

VARIANTS = ["raw", "modern", "modern_asp", "modern_prefix"]


def sh(*args):
    print("\n$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "analyse"])
    ap.add_argument("--model-tag", default="LaBSE")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    tag = ["--model-tag", a.model_tag]

    if a.phase == "prepare":
        sh("src/prepare_data.py")
        sh("src/build_lexicon.py")
        sh("src/normalise.py", "candidates")
        if Path("labels/normaliser_labelled.csv").exists():
            sh("src/normalise.py", "tune")
        for rules in ["all", "asp", "prefix"]:
            sh("src/normalise.py", "apply", "--rules", rules)
        sh("src/tool_metrics.py")
        sh("src/synthetic_check.py")
        return

    # only the variants embedded with this encoder (the Colab notebook embeds raw + modern for E5)
    variants = [v for v in VARIANTS if Path(f"data/embeddings/{v}_{a.model_tag}.npy").exists()]
    if "raw" not in variants:
        sys.exit(f"\nNo data/embeddings/raw_{a.model_tag}.npy: run the embedding step first (see README).")
    skipped = sorted(set(VARIANTS) - set(variants))
    if skipped:
        print(f"\n{a.model_tag}: no embeddings for {skipped}; analysing {variants}")
    for v in variants:
        sh("src/pairs.py", "candidates", "--variant", v, *tag)
    if Path("labels/pairs_labelled.csv").exists():
        sh("src/pairs.py", "tune", *tag)
        thr = []
    elif a.threshold is not None:
        thr = ["--threshold", str(a.threshold)]
        print(f"\nNo labelled pairs yet: using provisional threshold {a.threshold}")
    else:
        sh("src/pairs.py", "sheet", *tag)
        sys.exit("\nLabel labels/pairs_to_label.csv (see labels/LABELLING_GUIDE.md), save it as "
                 "labels/pairs_labelled.csv and rerun, or pass --threshold for a provisional run.")
    sh("src/classify.py", "--variant", "raw", *tag, *thr)
    for v in variants[1:]:
        sh("src/classify.py", "--variant", v, "--pairs-from", "raw", "--k-from", "raw", *tag, *thr)
    if len(variants) > 1:
        sh("src/compare.py", "--variants", *variants, *tag)
    sh("src/qualitative.py", *tag)


if __name__ == "__main__":
    main()
