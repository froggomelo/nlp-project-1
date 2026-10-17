"""End-to-end smoke test of Stages 1-3 without LaBSE.

Builds stand-in embeddings (hashed bag of words -> random projection, plus a language offset) under
the model tag FAKE, fakes pair labels from shared names/numbers, then runs every step of the pipeline.
It checks that the code runs and the files line up; the numbers mean nothing.
Outputs go to results/*__FAKE and data/embeddings/*_FAKE.npy; remove them with --clean.

Usage:
    python src/smoke_test.py            # run
    python src/smoke_test.py --clean    # delete FAKE outputs
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from common import load_usable, run_dir

VARIANTS = ["raw", "modern", "modern_asp", "modern_prefix"]
TAG = "FAKE"


def fake_embed(df, dim=64, seed=0):
    hv = HashingVectorizer(n_features=2 ** 16, alternate_sign=False, norm=None, token_pattern=r"[A-Za-z0-9]{2,}")
    R = np.random.default_rng(seed).standard_normal((2 ** 16, dim)).astype(np.float32)
    E = hv.transform(df["text"]) @ R
    lang = np.random.default_rng(seed + 1).standard_normal(dim).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E += np.where(df["lang"].to_numpy()[:, None] == "xh", 0.3, -0.3) * lang / np.linalg.norm(lang)
    return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float16)


def sh(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        for v in VARIANTS + ["compare"]:
            shutil.rmtree(run_dir(v, TAG), ignore_errors=True)
            Path(f"data/embeddings/{v}_{TAG}.npy").unlink(missing_ok=True)
        print("removed FAKE outputs")
        return

    for v in VARIANTS:
        chunks = "data/processed/chunks.parquet" if v == "raw" else f"data/processed/chunks_{v}.parquet"
        df = load_usable(chunks)
        np.save(f"data/embeddings/{v}_{TAG}.npy", fake_embed(df))
        Path(f"data/embeddings/{v}_ids.txt").write_text("\n".join(df["chunk_id"]), encoding="utf-8")
        sh("src/pairs.py", "candidates", "--variant", v, "--model-tag", TAG)

    # fake labels: 2 if the pair shares >= 3 names/numbers, else 0
    rd = run_dir("raw", TAG)
    sh("src/pairs.py", "sheet", "--model-tag", TAG, "--n", "300")
    sheet = pd.read_csv("labels/pairs_to_label.csv")
    sheet["label"] = np.where(sheet["shared"].fillna("").str.split().str.len() >= 3, 2, 0)
    fake_labels = rd / "fake_pairs_labelled.csv"
    sheet.to_csv(fake_labels, index=False)
    Path("labels/pairs_to_label.csv").unlink()  # don't leave a FAKE-based sheet where labellers would find it
    sh("src/pairs.py", "tune", "--model-tag", TAG, "--labels", str(fake_labels), "--min-precision", "0.6")

    sh("src/classify.py", "--variant", "raw", "--model-tag", TAG, "--n-boot", "100")
    for v in VARIANTS[1:]:
        sh("src/classify.py", "--variant", v, "--model-tag", TAG, "--pairs-from", "raw", "--k-from", "raw",
           "--n-boot", "100")
    sh("src/compare.py", "--model-tag", TAG, "--n-boot", "100", "--labels", str(fake_labels))
    sh("src/qualitative.py", "--model-tag", TAG, "--n-perm", "10")
    print("\nSMOKE TEST PASSED: every stage ran end to end. Remove outputs with: python src/smoke_test.py --clean")


if __name__ == "__main__":
    main()
