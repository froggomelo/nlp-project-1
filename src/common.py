"""Shared loaders so every stage sees chunks and embeddings in the same order."""
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_ORDER = {"val": 0, "test": 1, "train": 2}


def load_usable(chunks_path="data/processed/chunks.parquet"):
    """Usable chunks in embedding order: val, then test, then train; chunk_id within each split."""
    df = pd.read_parquet(chunks_path)
    df = df[df["usable"]]
    df = df.sort_values(["split", "chunk_id"], key=lambda c: c.map(SPLIT_ORDER) if c.name == "split" else c)
    return df.reset_index(drop=True)


def run_dir(variant, model_tag="LaBSE"):
    """results/<variant> for LaBSE, results/<variant>__<tag> for any other encoder."""
    name = variant if model_tag == "LaBSE" else f"{variant}__{model_tag}"
    return Path("results") / name


def load_embeddings(df, variant="raw", model_tag="LaBSE", emb_dir="data/embeddings"):
    """Return (df, emb) restricted to the rows embedded so far (reads shards if the run is unfinished)."""
    emb_dir = Path(emb_dir)
    full = emb_dir / f"{variant}_{model_tag}.npy"
    if full.exists():
        emb = np.load(full)
        ids_file = emb_dir / f"{variant}_ids.txt"
        if ids_file.exists():
            ids = ids_file.read_text(encoding="utf-8").split("\n")
            if len(ids) != len(emb) or ids != df["chunk_id"].iloc[:len(ids)].tolist():
                raise SystemExit(f"{full} does not match the chunks on disk (different bundle or chunks "
                                 "file): rebuild the Colab bundle from these chunks and re-embed")
    else:
        shards = sorted((emb_dir / f"{variant}_{model_tag}_shards").glob("*.npy"))
        if not shards:
            raise FileNotFoundError(f"no embeddings for {variant}_{model_tag} in {emb_dir}")
        emb = np.concatenate([np.load(s) for s in shards])
    emb = emb.astype(np.float32)
    return df.iloc[:len(emb)].reset_index(drop=True), emb
