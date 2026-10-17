"""Embed usable chunks with a sentence encoder (default LaBSE), resumably.

Writes <out>/<variant>_<model-tag>.npy (float16, L2-normalised, row order = chunk order in
<out>/<variant>_ids.txt). Work is saved in shards, so an interrupted run picks up where it stopped.

With --reuse raw, only chunks whose text differs from the raw run are embedded; all other rows
(every English chunk and any isiXhosa chunk the normaliser left unchanged) are copied from the raw
embeddings. This keeps the comparison exact for unchanged text and cuts GPU time.

Usage:
    python src/embed.py --variant raw
    python src/embed.py --variant modern --chunks data/processed/chunks_modern.parquet --reuse raw
    python src/embed.py --variant raw --model intfloat/multilingual-e5-base --prefix "query: "   (second encoder)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import load_usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.parquet")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--prefix", default="", help='text prepended to every chunk (e5 models need "query: ")')
    ap.add_argument("--reuse", default="", help="variant whose embeddings are copied for unchanged chunks")
    ap.add_argument("--reuse-chunks", default="data/processed/chunks.parquet")
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--shard", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (default: all cores)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N chunks (speed test)")
    args = ap.parse_args()

    out = Path(args.out)
    tag = args.model.split("/")[-1]
    shard_dir = out / f"{args.variant}_{tag}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # val/test first so pair labelling can start before the (larger) train split is done
    df = load_usable(args.chunks)
    if args.limit:
        df = df.head(args.limit)
    texts = df["text"].tolist()

    emb_full = None
    todo = np.arange(len(df))
    if args.reuse:
        ref = load_usable(args.reuse_chunks)
        ref_emb = np.load(out / f"{args.reuse}_{tag}.npy")
        ref_ids = (out / f"{args.reuse}_ids.txt").read_text(encoding="utf-8").split("\n")
        assert ref_ids == ref["chunk_id"].tolist(), "reuse embeddings do not match the reference chunks"
        ref_row = dict(zip(ref["chunk_id"], range(len(ref))))
        ref_text = dict(zip(ref["chunk_id"], ref["text"]))
        emb_full = np.zeros((len(df), ref_emb.shape[1]), dtype=np.float16)
        same = np.array([ref_text.get(c) == t for c, t in zip(df["chunk_id"], texts)])
        emb_full[same] = ref_emb[[ref_row[c] for c in df["chunk_id"][same]]]
        todo = np.flatnonzero(~same)
        print(f"reusing {same.sum():,} unchanged chunks from '{args.reuse}'; embedding {len(todo):,} changed chunks")

    torch.set_num_threads(args.threads or os.cpu_count())
    model = SentenceTransformer(args.model)
    n_shards = (len(todo) + args.shard - 1) // args.shard
    t0 = time.time()
    for s in range(n_shards):
        path = shard_dir / f"{s:04d}.npy"
        if path.exists():
            continue
        batch = [args.prefix + texts[i] for i in todo[s * args.shard:(s + 1) * args.shard]]
        emb = model.encode(batch, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True)
        np.save(path, emb.astype(np.float16))
        done = min((s + 1) * args.shard, len(todo))
        print(f"shard {s + 1}/{n_shards}: {done:,}/{len(todo):,} chunks, {time.time() - t0:,.0f}s", flush=True)

    if args.limit:
        print(f"speed test: {len(todo)} chunks in {time.time() - t0:.1f}s")
        return
    new = np.concatenate([np.load(shard_dir / f"{s:04d}.npy") for s in range(n_shards)]) if n_shards else None
    if emb_full is None:
        emb_full = new
    elif new is not None:
        emb_full[todo] = new
    np.save(out / f"{args.variant}_{tag}.npy", emb_full)
    (out / f"{args.variant}_ids.txt").write_text("\n".join(df["chunk_id"]), encoding="utf-8")
    print(f"wrote {emb_full.shape} to {out / f'{args.variant}_{tag}.npy'}")


if __name__ == "__main__":
    main()
