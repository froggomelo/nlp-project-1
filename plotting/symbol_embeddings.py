"""How a trained model embeds the symbols, read from its saved checkpoints.

A symbol is an Omniglot drawing, stored as a fixed 512-d ResNet18 feature. The
model's example_embed layer (linear, with bias) maps it into the 64-d residual
stream, where it is interleaved with the label embeddings. This module applies
that layer to the drawings of the 50 train classes and of held-out classes and
measures how the embedding groups drawings of the same character.

  python -m plotting.symbol_embeddings results/<run>   # prints the metrics
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from src import main_utils

N_TEST_SCATTER = 20     # held-out classes drawn in the scatter (all 100 enter the metrics)
TRAINING_SNAPSHOTS = [0, 100_000, 200_000, 300_000, 500_000, 1_000_000]  # single runs
N_TRAINING_METRIC_POINTS = 40


class Embedder:
  """Loads checkpoints of one run and embeds its symbols."""

  def __init__(self, folder):
    argv, sys.argv = sys.argv, sys.argv[:1]  # the option parser reads sys.argv
    try:
      self.opts = main_utils.get_opts_from_json_file(f'{folder}/config.json')
    finally:
      sys.argv = argv
    self.feats = np.asarray(main_utils.get_data_from_opts(self.opts), dtype=np.float32)  # (class, exemplar, 512)
    n_train = self.opts.class_split[0]
    self.train_classes = np.arange(n_train)
    self.test_classes = np.arange(self.feats.shape[0] - self.opts.class_split[2], self.feats.shape[0])
    model = main_utils.get_model_from_opts(self.opts, (self.feats.shape[-1],))
    opt_state = main_utils.get_optimizer_from_opts(self.opts).init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(0)
    self.ckpt_fmt = {'iter': -1, 'opt_state': opt_state, 'model': model,
                     'seeds': {'eval_model_seed': key, 'train_data_seed': key, 'train_model_seed': key}}

  def load(self, path):
    return eqx.tree_deserialise_leaves(str(path), self.ckpt_fmt)['model']

  def embed(self, model, classes):
    w = np.asarray(model.example_embed.weight)  # (64, 512)
    b = np.asarray(model.example_embed.bias)
    return self.feats[classes] @ w.T + b        # (class, exemplar, 64)


def _unit(x):
  return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def class_metrics(e):
  """e: (class, exemplar, dim). Exemplar 0 is the one seen in training.

  Similarities use embeddings centred on the set's mean: every symbol shares a large
  mean direction (ResNet features are non-negative), which would otherwise dominate.
  """
  e = e - e.reshape(-1, e.shape[-1]).mean(0)
  u = _unit(e)
  n_cls, n_ex, _ = u.shape
  flat = u.reshape(-1, u.shape[-1])
  sim = flat @ flat.T
  same = np.repeat(np.arange(n_cls), n_ex)
  same = same[:, None] == same[None, :]
  off_diag = ~np.eye(len(flat), dtype=bool)
  within = sim[same & off_diag].mean()
  between = sim[~same].mean()
  # One-shot matching: each other exemplar finds the closest exemplar 0 among all classes.
  q = u[:, 1:].reshape(-1, u.shape[-1])
  match = (q @ u[:, 0].T).argmax(-1) == np.repeat(np.arange(n_cls), n_ex - 1)
  # Effective dimension (participation ratio).
  c = e.reshape(-1, e.shape[-1])
  ev = np.linalg.eigvalsh(c.T @ c)
  pr = ev.sum() ** 2 / (ev ** 2).sum()
  return {'within': float(within), 'between': float(between), 'gap': float(within - between),
          'match': float(match.mean()), 'dim': float(pr)}


def run_metrics(emb, model):
  tr = emb.embed(model, emb.train_classes)
  te = emb.embed(model, emb.test_classes)
  out = {'train': class_metrics(tr), 'test': class_metrics(te)}
  lab = np.asarray(model.label_embed.weight).T  # (labels, 64)
  out['label_cos'] = (_unit(lab) @ _unit(lab).T).round(3).tolist()
  out['label_norm_ratio'] = float(np.linalg.norm(lab, axis=-1).mean() / np.linalg.norm(tr, axis=-1).mean())
  if getattr(model, 'symbol_unembed', None) is not None:
    # Extended model: does the next-symbol readout row of a class point along its input embedding?
    w = _unit(np.asarray(model.symbol_unembed.weight)[emb.train_classes])  # (50, 64)
    m = _unit(tr.mean(1))
    cos = w @ m.T
    out['readout_alignment'] = float(np.diag(cos).mean() - cos[~np.eye(len(cos), dtype=bool)].mean())
  return out


def scatter_points(emb, model):
  classes = np.concatenate([emb.train_classes, emb.test_classes[:N_TEST_SCATTER]])
  e = emb.embed(model, classes)
  return e.reshape(-1, e.shape[-1]), classes


def project(points, ref=None):
  """2-D PCA, rotated (orthogonal Procrustes) onto ref so snapshots stay comparable."""
  c = points - points.mean(0)
  _, _, vt = np.linalg.svd(c, full_matrices=False)
  xy = c @ vt[:2].T
  if ref is not None:
    u, _, wt = np.linalg.svd(xy.T @ ref)
    xy = xy @ (u @ wt)
  return xy / np.abs(xy).max()


def raw_baseline(emb):
  return {'train': class_metrics(emb.feats[emb.train_classes]), 'test': class_metrics(emb.feats[emb.test_classes])}


def checkpoints(folder):
  files = sorted(Path(folder, 'checkpoints').glob('*.eqx'))
  return files, np.array([int(f.stem) for f in files])


def snapshot(emb, model, ref):
  pts, classes = scatter_points(emb, model)
  xy = project(pts, ref)
  return xy, {'metrics': run_metrics(emb, model), 'xy': np.round(xy, 3).ravel().tolist()}


def collect(folder, gen_folders):
  """Embedding data for one run. gen_folders: [(gen, folder)]; one entry for a single run."""
  emb = Embedder(folder)
  _, classes = scatter_points(emb, emb.ckpt_fmt['model'])
  n_ex = emb.feats.shape[1]
  out = {'raw': raw_baseline(emb),
         'points': {'class': np.repeat(classes, n_ex).tolist(), 'exemplar': np.tile(np.arange(n_ex), len(classes)).tolist(),
                    'n_train': len(emb.train_classes) * n_ex},
         'snapshots': []}
  ref = None
  if len(gen_folders) == 1 and gen_folders[0][1] == Path(folder):
    # Single run: snapshots across training, plus metrics at evenly spaced checkpoints.
    files, iters = checkpoints(folder)
    final = emb.load(files[-1])
    ref, _ = snapshot(emb, final, None)  # align every snapshot to the trained model
    for it in TRAINING_SNAPSHOTS:
      i = int(np.abs(iters - it).argmin())
      _, s = snapshot(emb, emb.load(files[i]), ref)
      out['snapshots'].append(dict(s, label=f'{int(iters[i]):,} sequences', iter=int(iters[i])))
    idx = np.unique(np.linspace(0, len(files) - 1, N_TRAINING_METRIC_POINTS).round().astype(int))
    out['over_training'] = [{'iter': int(iters[i]), **run_metrics(emb, emb.load(files[i]))} for i in idx]
  else:
    for g, gf in gen_folders:
      files, _ = checkpoints(gf)
      if not files:
        continue
      xy, s = snapshot(emb, emb.load(files[-1]), ref)
      ref = xy if ref is None else ref
      out['snapshots'].append(dict(s, label=f'Gen {g}', gen=g))
  return out


if __name__ == '__main__':
  import json
  folder = Path(sys.argv[1])
  gens = sorted((int(d.name[4:]), d) for d in folder.glob('gen_*') if d.is_dir()) or [(0, folder)]
  d = collect(folder, gens)
  print(json.dumps({'raw': d['raw'], 'snapshots': [{'label': s['label'], **s['metrics']} for s in d['snapshots']]}, indent=1))
