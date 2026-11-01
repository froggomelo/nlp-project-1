'''
This file contains the model collapse pipeline pieces: a fixed-size training
pool, partial replacement of that pool with model-generated sequences, and
metrics describing how the pool drifts from the real data over generations.

The training loop itself stays in main.py; main.run_with_opts calls these
between generations.
'''

from functools import partial

import numpy as np
import h5py as h5

import jax
import jax.numpy as jnp
import equinox as eqx


POOL_KEYS = ['examples', 'labels', 'symbols']


def build_pool(data_sampler, size, key):
  '''Draws `size` real sequences once from `data_sampler` (a key -> batch fn).

  Returns a dict with 'examples', 'labels', 'symbols', 'n_in_ctx' (if the
  sampler provides it) and 'origin_gen' (0 for all items: real data).
  '''
  batches = []
  total = 0
  while total < size:
    key, use = jax.random.split(key)
    batches.append(data_sampler(use))
    total += batches[-1]['labels'].shape[0]
  pool = {k: jnp.concatenate([b[k] for b in batches])[:size] for k in batches[0]}
  pool['origin_gen'] = jnp.zeros(size, dtype=int)
  return pool


@partial(jax.jit, static_argnums=2)
def _index_pool(key, arrays, batch_size):
  idx = jax.random.randint(key, (batch_size,), 0, arrays['labels'].shape[0])
  return {k: v[idx] for k, v in arrays.items()}


def make_pool_sampler(pool, batch_size):
  '''Returns a key -> batch fn drawing sequences uniformly (with replacement) from the pool.'''
  arrays = {k: pool[k] for k in POOL_KEYS}
  return lambda key: _index_pool(key, arrays, batch_size)


@eqx.filter_jit
def _logits(model, fwd_fn, x, y, key):
  keys = jax.random.split(key, x.shape[0])
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)
  return out['out'], out.get('symbol_out')


def _choose(key, logits, sampling, temperature):
  '''Picks one token per row: a sample from softmax(logits / T), or the argmax.'''
  if sampling == 'argmax':
    return jnp.argmax(logits, axis=-1)
  return jax.random.categorical(key, logits / temperature, axis=-1)


def generate_sequences(model, fwd_fn, prompts, key, opts, data, exemplar_inds, num_train_classes):
  '''Regenerates the targets of `prompts` with `model`, one token at a time.

  The context and query (the prompt) are kept. Base: sample the query's value.
  Extended: sample the query's value, feed it in, sample the next symbol's
  class (restricted to train classes), give it a random train exemplar, feed
  it in, then sample that symbol's value. The model is causal, so later
  placeholder tokens do not affect earlier logits.
  '''
  extended = opts.model_variant == 'extended'
  n = prompts['labels'].shape[0]
  out = {k: [] for k in POOL_KEYS}
  for start in range(0, n, opts.eval_bs):
    x, y, s = (prompts[k][start:start+opts.eval_bs] for k in POOL_KEYS)
    key, *ks = jax.random.split(key, 8)
    choose = partial(_choose, sampling=opts.gen_sampling, temperature=opts.gen_temperature)
    if extended:
      label_logits, _ = _logits(model, fwd_fn, x, y, ks[0])
      y = y.at[:, -2].set(choose(ks[1], label_logits[:, -2]))
      _, symbol_logits = _logits(model, fwd_fn, x, y, ks[2])
      # Only train classes may be generated, so val/test classes never leak into training
      symbol_logits = symbol_logits[:, -1].at[:, num_train_classes:].set(-jnp.inf)
      next_symbol = choose(ks[3], symbol_logits)
      exemplar = exemplar_inds[jax.random.randint(ks[4], (x.shape[0],), 0, len(exemplar_inds))]
      x = x.at[:, -1].set(data[next_symbol, exemplar])
      s = s.at[:, -1].set(next_symbol)
      label_logits, _ = _logits(model, fwd_fn, x, y, ks[5])
      y = y.at[:, -1].set(choose(ks[6], label_logits[:, -1]))
    else:
      label_logits, _ = _logits(model, fwd_fn, x, y, ks[0])
      y = y.at[:, -1].set(choose(ks[1], label_logits[:, -1]))
    for k, v in zip(POOL_KEYS, (x, y, s)):
      out[k].append(v)
  return {k: jnp.concatenate(v) for k, v in out.items()}


def _n_in_ctx(symbols):
  return jnp.any(symbols[:, :-2] == symbols[:, -1:], axis=-1)


def synthetic_indices(size, frac, key):
  '''The fixed floor(frac*size) pool items that replace mode regenerates every
  generation. The other items stay real.'''
  return jax.random.permutation(key, size)[:int(np.floor(frac * size))]


def replace_portion(pool, idx, model, fwd_fn, opts, data, exemplar_inds, num_train_classes, gen, key):
  '''Regenerates the pool items at `idx` from their own (real) prompts with
  `model`, marking them with origin_gen = gen. The pool size stays N and the
  items outside `idx` stay real.'''
  if len(idx) == 0:
    return pool
  new = generate_sequences(model, fwd_fn, {k: pool[k][idx] for k in POOL_KEYS},
                           key, opts, data, exemplar_inds, num_train_classes)
  pool = dict(pool)
  for k in POOL_KEYS:
    pool[k] = pool[k].at[idx].set(new[k])
  pool['origin_gen'] = pool['origin_gen'].at[idx].set(gen)
  if 'n_in_ctx' in pool:
    pool['n_in_ctx'] = _n_in_ctx(pool['symbols'])
  return pool


def accumulate(pool, real_pool, model, fwd_fn, opts, data, exemplar_inds, num_train_classes, gen, key):
  '''Appends one model-generated sequence per real pool item (generated from
  that item's prompt), marked with origin_gen = gen. The pool grows by N.'''
  new = generate_sequences(model, fwd_fn, {k: real_pool[k] for k in POOL_KEYS},
                           key, opts, data, exemplar_inds, num_train_classes)
  new['origin_gen'] = jnp.full(new['labels'].shape[0], gen, dtype=pool['origin_gen'].dtype)
  if 'n_in_ctx' in pool:
    new['n_in_ctx'] = _n_in_ctx(new['symbols']).astype(pool['n_in_ctx'].dtype)
  return {k: jnp.concatenate([pool[k], new[k]]) for k in pool}


def _label_err(symbols, labels, pos, ctx_end):
  '''Fraction of items whose label at `pos` differs from the label of the same
  symbol in context (items whose symbol is not in context are skipped).'''
  match = symbols[:, :ctx_end] == symbols[:, pos][:, None]
  has = match.any(axis=-1)
  if not has.any():
    return np.nan
  ctx_label = labels[np.arange(len(labels)), match.argmax(axis=-1)]
  return float(np.mean(ctx_label[has] != labels[has, pos]))


def _dist_metrics(hist, ref_hist, eps=1e-10):
  p = hist / max(hist.sum(), 1)
  q = ref_hist / max(ref_hist.sum(), 1)
  nz = p > 0
  order = np.argsort(q, kind='stable')  # rarest reference classes first
  top = np.sort(p)[::-1]
  return {'entropy': float(-np.sum(p[nz] * np.log(p[nz]))),
          'kl_to_gen0': float(np.sum(p[nz] * (np.log(p[nz]) - np.log(np.where(q[nz] > 0, q[nz], eps))))),
          'n_unique': float(np.sum(hist > 0)),
          'top10_mass': float(top[:10].sum()),
          'tail_mass': float(p[order[:len(p) // 2]].sum())}


def pool_metrics(pool, gen, opts, num_train_classes, ref_hist=None):
  '''Collapse metrics for the pool and for the items generated this generation.

  Returns (scalar metrics, histograms). For the extended variant the next
  symbol histogram is over train classes, compared against `ref_hist` (the
  generation 0 histogram; defaults to this pool's own).
  '''
  extended = opts.model_variant == 'extended'
  q = -2 if extended else -1
  symbols = np.asarray(pool['symbols'])
  labels = np.asarray(pool['labels'])
  origin = np.asarray(pool['origin_gen'])

  # At generation 0 the "new" items are the real data, so every row has all keys
  groups = {'': np.ones(len(origin), dtype=bool), '_new': origin == gen}

  scalars = {'real_frac': float(np.mean(origin == 0))}
  hists = {}
  for suffix, sel in groups.items():
    s, l = symbols[sel], labels[sel]
    scalars['query_label_err' + suffix] = _label_err(s, l, q, q)
    hists['query_label_hist' + suffix] = np.bincount(l[:, q], minlength=opts.model_output_classes)[:opts.model_output_classes]
    if extended:
      next_hist = np.bincount(s[:, -1], minlength=num_train_classes)[:num_train_classes]
      hists['next_symbol_hist' + suffix] = next_hist
      ref = next_hist if ref_hist is None else ref_hist
      for m, v in _dist_metrics(next_hist, ref).items():
        scalars['next_symbol_' + m + suffix] = v
      scalars['next_offcontext_rate' + suffix] = float(np.mean(~(s[:, :-2] == s[:, -1:]).any(axis=-1)))
      scalars['next_value_err' + suffix] = _label_err(s, l, -1, -2)
  return scalars, hists


def append_to_h5(fname, gen, scalars, hists):
  '''Appends one row per metric (indexed by generation) to `fname`.'''
  with h5.File(fname, 'a') as f:
    rows = {'gen': np.array(gen), **{k: np.array(v) for k, v in {**scalars, **hists}.items()}}
    for k, v in rows.items():
      if k in f:
        f[k].resize(f[k].shape[0] + 1, axis=0)
        f[k][-1] = v
      else:
        f.create_dataset(k, data=v[None], maxshape=(None,) + v.shape)
