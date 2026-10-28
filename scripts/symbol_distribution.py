"""Per-generation next-symbol analysis of extended-model collapse runs.

For every generation's final model, on held-out real prompts (fresh sequences of
the training distribution, not in any pool), it measures:

Conditional symbol distribution, with the true query label supplied (no sampling):
  ctx_mass       probability of the unmasked symbol head (T = 1, train classes) on
                 the classes in the context
  q_entropy      entropy of the symbol distribution renormalised over the distinct
                 context classes, in nats (ln 2 for two classes)
  q_kl_uniform   KL divergence of that distribution from the uniform distribution
                 over the distinct context classes (the real next-symbol rule)
  q_pos1         renormalised probability of the class at context position 1
  q_query        renormalised probability of the class that matches the query
  q_absbias      mean |q(x1) - 1/2| over rows with two distinct context classes

Fully autoregressive generation at temperature --temperature (query label sampled
and fed back, as in the pipeline), unconstrained and constrained to the context:
  ar_offcontext            share of unconstrained next symbols absent from the context
  ar_next_err_none/_ctx    share of in-context next symbols given a wrong label
  ar_query_err             share of sampled query labels that are wrong

python -m scripts.symbol_distribution RUN [RUN ...] --out results/constrained/analysis
writes <out>/<run name>.tsv, one row per generation.
"""
import argparse
import glob
import os
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from src import circuits, generations, main_utils, opto
from src.main import make_train_data_sampler

FWD = opto.default_model_fwd_fn
HELD_OUT_SEED = 2027  # differs from every pool, evaluation and circuit seed


def checkpoint(run, gen):
  return sorted(glob.glob(os.path.join(run, 'gen_{}'.format(gen), 'checkpoints', '*.eqx')))[-1]


def conditional(model, prompts, opts, n_train):
  '''Symbol-head distribution at the next-symbol position with the true query label.'''
  x, y, s = (prompts[k] for k in generations.POOL_KEYS)
  rows = []
  for start in range(0, x.shape[0], opts.eval_bs):
    sl = slice(start, start + opts.eval_bs)
    _, logits = generations._logits(model, FWD, x[sl], y[sl], jax.random.PRNGKey(0))
    rows.append(logits[:, -1].at[:, n_train:].set(-jnp.inf))
  logits = jnp.concatenate(rows)
  ctx = s[:, :-2]
  mask = generations.context_mask(ctx, logits.shape[-1])
  p = jax.nn.softmax(logits, axis=-1)
  q = jax.nn.softmax(generations.mask_to_context(logits, ctx), axis=-1)
  k = mask.sum(-1)
  ent = -jnp.sum(jnp.where(q > 0, q * jnp.log(q), 0.0), axis=-1)
  r = jnp.arange(x.shape[0])
  two = np.asarray(ctx[:, 0] != ctx[:, 1])
  q1 = np.asarray(q[r, ctx[:, 0]])
  return {'ctx_mass': float(jnp.mean(jnp.sum(jnp.where(mask, p, 0.0), -1))),
          'q_entropy': float(jnp.mean(ent)),
          'q_kl_uniform': float(jnp.mean(jnp.log(k) - ent)),
          'q_pos1': float(np.mean(q1[two])),
          'q_query': float(jnp.mean(q[r, s[:, -2]])),
          'q_absbias': float(np.mean(np.abs(q1[two] - 0.5)))}


def autoregressive(model, prompts, opts, data, exemplars, n_train, temperature, key):
  out = {}
  for constraint in ('none', 'context'):
    o = deepcopy(opts)
    o.gen_sampling, o.gen_temperature, o.gen_symbol_constraint = 'sample', temperature, constraint
    new = generations.generate_sequences(model, FWD, prompts, key, o, data, exemplars, n_train)
    new['origin_gen'] = jnp.zeros(new['labels'].shape[0], dtype=int)
    m = generations.pool_metrics(new, 0, o, n_train)[0]
    if constraint == 'none':
      out['ar_offcontext'] = m['next_offcontext_rate']
      out['ar_query_err'] = m['query_label_err']
    else:
      assert m['next_offcontext_rate'] == 0.0, 'constrained generation left the context'
    out['ar_next_err_' + constraint] = m['next_value_err']
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('runs', nargs='+')
  parser.add_argument('--out', default='results/constrained/analysis')
  parser.add_argument('--n_prompts', type=int, default=5000)
  parser.add_argument('--temperature', type=float, default=None, help="default: each run's own")
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cols = ['gen', 'ctx_mass', 'q_entropy', 'q_kl_uniform', 'q_pos1', 'q_query', 'q_absbias',
          'ar_offcontext', 'ar_query_err', 'ar_next_err_none', 'ar_next_err_context']
  prompts = data = None
  for run in args.runs:
    run = run.rstrip('/')
    name = os.path.basename(run)
    out_file = os.path.join(args.out, name + '.tsv')
    n_gens = len(glob.glob(os.path.join(run, 'gen_*')))
    if os.path.exists(out_file) and sum(1 for _ in open(out_file)) == n_gens + 1:
      print('skip', name)
      continue
    opts, model = circuits.load_checkpoint(checkpoint(run, 0), os.path.join(run, 'config.json'))
    assert opts.model_variant == 'extended'
    if prompts is None:
      # The same held-out prompts for every run (train classes and exemplars, as in training)
      data = main_utils.get_data_from_opts(opts)
      splits = main_utils.get_splits_from_opts(opts, data.shape)
      exemplars, n_train = splits['exemplar']['train'], opts.class_split[0]
      held = generations.build_pool(make_train_data_sampler(opts, data, splits), args.n_prompts,
                                    jax.random.PRNGKey(HELD_OUT_SEED))
      prompts = {k: held[k] for k in generations.POOL_KEYS}
    t = args.temperature if args.temperature is not None else opts.gen_temperature
    with open(out_file + '.tmp', 'w') as f:
      f.write('\t'.join(cols) + '\n')
      for g in range(n_gens):
        if g > 0:
          opts, model = circuits.load_checkpoint(checkpoint(run, g), os.path.join(run, 'config.json'))
        row = {'gen': g, **conditional(model, prompts, opts, n_train),
               **autoregressive(model, prompts, opts, data, exemplars, n_train, t, jax.random.PRNGKey(g))}
        f.write('\t'.join(str(row[c]) for c in cols) + '\n')
        f.flush()
    os.replace(out_file + '.tmp', out_file)
    print('wrote', out_file, flush=True)


if __name__ == '__main__':
  main()
