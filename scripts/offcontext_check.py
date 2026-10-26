"""Measures how often the extended model generates a next symbol that is not in its
context, for a chosen generation's model and sampling temperature.

Off-context symbols can come from the temperature flattening a sound distribution
or from the learned distribution itself degrading. Sampling the same model at
T = 1 separates the two, and the probability mass the symbol head puts on the
context symbols measures the learned distribution directly, without sampling.

For each run, the script rebuilds the generation 0 real pool of the run (from its
config's train seed, as src/main.py does), regenerates the targets of all of its
prompts with the final checkpoint of each requested generation at each requested
temperature (src.generations.generate_sequences), and reports:
  offcontext     share of generated next symbols absent from their context
  ctx_mass       mean probability the symbol head gives the context symbols at
                 T = 1, with the real query label fed in (no sampling)
  label_err      share of generated query values that differ from the context
  next_value_err share of in-context next symbols whose generated value differs
  entropy, kl    entropy of the generated next-symbol distribution and its KL
                 divergence from the real pool's, in nats

--check reproduces the pool the run itself generated for generation 1 (same
prompts, order and key) and compares its off-context rate with generations.h5.

.venv/bin/python -m scripts.offcontext_check results/*_collapse_extended_G20_T3_replace100* \
  --generations 0 19 --temperatures 1 3
"""
import argparse
import glob
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src import circuits, generations, main_utils, opto
from src.main import make_train_data_sampler


def last_checkpoint(run, gen):
  ckpts = sorted(glob.glob(os.path.join(run, 'gen_{}'.format(gen), 'checkpoints', '*.eqx')))
  if not ckpts:
    raise SystemExit('no checkpoint in {}/gen_{}'.format(run, gen))
  return ckpts[-1]


def context_mass(model, pool, opts, num_train_classes):
  '''Mean probability the symbol head gives the context symbols, at T = 1.'''
  fwd = opto.default_model_fwd_fn
  masses = []
  for start in range(0, pool['labels'].shape[0], opts.eval_bs):
    x, y, s = (pool[k][start:start + opts.eval_bs] for k in generations.POOL_KEYS)
    _, symbol_logits = generations._logits(model, fwd, x, y, jax.random.PRNGKey(0))
    logits = symbol_logits[:, -1].at[:, num_train_classes:].set(-jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    in_ctx = jnp.zeros_like(probs, dtype=bool)
    in_ctx = in_ctx.at[jnp.arange(probs.shape[0])[:, None], s[:, :-2]].set(True)
    masses.append(np.asarray(jnp.sum(jnp.where(in_ctx, probs, 0.0), axis=-1)))
  return float(np.mean(np.concatenate(masses)))


def metrics(pool, real_hist, opts, num_train_classes):
  scalars, hists = generations.pool_metrics(pool, 0, opts, num_train_classes, real_hist)
  return {'offcontext': scalars['next_offcontext_rate'],
          'label_err': scalars['query_label_err'],
          'next_value_err': scalars['next_value_err'],
          'entropy': scalars['next_symbol_entropy'],
          'kl': scalars['next_symbol_kl_to_gen0']}


def load(run, gen):
  opts, model = circuits.load_checkpoint(last_checkpoint(run, gen), os.path.join(run, 'config.json'))
  return opts, model


def real_pool(opts, data, splits):
  '''The run's generation 0 pool, built as in src/main.py (config.json stores the integer seed).'''
  pool_seed, replace_seed = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(opts.train_seed), 1))
  pool = generations.build_pool(make_train_data_sampler(opts, data, splits), opts.gen_dataset_size, pool_seed)
  return pool, replace_seed


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('runs', nargs='+', help='extended-model collapse run folders')
  parser.add_argument('--generations', nargs='+', type=int, default=[0, 19])
  parser.add_argument('--temperatures', nargs='+', type=float, default=[1.0, 3.0])
  parser.add_argument('--seed', type=int, default=123, help='sampling key, the same for every measurement')
  parser.add_argument('--check', action='store_true', help='reproduce the logged generation 1 pool first')
  args = parser.parse_args()

  print('run\tgen\tT\toffcontext\tctx_mass\tlabel_err\tnext_value_err\tentropy\tkl')
  for run in args.runs:
    run = run.rstrip('/')
    opts, model = load(run, 0)
    assert opts.model_variant == 'extended', run + ' is not an extended-model run'
    data = main_utils.get_data_from_opts(opts)
    splits = main_utils.get_splits_from_opts(opts, data.shape)
    exemplars = splits['exemplar']['train']
    n_train = opts.class_split[0]
    pool, replace_seed = real_pool(opts, data, splits)
    real_hist = generations.pool_metrics(pool, 0, opts, n_train)[1]['next_symbol_hist']
    name = os.path.basename(run)

    if args.check:
      # The prompts, order and key with which the run generated its generation 1 pool
      if opts.gen_mode == 'accumulate':
        prompts = {k: pool[k] for k in generations.POOL_KEYS}
      else:
        idx = generations.synthetic_indices(opts.gen_dataset_size, opts.gen_replace_frac,
                                            jax.random.fold_in(replace_seed, 0))
        prompts = {k: pool[k][idx] for k in generations.POOL_KEYS}
      new = generations.generate_sequences(model, opto.default_model_fwd_fn, prompts,
                                           jax.random.fold_in(replace_seed, 1), opts, data, exemplars, n_train)
      new['origin_gen'] = jnp.zeros(new['labels'].shape[0], dtype=int)
      rate = metrics(new, real_hist, opts, n_train)['offcontext']
      with h5py.File(os.path.join(run, 'generations.h5'), 'r') as h:
        logged = h['next_offcontext_rate_new'][1]
      print('# check {}: reproduced {:.6f}, logged {:.6f}'.format(name, rate, logged))

    prompts = {k: pool[k] for k in generations.POOL_KEYS}
    for gen in args.generations:
      if gen != 0:
        opts, model = load(run, gen)
      mass = context_mass(model, pool, opts, n_train)
      for t in args.temperatures:
        opts.gen_sampling, opts.gen_temperature = 'sample', t
        new = generations.generate_sequences(model, opto.default_model_fwd_fn, prompts,
                                             jax.random.PRNGKey(args.seed), opts, data, exemplars, n_train)
        new['origin_gen'] = jnp.zeros(new['labels'].shape[0], dtype=int)
        m = metrics(new, real_hist, opts, n_train)
        print('{}\t{}\t{:g}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.3f}\t{:.4f}'.format(
          name, gen, t, m['offcontext'], mass, m['label_err'], m['next_value_err'], m['entropy'], m['kl']),
          flush=True)


if __name__ == '__main__':
  main()
