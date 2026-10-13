'''
This file contains code for training models. It relies on main_utils.py
for argparse options. It constructs training and evaluation data iterators,
creates a model (supporting loading from checkpoint logic), and trains the
model. Evaluations are run throughout training and checkpoints are saved.

It also factors in any optogenetic clamps that may be used throughout training.
See artificial_optogenetics_guide.md and opto.py for more details.
'''

from typing import Any
from typing import Sequence
from typing import Generator
from typing import List
from typing import Tuple
from jax import Array
from nptyping import NDArray
from nptyping import Floating

import argparse
from functools import partial
from time import time
import os
import json
from datetime import datetime
import pdb

import numpy as np
import h5py as h5
import pickle as pkl

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from src import samplers
from src import main_utils
from src import opto
from src import generations


ALL_TRAIN_METRICS = ['loss', 'grad_norm', 'grad_batch_stddev']


def smart_index(arr_maybe_none, i, default):
  if arr_maybe_none is None:
    return default
  elif isinstance(arr_maybe_none, str):
    return arr_maybe_none
  else:
    return arr_maybe_none[i]


def make_train_data_sampler(opts, data, splits):
  '''Returns a jitted key -> batch fn drawing fresh real training sequences.'''
  extended = opts.model_variant == 'extended'
  next_symbol = opts.next_symbol_source if extended else None

  zipf_ranks = jnp.arange(1, opts.class_split[0]+1, dtype=jnp.float32)
  zipf_probs = 1 / zipf_ranks**opts.zipf_alpha
  train_class_distr = zipf_probs / jnp.sum(zipf_probs)


  assert len(opts.pt_burstiness) == len(opts.mixing_coeffs)

  train_samplers = []
  assert len(opts.mixing_coeffs) > 0, "Must have at least 1 train sampler"
  for i in range(len(opts.mixing_coeffs)):
    train_samplers.append(partial(samplers.get_constant_burst_seq_idxs,
                                  classes=splits['class']['train'],
                                  class_distr=train_class_distr,
                                  num_seqs=opts.train_bs,
                                  context_len=opts.train_context_len,
                                  burstiness=opts.pt_burstiness[i],
                                  distractor=smart_index(opts.pt_distract, i, 1),
                                  no_support=smart_index(opts.pt_no_support, i, 0),
                                  unique_rest=smart_index(opts.pt_unique_rest, i, 0),
                                  next_symbol=next_symbol))
  train_class_sampler = partial(samplers.get_mixed_seq_idxs,
                            mix_probabilities=jnp.array(opts.mixing_coeffs),
                            mix_substrate_fns=train_samplers)

  train_exemplar_sampler = partial(
    samplers.get_exemplar_inds,
    allowed_inds=splits['exemplar']['train'],
    match_query_and_distractors=opts.match_query_and_distractors)
  # the below will now take in a key, and output a batch, all jitted up!
  return jax.jit(partial(samplers.make_data_sampler(
    train_class_sampler,
    train_exemplar_sampler,
    fs_relabel=splits['relabeling']['train'],
    noise_scale=opts.noise_scale_train,
    assign_query_label_random=opts.assign_query_label_random,
    next_symbol=extended,
    num_labels=opts.model_output_classes),
    data=data))


def accuracy(pred_y: Array, y: Array) -> Array:
  predicted_class = jnp.argmax(pred_y, axis=-1)
  return predicted_class == y


def ce(pred_y: Array, y: Array) -> Array:
  pred_y = jax.nn.log_softmax(pred_y, axis=-1)
  num_classes = pred_y.shape[-1]
  onehot_y = jax.nn.one_hot(y, num_classes)
  return -jnp.sum(pred_y * onehot_y, axis=-1)


def extended_ces(out, y: Array, s: Array) -> Tuple[Array, Array, Array]:
  '''Per-sequence CEs of the extended variant's 3 predictions: the query's
  value (at the query), the next symbol (at the query's label) and the next
  symbol's value (at the next symbol).'''
  return (ce(out['out'][:, -2, :], y[:, -2]),
          ce(out['symbol_out'][:, -1, :], s[:, -1]),
          ce(out['out'][:, -1, :], y[:, -1]))


# Uncomment this jit if looking to microbatch
# @eqx.filter_jit
# Remember that this only yields the grad w.r.t. first positional arg!
@eqx.filter_value_and_grad
def compute_loss(
  model: eqx.Module,
  fwd_fn,
  weight_decay: float,
  x: Array, 
  y: Array, 
  keys: Array,
  s: Array = None
) -> Array:
  '''s (symbols) is only given for the extended variant.'''
  out = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)
  if s is None:
    data_loss = ce(out['out'][:, -1, :], y[:, -1]).mean()
  else:
    data_loss = jnp.mean(jnp.stack(extended_ces(out, y, s)))
  # Hacky, but prevents nan'ing gradients of 0 (which biases are initialized too)
  weight_norm = jnp.sum(jnp.stack(jax.tree_map(lambda x: jnp.linalg.norm(jnp.where(x != 0, x, 0)), jax.tree_leaves(eqx.filter(model, eqx.is_array)))))
  return data_loss + weight_decay * weight_norm


# Comment this jit if looking to microbatch
@eqx.filter_jit
def train_step(
  model: eqx.Module,
  fwd_fn,
  optimizer: optax.GradientTransformation,
  opt_state: Array,
  microbs: int,
  weight_decay: float,
  x: Array,
  y: Array,
  key: Array,
  s: Array = None,
) -> Tuple[Array, eqx.Module, Array]:
  keys = jax.random.split(key, x.shape[0])

  losses = []
  grads = []
  lengths = []
  for i in range(0, x.shape[0], microbs):
    use_x = x[i:i+microbs]
    use_y = y[i:i+microbs]
    use_keys = keys[i:i+microbs]
    use_s = None if s is None else s[i:i+microbs]
    # For some reason using keywords for the argument on the next line breaks things
    # I think it has to do with how equinox implements the value_and_grad wrapper
    # Namely, it "priveleges" 'x' I think
    mini_loss, mini_grad = compute_loss(model, fwd_fn, weight_decay, use_x, use_y, use_keys, use_s)
    losses.append(mini_loss)
    grads.append(mini_grad)

  loss = jnp.mean(jnp.array(losses))
  avg_fn = lambda *args: jnp.mean(jnp.array(args),axis=0)
  norm_fn = lambda model_like: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(arr) for arr in jax.tree_util.tree_flatten(model_like)[0]])))

  grad = jax.tree_map(avg_fn, *grads)
  grad_norm = norm_fn(jax.tree_map(lambda x: x**2, grad))
  grads_dev = [jax.tree_map(lambda x,y: (x-y)**2, g, grad) for g in grads]
  grad_batch_var = jax.tree_map(avg_fn, *grads_dev)
  grad_batch_stddev = norm_fn(grad_batch_var)

  update, opt_state = optimizer.update(grad, opt_state)
  model = eqx.apply_updates(model, update)
  # Make sure that ALL_TRAIN_METRICS is correspondingly updated
  # if more metrics are added to this function
  return {'loss': loss, 'grad_norm': grad_norm, 'grad_batch_stddev': grad_batch_stddev}, model, opt_state


@eqx.filter_jit
def eval_step(
  model: eqx.Module, 
  fwd_fn,
  x: Array, 
  y: Array, 
  key: Array,
  s: Array = None
) -> Tuple[Array, Array]:
  '''s (symbols) is only given for the extended variant, in which case the
  standard metrics are computed at the query (second to last position).'''

  keys = jax.random.split(key, x.shape[0])
  fwd = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)
  pred_y = fwd['out']
  full_y, full_pred_y = y, pred_y
  q = -1 if s is None else -2

  in_context_mask = jnp.sum(jax.nn.one_hot(y[:, :q], pred_y.shape[-1]), axis=1) > 0

  # We don't care about non-query sequence outputs from hereon out
  y = y[:, q]
  pred_y = pred_y[:, q, :]

  loss = ce(pred_y, y)
  accs = accuracy(pred_y, y)
  prob_y = jax.nn.softmax(pred_y)

  output_prob = jnp.einsum('bc,bc->b', 
                            prob_y, 
                            jax.nn.one_hot(y, pred_y.shape[-1]))

  # mask out the logits for items that don't appear in context
  # by setting them to a very negative number
  in_context_pred_y = in_context_mask*pred_y - (1-in_context_mask)*1e20
  fsl_accs = accuracy(in_context_pred_y, y)
  fsl_output_prob = jnp.einsum('bc,bc->b', 
                                jax.nn.softmax(in_context_pred_y), 
                                jax.nn.one_hot(y, in_context_pred_y.shape[-1]))

  out_context_pred_y = (1-in_context_mask)*pred_y - in_context_mask*1e20
  out_context_accs = accuracy(out_context_pred_y, y)
  out_context_prob = jnp.einsum('bc,bc->b', 
                                jax.nn.softmax(out_context_pred_y), 
                                jax.nn.one_hot(y, out_context_pred_y.shape[-1]))

  use_context_prob = jnp.sum(in_context_mask*prob_y, axis=-1)

  metrics = {'loss': loss, 
             'acc': accs, 
             'in_context_acc': fsl_accs, 
             'out_context_acc': out_context_accs,
             'prob': output_prob,
             'in_context_prob': fsl_output_prob,
             'out_context_prob': out_context_prob,
             'use_context_prob': use_context_prob}
  if s is None:
    return metrics

  # Extended variant (teacher-forced): next symbol and its value
  _, loss_symbol, loss_next_value = extended_ces(fwd, full_y, s)
  symbol_pred = jnp.argmax(fwd['symbol_out'][:, -1, :], axis=-1)
  symbol_accs = symbol_pred == s[:, -1]
  next_value_accs = accuracy(full_pred_y[:, -1, :], full_y[:, -1])
  metrics.update({
    'loss_symbol': loss_symbol,
    'loss_next_value': loss_next_value,
    'loss_total': (loss + loss_symbol + loss_next_value) / 3,
    'symbol_acc': symbol_accs,
    'next_value_acc': next_value_accs,
    'seq_acc': accs & symbol_accs & next_value_accs,
    # How often the predicted next symbol is one of the context symbols
    'symbol_pred_in_ctx': jnp.any(s[:, :-2] == symbol_pred[:, None], axis=-1),
    'next_in_ctx': jnp.any(s[:, :-2] == s[:, -1:], axis=-1),
  })
  return metrics


def make_batched_fn(fn, batch_size):
  def batched_fn(model, x, y, key, s=None):
    total = x.shape[0]
    metrics = dict()
    seed = key
    for it in range(0, total, batch_size):
      seed, use = jax.random.split(seed)
      cap = min(it+batch_size, total)
      out = fn(model=model, x=x[it:cap], y=y[it:cap], key=use,
               s=None if s is None else s[it:cap])
      for m in out:
        metrics.setdefault(m, []).append(out[m])
    for m in metrics:
      metrics[m] = jnp.concatenate(metrics[m])
    return metrics
  return batched_fn


def evaluate(
  model: eqx.Module,
  fwd_fn,
  key: Array,
  eval_data,
  eval_batch_size,
  extended: bool = False
):
  seeds = jax.random.split(key, len(eval_data))
  eval_fn = make_batched_fn(partial(eval_step, fwd_fn=fwd_fn), eval_batch_size)
  retval = dict()
  for seed, k in zip(seeds, eval_data):
    retval[k] = eval_fn(model, eval_data[k]['examples'], eval_data[k]['labels'], key=seed,
                        s=eval_data[k]['symbols'] if extended else None)
    # Aggregate all query losses before exponentiating, including partial batches.
    # Keep a length-one array for the existing HDF5 metric logging interface.
    retval[k]['perplexity'] = jnp.exp(jnp.mean(retval[k]['loss']))[None]
    if extended:
      # Next value accuracy split by whether the next symbol is in context
      # (nan when a split is empty, e.g. unseen symbols with next_symbol_source=context)
      in_ctx = retval[k]['next_in_ctx']
      correct = retval[k]['next_value_acc']
      for name, sel in [('in_ctx', in_ctx), ('unseen', ~in_ctx)]:
        count = jnp.sum(sel)
        retval[k]['next_value_acc_' + name] = jnp.where(
          count > 0, jnp.sum(correct & sel) / jnp.maximum(count, 1), jnp.nan)[None]
  return retval


def train_generation(
  opts,
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state,
  seeds,
  start_iter: int,
  sample_batch,
  fwd_fn,
  eval_data,
  run_folder: str,
  wandb_prefix: str = '',
  wandb_step_offset: int = 0,
):
  '''Trains `model` for opts.train_iters sequences on batches from
  `sample_batch` (a key -> batch fn), running evals and checkpoints according
  to schedule. Writes log.h5 and checkpoints/ into `run_folder`.

  Returns the trained model, optimizer state and the advanced seeds.
  '''
  extended = opts.model_variant == 'extended'
  if opts.use_wandb:
    import wandb
  eval_model_seed = seeds['eval_model_seed']
  train_data_seed = seeds['train_data_seed']
  train_model_seed = seeds['train_model_seed']

  ### Setup log.h5 ###

  ckpt_folder = '/'.join([run_folder, 'checkpoints'])
  os.makedirs(ckpt_folder, exist_ok=True)

  eval_ind = np.searchsorted(opts.eval_sched, start_iter)
  ckpt_ind = np.searchsorted(opts.ckpt_sched, start_iter)

  train_metrics = {m: [] for m in ALL_TRAIN_METRICS}
  # We'll always want train iters
  train_metrics['iter'] = []

  results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
  for m in train_metrics:
    if 'train_{}'.format(m) not in results:
      results.create_dataset('train_{}'.format(m), shape=(0,), maxshape=(None,), dtype=float)
  if 'eval_iter' not in results:
    results.create_dataset('eval_iter', shape=(0,), maxshape=(None,), dtype=int)
  results.close()

  ### MAIN TRAIN LOOP ###
  # Runs eval and checkpoints according to schedule
  # Note i = iterations = # sequences seen
  
  for i in range(start_iter, opts.train_iters, opts.train_bs):
    if eval_ind < len(opts.eval_sched) and i >= opts.eval_sched[eval_ind]:
      if not opts.suppress_output:
        print('-'*10 + str(i))
      eval_model_seed, current_eval_seed = jax.random.split(eval_model_seed, 2)
      out = evaluate(model=model, 
                      fwd_fn=fwd_fn, 
                      key=current_eval_seed, 
                      eval_data=eval_data, 
                      eval_batch_size=opts.eval_bs,
                      extended=extended)
      results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
      for k in out:
        for m in out[k]:
          if not opts.suppress_output:
            print(k, m, jnp.mean(out[k][m]))
          if opts.use_wandb:
            wandb.log({wandb_prefix + '-'.join([k, m]): jnp.mean(out[k][m]), 'iteration': i}, step=wandb_step_offset + i//opts.train_bs)
          addr = '/'.join([k,m])
          if addr in results:
            results[addr].resize(results[addr].shape[0] + 1, axis=0)
            results[addr][-1] = out[k][m]
          else:
            results.create_dataset(addr, data=jnp.broadcast_to(out[k][m], (1, len(out[k][m]))), maxshape=(None, len(out[k][m])))
      results['eval_iter'].resize(results['eval_iter'].shape[0]+1, axis=0)
      results['eval_iter'][-1] = i
      if len(train_metrics['iter']) > 0:
        if not opts.suppress_output:
          print('-')
        if opts.use_wandb:
          wandb.log({**{wandb_prefix + m: train_metrics[m][-1] for m in ALL_TRAIN_METRICS}, 'iteration': i}, step=wandb_step_offset + i//opts.train_bs)
        for m in train_metrics:
          results['train_{}'.format(m)].resize(results['train_{}'.format(m)].shape[0]+len(train_metrics[m]), axis=0)
          results['train_{}'.format(m)][-len(train_metrics[m]):] = train_metrics[m]
          if not opts.suppress_output:
            print(m, train_metrics[m][-1])
          train_metrics[m] = []
      results.close()
      eval_ind += 1

    if ckpt_ind < len(opts.ckpt_sched) and i >= opts.ckpt_sched[ckpt_ind]:
      if not opts.suppress_output:
        print("Checkpointing...", i)
      ckpt = {'iter': i,
              'seeds': {'eval_model_seed': eval_model_seed,
                        'train_data_seed': train_data_seed,
                        'train_model_seed': train_model_seed}, 
              'opt_state': opt_state,
              'model': model}
      num = '{}'.format(i).zfill(11)
      eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)
      if len(train_metrics['iter']) > 0:
        if not opts.suppress_output:
          print(train_metrics['iter'][-1], train_metrics['loss'][-1])
      ckpt_ind += 1

    # Train step -- the train_data_seed is split and passed along (a la functional
    # programming)
    train_data_seed, current_data_seed = jax.random.split(train_data_seed)
    # batch is a dict with keys 'examples' and 'labels'
    # 'examples' is [batch_size, train_context_len + 1, embedding_dim]
    # 'labels' is [batch_size, train_context_len + 1]
    # (the +1 is for the target/query)
    batch = sample_batch(current_data_seed)

    train_model_seed, current_model_seed = jax.random.split(train_model_seed)
    metrics, model, opt_state = train_step(model=model, 
                                            fwd_fn=fwd_fn,
                                            optimizer=optimizer, 
                                            opt_state=opt_state, 
                                            microbs=opts.train_microbs, 
                                            weight_decay=opts.weight_decay,
                                            x=batch['examples'], 
                                            y=batch['labels'], 
                                            key=current_model_seed,
                                            s=batch['symbols'] if extended else None)
    train_metrics['iter'].append(i+opts.train_bs)
    for m in metrics:
      train_metrics[m].append(metrics[m])
    if opts.use_wandb:
      if i % 1000*opts.train_bs == 0:
        # Ideally we'd log learning rate, but hard to access with jax
        wandb.log({**{wandb_prefix + m: v for m, v in metrics.items()}, 'iteration': i}, step=wandb_step_offset + 1+i//opts.train_bs)

  print("End of training")
  eval_model_seed, current_eval_seed = jax.random.split(eval_model_seed, 2)
  out = evaluate(model=model, 
                  fwd_fn=fwd_fn,
                  key=current_eval_seed, 
                  eval_data=eval_data, 
                  eval_batch_size=opts.eval_bs,
                  extended=extended)
  results = h5.File('/'.join([run_folder, 'log.h5']), 'a')
  if len(train_metrics['iter']) > 0:
    print('-')
    for m in train_metrics:
      results['train_{}'.format(m)].resize(results['train_{}'.format(m)].shape[0]+len(train_metrics[m]), axis=0)
      results['train_{}'.format(m)][-len(train_metrics[m]):] = train_metrics[m]
      print(m, train_metrics[m][-1])
  for k in out:
    for m in out[k]:
      print(k, m, jnp.mean(out[k][m]))
      if opts.use_wandb:
        wandb.log({wandb_prefix + '-'.join([k, m]): jnp.mean(out[k][m]),
                   'iteration': i + opts.train_bs}, step=wandb_step_offset + 1+i//opts.train_bs)
      addr = '/'.join([k,m])
      if addr in results:
        results[addr].resize(results[addr].shape[0] + 1, axis=0)
        results[addr][-1] = out[k][m]
      else:
        results.create_dataset(addr, data=jnp.broadcast_to(out[k][m], (1, len(out[k][m]))), maxshape=(None,len(out[k][m])))
  results['eval_iter'].resize(results['eval_iter'].shape[0]+1, axis=0)
  results['eval_iter'][-1] = i + opts.train_bs
  results.close()

  if opts.ckpt_sched[-1] <= opts.train_iters:
    print("Checkpointing...", i + opts.train_bs)
    ckpt = {'iter': i + opts.train_bs,
            'seeds': {'eval_model_seed': eval_model_seed,
                      'train_data_seed': train_data_seed,
                      'train_model_seed': train_model_seed}, 
            'opt_state': opt_state,
            'model': model}
    num = '{}'.format(i + opts.train_bs).zfill(11)
    eqx.tree_serialise_leaves('/'.join([ckpt_folder, num+".eqx"]), ckpt)

  seeds = {'eval_model_seed': eval_model_seed,
           'train_data_seed': train_data_seed,
           'train_model_seed': train_model_seed}
  return model, opt_state, seeds


def run_with_opts(opts):

  ### Process input opts ###

  if opts.config_from_file is not None:
    opts = main_utils.get_opts_from_json_file(opts.config_from_file)
    opts.config_from_file = None

  if opts.train_microbs is None:
    opts.train_microbs = opts.train_bs
  else:
    if opts.train_bs % opts.train_microbs > 0:
      print("WARNING: using microbatching. Make sure jitting is configured properly -- see comments in main.py")
      print('Setting train_bs to a multiple of train_microbs')
      new_bs = opts.train_microbs*((opts.train_bs//opts.train_microbs) + 1)
      print("Updating train_bs from {} to {}".format(opts.train_bs, new_bs))
      opts.train_bs = new_bs

  main_utils.check_opts(opts)

  if opts.disable_jit:
    print("Disabling jit")
    jax.config.update('jax_disable_jit', True)

  if opts.eval_sched is not None:
    opts.eval_every = None
    opts.eval_sched_file = None
  elif opts.eval_every is not None:
    opts.eval_sched = np.arange(0, opts.train_iters, opts.eval_every)
    opts.eval_every = None
  elif opts.eval_sched_file is not None:
    with open(opts.eval_sched_file, 'rb') as f:
      opts.eval_sched = pkl.load(f)
    opts.eval_sched_file = None
  # Sort and canonicalize for json
  opts.eval_sched = [int(x) for x in sorted(opts.eval_sched)]

  if opts.ckpt_sched is not None:
    opts.ckpt_every = None
    opts.ckpt_sched_file = None
  elif opts.ckpt_every is not None:
    opts.ckpt_sched = np.arange(0, opts.train_iters, opts.ckpt_every)
    opts.ckpt_every = None
  elif opts.ckpt_sched_file is not None:
    with open(opts.ckpt_sched_file, 'rb') as f:
      opts.ckpt_sched = pkl.load(f) 
    opts.ckpt_sched_file = None
  else:
    # If all ckpt opts are None, avoid checkpointing
    opts.ckpt_sched = np.array([opts.train_iters*2]).astype(int)
  # Sort and canonicalize for json
  opts.ckpt_sched = [int(x) for x in sorted(opts.ckpt_sched)]

  # Model output classes default logic
  if opts.model_output_classes is None:
    if opts.fs_relabel > 0:
      opts.model_output_classes = opts.fs_relabel
    else:
      opts.model_output_classes = opts.class_split[0]
    print("Defaulting model output classes to", opts.model_output_classes)

  # It's important to set all the defaults before saving the opts
  # For example, when loading a checkpoint, we want things like model output classes
  # to be filled in

  if opts.raw_name:
    run_name = opts.run
  else:
    run_name = '_'.join([datetime.now().strftime("%Y%m%d%H%M%S"), opts.run])

  run_folder = '/'.join([opts.base_folder, run_name])
  os.makedirs(run_folder, exist_ok=True)
  with open('/'.join([run_folder, 'config.json']), 'w') as f:
    json.dump(vars(opts), f, indent='\t')

  if opts.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
      project="icl-transience", config=opts, name=opts.run, dir=opts.base_folder)
  
  opts.train_seed = jax.random.PRNGKey(opts.train_seed)
  opts.eval_seed = jax.random.PRNGKey(opts.eval_seed)

  extended = opts.model_variant == 'extended'
  next_symbol = opts.next_symbol_source if extended else None

  ### Make train data sampler ###

  data = main_utils.get_data_from_opts(opts)

  splits = main_utils.get_splits_from_opts(opts, data.shape)

  train_data_sampler = make_train_data_sampler(opts, data, splits)

  train_data_seed, train_model_seed = jax.random.split(opts.train_seed, 2)
  eval_data_seed, eval_model_seed = jax.random.split(opts.eval_seed, 2)

  ### Make eval data ###

  # We first make a sampler for each evaluator, 
  # then take one sample from it to use as our eval data
  # This allows us to re-use the machinery in samplers.py

  eval_data = {}
  if opts.load_eval_data is not None:
    for eval_file in opts.load_eval_data:
      # an evaluator is a mini dataset of sequences and labels
      print("loading some evaluators from file")
      eval_h5 = h5.File(eval_file, 'r')
      print("Found these eval sets in file:", eval_h5.keys())
      for k in eval_h5:
        if k in eval_data:
          raise ValueError("Found duplicate eval key. Behavior in this case is not defined")
        eval_data[k] = {'examples': jnp.array(eval_h5['/'.join([k, 'examples'])]),
                          'labels': jnp.array(eval_h5['/'.join([k, 'labels'])])}
      print("Loaded in file:", eval_h5.keys())
      eval_h5.close()

  eval_data_seeds = jax.random.split(eval_data_seed, len(opts.pe_names))
  for i, name in enumerate(opts.pe_names):
    if name in eval_data:
      print("ERROR: Found overlapping name of new evaluator with old evaluator from file! Please rename new evaluator or stop using file.")
      pdb.set_trace()

    assert name not in ['train', 'train_iter', 'eval_iter']

    classes = splits['class'][opts.pe_classes[i]]
    eval_class_sampler = partial(samplers.get_constant_burst_seq_idxs,
                                  classes=classes,
                                  class_distr=jnp.ones(len(classes))/len(classes),
                                  num_seqs=opts.eval_iters,
                                  context_len=smart_index(opts.pe_context_len, i, opts.train_context_len),
                                  burstiness=opts.pe_burstiness[i],
                                  distractor=smart_index(opts.pe_distract, i, 1),
                                  no_support=smart_index(opts.pe_no_support, i, 0),
                                  unique_rest=smart_index(opts.pe_unique_rest, i, 0),
                                  next_symbol=next_symbol)
    allowed_inds = splits['exemplar'][opts.pe_exemplars[i]]
    eval_exemplar_sampler = partial(
      samplers.get_exemplar_inds,
      allowed_inds=allowed_inds,
      match_query_and_distractors=opts.match_query_and_distractors)
    eval_data_sampler = samplers.make_data_sampler(
      eval_class_sampler,
      eval_exemplar_sampler,
      fs_relabel=splits['relabeling'][smart_index(opts.pe_fs_relabel_scheme, i, 'None')],
      noise_scale=smart_index(opts.pe_noise_scale, i, opts.noise_scale_train),
      assign_query_label_random=smart_index(opts.pe_assign_query_label_random, i, 0),
      next_symbol=extended,
      num_labels=opts.model_output_classes)
    eval_data[name] = eval_data_sampler(eval_data_seeds[i], data)

  if opts.save_eval_data is not None:
    fname = '/'.join([run_folder, opts.save_eval_data])
    print("Writing eval data to file:", fname)
    with h5.File(fname, 'w') as f:
      for k in eval_data:
        for s in eval_data[k]:
          f.create_dataset('/'.join([k,s]), data=eval_data[k][s])
    print("Done writing")


  ### Setup model and optimizer ###
  # Includes logic for loading from checkpoints

  fwd_fn_to_use = opto.make_fn_from_opts(opts)
  if extended:
    assert fwd_fn_to_use is opto.default_model_fwd_fn, 'Optogenetic clamps are only supported for the base variant'

  if opts.load_from_ckpt is not None:
    assert opts.load_from_ckpt_cfg is not None, "Must specify config for loading model checkpoint"
    print("Resetting config to checkpointed config")
    load_opts = main_utils.get_opts_from_json_file(opts.load_from_ckpt_cfg)
    model = main_utils.get_model_from_opts(load_opts)
  else:
    model = main_utils.get_model_from_opts(opts, input_shape=(data.shape[-1],))

  optimizer = main_utils.get_optimizer_from_opts(opts)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  start_iter = 0
  if opts.load_from_ckpt is not None:
    ckpt_fmt = {'iter': -1, 
                'seeds': {'eval_model_seed': eval_model_seed,
                          'train_data_seed': train_data_seed,
                          'train_model_seed': train_model_seed}, 
                'opt_state': opt_state,
                'model': model}
    ckpt = eqx.tree_deserialise_leaves(opts.load_from_ckpt, ckpt_fmt)

    start_iter = ckpt['iter']
    if not opts.no_load_seeds:
      eval_model_seed = ckpt['seeds']['eval_model_seed']
      train_data_seed = ckpt['seeds']['train_data_seed']
      train_model_seed = ckpt['seeds']['train_model_seed']
    if not opts.no_load_opt_state:
      opt_state = ckpt['opt_state']
    model = ckpt['model']

  seeds = {'eval_model_seed': eval_model_seed,
           'train_data_seed': train_data_seed,
           'train_model_seed': train_model_seed}
  train_kwargs = dict(opts=opts, optimizer=optimizer, fwd_fn=fwd_fn_to_use, eval_data=eval_data)

  # Runs after each generation trains; a no-op unless --circuit-discovery
  find_circuits = lambda model, gen_folder, g: None
  if opts.circuit_discovery:
    # Imported here so runs without the flag don't need torch
    from src import circuits
    if fwd_fn_to_use is not opto.default_model_fwd_fn:
      print("WARNING: circuit discovery ignores optogenetic clamps and analyses the unclamped weights")
    # The same real sequences for every generation, so circuits stay comparable
    circuit_prompts = generations.build_pool(train_data_sampler, opts.circuit_n_prompts,
                                             jax.random.PRNGKey(opts.circuit_seed))
    def find_circuits(model, gen_folder, g):
      circuits.run_circuit_discovery(
        model, opts, circuit_prompts, '/'.join([gen_folder, 'circuit']), data,
        splits['exemplar']['train'], opts.class_split[0], gen=g,
        h5_file='/'.join([run_folder, 'circuits.h5']))

  if opts.gen_dataset_size is None:
    # Original setup: fresh training data every step, a single generation
    model, _, _ = train_generation(model=model, opt_state=opt_state, seeds=seeds, start_iter=start_iter,
                                   sample_batch=train_data_sampler, run_folder=run_folder, **train_kwargs)
    find_circuits(model, run_folder, 0)
    return

  ### Model collapse pipeline ###
  # Generation 0 trains on a pool of gen_dataset_size (N) real sequences.
  # Before each later generation, the previous generation's model generates
  # new targets from real prompts:
  #   replace: a fixed gen_replace_frac portion of the pool is regenerated, so
  #     the pool stays at N and its real share stays 1 - gen_replace_frac.
  #   accumulate: N new sequences are appended, so the pool grows by N.

  num_train_classes = opts.class_split[0]
  pool_seed, replace_seed = jax.random.split(jax.random.fold_in(opts.train_seed, 1))
  pool = generations.build_pool(train_data_sampler, opts.gen_dataset_size, pool_seed)
  if extended and opts.next_symbol_source == 'context':
    assert bool(jnp.all(pool['n_in_ctx'])), 'Real next symbols must occur in context'
  real_pool = pool
  synth_idx = generations.synthetic_indices(opts.gen_dataset_size, opts.gen_replace_frac,
                                            jax.random.fold_in(replace_seed, 0))
  gen_args = (fwd_fn_to_use, opts, data, splits['exemplar']['train'], num_train_classes)

  init_model, init_opt_state = model, opt_state
  ref_hist = None
  wandb_gen_steps = opts.train_iters // opts.train_bs + 2
  gen_h5 = '/'.join([run_folder, 'generations.h5'])
  for g in range(opts.generations):
    gen_folder = '/'.join([run_folder, 'gen_{}'.format(g)])
    os.makedirs(gen_folder, exist_ok=True)
    if g > 0:
      gen_key = jax.random.fold_in(replace_seed, g)
      if opts.gen_mode == 'accumulate':
        print("Generation {}: adding {} sequences generated by generation {}'s model".format(
          g, opts.gen_dataset_size, g-1))
        pool = generations.accumulate(pool, real_pool, model, *gen_args, g, gen_key)
      else:
        print("Generation {}: regenerating {} of the pool with generation {}'s model".format(
          g, opts.gen_replace_frac, g-1))
        pool = generations.replace_portion(pool, synth_idx, model, *gen_args, g, gen_key)
      if opts.gen_init == 'fresh':
        model, opt_state = init_model, init_opt_state

    scalars, hists = generations.pool_metrics(pool, g, opts, num_train_classes, ref_hist)
    if g == 0 and extended:
      ref_hist = hists['next_symbol_hist']
    generations.append_to_h5(gen_h5, g, scalars, hists)
    for m in scalars:
      print('gen', g, m, scalars[m])
    if opts.use_wandb:
      wandb.log({'pool/' + m: v for m, v in scalars.items()}, step=g*wandb_gen_steps)
    np.savez('/'.join([gen_folder, 'pool_meta.npz']),
             **{k: np.asarray(pool[k]) for k in ['labels', 'symbols', 'origin_gen']})

    gen_seeds = {k: jax.random.fold_in(v, g) for k, v in seeds.items()}
    model, opt_state, _ = train_generation(
      model=model, opt_state=opt_state, seeds=gen_seeds,
      start_iter=start_iter if g == 0 else 0,
      sample_batch=generations.make_pool_sampler(pool, opts.train_bs),
      run_folder=gen_folder,
      wandb_prefix='gen{}/'.format(g) if opts.generations > 1 else '',
      wandb_step_offset=g*wandb_gen_steps,
      **train_kwargs)
    find_circuits(model, gen_folder, g)


if __name__ == "__main__":
  parser = main_utils.create_parser()
  opto.add_args_to_parser(parser)
  opts = parser.parse_args()
  start = time()
  run_with_opts(opts)
  end = time()
  print("total training time (min): ", (end-start) / 60.)
