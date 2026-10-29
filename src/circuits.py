'''
Circuit discovery with auto-circuit (Miller et al., 2024,
https://github.com/UFO-101/auto-circuit), run on the trained model of every
generation when --circuit-discovery is set.

Two methods run on every target:
  - edge attribution patching: auto-circuit's mask-gradient scores, which rank
    every edge, with a faithfulness curve over circuit sizes.
  - ACDC (Conmy et al., 2023, https://github.com/ArthurConmy/Automatic-Circuit-Discovery):
    the iterative pruning algorithm, run on auto-circuit's patchable graph. It
    returns one circuit per threshold tau.

auto-circuit works on PyTorch TransformerLens models, so the Equinox model is
first copied into an equivalent HookedTransformer, whose logits are checked
against the JAX model before any discovery. torch, transformer_lens and
auto_circuit are only imported here, so the rest of the code runs without them.
See docs/circuit_discovery.md.

Can also be run on a saved checkpoint:
  python -m src.circuits --checkpoint <ckpt.eqx> --run_config <config.json> --out_folder <dir>
'''

import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from auto_circuit.data import PromptDataset, PromptDataLoader
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import AblationType, PatchType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import edge_counts_util, patch_mode, patchable_model, set_all_masks
from auto_circuit.utils.tensor_ops import multibatch_kl_div
from auto_circuit.visualize import draw_seq_graph

from src import generations
from src import main_utils
from src import opto


# Max abs logit difference allowed between the JAX model and its conversion
EQUIVALENCE_TOL = 1e-4
# A circuit "recovers" the behaviour once it restores this share of the logit diff
RECOVERY_TARGET = 0.8
MAX_DRAWN_EDGES = 40
# ACDC thresholds on the KL divergence to the full model; the default is the reported circuit
ACDC_THRESHOLD = 0.001
ACDC_SWEEP = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


class EmbeddingInputTransformer(HookedTransformer):
  '''HookedTransformer whose input is the residual stream rather than tokens.

  Our model has no token vocabulary: it embeds image features and labels
  itself, so we feed it the interleaved embedding sequence directly.
  '''

  def forward(self, input, *args, **kwargs):
    if t.is_floating_point(input):
      kwargs.setdefault('start_at_layer', 0)
    return super().forward(input, *args, **kwargs)


def _t(x):
  return t.tensor(np.array(x), dtype=t.float32)


def to_hooked_transformer(model, opts, head='label', device='cpu'):
  '''Copies a SequenceClassifier into an EmbeddingInputTransformer.

  head: 'label' uses the value unembedding, 'symbol' the extended variant's
  next-symbol unembedding.
  '''
  blocks = model.transformer.blocks
  C, H = opts.d_model, opts.num_heads
  dh = C // H
  attn_only = opts.mlp_ratio is None
  rope = opts.pos_embedding_type == 'rope'
  unembed = model.unembed if head == 'label' else model.symbol_unembed
  cfg = HookedTransformerConfig(
    n_layers=len(blocks), d_model=C, n_heads=H, d_head=dh,
    d_mlp=None if attn_only else int(C * opts.mlp_ratio),
    attn_only=attn_only, act_fn='gelu_new',  # jax.nn.gelu defaults to the tanh approximation
    d_vocab=unembed.out_features, n_ctx=64,
    normalization_type=None if opts.no_norm else 'LN',
    eps=1e-5 if opts.no_norm else float(model.transformer.norm.eps),
    attention_dir='bidirectional' if opts.not_causal else 'causal',
    # Non-rope position embeddings are already part of the input embedding
    positional_embedding_type='rotary' if rope else 'standard',
    rotary_dim=dh if rope else None, rotary_base=opts.sin_time, rotary_adjacent_pairs=True,
    default_prepend_bos=False, device=str(device))
  tl = EmbeddingInputTransformer(cfg)

  sd = {k: t.zeros_like(v) if k.startswith(('embed.', 'pos_embed.')) else v
        for k, v in tl.state_dict().items()}
  def set_norm(name, norm):
    if not opts.no_norm:
      sd[name + '.w'], sd[name + '.b'] = _t(norm.weight), _t(norm.bias)
  for l, block in enumerate(blocks):
    p = 'blocks.{}.'.format(l)
    set_norm(p + 'ln1', block.norm1)
    # qkv rows are ordered (q/k/v, head, d_head), see AttentionBlock
    qkv = np.asarray(block.attn.qkv.weight).reshape(3, H, dh, C).transpose(0, 1, 3, 2)
    qkv_b = (np.zeros((3, H, dh)) if block.attn.qkv.bias is None
             else np.asarray(block.attn.qkv.bias).reshape(3, H, dh))
    for i, s in enumerate('QKV'):
      sd[p + 'attn.W_' + s], sd[p + 'attn.b_' + s] = _t(qkv[i]), _t(qkv_b[i])
    # proj columns are ordered (head, d_head)
    sd[p + 'attn.W_O'] = _t(np.asarray(block.attn.proj.weight).reshape(C, H, dh).transpose(1, 2, 0))
    sd[p + 'attn.b_O'] = t.zeros(C)
    if not attn_only:
      set_norm(p + 'ln2', block.norm2)
      sd[p + 'mlp.W_in'], sd[p + 'mlp.b_in'] = _t(block.mlp.fc1.weight.T), _t(block.mlp.fc1.bias)
      sd[p + 'mlp.W_out'], sd[p + 'mlp.b_out'] = _t(block.mlp.fc2.weight.T), _t(block.mlp.fc2.bias)
  set_norm('ln_final', model.transformer.norm)
  sd['unembed.W_U'], sd['unembed.b_U'] = _t(unembed.weight.T), _t(unembed.bias)
  tl.load_state_dict({k: v.to(device) for k, v in sd.items()})

  # Same setup as auto-circuit's load_tl_model: separate head outputs and q/k/v inputs
  tl.cfg.use_attn_result = True
  tl.cfg.use_split_qkv_input = True
  tl.cfg.use_hook_mlp_in = not attn_only
  tl.eval()
  tl.requires_grad_(False)
  return tl


@eqx.filter_jit
def _jax_fwd(model, x, y):
  keys = jax.random.split(jax.random.PRNGKey(0), x.shape[0])
  r = jax.vmap(partial(opto.default_model_fwd_fn, model=model))(x=x, y=y, key=keys)
  return r['embedding'], r['out'], r.get('symbol_out')


def _embed(model, x, y):
  '''Input embeddings, value logits and symbol logits (or None) from the JAX model.'''
  emb, out, symbol_out = _jax_fwd(model, x, y)
  return _t(emb), _t(out), None if symbol_out is None else _t(symbol_out)


def _swap_labels(labels, a, b):
  '''Swaps label ids a and b (per row) at every position.'''
  a, b = a[:, None], b[:, None]
  return jnp.where(labels == a, b, jnp.where(labels == b, a, labels))


def _swap_exemplars(x, symbols, a, b):
  '''Swaps the exemplars of classes a and b (per row) at every position but the last.'''
  rows = jnp.arange(len(x))
  x_a = x[rows, jnp.argmax(symbols == a[:, None], axis=-1)]
  x_b = x[rows, jnp.argmax(symbols == b[:, None], axis=-1)]
  before_last = (jnp.arange(symbols.shape[1]) < symbols.shape[1] - 1)[None, :, None]
  is_a = (symbols == a[:, None])[..., None] & before_last
  is_b = (symbols == b[:, None])[..., None] & before_last
  return jnp.where(is_a, x_b[:, None], jnp.where(is_b, x_a[:, None], x))


def value_at_query(opts):
  '''Whether the value targets read the query's value instead of the next symbol's.

  Only for the extended variant with next_symbol_source=global: its next symbol
  is usually not in context and then gets a fresh label, so its value cannot be
  copied in-context (the model learns to avoid the context labels there). The
  query's value can, as in the base model and the fsl evaluators.
  '''
  return opts.model_variant == 'extended' and getattr(opts, 'next_symbol_source', 'context') == 'global'


def value_pairs(model, prompts, extended, corruption, at_query=False):
  '''Clean/corrupt pairs for the value prediction at the final position.

  at_query (extended only): cut the sequence after the query, making it the
  final position; the model is causal, so this leaves its activations there
  unchanged. Otherwise the extended variant's final position is the next symbol.

  Both corruptions rebind the target symbol to another context value, so
  copying in-context gives the wrong answer (the other value):
    exemplar_swap: swaps the exemplars of the target's class and the other
      class before the final position, keeping the labels in place. Changes
      which position the induction head should attend to (Q/K path).
    label_swap: swaps the two label ids everywhere. Changes what the attended
      position carries (V path).
  '''
  x, y, s = prompts['examples'], prompts['labels'], prompts['symbols']
  if extended and at_query:
    x, y, s, extended = x[:, :-1], y[:, :-1], s[:, :-1], False
  n_ctx = y.shape[1] - (2 if extended else 1)
  ctx = y[:, :n_ctx]
  answer = y[:, -1]
  differs = ctx != answer[:, None]
  other = jnp.argmax(differs, axis=-1)
  wrong = ctx[jnp.arange(len(ctx)), other]
  keep = jnp.any(differs, axis=-1) & jnp.any(ctx == answer[:, None], axis=-1)
  if corruption == 'exemplar_swap':
    target_class, other_class = s[:, -1], s[jnp.arange(len(s)), other]
    keep &= jnp.any(s[:, :n_ctx] == target_class[:, None], axis=-1) & (target_class != other_class)
    x_corrupt, y_corrupt = _swap_exemplars(x, s, target_class, other_class), y
  else:
    x_corrupt, y_corrupt = x, _swap_labels(y, answer, wrong)
  keep = np.array(keep)
  clean, out, _ = _embed(model, x, y)
  corrupt, _, _ = _embed(model, x_corrupt, y_corrupt)
  return dict(clean=clean[keep], corrupt=corrupt[keep], jax_logits=out[keep, -1],
              answers=t.tensor(np.array(answer)[keep, None]),
              wrong_answers=t.tensor(np.array(wrong)[keep, None]),
              symbols=np.asarray(s)[keep])


def symbol_pairs(model, prompts, data, exemplar_inds, num_train_classes, seed):
  '''Clean/corrupt pairs for the extended variant's next-symbol prediction (at lq).

  The sequence is cut after lq, so lq is the final position; the model is
  causal, so this leaves its activations unchanged. Corrupt keeps the labels
  but replaces every symbol with an exemplar of a different train class.
  Answers are the context symbols, wrong answers their replacements.
  '''
  x, y = prompts['examples'], prompts['labels']
  symbols = np.asarray(prompts['symbols'])
  rng = np.random.default_rng(seed)
  new = symbols.copy()
  for i, row in enumerate(symbols):
    present = np.unique(row)
    choices = rng.permutation(np.setdiff1d(np.arange(num_train_classes), present))[:len(present)]
    new[i] = choices[np.searchsorted(present, row)]
  exemplars = np.asarray(exemplar_inds)[rng.integers(0, len(exemplar_inds), new.shape)]
  clean, _, symbol_out = _embed(model, x, y)
  corrupt, _, _ = _embed(model, data[new, exemplars], y)
  n_ctx = symbols.shape[1] - 2
  return dict(clean=clean[:, :-1], corrupt=corrupt[:, :-1], jax_logits=symbol_out[:, -1],
              answers=t.tensor(symbols[:, :n_ctx]), wrong_answers=t.tensor(new[:, :n_ctx]),
              symbols=symbols[:, :-1])


def _loaders(pairs, bs):
  '''Splits pairs in half: one loader for discovery, one for faithfulness.'''
  n = len(pairs['clean']) // 2
  bs = min(bs, n)
  def loader(sl):
    ds = PromptDataset(pairs['clean'][sl], pairs['corrupt'][sl],
                       list(pairs['answers'][sl]), list(pairs['wrong_answers'][sl]))
    return PromptDataLoader(ds, seq_len=None, diverge_idx=0, batch_size=bs, shuffle=False)
  return loader(slice(0, n)), loader(slice(n, 2 * n))


def discover(tl, pairs, bs, device):
  '''Runs mask-gradient edge attribution and measures circuit faithfulness.'''
  train_loader, test_loader = _loaders(pairs, bs)
  model = patchable_model(tl, factorized=True, slice_output='last_seq',
                          separate_qkv=True, device=device)
  scores = mask_gradient_prune_scores(
    model=model, dataloader=train_loader, official_edges=None,
    grad_function='logit', answer_function='avg_diff', mask_val=0.0)
  counts = edge_counts_util(model.edges, prune_scores=scores, zero_edges=True, all_edges=True)
  # Tree patching keeps the top-k edges clean and ablates the rest
  outs = run_circuits(model, test_loader, counts, scores,
                      patch_type=PatchType.TREE_PATCH, ablation_type=AblationType.RESAMPLE)
  faithfulness = dict(measure_answer_diff(model, test_loader, outs))
  return model, scores, faithfulness


def acdc(model, loader, threshold):
  '''ACDC (Conmy et al., 2023) on auto-circuit's patchable model.

  Follows the reference implementation: starting from the full model, edges are
  visited in reverse topological order. Destinations run from the output back,
  and each destination's sources from later layers and higher head indices to
  earlier ones. Each edge is ablated by patching in its corrupt
  activation. The ablation is kept, removing the edge, if it raises the KL
  divergence of the output from the full model's by less than `threshold`;
  otherwise the edge is restored and stays in the circuit. Uses the first
  batch of `loader` only.

  Returns {edge name: KL increase when the edge was tried}, the names of the
  kept edges and the KL divergence of the final circuit.
  '''
  batch = next(iter(loader))
  patch_outs = src_ablations(model, batch.corrupt, AblationType.RESAMPLE)
  with t.inference_mode():
    clean_logprobs = t.log_softmax(model(batch.clean)[model.out_slice], dim=-1)
  edges = sorted(model.edges, key=lambda e: (-e.dest.layer, -(e.dest.head_idx or 0), e.dest.name,
                                             -e.src.layer, -(e.src.head_idx or 0)))
  effects, kept, kl = {}, [], 0.0
  set_all_masks(model, val=0.0)
  with patch_mode(model, patch_outs):
    for edge in edges:
      edge.patch_mask(model).data[edge.patch_idx] = 1.0
      with t.inference_mode():
        out = model(batch.clean)[model.out_slice]
      new_kl = multibatch_kl_div(t.log_softmax(out, dim=-1), clean_logprobs).item()
      effects[edge.name] = new_kl - kl
      if new_kl - kl < threshold:
        kl = new_kl
      else:
        edge.patch_mask(model).data[edge.patch_idx] = 0.0
        kept.append(edge.name)
  set_all_masks(model, val=0.0)
  return effects, kept, kl


def circuit_logit_diffs(model, loader, kept):
  '''Logit differences of the fully ablated model, of only the `kept` edges and of the full model.'''
  scores = model.new_prune_scores()
  for edge in model.edges:
    if edge.name in kept:
      scores[edge.dest.module_name][edge.patch_idx] = 1.0
  counts = sorted({0, len(kept), len(model.edges)})
  outs = run_circuits(model, loader, counts, scores,
                      patch_type=PatchType.TREE_PATCH, ablation_type=AblationType.RESAMPLE)
  diffs = dict(measure_answer_diff(model, loader, outs))
  return diffs[0], diffs[len(kept)], diffs[len(model.edges)]


def discover_acdc(model, pairs, threshold, sweep=ACDC_SWEEP):
  '''Runs ACDC at every threshold in `sweep` and at `threshold`, the reported circuit.

  The circuit is found on the discovery half of the pairs, in one batch as in
  the reference implementation, and its faithfulness measured on the other half.
  '''
  discovery_loader, test_loader = _loaders(pairs, len(pairs['clean']))
  runs = {}
  for tau in sorted(set(sweep) | {threshold}):
    effects, kept, kl = acdc(model, discovery_loader, tau)
    ablated, circuit, full = circuit_logit_diffs(model, test_loader, kept)
    runs[tau] = dict(threshold=tau, n_edges=len(kept), kl_divergence=kl,
                     logit_diff_ablated=ablated, logit_diff_circuit=circuit, logit_diff_full=full,
                     recovery=(circuit - ablated) / (full - ablated), effects=effects, kept=kept)
  return runs


def _summarise_acdc(model, runs, threshold):
  '''Edge list and faithfulness summary of the reported ACDC circuit.'''
  run = runs[threshold]
  kept = set(run['kept'])
  edges = sorted(({'edge': e.name, 'src': e.src.name, 'dest': e.dest.name,
                   'score': run['effects'][e.name], 'in_circuit': e.name in kept} for e in model.edges),
                 key=lambda e: (not e['in_circuit'], -abs(e['score'])))
  scalars = {k: run[k] for k in ['threshold', 'n_edges', 'kl_divergence', 'recovery', 'logit_diff_full',
                                 'logit_diff_ablated', 'logit_diff_circuit']}
  sweep = [{k: v for k, v in r.items() if k not in ('effects', 'kept')} for r in runs.values()]
  return edges, scalars, sweep


def _summarise(model, scores, faithfulness):
  edges = sorted(({'edge': e.name, 'src': e.src.name, 'dest': e.dest.name,
                   'score': e.prune_score(scores).item()} for e in model.edges),
                 key=lambda e: -abs(e['score']))
  counts = sorted(faithfulness)
  ablated, full = faithfulness[counts[0]], faithfulness[counts[-1]]
  recovery = {k: (faithfulness[k] - ablated) / (full - ablated) for k in counts}
  n_recover = next((k for k in counts if recovery[k] >= RECOVERY_TARGET), counts[-1])
  scalars = {'logit_diff_full': full, 'logit_diff_ablated': ablated,
             'n_edges': len(edges), 'n_edges_to_recover': n_recover}
  curve = [{'n_edges': k, 'logit_diff': faithfulness[k], 'recovery': recovery[k]} for k in counts]
  return edges, curve, scalars


def run_circuit_discovery(model, opts, prompts, out_folder, data, exemplar_inds,
                          num_train_classes, gen=0, h5_file=None):
  '''Discovers and saves circuits of `model` on the fixed real `prompts`.

  Targets: the final value under both corruptions (value_exemplar_swap,
  value_label_swap) and, for the extended variant, the next symbol
  (next_symbol). Writes {target}_prune_scores.pt, _edges.json,
  _faithfulness.json and _graph.html for edge attribution patching and
  {target}_acdc_edges.json and _acdc_faithfulness.json for ACDC to
  out_folder, and appends summary scalars (one row per generation) to h5_file
  if given.
  '''
  os.makedirs(out_folder, exist_ok=True)
  # CPU even with a GPU: auto-circuit keeps its answer tensors on the CPU, and the
  # report's circuits were found on the CPU.
  device = t.device('cpu')
  extended = opts.model_variant == 'extended'
  bs = getattr(opts, 'circuit_bs', 64)
  acdc_threshold = getattr(opts, 'circuit_acdc_threshold', ACDC_THRESHOLD)
  # target name -> (output head, pair builder)
  targets = {'value_' + c: ('label', partial(value_pairs, model, prompts, extended, c, value_at_query(opts)))
             for c in ['exemplar_swap', 'label_swap']}
  if extended:
    targets['next_symbol'] = ('symbol', partial(symbol_pairs, model, prompts, data, exemplar_inds,
                                                num_train_classes, getattr(opts, 'circuit_seed', 7)))
  all_scalars = {}
  for target, (head, make_pairs) in targets.items():
    pairs = make_pairs()
    tl = to_hooked_transformer(model, opts, head=head, device=device)
    with t.no_grad():
      tl_logits = tl(pairs['clean'][:bs].to(device))[:, -1].cpu()
    max_diff = (tl_logits - pairs['jax_logits'][:bs]).abs().max().item()
    if max_diff > EQUIVALENCE_TOL:
      raise RuntimeError('Converted {} model disagrees with JAX: max logit diff {:.2e}'.format(target, max_diff))

    model_tl, scores, faithfulness = discover(tl, pairs, bs, device)
    edges, curve, scalars = _summarise(model_tl, scores, faithfulness)
    scalars['conversion_max_diff'] = max_diff
    scalars['n_prompts'] = len(pairs['clean'])

    prefix = os.path.join(out_folder, target)
    t.save({k: v.cpu() for k, v in scores.items()}, prefix + '_prune_scores.pt')
    with open(prefix + '_edges.json', 'w') as f:
      json.dump(edges, f, indent='\t')
    with open(prefix + '_faithfulness.json', 'w') as f:
      json.dump({'summary': scalars, 'curve': curve}, f, indent='\t')
    # Draw the recovering circuit, capped so the graph stays readable
    n_shown = min(max(scalars['n_edges_to_recover'], 5), MAX_DRAWN_EDGES, len(edges))
    threshold = abs(edges[n_shown - 1]['score']) * (1 - 1e-6)
    fig = draw_seq_graph(model_tl, scores, score_threshold=threshold, layer_spacing=True,
                         display_ipython=False)
    fig.write_html(prefix + '_graph.html')
    # Two publication-friendly static views of the same selected edge set.
    # Import lazily so runs without circuit discovery do not load matplotlib.
    from plotting.circuit_plots import render_circuit_views
    render_circuit_views(edges, scalars, prefix, title=target.replace('_', ' ').title(),
                         max_edges=MAX_DRAWN_EDGES)

    print('circuit', target, 'gen', gen, {k: float('{:.4g}'.format(v)) for k, v in scalars.items()})
    print('circuit', target, 'top edges:', ', '.join(
      '{} ({:.3f})'.format(e['edge'], e['score']) for e in edges[:10]))
    all_scalars.update({target + '/' + k: v for k, v in scalars.items()})

    # ACDC on the same graph and prompt pairs
    acdc_runs = discover_acdc(model_tl, pairs, acdc_threshold)
    acdc_edges, acdc_scalars, acdc_sweep = _summarise_acdc(model_tl, acdc_runs, acdc_threshold)
    with open(prefix + '_acdc_edges.json', 'w') as f:
      json.dump(acdc_edges, f, indent='\t')
    with open(prefix + '_acdc_faithfulness.json', 'w') as f:
      json.dump({'summary': acdc_scalars, 'sweep': acdc_sweep}, f, indent='\t')
    render_circuit_views(acdc_edges, acdc_scalars, prefix + '_acdc',
                         title=target.replace('_', ' ').title() + ' (ACDC)', max_edges=MAX_DRAWN_EDGES)
    print('circuit', target, 'ACDC sweep:', ', '.join(
      'tau={} {} edges {:.0%}'.format(r['threshold'], r['n_edges'], r['recovery']) for r in acdc_sweep))
    print('circuit', target, 'ACDC edges:', ', '.join(e['edge'] for e in acdc_edges if e['in_circuit']))
    all_scalars.update({target + '/acdc_' + k: v for k, v in acdc_scalars.items()})

  if h5_file is not None:
    generations.append_to_h5(h5_file, gen, all_scalars, {})
  return all_scalars


def load_checkpoint(checkpoint, run_config):
  '''Loads the model of a saved checkpoint, as in main.run_with_opts.'''
  argv, sys.argv = sys.argv, sys.argv[:1]  # get_opts_from_json_file parses sys.argv
  try:
    opts = main_utils.get_opts_from_json_file(run_config)
  finally:
    sys.argv = argv
  model = main_utils.get_model_from_opts(opts)
  opt_state = main_utils.get_optimizer_from_opts(opts).init(eqx.filter(model, eqx.is_array))
  key = jax.random.PRNGKey(0)
  ckpt_fmt = {'iter': -1,
              'seeds': {'eval_model_seed': key, 'train_data_seed': key, 'train_model_seed': key},
              'opt_state': opt_state, 'model': model}
  return opts, eqx.tree_deserialise_leaves(checkpoint, ckpt_fmt)['model']


def real_prompts(opts, data, splits, n, seed):
  '''Draws a fixed set of n real training sequences.'''
  from src.main import make_train_data_sampler
  return generations.build_pool(make_train_data_sampler(opts, data, splits), n, jax.random.PRNGKey(seed))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Discover circuits of a saved checkpoint')
  parser.add_argument('--checkpoint', required=True, type=str)
  parser.add_argument('--run_config', required=True, type=str, help="The run's config.json")
  parser.add_argument('--out_folder', required=True, type=str)
  parser.add_argument('--n_prompts', default=512, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--seed', default=7, type=int)
  parser.add_argument('--acdc_threshold', default=ACDC_THRESHOLD, type=float)
  args = parser.parse_args()

  opts, model = load_checkpoint(args.checkpoint, args.run_config)
  opts.circuit_bs, opts.circuit_seed, opts.circuit_acdc_threshold = args.batch_size, args.seed, args.acdc_threshold
  data = main_utils.get_data_from_opts(opts)
  splits = main_utils.get_splits_from_opts(opts, data.shape)
  prompts = real_prompts(opts, data, splits, args.n_prompts, args.seed)
  run_circuit_discovery(model, opts, prompts, args.out_folder, data, splits['exemplar']['train'],
                        opts.class_split[0])
