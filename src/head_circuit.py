'''
Which attention heads talk to each other: causal tests of the induction circuit.

Circuit discovery (src/circuits.py) ranks edges by attribution and finds ACDC
circuits. This module tests the candidate circuit with interventions on the
same converted model, prompt pairs and logit difference D:

  1. head ablation: every head's outgoing edges zeroed, mean-ablated or
     resample-ablated, with the change in D and in accuracy;
  2. edge matrix: each single edge into a layer-1 head's Q, K or V input
     carries its corrupt activation inside the clean run (noising), giving
     its share of the clean-corrupt gap and the change in the receiver's
     retrieval, next to weight-based composition scores;
  3. the previous-token head P -> induction head I connection, per channel:
     disrupted alone, and restored alone after every input to I.X is
     corrupted, with controls for the other heads;
  4. the candidate circuit kept clean while every other edge is patched from
     the corrupt run: recovery R and accuracy.

P and I come from attention scores on the clean prompts, not from a fixed
list. Retrieval is the attention from the final position to the label after
the matching exemplar minus the attention to the other labels. See
docs/head_circuit.md.

Can also be run on a saved checkpoint:
  python -m src.head_circuit --checkpoint <ckpt.eqx> --run_config <config.json> --out_folder <dir>
'''

import argparse
import json
import os
from functools import partial

import numpy as np
import jax
import torch as t

from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, set_all_masks

from src import circuits
from src import generations
from src import main_utils

CHANNELS = ['Q', 'K', 'V']
# Edges whose noising effect reaches this share of the clean-corrupt gap join the extended circuit
EDGE_EFFECT_THRESHOLD = 0.02
# Heads with a previous-token (layer 0) or induction (layer 1) score above this take that role
ROLE_SCORE = 0.5
# Restoring an edge is only scored when corrupting the receiver's input costs this share of the gap
MIN_INPUT_DAMAGE = 0.05
# Below this clean-corrupt logit difference gap (e.g. an untrained model) ratios are not meaningful
MIN_GAP = 1.0
ABLATIONS = ['zero', 'mean', 'resample']


def patchable(tl, device):
  '''auto-circuit's patchable graph of `tl`, as in circuits.discover, with hooks that keep
  the attention patterns, the last layer's head outputs and the final LayerNorm scale.

  patchable_model wraps tl's modules in place, so build it once per converted model.
  '''
  model = patchable_model(tl, factorized=True, slice_output='last_seq', separate_qkv=True, device=device)
  model.cache = {}
  def store(name, module, inputs, output):
    model.cache[name] = output.detach()
  for l in range(tl.cfg.n_layers):
    tl.hook_dict['blocks.{}.attn.hook_pattern'.format(l)].register_forward_hook(partial(store, 'pattern{}'.format(l)))
  tl.hook_dict['blocks.{}.attn.hook_result'.format(tl.cfg.n_layers - 1)].register_forward_hook(partial(store, 'result'))
  tl.hook_dict['ln_final.hook_scale'].register_forward_hook(partial(store, 'scale'))
  return model


class Analysis:
  '''One clean/corrupt batch on a patchable model.'''

  def __init__(self, model, tl, pairs, device):
    self.model = model
    self.tl = tl
    self.n_layers, self.n_heads = tl.cfg.n_layers, tl.cfg.n_heads
    self.clean = pairs['clean'].to(device)
    self.corrupt = pairs['corrupt'].to(device)
    self.answers = pairs['answers'].to(device)
    self.wrong = pairs['wrong_answers'].to(device)
    self.symbols = np.asarray(pairs['symbols'])
    self.edges = {e.name: e for e in self.model.edges}
    clean_acts = src_ablations(self.model, self.clean, AblationType.RESAMPLE)
    self.patch = {
      'clean': clean_acts,
      'resample': src_ablations(self.model, self.corrupt, AblationType.RESAMPLE),
      'zero': t.zeros_like(clean_acts),
      'mean': clean_acts.mean(1, keepdim=True).expand_as(clean_acts).contiguous(),
    }

  def run(self, patched=(), ablation='resample', on_corrupt=False):
    '''Forward pass with the edges in `patched` taking the ablation values.

    on_corrupt runs the corrupt batch instead and patches from the clean run, so
    `patched` edges are restored to their clean activations.
    '''
    set_all_masks(self.model, val=0.0)
    with patch_mode(self.model, self.patch['clean' if on_corrupt else ablation]), t.inference_mode():
      for name in patched:
        e = self.edges[name]
        e.patch_mask(self.model).data[e.patch_idx] = 1.0
      out = self.model(self.corrupt if on_corrupt else self.clean)[self.model.out_slice]
    set_all_masks(self.model, val=0.0)
    return self.measure(out)

  def measure(self, out):
    '''Logit difference D, accuracy and the cached attention of this pass.'''
    right = out.gather(-1, self.answers).mean(-1)
    wrong = out.gather(-1, self.wrong).mean(-1)
    pred = out.argmax(-1, keepdim=True)
    return {'logit_diff': (right - wrong).mean().item(),
            'accuracy': (pred == self.answers).any(-1).float().mean().item(),
            'patterns': dict(self.model.cache)}

  # --- attention scores ---

  def label_layout(self):
    '''Label positions and, per prompt, which of them follow the final exemplar's class.'''
    L = self.clean.shape[1]
    n_pairs = min(self.symbols.shape[1] - 1, (L - 1) // 2)
    labels = 2 * np.arange(n_pairs) + 1
    final_is_exemplar = (L - 1) % 2 == 0
    match = self.symbols[:, :n_pairs] == self.symbols[:, [-1]] if final_is_exemplar else None
    return labels, match

  def previous_token_scores(self, patterns):
    '''Layer-0 attention from each label to its exemplar, per head.'''
    labels, _ = self.label_layout()
    p = patterns['pattern0']  # (batch, head, query, key)
    return p[:, :, labels, labels - 1].mean((0, 2)).cpu().numpy()

  def retrieval(self, patterns, layer=None):
    '''Attention from the final position to the matching label minus to the other labels.

    Per head of `layer` (default: last), or None when the final position is not an exemplar.
    '''
    layer = self.n_layers - 1 if layer is None else layer
    labels, match = self.label_layout()
    if match is None:
      return None
    keep = match.any(1) & (~match).any(1)
    if not keep.any():
      return None
    a = patterns['pattern{}'.format(layer)][:, :, -1, :][:, :, labels].cpu().numpy()[keep]  # (b, head, pairs)
    m = match[keep][:, None, :]
    correct = (a * m).sum(-1) / m.sum(-1)
    other = (a * ~m).sum(-1) / (~m).sum(-1)
    return (correct - other).mean(0)

  def direct_effect(self, patterns, head):
    '''Direct logit attribution of a last-layer head: its output through the final LayerNorm and unembedding.'''
    res = patterns['result'][:, -1, head]  # (batch, d_model)
    scale = patterns['scale'][:, -1]
    w = self.tl.W_U
    proj = (res / scale) @ w  # (batch, vocab)
    return (proj.gather(-1, self.answers).mean(-1) - proj.gather(-1, self.wrong).mean(-1)).mean().item()


def edge_name(src, dest_head, channel, layer=1):
  return '{}->A{}.{}.{}'.format(src, layer, dest_head, channel)


def composition_scores(tl):
  '''Q, K and V composition (Elhage et al., 2021) of every layer-0 head into every layer-1 head.

  The layer-1 LayerNorm weight is folded into the receiving weights; its
  centring and scaling by the input norm are ignored.
  '''
  g = tl.blocks[1].ln1.w.detach() if hasattr(tl.blocks[1], 'ln1') and hasattr(tl.blocks[1].ln1, 'w') else None
  def fold(w):
    return w if g is None else g[:, None] * w
  WV0, WO0 = tl.W_V[0].detach(), tl.W_O[0].detach()
  WQ1, WK1, WV1, WO1 = (x[1].detach() for x in (tl.W_Q, tl.W_K, tl.W_V, tl.W_O))
  H = tl.cfg.n_heads
  out = {c: np.zeros((H, H)) for c in CHANNELS}
  norm = lambda m: t.linalg.matrix_norm(m).item()
  for i in range(H):
    ov_i = WV0[i] @ WO0[i]
    for j in range(H):
      qk_j = fold(WQ1[j]) @ fold(WK1[j]).T
      ov_j = fold(WV1[j]) @ WO1[j]
      for c, m, other in [('Q', ov_i @ qk_j, qk_j), ('K', qk_j @ ov_i.T, qk_j), ('V', ov_i @ ov_j, ov_j)]:
        out[c][i, j] = norm(m) / (norm(ov_i) * norm(other))
  return {c: v.tolist() for c, v in out.items()}


def gap_share(clean, corrupt, value):
  '''Share of the clean-corrupt gap that `value` has lost: 0 = clean, 1 = corrupt.'''
  return (clean - value) / (clean - corrupt)


def recovery(clean, corrupt, value):
  '''R: share of the clean-corrupt gap that `value` restores: 0 = corrupt, 1 = clean.'''
  return (value - corrupt) / (clean - corrupt)


def analyse_target(tl, pairs, eval_pairs, acdc_kept, device):
  '''All four experiments for one target. Returns a JSON-able dict.'''
  n = len(pairs['clean']) // 2
  half = lambda sl: {k: v[sl] for k, v in pairs.items() if k != 'jax_logits'}
  model = patchable(tl, device)
  disc, test = Analysis(model, tl, half(slice(0, n)), device), Analysis(model, tl, half(slice(n, 2 * n)), device)
  H = disc.n_heads
  heads = ['A{}.{}'.format(l, h) for l in range(disc.n_layers) for h in range(H)]
  senders = ['Resid Start'] + ['A0.{}'.format(h) for h in range(H)]
  all_edges = list(disc.edges)

  def base_runs(a):
    clean = a.run()
    corrupt = a.run(all_edges)
    return clean, corrupt
  out = {'n_prompts': {'discovery': n, 'test': n}}

  # Head roles on clean prompts (test half)
  clean_t, corrupt_t = base_runs(test)
  prev = test.previous_token_scores(clean_t['patterns'])
  ind = test.retrieval(clean_t['patterns'])
  P = int(np.argmax(prev))
  I = int(np.argmax(ind)) if ind is not None else None
  out['clean'] = {k: clean_t[k] for k in ('logit_diff', 'accuracy')}
  out['corrupt'] = {k: corrupt_t[k] for k in ('logit_diff', 'accuracy')}
  out['previous_token_scores'] = prev.tolist()
  out['induction_scores'] = None if ind is None else ind.tolist()
  Dc, Dx = clean_t['logit_diff'], corrupt_t['logit_diff']
  out['well_conditioned'] = Dc - Dx >= MIN_GAP

  # 1. Head ablation: every edge leaving the head
  def outgoing(head):
    return [e for e in all_edges if test.edges[e].src.name == head]
  ablation = {}
  for head in heads:
    ablation[head] = {}
    for kind in ABLATIONS:
      r = test.run(outgoing(head), ablation=kind)
      ablation[head][kind] = {'logit_diff': r['logit_diff'], 'accuracy': r['accuracy'],
                              'gap_share': gap_share(Dc, Dx, r['logit_diff'])}
  if eval_pairs is not None:
    ev = Analysis(model, tl, {k: v for k, v in eval_pairs.items() if k != 'jax_logits'}, device)
    ev_clean = ev.run()
    out['eval_clean'] = {k: ev_clean[k] for k in ('logit_diff', 'accuracy')}
    for head in heads:
      for kind in ABLATIONS:
        r = ev.run([e for e in all_edges if ev.edges[e].src.name == head], ablation=kind)
        ablation[head][kind]['eval_accuracy'] = r['accuracy']
        ablation[head][kind]['eval_logit_diff'] = r['logit_diff']
  out['head_ablation'] = ablation

  # 2. Edge matrix (discovery half): noise one edge into a layer-1 head's input
  clean_d, corrupt_d = base_runs(disc)
  Dc_d, Dx_d = clean_d['logit_diff'], corrupt_d['logit_diff']
  ret_clean_d = disc.retrieval(clean_d['patterns'])
  matrix = {c: np.zeros((len(senders), H)) for c in CHANNELS}
  ret_drop = {c: np.full((len(senders), H), np.nan) for c in CHANNELS}
  for c in CHANNELS:
    for si, s in enumerate(senders):
      for j in range(H):
        r = disc.run([edge_name(s, j, c)])
        matrix[c][si, j] = gap_share(Dc_d, Dx_d, r['logit_diff'])
        ret = disc.retrieval(r['patterns'])
        if ret is not None:
          ret_drop[c][si, j] = ret_clean_d[j] - ret[j]
  out['edge_matrix'] = {'senders': senders, 'receivers': ['A1.{}'.format(j) for j in range(H)],
                        'gap_share': {c: m.tolist() for c, m in matrix.items()},
                        'retrieval_drop': {c: np.where(np.isnan(m), None, m).tolist() for c, m in ret_drop.items()}}
  out['composition'] = composition_scores(tl)

  # The strongest head-to-head edge, as found by noising
  hh = [(matrix[c][si, j], 'A0.{}'.format(si - 1), j, c) for c in CHANNELS for si in range(1, len(senders)) for j in range(H)]
  top = max(hh)
  out['top_head_edge'] = {'edge': edge_name(top[1], top[2], top[3]), 'gap_share': top[0]}

  # 3. P -> I, per channel, on the test half
  pairs_to_test = []
  if I is not None:
    pairs_to_test.append(('A0.{}'.format(P), I))
  top_pair = (top[1], top[2])
  if top_pair not in pairs_to_test:
    pairs_to_test.append(top_pair)
  ret_clean_t = test.retrieval(clean_t['patterns'])

  def summary(r, receiver):
    d = {'logit_diff': r['logit_diff'], 'accuracy': r['accuracy'],
         'gap_share': gap_share(Dc, Dx, r['logit_diff']),
         'direct_effect': test.direct_effect(r['patterns'], receiver)}
    ret = test.retrieval(r['patterns'])
    d['retrieval'] = None if ret is None else float(ret[receiver])
    return d

  connection = {}
  for sender, receiver in pairs_to_test:
    key = '{}->A1.{}'.format(sender, receiver)
    res = {'sender': sender, 'receiver': 'A1.{}'.format(receiver),
           'clean': summary(clean_t, receiver), 'corrupt': summary(corrupt_t, receiver)}
    for c in CHANNELS:
      target = edge_name(sender, receiver, c)
      into = [edge_name(s, receiver, c) for s in senders]
      disrupt = summary(test.run([target]), receiver)
      inputs_corrupt = summary(test.run(into), receiver)
      restored = summary(test.run([e for e in into if e != target]), receiver)
      # On the corrupt prompt, only this edge carries the sender's clean output
      restored_alone = summary(test.run([target], on_corrupt=True), receiver)
      span = inputs_corrupt['logit_diff'] - Dc
      res[c] = {
        'disrupt': disrupt, 'input_corrupt': inputs_corrupt, 'restore_in_input': restored,
        'restore_alone': restored_alone,
        # Share of the damage from corrupting all of I.X that restoring this one edge undoes
        'restore_recovery': (None if not out['well_conditioned'] or -span < MIN_INPUT_DAMAGE * (Dc - Dx)
                             else (restored['logit_diff'] - inputs_corrupt['logit_diff']) / -span),
        'restore_alone_recovery': recovery(Dc, Dx, restored_alone['logit_diff']),
      }
      # Specificity: other senders into the same receiver input, and this sender into other receivers
      res[c]['controls'] = {
        'other_senders': {s: gap_share(Dc, Dx, test.run([edge_name(s, receiver, c)])['logit_diff'])
                          for s in senders if s != sender},
        'other_receivers': {'A1.{}'.format(j): gap_share(Dc, Dx, test.run([edge_name(sender, j, c)])['logit_diff'])
                            for j in range(H) if j != receiver},
      }
    connection[key] = res
  out['connections'] = connection
  out['P'], out['I'] = 'A0.{}'.format(P), None if I is None else 'A1.{}'.format(I)

  # 4. Candidate circuit on the test half: kept clean, every other edge resample-patched
  def circuit_result(kept):
    kept = set(kept) & set(all_edges)
    r = test.run([e for e in all_edges if e not in kept])
    d = {'n_edges': len(kept), 'logit_diff': r['logit_diff'], 'accuracy': r['accuracy'],
         'recovery': recovery(Dc, Dx, r['logit_diff'])}
    if eval_pairs is not None:
      re = ev.run([e for e in all_edges if e not in kept])
      d['eval_accuracy'], d['eval_logit_diff'] = re['accuracy'], re['logit_diff']
    return d, sorted(kept)
  def induction_circuit(prev_heads, ind_heads):
    '''Previous-token heads read their inputs and feed the induction heads' keys;
    induction heads read query and value from the input and write to the output.'''
    edges = ['Resid Start->Resid End']
    edges += [edge_name('Resid Start', p, c, layer=0) for p in prev_heads for c in CHANNELS]
    edges += [edge_name('A0.{}'.format(p), i, 'K') for p in prev_heads for i in ind_heads]
    edges += [edge_name('Resid Start', i, c) for i in ind_heads for c in ('Q', 'V')]
    edges += ['A1.{}->Resid End'.format(i) for i in ind_heads]
    return edges
  def with_support(edges):
    '''Adds each layer-0 sender's inputs and each layer-1 receiver's output edge.'''
    heads = {e.split('->')[0] for e in edges if e.startswith('A0.')}
    heads |= {e.split('->')[1].rsplit('.', 1)[0] for e in edges if '->A' in e}
    out_ = set(edges) | {'Resid Start->Resid End'}
    for h in heads:
      l, i = (int(x) for x in h[1:].split('.'))
      if l == 0:
        out_ |= {edge_name('Resid Start', i, c, layer=0) for c in CHANNELS}
      out_.add('{}->Resid End'.format(h))
    return sorted(out_)
  prev_set = [h for h in range(H) if prev[h] >= ROLE_SCORE] or [P]
  ind_set = ([] if ind is None else [h for h in range(H) if ind[h] >= ROLE_SCORE]) or ([] if I is None else [I])
  strong = [edge_name(senders[si], j, c) for c in CHANNELS for si in range(len(senders)) for j in range(H)
            if matrix[c][si, j] >= EDGE_EFFECT_THRESHOLD]
  out['roles'] = {'previous_token_heads': ['A0.{}'.format(h) for h in prev_set],
                  'induction_heads': ['A1.{}'.format(h) for h in ind_set]}
  candidates = []
  if I is not None:
    candidates.append(('single_path', induction_circuit([P], [I])))
  if ind_set:
    candidates.append(('induction_heads', induction_circuit(prev_set, ind_set)))
    candidates.append(('induction_heads_plus', with_support(induction_circuit(prev_set, ind_set) + strong)))
  else:
    candidates.append(('strong_edges', with_support(strong)))
  if acdc_kept:
    candidates.append(('acdc', acdc_kept))
  candidates.append(('full', all_edges))
  circuits_out = {}
  for name, kept in candidates:
    circuits_out[name], circuits_out[name + '_edges'] = circuit_result(kept)
  out['circuits'] = circuits_out
  if eval_pairs is not None:
    ev_corrupt = ev.run(all_edges)
    out['eval_corrupt'] = {k: ev_corrupt[k] for k in ('logit_diff', 'accuracy')}
  return out


def h5_row(target, res):
  '''Scalars for one row of head_circuit.h5.'''
  ratio = (lambda v: v) if res['well_conditioned'] else (lambda v: np.nan)
  row = {target + '/P': int(res['P'].split('.')[1]),
         target + '/logit_diff_gap': res['clean']['logit_diff'] - res['corrupt']['logit_diff'],
         target + '/I': -1 if res['I'] is None else int(res['I'].split('.')[1]),
         target + '/P_prev_score': max(res['previous_token_scores']),
         target + '/clean_accuracy': res['clean']['accuracy'],
         target + '/logit_diff_clean': res['clean']['logit_diff']}
  if res['induction_scores'] is not None:
    row[target + '/I_induction_score'] = max(res['induction_scores'])
  key = '{}->{}'.format(res['P'], res['I'])
  if key in res['connections']:
    for c in CHANNELS:
      r = res['connections'][key][c]
      row['{}/PI_{}_disrupt'.format(target, c)] = ratio(r['disrupt']['gap_share'])
      row['{}/PI_{}_restore'.format(target, c)] = ratio(r['restore_recovery']) if r['restore_recovery'] is not None else np.nan
      row['{}/PI_{}_restore_alone'.format(target, c)] = ratio(r['restore_alone_recovery'])
      if r['disrupt']['retrieval'] is not None:
        row['{}/PI_{}_retrieval'.format(target, c)] = r['disrupt']['retrieval']
  for name in ('single_path', 'induction_heads', 'induction_heads_plus', 'strong_edges', 'acdc'):
    if name in res['circuits']:
      row['{}/{}_recovery'.format(target, name)] = ratio(res['circuits'][name]['recovery'])
      row['{}/{}_accuracy'.format(target, name)] = res['circuits'][name]['accuracy']
      if 'eval_accuracy' in res['circuits'][name]:
        row['{}/{}_eval_accuracy'.format(target, name)] = res['circuits'][name]['eval_accuracy']
  return row


def run_head_circuit(model, opts, prompts, eval_prompts, out_folder, data, exemplar_inds,
                     num_train_classes, gen=0, h5_file=None, circuit_folder=None):
  '''Runs the experiments for every target of `model` and saves them to out_folder.

  prompts: the fixed real sequences of circuit discovery. eval_prompts: held-out
  class sequences (the fsl_test_class evaluator) for in-context accuracy, or None.
  ACDC circuits are read from circuit_folder (default: the sibling circuit/
  folder) when present.
  '''
  os.makedirs(out_folder, exist_ok=True)
  # CPU even with a GPU: auto-circuit keeps its answer tensors on the CPU, and the
  # report's circuits were found on the CPU.
  device = t.device('cpu')
  extended = opts.model_variant == 'extended'
  at_query = circuits.value_at_query(opts)
  targets = {'value_' + c: ('label', partial(circuits.value_pairs, model, prompts, extended, c, at_query),
                            None if eval_prompts is None else partial(circuits.value_pairs, model, eval_prompts, extended, c, at_query))
             for c in ['exemplar_swap', 'label_swap']}
  if extended:
    sym = partial(circuits.symbol_pairs, data=data, exemplar_inds=exemplar_inds,
                  num_train_classes=num_train_classes, seed=getattr(opts, 'circuit_seed', 7))
    targets['next_symbol'] = ('symbol', partial(sym, model, prompts), None)
  if circuit_folder is None:
    circuit_folder = os.path.join(os.path.dirname(out_folder.rstrip('/')), 'circuit')
  row = {}
  for target, (head, make_pairs, make_eval) in targets.items():
    tl = circuits.to_hooked_transformer(model, opts, head=head, device=device)
    acdc_file = os.path.join(circuit_folder, target + '_acdc_edges.json')
    acdc_kept = None
    if os.path.exists(acdc_file):
      with open(acdc_file) as f:
        acdc_kept = [e['edge'] for e in json.load(f) if e['in_circuit']]
    res = analyse_target(tl, make_pairs(), None if make_eval is None else make_eval(), acdc_kept, device)
    res['target'] = target
    with open(os.path.join(out_folder, target + '_head_circuit.json'), 'w') as f:
      json.dump(res, f, indent='\t', default=float)
    from plotting.head_circuit_plots import plot_target
    plot_target(res, os.path.join(out_folder, target))
    pi = res['connections'].get('{}->{}'.format(res['P'], res['I']), {})
    print('head circuit', target, 'gen', gen, 'P', res['P'], 'I', res['I'], 'top edge', res['top_head_edge']['edge'],
          '' if res['well_conditioned'] else '(clean-corrupt gap below {}: ratios unreliable)'.format(MIN_GAP),
          ' '.join('{} disrupt {:.0%} restore {}'.format(c, pi[c]['disrupt']['gap_share'],
                   'n/a' if pi[c]['restore_recovery'] is None else '{:.0%}'.format(pi[c]['restore_recovery']))
                   for c in CHANNELS if c in pi),
          ' '.join('{} R {:.0%}'.format(k, v['recovery']) for k, v in res['circuits'].items() if not k.endswith('_edges')))
    row.update(h5_row(target, res))
  if h5_file is not None:
    # A measure missing in this generation (e.g. an ACDC circuit with no edges) is
    # recorded as NaN, so every column keeps one row per generation
    if os.path.exists(h5_file):
      import h5py
      with h5py.File(h5_file, 'r') as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
      row.update({n: np.nan for n in names if n != 'gen' and n not in row})
    generations.append_to_h5(h5_file, gen, row, {})
  return row


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Head-to-head patching experiments on a saved checkpoint')
  parser.add_argument('--checkpoint', required=True, type=str)
  parser.add_argument('--run_config', required=True, type=str, help="The run's config.json")
  parser.add_argument('--out_folder', required=True, type=str)
  parser.add_argument('--n_prompts', default=512, type=int)
  parser.add_argument('--seed', default=7, type=int)
  args = parser.parse_args()

  opts, model = circuits.load_checkpoint(args.checkpoint, args.run_config)
  opts.circuit_seed = args.seed
  data = main_utils.get_data_from_opts(opts)
  splits = main_utils.get_splits_from_opts(opts, data.shape)
  prompts = circuits.real_prompts(opts, data, splits, args.n_prompts, args.seed)
  from src.main import make_eval_data
  # As in main.run_with_opts; config.json stores the integer seed
  eval_data_seed, _ = jax.random.split(jax.random.PRNGKey(opts.eval_seed), 2)
  eval_data = make_eval_data(opts, data, splits, eval_data_seed)
  run_head_circuit(model, opts, prompts, eval_data.get('fsl_test_class'), args.out_folder, data,
                   splits['exemplar']['train'], opts.class_split[0])
