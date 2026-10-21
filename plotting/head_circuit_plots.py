"""Figures for the head-to-head patching experiments (src/head_circuit.py).

plot_target draws one model's figures from its <target>_head_circuit.json
results. plot_generations and plot_conditions read head_circuit.h5 to follow
the P -> I connection across collapse generations and conditions.

  python -m plotting.head_circuit_plots results/<run>/head_circuit   # redraw one model
  python -m plotting.head_circuit_plots --conditions results            # compare main collapse runs
"""
import argparse
import glob
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Same palette as plotting/nlp_project.ipynb
SEQ_CMAP = LinearSegmentedColormap.from_list('attn', ['#fcfcfb', '#9ec5f0', '#2a78d6', '#0d3a73'])
DIV_CMAP = LinearSegmentedColormap.from_list('score', ['#eb6834', '#f3f2ee', '#2a78d6'])
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MUTED = '#c3c2b7'
INK = '#0b0b0b'
CHANNELS = ['Q', 'K', 'V']
CONDITIONS = ['real_only', 'replace50', 'replace90', 'replace100', 'accumulate']
STYLE = {'font.size': 10, 'axes.titlesize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e',
         'ytick.color': '#52514e', 'figure.dpi': 110, 'savefig.bbox': 'tight'}


def _save(fig, prefix, name):
  for ext in ('png', 'pdf'):
    fig.savefig('{}_{}.{}'.format(prefix, name, ext))
  plt.close(fig)


def _annotate(ax, m, fmt='{:.2f}', min_abs=0.05, scale=1.0):
  for (i, j), v in np.ndenumerate(m):
    if v is None or not np.isfinite(v) or abs(v) < min_abs:
      continue
    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7.5,
            color='white' if abs(v) / scale > 0.6 else INK)


def _heads_grid(ax, m, rows, cols, cmap, norm, title):
  im = ax.imshow(m, cmap=cmap, norm=norm, aspect='auto')
  ax.set_xticks(range(len(cols)), cols, fontsize=8)
  ax.set_yticks(range(len(rows)), rows, fontsize=8)
  ax.tick_params(length=0)
  for side in ax.spines.values():
    side.set_visible(False)
  ax.set_title(title)
  return im


def plot_edge_matrix(res, prefix):
  em = res['edge_matrix']
  rows = ['Embed' if s == 'Resid Start' else s for s in em['senders']]
  m = {c: np.array(em['gap_share'][c], dtype=float) for c in CHANNELS}
  comp = {c: np.array(res['composition'][c], dtype=float) for c in CHANNELS}
  vmax = max(0.1, max(np.nanmax(np.abs(v)) for v in m.values()))
  cmax = max(np.nanmax(v) for v in comp.values())
  with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw={'height_ratios': [len(rows), len(rows) - 1]})
    for k, c in enumerate(CHANNELS):
      im = _heads_grid(axs[0, k], m[c], rows, em['receivers'], DIV_CMAP, TwoSlopeNorm(0, -vmax, vmax),
                       'noising one edge into {} input'.format(c))
      _annotate(axs[0, k], m[c], scale=vmax)
      im2 = _heads_grid(axs[1, k], comp[c], rows[1:], em['receivers'], SEQ_CMAP, plt.Normalize(0, cmax),
                        '{}-composition (weights)'.format(c))
      _annotate(axs[1, k], comp[c], min_abs=0.0, scale=cmax)
    fig.colorbar(im, ax=axs[0], shrink=0.8, label='share of clean-corrupt gap lost')
    fig.colorbar(im2, ax=axs[1], shrink=0.8, label='composition score')
    axs[0, 0].set_ylabel('sender')
    axs[1, 0].set_ylabel('layer-0 head')
    fig.suptitle('{}: which layer-0 heads feed which layer-1 heads (P = {}, I = {})'.format(
      res['target'], res['P'], res['I']))
    _save(fig, prefix, 'edge_matrix')


def plot_head_ablation(res, prefix):
  heads = list(res['head_ablation'])
  kinds = ['zero', 'mean', 'resample']
  has_eval = 'eval_accuracy' in res['head_ablation'][heads[0]]['zero']
  with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2 if has_eval else 1, figsize=(14 if has_eval else 8, 3.8), squeeze=False)
    x = np.arange(len(heads))
    w = 0.26
    for k, kind in enumerate(kinds):
      axs[0, 0].bar(x + (k - 1) * w, [res['head_ablation'][h][kind]['gap_share'] for h in heads], w,
                    color=SERIES[k], label=kind)
    axs[0, 0].set_ylabel('share of clean-corrupt gap lost')
    axs[0, 0].set_title('Ablating one head: logit difference')
    if has_eval:
      base = res['eval_clean']['accuracy']
      for k, kind in enumerate(kinds):
        axs[0, 1].bar(x + (k - 1) * w, [base - res['head_ablation'][h][kind]['eval_accuracy'] for h in heads], w,
                      color=SERIES[k], label=kind)
      axs[0, 1].set_ylabel('accuracy drop, held-out classes')
      axs[0, 1].set_title('Ablating one head: in-context accuracy (clean {:.0%})'.format(base))
    for ax in axs[0]:
      ax.set_xticks(x, heads, rotation=90, fontsize=8)
      ax.axhline(0, color=MUTED, lw=1)
      for h, name in enumerate(heads):
        if name in (res['P'], res['I']):
          ax.get_xticklabels()[h].set_fontweight('bold')
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('{}: single-head ablations (bold: P and I)'.format(res['target']))
    fig.tight_layout()
    _save(fig, prefix, 'head_ablation')


def plot_connection(res, prefix):
  for key, con in res['connections'].items():
    steps = [('clean', lambda c: con['clean']), ('corrupt', lambda c: con['corrupt']),
             ('disrupt edge', lambda c: con[c]['disrupt']), ('corrupt all inputs', lambda c: con[c]['input_corrupt']),
             ('…restore edge', lambda c: con[c]['restore_in_input']), ('restore edge only', lambda c: con[c]['restore_alone'])]
    metrics = [('logit_diff', 'logit difference D'), ('accuracy', 'accuracy'), ('retrieval', 'receiver retrieval')]
    if con['clean']['retrieval'] is None:
      metrics = metrics[:2]
    with plt.rc_context(STYLE):
      fig, axs = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.8), squeeze=False)
      x = np.arange(len(steps))
      w = 0.26
      for ax, (m, label) in zip(axs[0], metrics):
        for k, c in enumerate(CHANNELS):
          ax.bar(x + (k - 1) * w, [f(c)[m] if f(c)[m] is not None else np.nan for _, f in steps], w,
                 color=SERIES[k], label='{} input'.format(c))
        ax.set_xticks(x, [s for s, _ in steps], rotation=30, ha='right', fontsize=8)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(label)
      axs[0, 0].legend(frameon=False, fontsize=8)
      fig.suptitle('{}: {} -> {} per input channel'.format(res['target'], con['sender'], con['receiver']))
      fig.tight_layout()
      _save(fig, prefix, 'connection_{}_{}'.format(con['sender'].replace('.', ''), con['receiver'].replace('.', '')))


def plot_circuits(res, prefix):
  names = [n for n in ('single_path', 'induction_heads', 'induction_heads_plus', 'strong_edges', 'acdc', 'full')
           if n in res['circuits']]
  with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    vals = [res['circuits'][n]['recovery'] for n in names]
    axs[0].bar(range(len(names)), vals, color=SERIES[0], width=0.5)
    axs[0].set_xticks(range(len(names)), [n.replace('_', ' ') for n in names], rotation=20, ha='right')
    axs[0].axhline(0.8, color=SERIES[1], lw=1, ls='--')
    axs[0].text(len(names) - 0.5, 0.8, '80%', color=SERIES[1], va='bottom', ha='right', fontsize=8)
    axs[0].set_title('Recovery R of the logit difference')
    for i, (n, v) in enumerate(zip(names, vals)):
      axs[0].text(i, v, '{:.0%}\n{} edges'.format(v, res['circuits'][n]['n_edges']), ha='center', va='bottom', fontsize=8)
    accs = [res['circuits'][n]['accuracy'] for n in names]
    axs[1].bar(np.arange(len(names)) - 0.18, accs, 0.36, color=SERIES[0], label='circuit prompts')
    if 'eval_accuracy' in res['circuits'][names[0]]:
      axs[1].bar(np.arange(len(names)) + 0.18, [res['circuits'][n]['eval_accuracy'] for n in names], 0.36,
                 color=SERIES[2], label='held-out classes')
    axs[1].set_xticks(range(len(names)), [n.replace('_', ' ') for n in names], rotation=20, ha='right')
    axs[1].set_ylim(0, 1)
    axs[1].set_title('Accuracy with only the circuit clean')
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle('{}: candidate circuit, everything else patched from the corrupt run'.format(res['target']))
    fig.tight_layout()
    _save(fig, prefix, 'circuits')


def plot_target(res, prefix):
  plot_edge_matrix(res, prefix)
  plot_head_ablation(res, prefix)
  plot_connection(res, prefix)
  plot_circuits(res, prefix)


def _read_h5(path):
  '''Every dataset, keyed by its full path ('<target>/<metric>' rows are nested groups).'''
  out = {}
  with h5py.File(path, 'r') as f:
    f.visititems(lambda name, obj: out.__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
  return out


def plot_generations(h5_file, prefix, target='value_exemplar_swap'):
  '''P -> I disruption, restoration and circuit recovery over the generations of one run.'''
  d = _read_h5(h5_file)
  gens = d['gen']
  with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for k, c in enumerate(CHANNELS):
      for ax, kind in zip(axs[:2], ('disrupt', 'restore')):
        key = '{}/PI_{}_{}'.format(target, c, kind)
        if key in d:
          ax.plot(gens, d[key], marker='o', color=SERIES[k], label='{} input'.format(c))
    for k, name in enumerate(('single_path', 'induction_heads', 'induction_heads_plus', 'acdc')):
      key = '{}/{}_recovery'.format(target, name)
      if key in d:
        axs[2].plot(gens, d[key], marker='o', color=SERIES[k], label=name)
    axs[2].axhline(0.8, color=MUTED, lw=1, ls='--')
    axs[0].set_title('Disrupting P -> I: share of gap lost')
    axs[1].set_title('Restoring P -> I: share of damage undone')
    axs[2].set_title('Circuit recovery R')
    pi = ['A0.{}->A1.{}'.format(int(p), int(i)) for p, i in zip(d['{}/P'.format(target)], d['{}/I'.format(target)])]
    for ax in axs:
      ax.set_xlabel('generation')
      ax.set_xticks(gens)
      ax.legend(frameon=False, fontsize=8)
    fig.suptitle('{}: P -> I per generation ({})'.format(target, ', '.join(pi)))
    fig.tight_layout()
    _save(fig, prefix, target + '_generations')


def plot_conditions(results_folder, out_folder, variant='base', target='value_exemplar_swap'):
  '''The P -> I.K connection and circuit recovery across generations, one line per main collapse condition.'''
  found = {}
  for c in CONDITIONS:
    runs = sorted(r for r in glob.glob('{}/*_collapse_{}_{}'.format(results_folder, variant, c)) if os.path.isdir(r))
    if runs and os.path.exists(runs[-1] + '/head_circuit.h5'):
      found[c] = _read_h5(runs[-1] + '/head_circuit.h5')
  if not found:
    return
  panels = [('PI_K_disrupt', 'Disrupting P -> I.K: share of gap lost'),
            ('PI_K_restore', 'Restoring P -> I.K: share of damage undone'),
            ('induction_heads_recovery', 'Induction-head circuit recovery R'),
            ('induction_heads_eval_accuracy', 'Induction-head circuit accuracy, held-out classes')]
  with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6))
    for k, (c, d) in enumerate(found.items()):
      for ax, (m, title) in zip(axs, panels):
        key = '{}/{}'.format(target, m)
        if key in d:
          ax.plot(d['gen'], d[key], marker='o', color=MUTED if c == 'real_only' else SERIES[(k - 1) % len(SERIES)], label=c)
        ax.set_title(title)
        ax.set_xlabel('generation')
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('{} model, {}: the induction connection across collapse conditions'.format(variant, target))
    fig.tight_layout()
    os.makedirs(out_folder, exist_ok=True)
    _save(fig, os.path.join(out_folder, 'collapse_{}'.format(variant)), target + '_head_circuit')


if __name__ == '__main__':
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('folder', help='a head_circuit/ folder, or the results folder with --conditions')
  ap.add_argument('--conditions', action='store_true')
  args = ap.parse_args()
  if args.conditions:
    for v in ('base', 'extended'):
      for tgt in ('value_exemplar_swap', 'value_label_swap'):
        plot_conditions(args.folder, os.path.join(args.folder, 'figures', 'head_circuit'), v, tgt)
  else:
    for path in sorted(glob.glob(os.path.join(args.folder, '*_head_circuit.json'))):
      with open(path) as f:
        res = json.load(f)
      plot_target(res, path[:-len('_head_circuit.json')])
