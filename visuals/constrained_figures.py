'''
Constrained next-symbol ablation: compares the extended model's 20-generation runs
with next symbols constrained to the context (scripts/constrained_run.sh, in
results/constrained/) against the unconstrained runs of Table 1, over three seeds.

Prints the generation 19 table (mean ± sd over seeds), the LaTeX rows of Table 1 with
the extended model's runs constrained, and draws the trends over
generations into visuals/figures/constrained_trends.{pdf,png}. The symbol
distribution measures come from scripts/symbol_distribution.py
(results/constrained/analysis/<run>.tsv).

python -I visuals/constrained_figures.py
'''
import csv
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import results_figures as rf

RESULTS = os.path.join(HERE, '..', 'results')
CONSTRAINED = os.path.join(RESULTS, 'constrained')
ANALYSIS = os.path.join(CONSTRAINED, 'analysis')
SEEDS = ('', '_seed15', '_seed25')
CONDITIONS = [  # label, unconstrained name, constrained name (None: identical, nothing sampled)
  ('Real only', 'G20_real_only', None),
  ('50% replaced', 'G20_T3_replace50', 'G20_T3ctx_replace50'),
  ('90% replaced', 'G20_T3_replace90', 'G20_T3ctx_replace90'),
  ('100% replaced', 'G20_T3_replace100', 'G20_T3ctx_replace100'),
  ('Accumulate', 'G20_T3_accumulate', 'G20_T3ctx_accumulate'),
]
T1_CONDITIONS = [  # T = 1 rows: label, unconstrained name, constrained name
  ('100% replaced, $T = 1$', 'G20_replace100', 'G20_T1ctx_replace100'),
  ('Accumulate, $T = 1$', 'G20_accumulate', 'G20_T1ctx_accumulate'),
]
# Table 1 of the report: row label, run suffix at T = 1 or T = 3, and its constrained
# counterpart (None: nothing is sampled, so constrained and unconstrained are identical)
TABLE1_ROWS = [
  ('\\multicolumn{7}{l}{\\emph{No forced errors} ($T = 1$)} \\\\', None, None),
  ('Real only', 'G20_real_only', None),
  ('100\\% replaced', 'G20_replace100', 'G20_T1ctx_replace100'),
  ('Accumulate', 'G20_accumulate', 'G20_T1ctx_accumulate'),
  ('\\midrule\n\\multicolumn{7}{l}{\\emph{Forced errors} ($T = 3$)} \\\\', None, None),
  ('50\\% replaced', 'G20_T3_replace50', 'G20_T3ctx_replace50'),
  ('90\\% replaced', 'G20_T3_replace90', 'G20_T3ctx_replace90'),
  ('100\\% replaced', 'G20_T3_replace100', 'G20_T3ctx_replace100'),
  ('Accumulate', 'G20_T3_accumulate', 'G20_T3ctx_accumulate'),
]
COLOURS = {'Real only': rf.INK_2, '50% replaced': '#86b6ef', '90% replaced': '#2a78d6',
           '100% replaced': '#104281', 'Accumulate': '#eb6834'}


def run_folder(folder, name):
  runs = sorted(r for r in glob.glob(os.path.join(folder, '[0-9]*_collapse_extended_' + name))
                if os.path.isdir(r) and os.path.basename(r).endswith('_collapse_extended_' + name))
  return runs[-1] if runs and rf.finished(runs[-1]) else None


def metrics(run):
  '''Per-generation arrays of one run: query and next-label accuracy, margin, ACDC
  edges, induction-head match and copy, generated-data errors, and the symbol
  distribution measures.'''
  m = rf.run_metrics(run)
  n = len(m['accuracy'])
  next_acc = []
  for g in range(n):
    with h5py.File(os.path.join(run, 'gen_{}'.format(g), 'log.h5'), 'r') as f:
      next_acc.append(f['fsl_test_class/next_value_acc'][-5:].mean())
  m['next_acc'] = np.array(next_acc)
  m['retrieval'], m['direct'] = rf.head_metrics(run)
  with h5py.File(os.path.join(run, 'generations.h5'), 'r') as g:
    for k in ('next_value_err_new', 'next_value_err', 'next_offcontext_rate_new', 'query_label_err_new'):
      m[k] = g[k][:]
  tsv = os.path.join(ANALYSIS, os.path.basename(run) + '.tsv')
  if os.path.exists(tsv):
    rows = list(csv.DictReader(open(tsv), delimiter='\t'))
    for k in rows[0]:
      if k != 'gen':
        m[k] = np.array([float(r[k]) for r in rows])
  return m


def seed_metrics(folder, name):
  runs = [run_folder(folder, name + s) for s in SEEDS]
  return [metrics(r) for r in runs if r]


def cell(ms, key, digits, g=-1):
  vals = [m[key][g] for m in ms if key in m and np.isfinite(m[key][g])]
  if not vals:  # missing, or nothing generated (real only)
    return '--'
  return '{:.{d}f} ± {:.{d}f}'.format(np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0, d=digits)


def table():
  cols = [('accuracy', 'query acc.', 2), ('next_acc', 'next-label acc.', 2), ('margin', 'margin', 1),
          ('edges', 'ACDC edges', 0), ('ctx_mass', 'context mass', 3), ('q_kl_uniform', 'KL to uniform', 3),
          ('q_absbias', '|q - 1/2|', 3), ('next_value_err', 'pool next-label err.', 2),
          ('next_offcontext_rate_new', 'generated off-context', 3)]
  print('| Condition | Generation | n | ' + ' | '.join(c[1] for c in cols) + ' |')
  print('|' + '---|' * (len(cols) + 3))
  rows = []
  for label, unc, con in CONDITIONS:
    rows.append((label, 'unconstrained', seed_metrics(RESULTS, unc)))
    if con:
      rows.append((label, 'constrained', seed_metrics(CONSTRAINED, con)))
  for label, unc, con in T1_CONDITIONS:
    rows.append((label, 'unconstrained', seed_metrics(RESULTS, unc)))
    rows.append((label, 'constrained', seed_metrics(CONSTRAINED, con)))
  for label, arm, ms in rows:
    print('| {} | {} | {} | '.format(label, arm, len(ms)) + ' | '.join(cell(ms, k, d) for k, _, d in cols) + ' |')
  gen0 = seed_metrics(RESULTS, 'G20_T3_replace100')
  print('| Generation 0 (shared) | -- | {} | '.format(len(gen0)) + ' | '.join(cell(gen0, k, d, 0) for k, _, d in cols) + ' |')


def table1():
  '''LaTeX rows of Table 1 with the extended model's runs constrained to the context.
  The base model has no next symbol, so its columns come from the original runs.'''
  def tex(ms, key, digits):
    vals = [m[key][-1] for m in ms]
    return '{:.{d}f}\\,{{\\scriptsize$\\pm${:.{d}f}}}'.format(np.mean(vals), np.std(vals, ddof=1), d=digits)

  def margin_edges(ms):
    if len(ms) < len(SEEDS):
      return '-- & --'
    dagger = '$^\\dagger$' if np.mean([m['margin'][-1] for m in ms]) < 1 else ''
    return '{} & {}{}'.format(tex(ms, 'margin', 1), tex(ms, 'edges', 0), dagger)

  for label, unc, con in TABLE1_ROWS:
    if unc is None:
      print(label)
      continue
    b = [rf.run_metrics(r) for r in (run_folder_base(unc + s) for s in SEEDS) if r]
    e = seed_metrics(CONSTRAINED, con) if con else seed_metrics(RESULTS, unc)
    print('{} & {:.0f}\\% & {} & {} & {} \\\\'.format(
      label, 100 * b[0]['real_frac'][-1], tex(b, 'accuracy', 2), margin_edges(b), margin_edges(e)))


def run_folder_base(name):
  runs = sorted(r for r in glob.glob(os.path.join(RESULTS, '[0-9]*_collapse_base_' + name))
                if os.path.isdir(r) and os.path.basename(r).endswith('_collapse_base_' + name))
  return runs[-1] if runs and rf.finished(runs[-1]) else None


def trends(out):
  panels = [('accuracy', 'query accuracy', (0, 1.05)), ('next_acc', 'next-label accuracy', (0, 1.05)),
            ('margin', 'margin', None), ('ctx_mass', 'unmasked context mass', (0, 1.05)),
            ('q_kl_uniform', 'KL to uniform (nats)', None), ('next_value_err', 'pool next-label error', (0, 0.85))]
  fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
  for ax, (key, ylabel, ylim) in zip(axes.flat, panels):
    for label, unc, con in CONDITIONS:
      for name, folder, ls in ((unc, RESULTS, (0, (3, 2))), (con, CONSTRAINED, '-')):
        if not name:
          continue
        ms = [m for m in seed_metrics(folder, name) if key in m]
        if not ms:
          continue
        y = np.mean([m[key] for m in ms], axis=0)
        ax.plot(np.arange(len(y)), y, color=COLOURS[label], lw=1.4, ls=ls)
    rf.style(ax, 'generation', ylabel, ylim)
    ax.set_xticks([0, 5, 10, 15, 19])
  handles = [plt.Line2D([], [], color=c, lw=1.5, label=l) for l, c in COLOURS.items()]
  handles += [plt.Line2D([], [], color=rf.INK_2, lw=1.5, label='constrained'),
              plt.Line2D([], [], color=rf.INK_2, lw=1.5, ls=(0, (3, 2)), label='unconstrained')]
  fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
  fig.tight_layout(rect=(0, 0.05, 1, 1))
  rf.save(fig, out, 'constrained_trends')


if __name__ == '__main__':
  out = os.path.join(HERE, 'figures')
  os.makedirs(out, exist_ok=True)
  table()
  print('\nTable 1 with the extended model constrained to the context (LaTeX rows):')
  table1()
  trends(out)
  print('wrote', os.path.join(out, 'constrained_trends.pdf'))
