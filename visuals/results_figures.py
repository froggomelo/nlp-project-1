'''
Draws the Results figures and prints the LaTeX for Table 1, from the tuned runs
in results/ (stamp 20261015142800 onwards).

  failure_order    base, full replacement, T = 3: accuracy, margin, induction
                   score and ACDC circuit size per generation, on one 0-1 axis
  retention_acc    base, T = 3: test accuracy per generation for each condition
  retention_edges  base, T = 3: ACDC circuit size per generation for each condition
  margin_error     predicted vs observed error rate of each generation's samples
  circuits         ACDC circuits under full replacement at T = 1 and T = 3
  circuits_two     ACDC circuits at generation 0 and at T = 3, generation 1 (Figure 1)
  circuits2        ACDC circuits under accumulation
  induction_head   the induction head's attention to the match and direct effect at
                   T = 3, mean and range over seeds (Figure 2)
  and prints the rows of Table 1 as the mean and standard deviation over seeds.
  failure_order and retention_acc are Figures S3 and S2 of the supplement.

python -I visuals/results_figures.py
python -I visuals/results_figures.py --results results --out visuals/figures
'''
import argparse
import glob
import json
import os
import warnings

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_STAMP = '20261015142800'   # the tuned run; older folders are untuned
N_EDGES = 257                    # edges in the two-layer, eight-head graph
N_LABELS = 5

# Colours from the dataviz reference palette (validated: categorical light mode,
# blue ordinal ramp). Text stays in ink colours, never the series colour.
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MEASURES = {  # failure_order lines, categorical slots 1-4
  'accuracy': ('#2a78d6', 'test accuracy'),
  'margin': ('#eb6834', 'margin / generation 0'),
  'induction': ('#1baf7a', 'induction score'),
  'edges': ('#eda100', 'ACDC edges / 257'),
}
CONDITIONS = {  # retention lines: control grey, replacement as a blue ordinal ramp
  'real_only': ('#52514e', 'real only', (0, (4, 3))),
  'replace50': ('#86b6ef', '50% replaced', '-'),
  'replace90': ('#2a78d6', '90% replaced', '-'),
  'replace100': ('#104281', '100% replaced', '-'),
  'accumulate': ('#eb6834', 'accumulate', '-'),
}

plt.rcParams.update({
  'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK_2,
  'axes.labelcolor': INK, 'xtick.color': INK_2, 'ytick.color': INK_2,
  'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
  'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False,
})


def find_run(results, name):
  '''Newest tuned run folder named <stamp>_<name>.'''
  runs = [r for r in glob.glob(os.path.join(results, '*_' + name))
          if os.path.isdir(r) and os.path.basename(r)[:14] >= FIRST_STAMP]
  if not runs:
    raise SystemExit('no tuned run named ' + name)
  return sorted(runs)[-1]


SEED_SUFFIXES = ('', '_seed15', '_seed25')  # seed 5 (the original runs), 15 and 25


def seed_runs(results, name):
  '''The run folders of `name` for every seed; find_run's glob ends at the name, so the
  seed-5 lookup never picks up a _seed folder.'''
  return [find_run(results, name + s) for s in SEED_SUFFIXES]


def run_metrics(run):
  '''Per-generation metrics of one collapse run, as numpy arrays.'''
  with h5py.File(os.path.join(run, 'generations.h5'), 'r') as g:
    n = len(g['gen'])
    out = {'label_err': g['query_label_err'][:], 'label_err_new': g['query_label_err_new'][:],
           'real_frac': g['real_frac'][:]}
  acc = []
  for k in range(n):
    with h5py.File(os.path.join(run, 'gen_{}'.format(k), 'log.h5'), 'r') as l:
      acc.append(l['fsl_test_class/acc'][-5:].mean())  # last five evaluations
  out['accuracy'] = np.array(acc)
  with h5py.File(os.path.join(run, 'head_circuit.h5'), 'r') as h:
    t = h['value_exemplar_swap']
    out['margin'] = t['logit_diff_clean'][:]
    out['induction'] = t['I_induction_score'][:]
  with h5py.File(os.path.join(run, 'circuits.h5'), 'r') as c:
    out['edges'] = c['value_exemplar_swap/acdc_n_edges'][:].astype(float)
  return out


def style(ax, xlabel, ylabel, ylim=None):
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(axis='y', color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  if ylim:
    ax.set_ylim(*ylim)


def end_labels(ax, ys, labels, x, min_gap):
  '''Direct labels at the line ends, nudged apart so they do not collide.'''
  order = np.argsort(ys)
  placed = []
  for i in order:
    y = ys[i] if not placed else max(ys[i], placed[-1] + min_gap)
    placed.append(y)
    ax.text(x, y, labels[i], va='center', ha='left', fontsize=7, color=INK)


def save(fig, out, name):
  for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(out, '{}.{}'.format(name, ext)), dpi=300, bbox_inches='tight')
  plt.close(fig)


def failure_order(results, out, last=5):
  # The first generations of full replacement at T = 3 for every seed, the margin
  # relative to each seed's own generation 0; the mean as the line, the range shaded
  ms = [run_metrics(r) for r in seed_runs(results, 'collapse_base_G20_T3_replace100')]
  per_seed = lambda f: np.array([f(m)[:last + 1] for m in ms])
  gens = np.arange(last + 1)
  series = {'accuracy': per_seed(lambda m: m['accuracy']),
            'margin': per_seed(lambda m: m['margin'] / m['margin'][0]),
            'induction': per_seed(lambda m: m['induction']),
            'edges': per_seed(lambda m: m['edges'] / N_EDGES)}
  fig, ax = plt.subplots(figsize=(3.3, 2.3))
  for key, ys in series.items():
    colour, _ = MEASURES[key]
    ax.fill_between(gens, ys.min(0), ys.max(0), color=colour, alpha=0.18, lw=0, zorder=2)
    ax.plot(gens, ys.mean(0), color=colour, lw=2, marker='o', ms=4, zorder=3)
  end_labels(ax, np.array([ys.mean(0)[-1] for ys in series.values()]),
             [MEASURES[k][1] for k in series], gens[-1] + 0.2, 0.075)
  ax.set_xlim(-0.2, gens[-1] + 2.6)
  ax.set_xticks(gens)
  style(ax, 'generation', 'value (0 to 1)', (-0.03, 1.05))
  save(fig, out, 'failure_order')


def retention(results, out, key, name, ylabel, ylim):
  fig, ax = plt.subplots(figsize=(3.3, 2.3))
  ends = []
  for cond, (colour, label, ls) in CONDITIONS.items():
    # real only never samples, so its T = 1 run is the control at every temperature
    run = 'collapse_base_G20_real_only' if cond == 'real_only' else 'collapse_base_G20_T3_' + cond
    # Mean over seeds with the range shaded; hollow markers where the mean margin is below 1
    ms = [run_metrics(r) for r in seed_runs(results, run)]
    ys = np.array([x[key] for x in ms]) / (N_EDGES if key == 'edges' else 1)
    y = ys.mean(0)
    margin = np.mean([x['margin'] for x in ms], axis=0)
    gens = np.arange(len(y))
    ax.fill_between(gens, ys.min(0), ys.max(0), color=colour, alpha=0.18, lw=0, zorder=2)
    # The dashed control sits on top, as accumulation runs along the same values
    ax.plot(gens, y, color=colour, lw=2, ls=ls, zorder=5 if cond == 'real_only' else 3)
    # Hollow markers where the margin is below 1: ACDC recovery is undefined there
    weak = margin < 1
    ax.scatter(gens[~weak], y[~weak], s=16, color=colour, zorder=4, linewidths=0)
    ax.scatter(gens[weak], y[weak], s=16, facecolor='white', edgecolor=colour, lw=1.2, zorder=4)
    ends.append((y[-1], label))
  last = len(gens) - 1
  end_labels(ax, np.array([e[0] for e in ends]), [e[1] for e in ends], last + 0.6,
             0.075 * (ylim[1] - ylim[0]))
  ax.set_xlim(-0.5, last + 8.5)
  ax.set_xticks(list(range(0, last, 5)) + [last])
  style(ax, 'generation', ylabel, ylim)
  save(fig, out, name)


def predicted_error(margin, temperature):
  '''Share of samples that are wrong if the correct label leads the other four by margin.'''
  w = (N_LABELS - 1) * np.exp(-margin / temperature)
  return w / (1 + w)


def margin_error(results, out):
  fig, ax = plt.subplots(figsize=(3.3, 2.6))
  colours = {3: '#2a78d6', 5: '#eb6834'}
  markers = {'base': 'o', 'extended': 's'}
  pairs = []
  for variant in ('base', 'extended'):
    for t in (3, 5):
      for cond in ('replace50', 'replace90', 'replace100', 'accumulate'):
        m = run_metrics(find_run(results, 'collapse_{}_T{}_{}'.format(variant, t, cond)))
        # The model of generation g writes the new items of generation g + 1
        pred = predicted_error(m['margin'][:-1], t)
        obs = m['label_err_new'][1:]
        ax.scatter(pred, obs, s=18, marker=markers[variant], color=colours[t],
                   edgecolor='white', lw=0.5, zorder=3)
        pairs.extend(zip(pred, obs))
  ax.plot([0, 0.8], [0, 0.8], color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=2)
  ax.text(0.52, 0.36, 'predicted = observed', fontsize=7, color=INK_2)
  for t, c in colours.items():
    ax.scatter([], [], s=18, color=c, label='T = {}'.format(t))
  for v, mk in markers.items():
    ax.scatter([], [], s=18, marker=mk, color=INK_2, label=v)
  ax.legend(loc='upper left', fontsize=7, handletextpad=0.2)
  ax.set_xlim(-0.02, 0.82)
  style(ax, 'error predicted from the previous margin', 'observed error of new samples', (-0.02, 0.85))
  save(fig, out, 'margin_error')
  pairs = np.array(pairs)
  return np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1], np.abs(pairs[:, 0] - pairs[:, 1]).mean()


CHANNELS = {  # edge colour by the receiving input; categorical slots 1-3, output in ink
  'Q': ('#2a78d6', 'query input'), 'K': ('#eb6834', 'key input'),
  'V': ('#1baf7a', 'value input'), 'out': (INK_2, 'to output'),
}
CIRCUIT_PANELS = [  # (run, generation, title); full replacement, one row per temperature
  # Top row, T = 1: generation 0 against generation 15, the best ACDC recovery (91.3%)
  ('collapse_base_G20_replace100', 0, '(a) $T = 1$, generation 0'),
  ('collapse_base_G20_replace100', 15, '(b) $T = 1$, 100% replaced, gen. 15'),
  # Bottom row, T = 3: generation 1, when the circuit has spread while accuracy is still
  # 0.99, against generation 3, the best recovery (99.96%) and the last with margin >= 1
  ('collapse_base_G20_T3_replace100', 1, '(c) $T = 3$, 100% replaced, gen. 1'),
  ('collapse_base_G20_T3_replace100', 3, '(d) $T = 3$, 100% replaced, gen. 3'),
]
# The two moments of the report: the circuit as formed, and tripled while accuracy holds
TWO_PANELS = [
  ('collapse_base_G20_replace100', 0, '(a) Generation 0'),
  ('collapse_base_G20_T3_replace100', 1, '(b) $\\tau = 3$, 100% replaced, gen. 1'),
]
# The same four moments under accumulation instead of full replacement
ACCUMULATE_PANELS = [
  ('collapse_base_G20_real_only', 0, '(a) Generation 0'),
  ('collapse_base_G20_accumulate', 19, '(b) $T = 1$, accumulate, gen. 19'),
  ('collapse_base_G20_T3_accumulate', 3, '(c) $T = 3$, accumulate, gen. 3'),
  ('collapse_base_G20_T3_accumulate', 19, '(d) $T = 3$, accumulate, gen. 19'),
]


def node_xy(name):
  '''Fixed layout: input, eight layer-0 heads, eight layer-1 heads, output.'''
  if name == 'Resid Start':
    return 0.0, 3.5
  if name == 'Resid End':
    return 3.0, 3.5
  layer, head = name[1:].split('.')
  return 1.0 + int(layer), 7 - int(head)


def circuit_graph(ax, run, gen, title):
  with open(os.path.join(run, 'gen_{}'.format(gen), 'circuit',
                         'value_exemplar_swap_acdc_edges.json')) as f:
    edges = [e for e in json.load(f) if e['in_circuit']]
  used = set()
  for e in sorted(edges, key=lambda e: e['score']):  # strongest drawn last, on top
    src = e['src']
    dest = e['dest'] if e['dest'] == 'Resid End' else e['dest'][:-2]
    channel = 'out' if e['dest'] == 'Resid End' else e['dest'][-1]
    used.update((src, dest))
    (x0, y0), (x1, y1) = node_xy(src), node_xy(dest)
    # Query, key and value edges into the same head would overlap exactly, so each
    # channel ends at its own point on the receiving node
    y1 += {'Q': 0.2, 'K': 0.0, 'V': -0.2, 'out': 0.0}[channel]
    # Edges that skip a layer bend away so they do not pass through the layer-0 nodes
    rad = -0.35 if abs(x1 - x0) > 1.5 else 0.0
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-',
                                 connectionstyle='arc3,rad={}'.format(rad),
                                 color=CHANNELS[channel][0], alpha=0.8, zorder=2,
                                 lw=0.25 + 0.6 * np.log1p(max(e['score'], 0))))
  nodes = ['Resid Start', 'Resid End'] + ['A{}.{}'.format(l, h) for l in (0, 1) for h in range(8)]
  for n in nodes:
    x, y = node_xy(n)
    on = n in used
    ax.scatter([x], [y], s=26 if n.startswith('Resid') else 13, zorder=3,
               facecolor=INK if on else 'white', edgecolor=INK if on else GRID, lw=0.7)
  for x, label in ((0, 'input'), (1, 'layer 0'), (2, 'layer 1'), (3, 'output')):
    ax.text(x, 7.9, label, ha='center', va='bottom', fontsize=5, color=INK_2)
  for h in range(8):
    ax.text(0.8, 7 - h, str(h), ha='right', va='center', fontsize=4.5, color=INK_2)
  with h5py.File(os.path.join(run, 'gen_{}'.format(gen), 'log.h5'), 'r') as l:
    acc = l['fsl_test_class/acc'][-5:].mean()
  ax.set_title('{}\n{} edges, accuracy {:.2f}'.format(title, len(edges), acc), fontsize=6, color=INK, pad=2)
  ax.set_xlim(-0.3, 3.3)
  ax.set_ylim(-0.6, 8.6)
  ax.axis('off')


def circuits(results, out, panels=CIRCUIT_PANELS, name='circuits'):
  # Sized for one column of the two-column template, two panels per row
  rows = (len(panels) + 1) // 2
  fig, axes = plt.subplots(rows, 2, figsize=(3.4, 0.35 + (1.675 if rows > 1 else 1.45) * rows))
  for ax, (run, gen, title) in zip(axes.flat, panels):
    circuit_graph(ax, find_run(results, run), gen, title)
  handles = [plt.Line2D([], [], color=c, lw=1.5, label=l) for c, l in CHANNELS.values()]
  fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=5.5, handlelength=1.2,
             columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
  fig.subplots_adjust(wspace=0.18, hspace=0.3)
  save(fig, out, name)


def head_metrics(run):
  '''Retrieval and direct effect of each generation's induction head, from the clean
  run of the previous-token -> induction head connection test (docs/head_circuit.md).'''
  retrieval, direct = [], []
  for g in range(len(glob.glob(os.path.join(run, 'gen_*')))):
    with open(os.path.join(run, 'gen_{}'.format(g), 'head_circuit',
                           'value_exemplar_swap_head_circuit.json')) as f:
      conn = next(iter(json.load(f)['connections'].values()))['clean']
    retrieval.append(conn['retrieval'])
    direct.append(conn['direct_effect'])
  return np.array(retrieval), np.array(direct)


def induction_head(results, out):
  '''Match (retrieval) against copy (direct effect) of the induction head at T = 3,
  each as a fraction of its generation 0 value, so both share one axis.'''
  fig, ax = plt.subplots(figsize=(3.3, 0.9))
  conditions = [('collapse_base_G20_T3_replace100', '#2a78d6', '100% replaced'),
                ('collapse_base_G20_T3_accumulate', '#eb6834', 'accumulate')]
  for run, colour, _ in conditions:
    # Each seed relative to its own generation 0; the mean as the line, the range shaded
    seeds = [head_metrics(r) for r in seed_runs(results, run)]
    for k, ls in ((0, '-'), (1, (0, (4, 2)))):
      curves = np.array([m[k] / m[k][0] for m in seeds])
      gens = np.arange(curves.shape[1])
      ax.fill_between(gens, curves.min(0), curves.max(0), color=colour, alpha=0.18, lw=0, zorder=2)
      ax.plot(gens, curves.mean(0), color=colour, lw=2, ls=ls, zorder=3)
  # Colours (condition) in the first column and line styles (measure) in the second, above
  # the axes so that the legend covers no line
  handles = [plt.Line2D([], [], color=c, lw=2, label=l) for _, c, l in conditions]
  handles += [plt.Line2D([], [], color=INK_2, lw=2, label='attention to match'),
              plt.Line2D([], [], color=INK_2, lw=2, ls=(0, (4, 2)), label='direct effect (copy)')]
  ax.legend(handles=handles, loc='lower center', fontsize=6.5, handlelength=2.8, ncol=2,
            columnspacing=1.5, frameon=False, bbox_to_anchor=(0.5, 0.98))
  ax.set_xticks([0, 5, 10, 15, 19])
  ax.set_yticks([0, 0.5, 1])
  ax.set_xlim(-0.5, 19.5)
  style(ax, 'generation', 'fraction of generation 0', (-0.05, 1.1))
  save(fig, out, 'induction_head')


def finished(run):
  '''Whether a run has written circuit results for every generation in its config.'''
  try:
    with open(os.path.join(run, 'config.json')) as f:
      n = json.load(f)['generations']
    with h5py.File(os.path.join(run, 'head_circuit.h5'), 'r') as h:
      return len(h['gen']) == n
  except (OSError, KeyError):
    return False


def summary_table(results):
  '''LaTeX rows of Table 1: generation 19 of every 20-generation run, as the mean over
  seeds with the standard deviation in small type.'''
  rows = [
    ('\\multicolumn{7}{l}{\\emph{No forced errors} ($T = 1$)} \\\\', None),
    ('Real only', 'G20_real_only'),
    ('100\\% replaced', 'G20_replace100'),
    ('Accumulate', 'G20_accumulate'),
    ('\\midrule\n\\multicolumn{7}{l}{\\emph{Forced errors} ($T = 3$)} \\\\', None),
    ('50\\% replaced', 'G20_T3_replace50'),
    ('90\\% replaced', 'G20_T3_replace90'),
    ('100\\% replaced', 'G20_T3_replace100'),
    ('Accumulate', 'G20_T3_accumulate'),
  ]

  def cell(values, digits):
    values = np.array(values)
    return '{:.{d}f}\\,{{\\scriptsize$\\pm${:.{d}f}}}'.format(values.mean(), values.std(ddof=1), d=digits)

  def margin_edges(ms):
    # A dagger marks a mean margin below 1, where the circuit size is less reliable
    margins = [m['margin'][-1] for m in ms]
    dagger = '$^\\dagger$' if np.mean(margins) < 1 else ''
    return '{} & {}{}'.format(cell(margins, 1), cell([m['edges'][-1] for m in ms], 0), dagger)

  lines = []
  for label, suffix in rows:
    if suffix is None:
      lines.append(label)
      continue
    b = [run_metrics(r) for r in seed_runs(results, 'collapse_base_' + suffix)]
    e = [run_metrics(r) for r in seed_runs(results, 'collapse_extended_' + suffix)]
    lines.append('{} & {:.0f}\\% & {} & {} & {} \\\\'.format(
      label, 100 * b[0]['real_frac'][-1], cell([m['accuracy'][-1] for m in b], 2),
      margin_edges(b), margin_edges(e)))
  return lines


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--results', default=os.path.join(HERE, '..', 'results'))
  parser.add_argument('--out', default=os.path.join(HERE, 'figures'))
  opts = parser.parse_args()
  os.makedirs(opts.out, exist_ok=True)

  failure_order(opts.results, opts.out)
  retention(opts.results, opts.out, 'accuracy', 'retention_acc', 'test accuracy', (0, 1.05))
  retention(opts.results, opts.out, 'edges', 'retention_edges', 'ACDC edges / 257', (0, 0.9))
  r, mae = margin_error(opts.results, opts.out)
  print('margin_error: correlation {:.3f}, mean absolute error {:.3f}'.format(r, mae))
  circuits(opts.results, opts.out)
  circuits(opts.results, opts.out, TWO_PANELS, 'circuits_two')
  induction_head(opts.results, opts.out)
  circuits(opts.results, opts.out, ACCUMULATE_PANELS, 'circuits2')
  print('\n'.join(summary_table(opts.results)))
  print('wrote figures to', os.path.abspath(opts.out))


if __name__ == '__main__':
  main()
