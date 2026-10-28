'''
Draws the quick test of scripts/quick_test.sh: test accuracy, margin and ACDC circuit
size per generation of the base model under real data only, full replacement at T = 3
and accumulation at T = 3, into visuals/figures/quick_test.{pdf,png}. Prints each
measure next to the report's value for the same run and generation.

python -I visuals/quick_test_figure.py
'''
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import results_figures as rf

RUNS = [  # quick test run, condition key of rf.CONDITIONS, label
  ('quick_base_real_only', 'real_only', 'real only'),
  ('quick_base_T3_replace100', 'replace100', '100% replaced, T = 3'),
  ('quick_base_T3_accumulate', 'accumulate', 'accumulate, T = 3'),
]
# Generations 0-3 of the report's runs with the same settings and seed:
# collapse_base_G20_real_only, collapse_base_T3_replace100, collapse_base_T3_accumulate
REPORT = {
  'quick_base_real_only': {'accuracy': [0.98, 0.98, 0.98, 0.99], 'margin': [19.9, 17.2, 19.4, 19.8],
                           'edges': [17, 25, 19, 19]},
  'quick_base_T3_replace100': {'accuracy': [0.98, 0.99, 0.88, 0.705], 'margin': [19.9, 10.8, 5.3, 2.2],
                               'edges': [17, 55, 188, 209]},
  'quick_base_T3_accumulate': {'accuracy': [0.98, 0.99, 0.99, 0.99], 'margin': [19.9, 9.5, 4.6, 3.1],
                               'edges': [17, 38, 51, 44]},
}
PANELS = [  # measure, axis label, y limits, value format
  ('accuracy', 'test accuracy', (0, 1.05), '{:.2f}'),
  ('margin', 'margin (logit difference)', (0, None), '{:.1f}'),
  ('edges', 'ACDC circuit edges', (0, None), '{:.0f}'),
]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--results', default=os.path.join(HERE, '..', 'results'))
  parser.add_argument('--out', default=os.path.join(HERE, 'figures'))
  opts = parser.parse_args()
  os.makedirs(opts.out, exist_ok=True)

  metrics = {run: rf.run_metrics(rf.find_run(opts.results, run)) for run, _, _ in RUNS}
  fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
  for ax, (measure, ylabel, ylim, _) in zip(axes, PANELS):
    for run, cond, label in RUNS:
      colour, _, ls = rf.CONDITIONS[cond]
      y = metrics[run][measure]
      ax.plot(range(len(y)), y, color=colour, ls=ls, lw=1.6, marker='o', ms=3, label=label)
    rf.style(ax, 'generation', ylabel, ylim)
    ax.set_xticks(range(len(y)))
  axes[0].legend(loc='lower left', fontsize=7)
  fig.tight_layout()
  rf.save(fig, opts.out, 'quick_test')

  for measure, _, _, fmt in PANELS:
    print('\n' + measure + ' per generation, this run (report)')
    for run, _, label in RUNS:
      cells = ['{} ({})'.format(fmt.format(v), fmt.format(r))
               for v, r in zip(metrics[run][measure], REPORT[run][measure])]
      print('  {:<22} {}'.format(label, '  '.join(cells)))
  print('\nwrote', os.path.abspath(os.path.join(opts.out, 'quick_test.png')))


if __name__ == '__main__':
  main()
