"""Picks the learning rate and training length per variant from the tuning runs.

Reads gen_0/log.h5 of the newest results/<timestamp>_tune_<variant>_lr<lr>
folder for every variant and learning rate. The score is query accuracy on the
fsl_val_class evaluator (validation classes, so the answer must come from the
context), averaged over the evaluations in the last 50,000 sequences before each
candidate length. A longer training or a larger learning rate is only chosen
when it beats the best cheaper setting by more than TOLERANCE. The table goes
to stderr and the chosen values to stdout as shell assignments, e.g.
BASE_LR=1e-4, which scripts/nlp_project_run.sh evaluates.

python scripts/select_hparams.py --results results --lrs 1e-5 3e-5 1e-4 3e-4 \
  --lengths 1000000 2000000 3000000
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

EVALUATOR = 'fsl_val_class'
WINDOW = 50_000
TOLERANCE = 0.01


def newest_run(results, variant, lr):
  folders = sorted(glob.glob(os.path.join(results, f'*_tune_{variant}_lr{lr}')))
  return folders[-1] if folders else None


def score_at(log_file, lengths):
  with h5py.File(log_file, 'r') as f:
    iters = f['eval_iter'][:]
    acc = f[EVALUATOR + '/acc'][:].mean(axis=1)
  scores = []
  for length in lengths:
    window = (iters > length - WINDOW) & (iters <= length)
    scores.append(acc[window].mean() if window.any() else np.nan)
  return scores


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--results', default='results')
  parser.add_argument('--variants', nargs='+', default=['base', 'extended'])
  parser.add_argument('--lrs', nargs='+', required=True)
  parser.add_argument('--lengths', nargs='+', type=int, required=True)
  opts = parser.parse_args()

  failed = False
  for variant in opts.variants:
    print(f'== {variant}: {EVALUATOR} accuracy by learning rate and training length',
          file=sys.stderr)
    print('lr       ' + ''.join(f'{n:>11,d}' for n in opts.lengths), file=sys.stderr)
    # Candidates in order of cost: shorter training first, then smaller learning rate.
    candidates = []
    for lr in opts.lrs:
      run = newest_run(opts.results, variant, lr)
      log_file = run and os.path.join(run, 'gen_0', 'log.h5')
      if not log_file or not os.path.exists(log_file):
        print(f'{lr:<9s}missing', file=sys.stderr)
        continue
      scores = score_at(log_file, opts.lengths)
      print(f'{lr:<9s}' + ''.join(f'{s:11.3f}' for s in scores), file=sys.stderr)
      candidates += [(length, i, lr, s) for i, (length, s) in enumerate(zip(opts.lengths, scores))
                     if not np.isnan(s)]
    if not candidates:
      print(f'no tuning runs found for {variant}', file=sys.stderr)
      failed = True
      continue
    candidates.sort(key=lambda c: (c[0], opts.lrs.index(c[2])))
    best = candidates[0]
    for c in candidates[1:]:
      if c[3] > best[3] + TOLERANCE:
        best = c
    print(f'chosen: lr {best[2]}, {best[0]:,d} sequences, accuracy {best[3]:.3f}', file=sys.stderr)
    print(f'{variant.upper()}_LR={best[2]}')
    print(f'{variant.upper()}_ITERS={best[0]}')
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
