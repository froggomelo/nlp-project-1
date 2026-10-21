"""Collects the nlp_project_run.sh results into a static results page.

Reads the newest results/<stamp>_<name>/ folder of every experiment and writes
results/site/: data.json (metrics, training curves, circuit summaries) and one
WebP sprite of circuit diagrams per run and view. index.html comes from
plotting/results_page.html.

  python -m plotting.results_page [--results ./results] [--out ./results/site]
"""
import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from plotting import symbol_embeddings

EVAL_SETS = ['fsl_train', 'fsl_val_rl', 'fsl_train_valex', 'fsl_test_class']
EVAL_METRICS = ['acc', 'in_context_acc', 'loss', 'symbol_acc', 'next_value_acc', 'loss_total']
TARGETS = ['value_exemplar_swap', 'value_label_swap', 'next_symbol']
METHODS = ['eap', 'acdc']
POOL_METRICS = ['real_frac', 'query_label_err', 'next_symbol_entropy', 'next_symbol_kl_to_gen0',
                'next_symbol_n_unique', 'next_symbol_top10_mass', 'next_symbol_tail_mass',
                'next_offcontext_rate', 'next_value_err']
CONDITIONS = ['real_only', 'replace50', 'replace90', 'replace100', 'accumulate']
CELL_W, CELL_H = 1200, 780
CURVE_POINTS = 60
LOSS_POINTS = 150

# Mirrors the experiment list in scripts/nlp_project_run.sh.
SUITES = [
  ('single', 'Single runs', 'One generation, fresh data every step.'),
  ('main', 'Pool N = 20,000', 'Main collapse runs: five pool conditions over six generations.'),
  ('N5000', 'Pool N = 5,000', 'Smaller pool, four replacement conditions.'),
  ('zipf0.5', 'Zipf α = 0.5', 'Zipfian train class distribution.'),
  ('zipf1.0', 'Zipf α = 1.0', 'Zipfian train class distribution.'),
  ('zipf1.5', 'Zipf α = 1.5', 'Zipfian train class distribution.'),
  ('L8B1', 'L = 8, B = 1', 'Context length 8, burstiness 1.'),
  ('L8B2', 'L = 8, B = 2', 'Context length 8, burstiness 2.'),
  ('L8B4', 'L = 8, B = 4', 'Context length 8, burstiness 4.'),
]


def expected_runs():
  runs = {'base': ('single', 'base', None), 'extended': ('single', 'extended', None)}
  for v in ('base', 'extended'):
    for c in CONDITIONS:
      runs[f'collapse_{v}_{c}'] = ('main', v, c)
  for suite, _, _ in SUITES[2:]:
    for v in ('base', 'extended'):
      for c in CONDITIONS[:4]:
        runs[f'collapse_{v}_{suite}_{c}'] = (suite, v, c)
  return runs


def r4(x):
  x = float(x)
  return None if not math.isfinite(x) else float(f'{x:.4g}')


def downsample(n, k):
  return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int))


def read_log(path, extended):
  with h5py.File(path, 'r') as f:
    iters = f['eval_iter'][:]
    n = len(iters)
    idx = downsample(n, CURVE_POINTS)
    curves = {'iter': [int(i) for i in iters[idx]]}
    final = {}
    for s in EVAL_SETS:
      if s not in f:
        continue
      curves[s], final[s] = {}, {}
      for m in EVAL_METRICS:
        if m not in f[s] or (m in ('symbol_acc', 'next_value_acc', 'loss_total') and not extended):
          continue
        v = f[s][m][:n].astype(np.float64).reshape(n, -1).mean(axis=1)
        curves[s][m] = [r4(x) for x in v[idx]]
        final[s][m] = r4(v[-1])
    loss = f['train_loss'][:]
    ti = f['train_iter'][:]
    # Mean over equal windows so the downsampled curve is smoothed, not aliased.
    k = min(LOSS_POINTS, len(loss))
    edges = np.linspace(0, len(loss), k + 1).astype(int)
    train = {'iter': [int(ti[min(b - 1, len(ti) - 1)]) for b in edges[1:]],
             'loss': [r4(loss[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]}
  return curves, final, train


def read_circuit(folder):
  out = {}
  for t in TARGETS:
    eap, acdc = folder / f'{t}_faithfulness.json', folder / f'{t}_acdc_faithfulness.json'
    if not eap.exists():
      continue
    d = json.loads(eap.read_text())
    s = d['summary']
    entry = {'eap': {'n_edges_to_recover': s.get('n_edges_to_recover'),
                     'logit_diff_full': r4(s['logit_diff_full']),
                     'logit_diff_ablated': r4(s['logit_diff_ablated']),
                     'curve': [[c['n_edges'], r4(c['recovery'])] for c in d.get('curve', [])]}}
    if acdc.exists():
      a = json.loads(acdc.read_text())
      s = a['summary']
      entry['acdc'] = {'threshold': s['threshold'], 'n_edges': s['n_edges'],
                       'recovery': r4(s['recovery']), 'kl': r4(s['kl_divergence']),
                       'logit_diff_circuit': r4(s['logit_diff_circuit']),
                       'sweep': [[w['threshold'], w['n_edges'], r4(w['recovery'])] for w in a.get('sweep', [])]}
    out[t] = entry
  return out


def read_head_circuit(folder):
  """Compact head-to-head results (src/head_circuit.py) per target; ratios are None when ill-conditioned."""
  out = {}
  for p in sorted(folder.glob('*_head_circuit.json')):
    r = json.loads(p.read_text())
    ok = r.get('well_conditioned', True)
    ratio = (lambda v: None if v is None else r4(v)) if ok else (lambda v: None)
    mat = lambda m: [[ratio(v) for v in row] for row in m]
    ablation = {h: {k: [ratio(v['gap_share']), r4(v['eval_accuracy']) if 'eval_accuracy' in v else None]
                    for k, v in kinds.items()} for h, kinds in r['head_ablation'].items()}
    connections = []
    for con in r['connections'].values():
      ch = {}
      for c in ('Q', 'K', 'V'):
        x = con[c]
        ret = lambda s: None if x[s]['retrieval'] is None else r4(x[s]['retrieval'])
        ch[c] = {'disrupt': ratio(x['disrupt']['gap_share']), 'disrupt_acc': r4(x['disrupt']['accuracy']),
                 'input_corrupt': ratio(x['input_corrupt']['gap_share']),
                 'restore': ratio(x['restore_recovery']), 'restore_alone': ratio(x['restore_alone_recovery']),
                 'ret': {s: ret(s) for s in ('disrupt', 'input_corrupt', 'restore_in_input', 'restore_alone')}}
      connections.append({'sender': con['sender'], 'receiver': con['receiver'],
                          'clean_ret': None if con['clean']['retrieval'] is None else r4(con['clean']['retrieval']),
                          'clean_direct': r4(con['clean']['direct_effect']), 'channels': ch})
    circuits = {k: {'n': v['n_edges'], 'R': ratio(v['recovery']), 'acc': r4(v['accuracy']),
                    'eval_acc': r4(v['eval_accuracy']) if 'eval_accuracy' in v else None}
                for k, v in r['circuits'].items() if not k.endswith('_edges')}
    out[r['target']] = {
      'ok': ok, 'P': r['P'], 'I': r['I'], 'roles': r.get('roles'),
      'prev': [r4(v) for v in r['previous_token_scores']],
      'ind': None if r['induction_scores'] is None else [r4(v) for v in r['induction_scores']],
      'clean': [r4(r['clean']['logit_diff']), r4(r['clean']['accuracy'])],
      'corrupt': [r4(r['corrupt']['logit_diff']), r4(r['corrupt']['accuracy'])],
      'eval_clean': r4(r['eval_clean']['accuracy']) if 'eval_clean' in r else None,
      'senders': r['edge_matrix']['senders'],
      'matrix': {c: mat(m) for c, m in r['edge_matrix']['gap_share'].items()},
      'comp': {c: [[r4(v) for v in row] for row in m] for c, m in r['composition'].items()},
      'top': r['top_head_edge']['edge'], 'ablation': ablation,
      'connections': connections, 'circuits': circuits}
  return out


def read_pool(path):
  rows = {}
  with h5py.File(path, 'r') as f:
    gens = f['gen'][:]
    for m in POOL_METRICS:
      if m in f:
        rows[m] = {int(g): r4(v) for g, v in zip(gens, f[m][:])}
  return rows


def diagram(folder, target, method, view):
  suffix = '_acdc' if method == 'acdc' else ''
  p = folder / f'{target}{suffix}_{view}.png'
  return p if p.exists() else None


def build_sprite(cells, rows, cols, out_path):
  """cells: {(row, col): png path}. Fits each diagram into a white CELL_W x CELL_H cell."""
  sheet = Image.new('RGB', (CELL_W * len(cols), CELL_H * len(rows)), 'white')
  for (r, c), p in cells.items():
    im = Image.open(p).convert('RGB')
    im.thumbnail((CELL_W, CELL_H), Image.LANCZOS)
    sheet.paste(im, (c * CELL_W + (CELL_W - im.width) // 2, r * CELL_H + (CELL_H - im.height) // 2))
  sheet.save(out_path, 'WEBP', quality=72, method=6)


NOTEBOOK_FIGURES = [
  ('attention_maps', 'Mean attention of every head', 'Rows are query positions, columns the positions they attend to, averaged over 1,024 real sequences.'),
  ('head_scores', 'Head scores', 'Previous-token score (left) and induction score (right) of every head.'),
  ('single_sequences', 'Key heads on single sequences', 'The strongest previous-token head (top) and induction head (bottom) on three sequences.'),
  ('scores_over_training', 'Head formation during training', 'Training loss and both head scores at 40 checkpoints.'),
  ('head_ablations', 'Single-head ablations', 'Accuracy drop when one head is ablated, ordered by head score.'),
  ('attention_from_yq', 'Attention at the next-symbol prediction', 'Where the extended model looks from the query label yq.'),
  ('heads_per_generation', 'Heads per generation', 'Last-layer head scores of each generation\'s final model, and the heads in its ACDC circuit (exemplar swap).'),
  ('induction_across_generations', 'Strongest induction head across generations', 'Largest last-layer induction score per generation for each main pool condition.'),
]


def collect_notebook_figures(fig_folder, out):
  """Copies the PNGs saved by plotting/nlp_project.ipynb, keyed by figure name."""
  figs = {}
  if not fig_folder.is_dir():
    return figs
  (out / 'notebook').mkdir(exist_ok=True)
  for p in sorted(fig_folder.glob('*.png')):
    kind = next((k for k, _, _ in NOTEBOOK_FIGURES if p.stem.endswith(k)), None)
    if kind is None:
      continue
    im = Image.open(p).convert('RGB')
    im.thumbnail((1800, 1800), Image.LANCZOS)
    fname = f'notebook/{p.stem}.webp'
    im.save(out / fname, 'WEBP', quality=82, method=6)
    owner = p.stem[:-len(kind)].rstrip('_')  # base, extended, collapse_<...>
    title, caption = next((t, c) for k, t, c in NOTEBOOK_FIGURES if k == kind)
    figs[p.stem] = {'file': fname, 'kind': kind, 'owner': owner, 'title': title,
                    'caption': caption, 'size': [im.width, im.height]}
  return figs


def collect_run(folder, name, meta, out, views):
  suite, variant, condition = meta
  extended = variant == 'extended'
  run = {'name': name, 'suite': suite, 'variant': variant, 'condition': condition,
         'stamp': folder.name.split('_')[0], 'gens': []}
  cfg = json.loads((folder / 'config.json').read_text())
  run['expected_gens'] = int(cfg.get('generations') or 1) if suite != 'single' else 1
  gen_dirs = sorted((d for d in folder.glob('gen_*') if d.is_dir()), key=lambda d: int(d.name[4:]))
  units = [(int(d.name[4:]), d) for d in gen_dirs] if gen_dirs else [(0, folder)]
  pool = read_pool(folder / 'generations.h5') if (folder / 'generations.h5').exists() else {}
  for g, d in units:
    if not (d / 'log.h5').exists() or not (d / 'circuit').exists():
      continue  # generation still training
    try:
      curves, final, train = read_log(d / 'log.h5', extended)
    except (OSError, KeyError):
      continue
    entry = {'gen': g, 'final': final, 'curves': curves, 'train': train,
             'circuits': read_circuit(d / 'circuit'),
             'head_circuit': read_head_circuit(d / 'head_circuit'),
             'pool': {m: v[g] for m, v in pool.items() if g in v}}
    run['gens'].append(entry)
  run['complete'] = len(run['gens']) >= run['expected_gens']
  try:
    done = {e['gen'] for e in run['gens']}
    run['embeddings'] = symbol_embeddings.collect(folder, [(g, d) for g, d in units if g in done])
  except Exception as err:  # keep the page buildable if a checkpoint is unreadable
    print(f'{name}: symbol embeddings skipped ({err})')

  # One sprite per view: rows are generations, columns are (target, method).
  targets = [t for t in TARGETS if any(t in e['circuits'] for e in run['gens'])]
  cols = [(t, m) for t in targets for m in METHODS]
  rows = [e['gen'] for e in run['gens']]
  run['targets'] = targets
  run['sprites'] = {}
  for view in views:
    cells = {}
    for ri, (g, d) in enumerate((g, d) for g, d in units if g in rows):
      for ci, (t, m) in enumerate(cols):
        p = diagram(d / 'circuit', t, m, view)
        if p is not None:
          cells[(ri, ci)] = p
    if not cells:
      continue
    fname = f'img/{name}_{view}.webp'
    build_sprite(cells, rows, cols, out / fname)
    run['sprites'][view] = {'file': fname, 'rows': rows, 'cols': [list(c) for c in cols],
                            'cell': [CELL_W, CELL_H], 'have': [[r, c] for r, c in cells]}
  return run


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--results', default='./results')
  ap.add_argument('--out', default='./results/site')
  ap.add_argument('--sweep-views', nargs='+', default=['component'],
                  help='diagram views kept for sweep runs (main and single runs keep both)')
  args = ap.parse_args()
  results, out = Path(args.results), Path(args.out)
  (out / 'img').mkdir(parents=True, exist_ok=True)

  expected = expected_runs()
  newest = {}
  for d in results.iterdir():
    m = re.fullmatch(r'(\d{14})_(.+)', d.name)
    if d.is_dir() and m and m.group(2) in expected and (d / 'config.json').exists():
      if m.group(2) not in newest or d.name > newest[m.group(2)].name:
        newest[m.group(2)] = d

  runs = {}
  for name, folder in sorted(newest.items()):
    meta = expected[name]
    views = ['component', 'sankey'] if meta[0] in ('single', 'main') else args.sweep_views
    run = collect_run(folder, name, meta, out, views)
    if run['gens']:
      runs[name] = run
      print(f'{name}: {len(run["gens"])}/{run["expected_gens"]} generations')

  summaries = sorted(results.glob('*_summary.log'))
  status = {}
  if summaries:
    for line in summaries[-1].read_text().splitlines():
      parts = line.split()
      if len(parts) >= 2 and parts[0] in ('OK', 'FAILED'):
        status[parts[1]] = parts[0]

  data = {
    'generated': datetime.now().isoformat(timespec='minutes'),
    'suites': [{'id': s, 'label': l, 'note': n} for s, l, n in SUITES],
    'expected': {k: {'suite': v[0], 'variant': v[1], 'condition': v[2]} for k, v in expected.items()},
    'status': status,
    'runs': runs,
    'notebook': collect_notebook_figures(results / 'figures' / 'nlp_project', out),
    'notebook_order': [k for k, _, _ in NOTEBOOK_FIGURES],
  }
  (out / 'data.json').write_text(json.dumps(data, separators=(',', ':')))
  shutil.copy(Path(__file__).with_name('results_page.html'), out / 'index.html')
  print(f'wrote {out}/data.json with {len(runs)} runs')


if __name__ == '__main__':
  main()
