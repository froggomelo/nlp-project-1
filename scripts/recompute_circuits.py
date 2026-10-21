'''
Recomputes circuit discovery and the head-circuit tests of finished collapse runs
from their saved checkpoints, without retraining.

Uses the final checkpoint of every gen_<g>/checkpoints/ folder, the same fixed
real prompts (circuit_seed) and fsl_test_class eval data as the run. Outputs go
next to the originals, which are left untouched:
  <run>/gen_<g>/circuit<suffix>/, <run>/gen_<g>/head_circuit<suffix>/
  <run>/circuits<suffix>.h5, <run>/head_circuit<suffix>.h5
  <run>/head_circuit<suffix>_<target>_generations.{png,pdf}

Written for the extended global runs, whose value targets changed (see
circuits.value_at_query). From the repository root:
  .venv/bin/python -m scripts.recompute_circuits results/*_collapse_extended_global_*/
'''

import argparse
import glob
import os

import jax

from src import circuits
from src import head_circuit
from src import main_utils
from src.main import make_eval_data


def final_checkpoint(gen_folder):
  ckpts = sorted(glob.glob(os.path.join(gen_folder, 'checkpoints', '*.eqx')))
  if not ckpts:
    raise FileNotFoundError('No checkpoint in ' + gen_folder)
  return ckpts[-1]


def recompute(run, suffix):
  config = os.path.join(run, 'config.json')
  circuits_h5 = os.path.join(run, 'circuits{}.h5'.format(suffix))
  head_h5 = os.path.join(run, 'head_circuit{}.h5'.format(suffix))
  for f in (circuits_h5, head_h5):
    if os.path.exists(f):
      raise FileExistsError('{} exists; pick another --suffix'.format(f))
  gens = sorted(int(d.split('_')[-1]) for d in os.listdir(run) if d.startswith('gen_'))
  data = splits = prompts = eval_prompts = None
  for g in gens:
    gen_folder = os.path.join(run, 'gen_{}'.format(g))
    ckpt = final_checkpoint(gen_folder)
    print('==', run, 'gen', g, ckpt, flush=True)
    opts, model = circuits.load_checkpoint(ckpt, config)
    if data is None:
      # As in main.run_with_opts: the same prompts and eval data for every generation
      data = main_utils.get_data_from_opts(opts)
      splits = main_utils.get_splits_from_opts(opts, data.shape)
      prompts = circuits.real_prompts(opts, data, splits, opts.circuit_n_prompts, opts.circuit_seed)
      eval_data_seed, _ = jax.random.split(jax.random.PRNGKey(opts.eval_seed), 2)
      eval_prompts = make_eval_data(opts, data, splits, eval_data_seed).get('fsl_test_class')
    circuit_folder = os.path.join(gen_folder, 'circuit' + suffix)
    circuits.run_circuit_discovery(model, opts, prompts, circuit_folder, data, splits['exemplar']['train'],
                                   opts.class_split[0], gen=g, h5_file=circuits_h5)
    if not opts.no_head_circuit:
      head_circuit.run_head_circuit(model, opts, prompts, eval_prompts,
                                    os.path.join(gen_folder, 'head_circuit' + suffix), data,
                                    splits['exemplar']['train'], opts.class_split[0], gen=g,
                                    h5_file=head_h5, circuit_folder=circuit_folder)
  if os.path.exists(head_h5) and len(gens) > 1:
    from plotting.head_circuit_plots import plot_generations
    for target in ('value_exemplar_swap', 'value_label_swap'):
      plot_generations(head_h5, os.path.join(run, 'head_circuit' + suffix), target)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Recompute circuits.h5 and head_circuit.h5 from saved checkpoints')
  parser.add_argument('runs', nargs='+', help='Run folders, e.g. results/<stamp>_collapse_extended_global_real_only')
  parser.add_argument('--suffix', default='_fixed', help='Appended to every output file and folder name')
  args = parser.parse_args()
  for run in args.runs:
    recompute(run.rstrip('/'), args.suffix)
