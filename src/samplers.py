'''
This file contains all utils to construct data samplers for
our synthetic datasets.
'''

import pdb
from typing import Any
from typing import Sequence
from typing import Union
from typing import Callable
from typing import Optional
from nptyping import NDArray
from nptyping import Int
from jax import Array

from enum import IntEnum
from enum import unique
from functools import partial

import jax
from jax import numpy as jnp


@unique
class ItemType(IntEnum):
  OTHER = 0
  BURSTY = 1
  DISTRACTOR = 2
  QUERY = 3
  # Next symbol after the query, only used by the extended model variant.
  NEXT = 4


def get_constant_burst_seq_idxs(
  key: Array,
  classes: Sequence[int],
  class_distr: Union[Array, NDArray],
  num_seqs: int,
  context_len: int,
  burstiness: int,
  distractor: bool = True,
  no_support: bool = False,
  unique_rest: bool = False,
  next_symbol: Optional[str] = None,
):
  """Samples a batch of sequences per various parameters

  Args:
    key: jax PRNG key
    classes: Sequence of class indices that will be present in the output
    class_distr: a multinomial distr over classes
    num_seqs: How many sequences to return
    context_len: length of context in returned sequences
    burstiness: # examples of query class present in context
    distractor: whether or not to include a distractor class that appears
      `burstiness` times
    no_support: whether or not to make the query not appear in context
    unique_rest: whether or not to make the every item in context (besides
      bursts and distractors) unique. Note when unique_rest is True, the
      query will not appear in the non-bursty context
    next_symbol: if not None, append a next symbol (ItemType.NEXT) after the
      query, for the extended model. 'context' picks uniformly among the
      distinct classes in context; 'global' draws from class_distr.

  return:
    a dictionary with two num_seqs x context_len+1 arrays (context_len+2
    if next_symbol is given):
    class_idxs: idxs from `classes` with the given conditions
    idx_types: ItemType's indicating which samples are from bursts, distractors, etc.
  """
  num_classes = len(classes)
  assert num_classes == len(class_distr)

  # Ensure normalization
  class_distr = class_distr / jnp.sum(class_distr)

  query_key, distractor_key, rest_key, shuffle_key = jax.random.split(key, 4)

  query = jax.random.categorical(
    query_key, jnp.log(class_distr), shape=(1, num_seqs)
  )
  mask = jnp.ones((num_seqs, num_classes)) - jnp.eye(num_classes)[query[0]]
  left_to_sample = context_len - burstiness

  def _sample_masked(k, m, n=1):
    new_probs = class_distr[None, :] * m
    new_probs = new_probs / jnp.sum(new_probs, axis=1)[:, None]
    out = jax.random.categorical(k, jnp.log(new_probs), shape=(n, num_seqs))
    return out

  if burstiness > 0 and distractor:
    assert burstiness * 2 <= context_len
    distractors = _sample_masked(distractor_key, mask)
    mask -= jnp.eye(num_classes)[distractors[0]]
    left_to_sample -= burstiness

  if unique_rest:
    new_probs = class_distr[None, :] * mask
    new_probs = new_probs / jnp.sum(new_probs, axis=1)[:, None]
    # Use a vmapped random choice without replacement to get remaining indices
    # Note we transpose as we want to concatenate and shuffle sequences in dim 0
    # (which gives slight performance boosts I think)
    context = jax.vmap(partial(jax.random.choice, 
                                a=num_classes, 
                                shape=(left_to_sample,), 
                                replace=False)
                      )(key=jax.random.split(rest_key, num_seqs), p=new_probs).T
  else:
    if no_support:
      context = _sample_masked(rest_key, mask, n=left_to_sample)
    else:
      context = jax.random.categorical(
        rest_key, jnp.log(class_distr), shape=(left_to_sample, num_seqs)
      )
  context_type = jnp.ones((left_to_sample, num_seqs), dtype=int)*ItemType.OTHER

  if burstiness > 0:
    context = jnp.concatenate(
      [context, jnp.broadcast_to(query, (burstiness, num_seqs))]
    )
    context_type = jnp.concatenate(
      [context_type, jnp.ones((burstiness, num_seqs), dtype=int)*ItemType.BURSTY]
    )
    if distractor:
      context = jnp.concatenate(
        [context, jnp.broadcast_to(distractors, (burstiness, num_seqs))]
      )
      context_type = jnp.concatenate(
        [context_type, jnp.ones((burstiness, num_seqs), dtype=int)*ItemType.DISTRACTOR]
      )

    # We only have to shuffle if we added bursty sequences.
    # If we didn't it's already random
    context = jax.random.permutation(
      shuffle_key, context, axis=0, independent=True
    )
    context_type = jax.random.permutation(
      shuffle_key, context_type, axis=0, independent=True
    )

  seq = jnp.concatenate([context, query])
  seq_type = jnp.concatenate([context_type,
                              jnp.ones((1,num_seqs), dtype=int)*ItemType.QUERY])

  if next_symbol is not None:
    # fold_in (rather than an extra split above) keeps base sequences unchanged
    next_key = jax.random.fold_in(key, ItemType.NEXT)
    if next_symbol == 'context':
      # Uniform over the distinct classes in context: only the first
      # occurrence of each class is eligible.
      ctx = context.T  # [num_seqs, context_len]
      earlier_same = (ctx[:, :, None] == ctx[:, None, :]) & jnp.tril(
        jnp.ones((context_len, context_len), dtype=bool), k=-1)
      first_occ = ~jnp.any(earlier_same, axis=-1)
      pos = jax.random.categorical(next_key, jnp.log(first_occ.astype(float)), axis=-1)
      nxt = ctx[jnp.arange(num_seqs), pos][None, :]
    elif next_symbol == 'global':
      nxt = jax.random.categorical(next_key, jnp.log(class_distr), shape=(1, num_seqs))
    else:
      raise NotImplementedError(next_symbol)
    seq = jnp.concatenate([seq, nxt])
    seq_type = jnp.concatenate([seq_type,
                                jnp.ones((1,num_seqs), dtype=int)*ItemType.NEXT])

  return {'class_idxs': classes[seq.T],
          'idx_types': seq_type.T}


def get_mixed_seq_idxs(
  key: Array,
  mix_probabilities: Union[Array, NDArray],
  mix_substrate_fns: Sequence[Any],
):
  """Returns a mixed batch using a bunch of underlying pieces

  Intended usage to mix a bunch of `get_constant_burst_seq_idxs` partials

  Args:
    mix_probabilities: what percentage to mix each substrate
    mix_substrate_fns: functions that map from a key to a batch of data

  Returns:
    a mixed batch of data with shape equal to that of a mix_substrate when
    called
  """
  assert len(mix_probabilities) == len(mix_substrate_fns)
  keys = jax.random.split(key, len(mix_substrate_fns) + 1)
  substrates = jax.tree_util.tree_map(lambda *args: jnp.stack(args), 
                  *[s(keys[i]) for i, s in enumerate(mix_substrate_fns)])
  num_seqs = substrates['class_idxs'][0].shape[0]
  which = jax.random.categorical(
    keys[-1], jnp.log(mix_probabilities), shape=(num_seqs,)
  )
  return jax.tree_util.tree_map(lambda x: x[which, jnp.arange(num_seqs)], substrates)


def get_exemplar_inds(
  key: Array,
  idx_types: NDArray[Any, Int],
  allowed_inds: Sequence[int],
  match_query_and_distractors: bool = False,
) -> NDArray[Any, Int]:
  """Samples a batch of exemplar indices.

  Even though this method currently just does a simple random choice,
  we abstract it away in case we want to do some more clever exemplar picking
  (such as e.g. ensuring the query exemplar is different from the support).

  Note we also assume all classes have the same # of exemplars (reasonable).

  Args:
    key: jax PRNG key
    idx_types: An array of ItemTypes (in case of clever picking), shaped 
      [num_seqs, context_len+1], describing the type at each index (see def above)
    allowed_inds: Allowed exemplar indices (in case we want to have)

  return:
    a num_seqs x context_len+1 array of indices from `allowed_inds`
  """
  if match_query_and_distractors:
    # the query and bursty exemplars should be the same, and the distractors
    # should also be the same for each sequence
    num_seqs, _ = idx_types.shape
    key, subkey = jax.random.split(key)
    
    # Sample query indices for each sequence
    query_indices = jax.random.choice(subkey, allowed_inds, shape=(num_seqs,))

    # Sample distractor indices for each sequence
    distractor_indices = jax.random.choice(key, allowed_inds, shape=(num_seqs,))

    # Create the output array with the same shape as idx_types
    exemplar_inds = jnp.zeros_like(idx_types)

    # Set the query and bursty indices
    exemplar_inds = jnp.where((idx_types == ItemType.BURSTY) | (idx_types == ItemType.QUERY), query_indices[:, None], exemplar_inds)

    # Set the distractor indices
    exemplar_inds = jnp.where(idx_types == ItemType.DISTRACTOR, distractor_indices[:, None], exemplar_inds)

    # Set the other indices
    other_mask = (idx_types == ItemType.OTHER) | (idx_types == ItemType.NEXT)
    other_indices = allowed_inds[jax.random.randint(key, shape=idx_types.shape, minval=0, maxval=len(allowed_inds))]
    exemplar_inds = jnp.where(other_mask, other_indices, exemplar_inds)

    return exemplar_inds

  return allowed_inds[jax.random.randint(key, shape=idx_types.shape, minval=0, maxval=len(allowed_inds))]


def fewshot_relabel(
  key: Array, 
  class_idxs: NDArray[Any, Int], 
  idx_types: NDArray[Any, Int],
  labels = jnp.array([[0,1]]),
  flip_labels: bool = False,
  num_labels: Optional[int] = None,):
  """Relabel class idxs randomly per sequence.

  We randomly pick ItemType.BURSTY, ItemType.DISTRACTOR -> fewshot_labels
  per sequence.

  key: jax PRNG key
  class_idxs: The output, shape [batch_size, context_len+1]
  idx_types: An array of ItemTypes (in case of clever picking), shape [batch_size, context_len+1]
  labels: Labels to choose from, shape [1, n_labels]
  num_labels: If given, ItemType.OTHER items (present when context_len >
    2 * burstiness) get a random label in 0..num_labels-1 not used by the
    pair, instead of keeping their class index (which can exceed the label
    range). Only the bursty items then carry the query's label.
  """
  assert len(labels) >= 1, "At least one label pair needed for fewshot relabeling"
  label_choice_key, order_choice_key = jax.random.split(key)
  # [batch_size, n_labels]
  labels_to_use = labels[jax.random.choice(label_choice_key, 
                                            labels.shape[0], 
                                            shape=(class_idxs.shape[0],))]
  # Randomly permute the labels for each sequence. [batch_size, n_labels]
  if flip_labels:
    assn = labels_to_use[:, ::-1]
  else:
    assn = jax.random.permutation(order_choice_key, labels_to_use, axis=1, independent=True)
  out = class_idxs
  # Assign exemplars of type BURSTY and DISTRACTOR to the first and second labels.
  out = jnp.where(idx_types == ItemType.BURSTY, assn[:, 0][:, None], out)
  out = jnp.where(idx_types == ItemType.DISTRACTOR, assn[:, 1][:, None], out)
  # Assign queries to the first label.
  out = jnp.where(idx_types == ItemType.QUERY, assn[:, 0][:, None], out)
  if num_labels is not None:
    used = jnp.any(jax.nn.one_hot(assn, num_labels) > 0, axis=1)  # [batch_size, num_labels]
    spare_logits = jnp.where(used, -jnp.inf, 0.0)[:, None, :]
    spare = jax.random.categorical(jax.random.fold_in(key, ItemType.OTHER),
                                   jnp.broadcast_to(spare_logits, out.shape + (num_labels,)), axis=-1)
    out = jnp.where(idx_types == ItemType.OTHER, spare, out)
  return out

def fewshot_fliplabels(
  class_idxs: NDArray[Any, Int], 
  idx_types: NDArray[Any, Int],):
  """Relabel class idxs randomly per sequence.

  We randomly pick ItemType.BURSTY, ItemType.DISTRACTOR -> fewshot_labels
  per sequence.

  class_idxs: The output, shape [batch_size, context_len+1]
  idx_types: An array of ItemTypes (in case of clever picking), shape [batch_size, context_len+1]
  """
  out = class_idxs.copy().flatten()
  # get indices of each ItemType
  bursty_idxs = jnp.where(idx_types.flatten() == ItemType.BURSTY)[0]
  query_idxs = jnp.where(idx_types.flatten() == ItemType.QUERY)[0]
  distractor_idxs = jnp.where(idx_types.flatten() == ItemType.DISTRACTOR)[0]
  # get assiacted bursty and distractor labels
  bursty_class_idxs = class_idxs.flatten()[bursty_idxs]
  distractor_class_idxs = class_idxs.flatten()[distractor_idxs]
  # derive burstiness
  bursty_fraction = len(bursty_idxs) / len(idx_types.flatten())
  seq_length = idx_types.shape[1]
  burstiness = int(bursty_fraction * seq_length)
  # set bursty indices to distractor class labels
  out = out.at[bursty_idxs].set(distractor_class_idxs)
  # set distractor indices to bursty class labels
  out = out.at[distractor_idxs].set(bursty_class_idxs)
  # set query indices to distractor class labels
  out = out.at[query_idxs].set(distractor_class_idxs[::burstiness])
  return out.reshape(*class_idxs.shape)


def label_next_symbol(
  key: Array,
  class_idxs: NDArray[Any, Int],
  labels: NDArray[Any, Int],
  num_labels: Optional[int],):
  """Labels the final (ItemType.NEXT) item of each sequence.

  If the next symbol's class occurs in context, it gets that item's label, so
  the value can be copied in-context. Otherwise it gets a label not used in
  context (drawn uniformly from 0..num_labels-1), or, when num_labels is None
  (no relabeling), its class index.

  Returns:
    labels with the final column replaced, and a [batch_size] bool array
    saying whether the next symbol occurs in context.
  """
  ctx_classes = class_idxs[:, :-2]
  ctx_labels = labels[:, :-2]
  match = ctx_classes == class_idxs[:, -1:]
  n_in_ctx = jnp.any(match, axis=-1)
  copied = ctx_labels[jnp.arange(labels.shape[0]), jnp.argmax(match, axis=-1)]
  if num_labels is None:
    fresh = class_idxs[:, -1]
  else:
    used = jnp.any(jax.nn.one_hot(ctx_labels, num_labels) > 0, axis=1)
    fresh = jax.random.categorical(key, jnp.where(used, -jnp.inf, 0.0), axis=-1)
  return labels.at[:, -1].set(jnp.where(n_in_ctx, copied, fresh)), n_in_ctx


def make_data_sampler(
  class_sampler: Callable, 
  exemplar_sampler: Callable, 
  fs_relabel: Optional[str] = None,
  noise_scale: float = 0.0,
  assign_query_label_random: bool = False,
  next_symbol: bool = False,
  num_labels: Optional[int] = None,):
  """Build a function which samples data as specified.

  Args:
      class_sampler (Callable): function which samples class labels for a batch 
        of contexts.
      exemplar_sampler (Callable): function which samples input examples for a 
        batch of contexts.
      fs_relabel (Optional[str], optional): Description of relabeling which 
        should be applied to examplars for few-shot eval. Defaults to None.
      noise_scale (float, optional): Std-dev of Gaussian noise to add to input 
        exemplars. Defaults to 0.0 (no noise).
      assign_query_label_random (bool, optional): Whether to assign the query 
        label to a random exemplar in-context. Defaults to False.
      next_symbol (bool, optional): Whether sequences end with a next symbol
        (extended model). If so, it is labeled by `label_next_symbol` and the
        output also contains 'n_in_ctx'. Defaults to False.
      num_labels (Optional[int], optional): Number of relabeled labels, used
        to pick fresh labels for next symbols not seen in context and for
        context items outside the burst and distractor.
  """
  assert not (next_symbol and assign_query_label_random), \
    "assign_query_label_random assumes the query is the final item"

  def sample(key: Array, data):
    keys = jax.random.split(key, 3)
    class_out = class_sampler(keys[0])
    exemplar_inds = exemplar_sampler(keys[1], class_out['idx_types'])
    labels = class_out['class_idxs']  # [batch_size, context_len+1]
    if assign_query_label_random:
      n_examples, input_length = labels.shape
      query_labels = labels[:, -1]
      # randomly pick an index from each context
      _, assign_key = jax.random.split(key)
      assign_idxs = jax.random.randint(
        assign_key, shape=(n_examples,), minval=0, maxval=input_length-1)
      # assign the query label to the randomly picked index
      labels = labels.at[jnp.arange(n_examples), assign_idxs].set(query_labels)
    if fs_relabel is not None and fs_relabel == 'flip':
      labels = fewshot_fliplabels(**class_out)
    elif fs_relabel is not None:
      labels = fewshot_relabel(keys[2], labels=fs_relabel, num_labels=num_labels, **class_out)
    out = dict()
    if next_symbol:
      labels, out['n_in_ctx'] = label_next_symbol(
        jax.random.fold_in(key, ItemType.NEXT), class_out['class_idxs'], labels,
        num_labels if fs_relabel is not None else None)
    examples = data[class_out['class_idxs'], exemplar_inds]
    if noise_scale > 0:
      _, noise_key = jax.random.split(key)
      examples += noise_scale * jax.random.normal(noise_key, examples.shape)
    return {'examples': examples, 'labels': labels,
            'symbols': class_out['class_idxs'], **out}
  return sample

