'''Readable static views of auto-circuit edge lists.

Each circuit is rendered twice:

* a compact component graph, similar to an ACDC diagram;
* a left-to-right weighted flow (Sankey-style) graph.

Both views use the same edges, selected by absolute attribution score.  The
module deliberately reads the JSON outputs rather than the PyTorch score file,
so old runs can be plotted without importing torch or auto-circuit.
'''

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


POSITIVE = '#4f46e5'
NEGATIVE = '#ef4444'
NODE_COLOURS = ['#fb923c', '#f472b6', '#a3e635', '#38bdf8', '#c084fc']


def _component(port_name):
  '''Returns (display component, input port, layer, within-layer index).'''
  if port_name == 'Resid Start':
    return 'Embed', '', -1, 0
  if port_name == 'Resid End':
    return 'Output', '', 10**6, 0
  match = re.fullmatch(r'(A(\d+)\.(\d+))(?:\.([QKV]))?', port_name)
  if match:
    return match.group(1), match.group(4) or '', int(match.group(2)), int(match.group(3))
  # auto-circuit calls MLP nodes M0, M1, ... when the model has MLPs.
  match = re.fullmatch(r'(?:M|MLP\s?)(\d+)(?:\.([A-Za-z]+))?', port_name)
  if match:
    layer = int(match.group(1))
    return 'M{}'.format(layer), match.group(2) or '', layer, 10**5
  return port_name, '', 10**5, 0


def _normalise_edges(edges):
  normalised = []
  for edge in edges:
    src, _, src_layer, src_index = _component(edge['src'])
    dest, port, dest_layer, dest_index = _component(edge['dest'])
    normalised.append({
      **edge, 'src_component': src, 'dest_component': dest,
      'dest_port': port, 'src_layer': src_layer, 'dest_layer': dest_layer,
      'src_index': src_index, 'dest_index': dest_index,
    })
  return normalised


def _select_edges(edges, summary, max_edges):
  if max_edges < 1:
    raise ValueError('max_edges must be at least 1')
  if any('in_circuit' in e for e in edges):
    # ACDC: the circuit is the set of kept edges, not a top-k by score
    circuit = [e for e in edges if e.get('in_circuit')]
    return _normalise_edges(circuit[:max_edges]), len(circuit)
  recovery_edges = int(summary.get('n_edges_to_recover', len(edges)))
  n_shown = min(max(recovery_edges, 5), max_edges, len(edges))
  return _normalise_edges(edges[:n_shown]), recovery_edges


def _nodes_and_columns(edges):
  meta = {}
  max_attention_layer = max(
    [e['src_layer'] for e in edges if e['src_layer'] < 10**5] +
    [e['dest_layer'] for e in edges if e['dest_layer'] < 10**5] + [-1])
  for edge in edges:
    meta[edge['src_component']] = (edge['src_layer'], edge['src_index'])
    meta[edge['dest_component']] = (edge['dest_layer'], edge['dest_index'])
  columns = defaultdict(list)
  for node, (layer, index) in meta.items():
    if node == 'Embed':
      column = 0
    elif node == 'Output':
      column = max_attention_layer + 2
    else:
      column = layer + 1
    columns[column].append(node)
  for column in columns:
    columns[column].sort(key=lambda n: (meta[n][1], n))
  return meta, dict(columns)


def _basic_layout(edges):
  _, columns = _nodes_and_columns(edges)
  max_column = max(columns, default=1)
  positions = {}
  for column, nodes in columns.items():
    x = 0.08 + 0.84 * column / max(max_column, 1)
    if len(nodes) == 1:
      ys = [0.5]
    else:
      ys = [0.88 - i * 0.76 / (len(nodes) - 1) for i in range(len(nodes))]
    positions.update({node: (x, y) for node, y in zip(nodes, ys)})
  return positions, columns


def _node_colour(node):
  if node == 'Embed':
    return '#fedf89'
  if node == 'Output':
    return '#a7f3d0'
  match = re.match(r'[AM](\d+)', node)
  layer = int(match.group(1)) if match else 0
  return NODE_COLOURS[layer % len(NODE_COLOURS)]


def _subtitle(summary, n_shown, recovery_edges):
  if 'threshold' in summary:
    selected = 'ACDC circuit, tau = {:g}: {} edges'.format(summary['threshold'], recovery_edges)
    if n_shown < recovery_edges:
      selected += ' (top {} shown)'.format(n_shown)
    return selected + '  |  {:.0%} recovery  |  width = KL increase when ablated'.format(summary['recovery'])
  recovery = 100 * float(summary.get('recovery_target', 0.8))
  if n_shown < recovery_edges:
    selected = 'top {} shown  |  first tested >= {:.0f}% recovery: {} edges'.format(
      n_shown, recovery, recovery_edges)
  else:
    selected = '{} edges shown  |  first tested >= {:.0f}% recovery'.format(
      n_shown, recovery)
  full = summary.get('logit_diff_full')
  if full is not None:
    selected += '  |  full-model logit difference {:.2f}'.format(full)
  return selected


def draw_component_graph(edges, summary, output_path, title='Circuit'):
  '''Draws an ACDC-style graph whose nodes are attention heads/components.'''
  positions, columns = _basic_layout(edges)
  fig, ax = plt.subplots(figsize=(14, 8))
  fig.patch.set_facecolor('white')
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis('off')

  max_score = max((abs(e['score']) for e in edges), default=1.0) or 1.0
  pairs = defaultdict(list)
  for edge in edges:
    pairs[(edge['src_component'], edge['dest_component'])].append(edge)

  for (src, dest), pair_edges in pairs.items():
    sx, sy = positions[src]
    dx, dy = positions[dest]
    for i, edge in enumerate(pair_edges):
      offset = i - (len(pair_edges) - 1) / 2
      rad = 0.08 * offset
      colour = POSITIVE if edge['score'] >= 0 else NEGATIVE
      width = 0.7 + 4.3 * math.sqrt(abs(edge['score']) / max_score)
      line_style = {'Q': ':', 'K': '--', 'V': '-'}.get(edge['dest_port'], '-')
      arrow = FancyArrowPatch(
        (sx, sy), (dx, dy), arrowstyle='-|>', mutation_scale=12,
        connectionstyle='arc3,rad={}'.format(rad), linewidth=width,
        linestyle=line_style, color=colour, alpha=0.55,
        shrinkA=25, shrinkB=25, zorder=1)
      ax.add_patch(arrow)

  for node, (x, y) in positions.items():
    width = 0.085 if node not in ('Embed', 'Output') else 0.095
    box = FancyBboxPatch(
      (x - width / 2, y - 0.028), width, 0.056,
      boxstyle='round,pad=0.008,rounding_size=0.014',
      facecolor=_node_colour(node), edgecolor='#374151', linewidth=1.3, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, node, ha='center', va='center', fontsize=10, color='#111827', zorder=4)

  for column, nodes in columns.items():
    x = positions[nodes[0]][0]
    label = 'Input' if column == min(columns) else ('Output' if column == max(columns)
            else 'Layer {}'.format(column - 1))
    ax.text(x, 0.965, label, ha='center', va='center', fontsize=11,
            fontweight='bold', color='#4b5563')

  n_shown = len(edges)
  recovery_edges = int(summary.get('n_edges_to_recover', n_shown))
  ax.set_title('{}\n{}'.format(title, _subtitle(summary, n_shown, recovery_edges)),
               fontsize=16, pad=18)
  ax.legend(handles=[
    Line2D([0], [0], color=POSITIVE, lw=3, label='positive attribution'),
    Line2D([0], [0], color=NEGATIVE, lw=3, label='negative attribution'),
    Line2D([0], [0], color='#6b7280', lw=2, ls=':', label='Q input'),
    Line2D([0], [0], color='#6b7280', lw=2, ls='--', label='K input'),
    Line2D([0], [0], color='#6b7280', lw=2, ls='-', label='V / output'),
  ], loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=5, frameon=False)
  fig.savefig(output_path, dpi=220, bbox_inches='tight', facecolor='white')
  plt.close(fig)


def _flow_layout(edges):
  '''Lays out weighted nodes and allocates input/output bands for ribbons.'''
  _, columns = _nodes_and_columns(edges)
  max_column = max(columns, default=1)
  incoming = defaultdict(float)
  outgoing = defaultdict(float)
  for edge in edges:
    value = abs(edge['score'])
    outgoing[edge['src_component']] += value
    incoming[edge['dest_component']] += value
  totals = {n: max(incoming[n], outgoing[n]) for nodes in columns.values() for n in nodes}

  boxes = {}
  for column, nodes in columns.items():
    gap = 0.025
    available = 0.78 - gap * (len(nodes) - 1)
    raw_total = sum(totals[n] for n in nodes) or 1.0
    heights = {n: max(0.024, available * totals[n] / raw_total) for n in nodes}
    scale = min(1.0, available / sum(heights.values()))
    heights = {n: h * scale for n, h in heights.items()}
    used = sum(heights.values()) + gap * (len(nodes) - 1)
    top = 0.5 + used / 2
    x = 0.07 + 0.86 * column / max(max_column, 1)
    for node in nodes:
      height = heights[node]
      boxes[node] = (x, top - height, height)
      top -= height + gap

  # Keep similarly directed ribbons adjacent to reduce crossings.
  outgoing_edges = defaultdict(list)
  incoming_edges = defaultdict(list)
  for edge in edges:
    outgoing_edges[edge['src_component']].append(edge)
    incoming_edges[edge['dest_component']].append(edge)
  for node, node_edges in outgoing_edges.items():
    node_edges.sort(key=lambda e: boxes[e['dest_component']][1])
  for node, node_edges in incoming_edges.items():
    node_edges.sort(key=lambda e: boxes[e['src_component']][1])

  source_bands, dest_bands = {}, {}
  for node, node_edges in outgoing_edges.items():
    _, y, height = boxes[node]
    total = outgoing[node] or 1.0
    cursor = y
    for edge in node_edges:
      band = height * abs(edge['score']) / total
      source_bands[id(edge)] = (cursor, cursor + band)
      cursor += band
  for node, node_edges in incoming_edges.items():
    _, y, height = boxes[node]
    total = incoming[node] or 1.0
    cursor = y
    for edge in node_edges:
      band = height * abs(edge['score']) / total
      dest_bands[id(edge)] = (cursor, cursor + band)
      cursor += band
  return boxes, columns, source_bands, dest_bands


def _ribbon_path(x0, x1, source_band, dest_band):
  sy0, sy1 = source_band
  dy0, dy1 = dest_band
  bend = 0.48 * (x1 - x0)
  vertices = [
    (x0, sy0), (x0 + bend, sy0), (x1 - bend, dy0), (x1, dy0),
    (x1, dy1), (x1 - bend, dy1), (x0 + bend, sy1), (x0, sy1), (x0, sy0),
  ]
  codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
           MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
           MplPath.CLOSEPOLY]
  return MplPath(vertices, codes)


def draw_sankey_graph(edges, summary, output_path, title='Circuit flow'):
  '''Draws a Sankey-style graph with ribbon width proportional to |score|.'''
  boxes, columns, source_bands, dest_bands = _flow_layout(edges)
  fig, ax = plt.subplots(figsize=(14, 8))
  fig.patch.set_facecolor('white')
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis('off')
  node_width = 0.022

  # Small ribbons first leaves the strongest routes visually dominant.
  for edge in sorted(edges, key=lambda e: abs(e['score'])):
    sx, _, _ = boxes[edge['src_component']]
    dx, _, _ = boxes[edge['dest_component']]
    path = _ribbon_path(sx + node_width / 2, dx - node_width / 2,
                        source_bands[id(edge)], dest_bands[id(edge)])
    colour = POSITIVE if edge['score'] >= 0 else NEGATIVE
    ax.add_patch(PathPatch(path, facecolor=colour, edgecolor='none', alpha=0.30, zorder=1))

  for node, (x, y, height) in boxes.items():
    ax.add_patch(Rectangle((x - node_width / 2, y), node_width, height,
                           facecolor=_node_colour(node), edgecolor='white',
                           linewidth=0.8, zorder=3))
    side = 1 if x < 0.82 else -1
    ax.text(x + side * (node_width / 2 + 0.007), y + height / 2, node,
            ha='left' if side == 1 else 'right', va='center', fontsize=11,
            color='#1f2937', zorder=4)

  for column, nodes in columns.items():
    x = boxes[nodes[0]][0]
    label = 'Input' if column == min(columns) else ('Output' if column == max(columns)
            else 'Layer {}'.format(column - 1))
    ax.text(x, 0.965, label, ha='center', va='center', fontsize=11,
            fontweight='bold', color='#4b5563')

  n_shown = len(edges)
  recovery_edges = int(summary.get('n_edges_to_recover', n_shown))
  ax.set_title('{}\n{}'.format(title, _subtitle(summary, n_shown, recovery_edges)),
               fontsize=16, pad=18)
  ax.legend(handles=[
    Line2D([0], [0], color=POSITIVE, lw=7, alpha=0.5, label='positive attribution'),
    Line2D([0], [0], color=NEGATIVE, lw=7, alpha=0.5, label='negative attribution'),
    Line2D([0], [0], color='#6b7280', lw=2, label='ribbon width = |edge score|'),
  ], loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False)
  fig.savefig(output_path, dpi=220, bbox_inches='tight', facecolor='white')
  plt.close(fig)


def render_circuit_views(edges, summary, prefix, title=None, max_edges=40,
                         formats=('png', 'pdf')):
  '''Writes both graph styles for one circuit and returns their paths.'''
  selected, recovery_edges = _select_edges(edges, summary, max_edges)
  if not selected:
    return []  # e.g. an ACDC circuit that kept no edges
  # Store the target explicitly so the subtitle is unambiguous when capped.
  summary = {**summary, 'n_edges_to_recover': recovery_edges, 'recovery_target': 0.8}
  prefix = Path(prefix)
  title = title or prefix.name.replace('_', ' ').title()
  written = []
  for extension in formats:
    component_path = prefix.with_name(prefix.name + '_component.' + extension)
    sankey_path = prefix.with_name(prefix.name + '_sankey.' + extension)
    draw_component_graph(selected, summary, component_path, title + ' — component view')
    draw_sankey_graph(selected, summary, sankey_path, title + ' — flow view')
    written.extend([component_path, sankey_path])
  return written


def render_from_json(edges_path, max_edges=40, formats=('png', 'pdf')):
  edges_path = Path(edges_path)
  suffix = '_edges.json'
  if not edges_path.name.endswith(suffix):
    raise ValueError('expected a file ending in {}'.format(suffix))
  prefix = edges_path.with_name(edges_path.name[:-len(suffix)])
  faithfulness_path = prefix.with_name(prefix.name + '_faithfulness.json')
  edges = json.loads(edges_path.read_text())
  summary = json.loads(faithfulness_path.read_text())['summary']
  return render_circuit_views(edges, summary, prefix, max_edges=max_edges, formats=formats)


def main():
  parser = argparse.ArgumentParser(description='Draw component and Sankey views of circuit JSON files')
  parser.add_argument('path', help='An *_edges.json file or a directory containing them')
  parser.add_argument('--max-edges', type=int, default=40,
                      help='Maximum edges to draw per view (default: 40)')
  parser.add_argument('--formats', nargs='+', choices=['png', 'pdf'], default=['png', 'pdf'])
  args = parser.parse_args()
  path = Path(args.path)
  edge_files = [path] if path.is_file() else sorted(path.rglob('*_edges.json'))
  if not edge_files:
    parser.error('no *_edges.json files found at {}'.format(path))
  for edge_file in edge_files:
    for output in render_from_json(edge_file, args.max_edges, args.formats):
      print(output)


if __name__ == '__main__':
  main()
