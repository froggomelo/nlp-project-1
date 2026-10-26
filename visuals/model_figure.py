'''
Draws the base and extended models side by side: the input tokens, the
two-layer attention-only transformer, and which output head predicts what at
which position.

Base:     [x1, l1, x2, l2, xq]          value head at xq -> lq
Extended: [x1, l1, x2, l2, xq, lq, xn]  value head at xq -> lq,
                                        symbol head at lq -> class of xn,
                                        value head at xn -> ln

Shaded boxes are model outputs. The extended model's lq and xn inputs (shaded,
dashed) are the true tokens during training (teacher forcing) and the model's own
sampled outputs when it generates data for the next generation.

python -I visuals/model_figure.py            # side by side
python -I visuals/model_figure.py --stack    # base on top, extended below
'''
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from task_figure import load_glyphs, BOX, STEP  # noqa: E402

OUTPUT_FILL = '#d6e4f5'
BLOCK_Y = 1.5          # transformer block bottom, above the token row
BLOCK_H = 1.25
HEAD_Y = BLOCK_Y + BLOCK_H + 0.45
HEAD_H = 0.62
OUT_Y = HEAD_Y + HEAD_H + 0.45
INCH_PER_UNIT = 0.42

# One entry per input token: (item, label, fed back). An item is a number or a
# (character, drawer) Omniglot glyph, as in task_figure.py.
BASE_TOKENS = [
  (('A', 1), r'$x_1$', False), (0, r'$\ell_1$', False),
  (('B', 1), r'$x_2$', False), (1, r'$\ell_2$', False),
  (('A', 2), r'$x_q$', False),
]
EXTENDED_TOKENS = BASE_TOKENS + [(0, r'$\ell_q$', True), (('B', 3), r'$x_n$', True)]
# Position -> (output item, head name, label)
BASE_OUTPUTS = {4: (0, 'value\nhead', r'$\ell_q$')}
EXTENDED_OUTPUTS = {
  4: (0, 'value\nhead', r'$\ell_q$'),
  5: (('B', 3), 'symbol\nhead', r'$x_n$'),
  6: (1, 'value\nhead', r'$\ell_n$'),
}


def draw_box(ax, x, y, item, glyphs, fill='white', dashed=False):
  ax.add_patch(FancyBboxPatch((x, y), BOX, BOX, boxstyle='round,pad=0,rounding_size=0.14',
                              linewidth=1.3, edgecolor='black', facecolor=fill, zorder=2,
                              linestyle=(0, (3, 2)) if dashed else '-'))
  if isinstance(item, tuple):
    inset = 0.17
    # Black strokes on a transparent background, so the box fill shows around them
    rgba = np.zeros(glyphs[item].shape + (4,))
    rgba[..., 3] = 1 - glyphs[item]
    ax.imshow(rgba, interpolation='lanczos', zorder=3,
              extent=(x + inset, x + BOX - inset, y + inset, y + BOX - inset))
  else:
    ax.text(x + BOX / 2, y + BOX / 2, str(item), ha='center', va='center', fontsize=14, zorder=3)


def arrow(ax, x, y0, y1):
  ax.annotate('', xy=(x, y1), xytext=(x, y0),
              arrowprops=dict(arrowstyle='-|>', lw=1.1, color='black', mutation_scale=11))


def draw_panel(ax, x0, y0, tokens, outputs, glyphs, title):
  '''Draws one model with its token row at (x0, y0). Returns the panel's right edge.'''
  n = len(tokens)
  right = x0 + (n - 1) * STEP + BOX
  for i, (item, label, fed_back) in enumerate(tokens):
    x = x0 + i * STEP
    draw_box(ax, x, y0, item, glyphs, fill=OUTPUT_FILL if fed_back else 'white', dashed=fed_back)
    ax.text(x + BOX / 2, y0 - 0.12, label, ha='center', va='top', fontsize=12)
    arrow(ax, x + BOX / 2, y0 + BOX + 0.05, y0 + BLOCK_Y - 0.03)

  ax.add_patch(FancyBboxPatch((x0 - 0.15, y0 + BLOCK_Y), right - x0 + 0.3, BLOCK_H,
                              boxstyle='round,pad=0,rounding_size=0.2', linewidth=1.3,
                              edgecolor='black', facecolor='0.94', zorder=1))
  ax.text((x0 + right) / 2, y0 + BLOCK_Y + BLOCK_H / 2,
          'attention-only transformer\n2 layers, 8 heads, causal',
          ha='center', va='center', fontsize=10, linespacing=1.4, zorder=2)

  for i, (item, head, label) in outputs.items():
    x = x0 + i * STEP
    arrow(ax, x + BOX / 2, y0 + BLOCK_Y + BLOCK_H + 0.03, y0 + HEAD_Y - 0.03)
    ax.add_patch(FancyBboxPatch((x - 0.06, y0 + HEAD_Y), BOX + 0.12, HEAD_H,
                                boxstyle='round,pad=0,rounding_size=0.1', linewidth=0.9,
                                edgecolor='0.35', facecolor='white', zorder=2))
    ax.text(x + BOX / 2, y0 + HEAD_Y + HEAD_H / 2, head, ha='center', va='center',
            fontsize=7.5, linespacing=1.1, color='0.25', zorder=3)
    arrow(ax, x + BOX / 2, y0 + HEAD_Y + HEAD_H + 0.03, y0 + OUT_Y - 0.03)
    draw_box(ax, x, y0 + OUT_Y, item, glyphs, fill=OUTPUT_FILL)
    ax.text(x + BOX / 2, y0 + OUT_Y + BOX + 0.1, label, ha='center', va='bottom', fontsize=12)

  ax.text(x0 - 0.15, y0 + OUT_Y + BOX + 0.15, title, ha='left', va='bottom',
          fontsize=13, fontweight='bold')
  return right


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--zip', default=os.path.join(HERE, 'omniglot', 'images_background.zip'))
  parser.add_argument('--out', default=os.path.join(HERE, 'model_figure'),
                      help='output path without extension; writes .pdf and .png')
  parser.add_argument('--stack', action='store_true', help='base on top, extended below')
  opts = parser.parse_args()

  glyphs = load_glyphs(opts.zip)
  fig, ax = plt.subplots()
  ax.set_aspect('equal')
  ax.axis('off')

  top = OUT_Y + BOX + 0.75
  if opts.stack:
    gap = 0.7
    lower = -(top + 0.6 + gap)
    base_right = draw_panel(ax, 0, 0, BASE_TOKENS, BASE_OUTPUTS, glyphs, 'Base')
    right = draw_panel(ax, 0, lower, EXTENDED_TOKENS, EXTENDED_OUTPUTS, glyphs, 'Extended')
    split = lower + top + gap / 2 - 0.1
    ax.plot([-0.3, right + 0.3], [split, split], color='0.55', lw=1.0, ls=(0, (5, 4)))
    xlim, ylim = (-0.35, right + 0.35), (lower - 0.6, top)
  else:
    gap = 1.3
    base_right = draw_panel(ax, 0, 0, BASE_TOKENS, BASE_OUTPUTS, glyphs, 'Base')
    ext_x = base_right + gap
    right = draw_panel(ax, ext_x, 0, EXTENDED_TOKENS, EXTENDED_OUTPUTS, glyphs, 'Extended')
    split = base_right + gap / 2
    ax.plot([split, split], [-0.6, top], color='0.55', lw=1.0, ls=(0, (5, 4)))
    xlim, ylim = (-0.35, right + 0.35), (-0.6, top)

  ax.set_xlim(*xlim)
  ax.set_ylim(*ylim)
  fig.set_size_inches((xlim[1] - xlim[0]) * INCH_PER_UNIT, (ylim[1] - ylim[0]) * INCH_PER_UNIT)
  fig.tight_layout(pad=0.1)
  for ext in ('pdf', 'png'):
    fig.savefig('{}.{}'.format(opts.out, ext), dpi=200, bbox_inches='tight')
  print('wrote {}.pdf and {}.png'.format(opts.out, opts.out))


if __name__ == '__main__':
  main()
