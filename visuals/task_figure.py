'''
Draws the in-context learning task in the style of Singh et al. (2024), Figure 1b,
with the base task on top and the extended task below.

Base:     [x1, l1, x2, l2, xq] -> lq
Extended: [x1, l1, x2, l2, xq] -> lq, xn, ln
          (next symbol xn is a fresh exemplar of a context class, ln its context value)

The symbols are real Omniglot characters, read straight from images_background.zip
(https://github.com/brendenlake/omniglot), so nothing needs extracting.

python -I visuals/task_figure.py
python -I visuals/task_figure.py --zip visuals/omniglot/images_background.zip --out visuals/task_figure
'''
import argparse
import io
import os
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# The four characters, as <alphabet>/<character>. A and B are the bursty pair of the
# opening rows, C and D the pair of the last row.
CHARACTERS = {
  'A': 'Bengali/character05',
  'B': 'Tifinagh/character11',
  'C': 'Gujarati/character20',
  'D': 'Inuktitut_(Canadian_Aboriginal_Syllabics)/character03',
}

# One sequence per row: context pairs, query, targets. A symbol is (character, drawer),
# so the same class can appear as a different handwritten exemplar. None draws dots.
BASE_ROWS = [
  [('A', 1), 0, ('B', 1), 1, ('A', 2), [0]],
  [('A', 1), 3, ('B', 1), 0, ('A', 2), [3]],
  [('B', 1), 0, ('A', 1), 1, ('A', 2), [1]],
  None,
  [('C', 1), 4, ('D', 1), 3, ('D', 2), [3]],
]
EXTENDED_ROWS = [
  [('A', 1), 0, ('B', 1), 1, ('A', 2), [0, ('B', 3), 1]],
  [('A', 1), 3, ('B', 1), 0, ('A', 2), [3, ('A', 3), 3]],
  [('B', 1), 0, ('A', 1), 1, ('A', 2), [1, ('B', 3), 0]],
  None,
  [('C', 1), 4, ('D', 1), 3, ('D', 2), [3, ('C', 3), 4]],
]

BOX = 1.0              # box side
STEP = 1.22            # distance between box left edges
QUERY_X = 4 * STEP + 0.3
ARROW_GAP = 1.15       # space for the arrow between query and first target
TARGET_X = QUERY_X + BOX + ARROW_GAP
ROW_STEP = 1.3
DOT_STEP = 0.9         # height of a row of dots
PANEL_GAP = 1.6        # space between the panels, holding the dividing line
INCH_PER_UNIT = 0.55   # figure size per box side; the font sizes are set for this scale


def load_glyphs(zip_path):
  '''Returns (character, drawer) -> greyscale image cropped to its strokes.'''
  glyphs = {}
  with zipfile.ZipFile(zip_path) as z:
    for key, char in CHARACTERS.items():
      folder = 'images_background/{}/'.format(char)
      files = sorted(n for n in z.namelist() if n.startswith(folder) and n.endswith('.png'))
      if not files:
        raise SystemExit('no images for {} in {}'.format(char, zip_path))
      for drawer, name in enumerate(files, start=1):
        img = np.asarray(Image.open(io.BytesIO(z.read(name))).convert('L'), dtype=float) / 255
        ys, xs = np.nonzero(img < 0.5)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        crop = img[y0:y1, x0:x1]
        side = max(crop.shape)  # pad to a square so the glyph keeps its aspect ratio
        square = np.ones((side, side))
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        glyphs[(key, drawer)] = square
  return glyphs


def draw_box(ax, x, y, item, glyphs):
  ax.add_patch(FancyBboxPatch((x, y), BOX, BOX, boxstyle='round,pad=0,rounding_size=0.14',
                              linewidth=1.3, edgecolor='black', facecolor='white', zorder=2))
  if isinstance(item, tuple):
    inset = 0.17
    ax.imshow(glyphs[item], cmap='gray', vmin=0, vmax=1, interpolation='lanczos', zorder=3,
              extent=(x + inset, x + BOX - inset, y + inset, y + BOX - inset))
  else:
    ax.text(x + BOX / 2, y + BOX / 2, str(item), ha='center', va='center', fontsize=15, zorder=3)


def draw_row(ax, y, row, glyphs):
  *context, query, targets = row
  for i, item in enumerate(context):
    draw_box(ax, i * STEP, y, item, glyphs)
  draw_box(ax, QUERY_X, y, query, glyphs)
  ax.annotate('', xy=(TARGET_X - 0.04, y + BOX / 2), xytext=(QUERY_X + BOX + 0.04, y + BOX / 2),
              arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black', mutation_scale=14))
  for j, item in enumerate(targets):
    draw_box(ax, TARGET_X + j * STEP, y, item, glyphs)


def draw_dots(ax, y):
  x = QUERY_X - (STEP - BOX) / 2 - 0.05
  for k in range(3):
    ax.plot(x, y + DOT_STEP * (0.25 + 0.25 * k), 'o', color='black', markersize=2.2)


def underline(ax, x0, x1, y, text):
  ax.plot([x0, x1], [y, y], color='black', lw=1.0)
  ax.text((x0 + x1) / 2, y - 0.22, text, ha='center', va='top', fontsize=11, linespacing=1.25)


def draw_panel(ax, top, rows, glyphs, n_targets, title, target_names):
  '''Draws one panel with its first row at height `top` and returns the panel bottom.'''
  y = top
  for row in rows:
    if row is None:
      y -= DOT_STEP
      draw_dots(ax, y)
    else:
      y -= ROW_STEP
      draw_row(ax, y, row, glyphs)
  ax.text(-0.35, top - 0.2, title, ha='left', va='bottom', fontsize=13, fontweight='bold')
  line_y = y - 0.25
  underline(ax, 0, 3 * STEP + BOX, line_y, 'context\n(2 exemplar-label pairs)')
  underline(ax, QUERY_X, QUERY_X + BOX, line_y, 'query')
  if n_targets == 1:
    underline(ax, TARGET_X, TARGET_X + BOX, line_y, 'target\noutput')
  else:
    # A shared line under the target boxes, with one name per box beneath it.
    underline(ax, TARGET_X, TARGET_X + (n_targets - 1) * STEP + BOX, line_y, '')
    for j, name in enumerate(target_names):
      ax.text(TARGET_X + j * STEP + BOX / 2, line_y - 0.22, name, ha='center', va='top',
              fontsize=11, linespacing=1.25)
  return line_y - 1.0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--zip', default=os.path.join(HERE, 'omniglot', 'images_background.zip'))
  parser.add_argument('--out', default=os.path.join(HERE, 'task_figure'),
                      help='output path without extension; writes .pdf and .png')
  opts = parser.parse_args()

  glyphs = load_glyphs(opts.zip)
  right = TARGET_X + 2 * STEP + BOX + 0.3
  fig, ax = plt.subplots()
  ax.set_aspect('equal')
  ax.axis('off')

  bottom = draw_panel(ax, 0, BASE_ROWS, glyphs, 1, 'Base', None)
  split = bottom - PANEL_GAP / 2 + 0.6
  ax.plot([-0.35, right], [split, split], color='0.55', lw=1.0, ls=(0, (5, 4)))
  bottom = draw_panel(ax, split - 0.5, EXTENDED_ROWS, glyphs, 3, 'Extended',
                      ['query\nvalue', 'next\nsymbol', 'next\nvalue'])

  ax.set_xlim(-0.45, right)
  ax.set_ylim(bottom + 0.4, 0.6)
  # The figure follows the content, at INCH_PER_UNIT inches per box side.
  fig.set_size_inches((right + 0.45) * INCH_PER_UNIT, (0.6 - bottom - 0.4) * INCH_PER_UNIT)
  fig.tight_layout(pad=0.2)
  for ext in ('pdf', 'png'):
    fig.savefig('{}.{}'.format(opts.out, ext), dpi=200, bbox_inches='tight')
  print('wrote {}.pdf and {}.png'.format(opts.out, opts.out))


if __name__ == '__main__':
  main()
