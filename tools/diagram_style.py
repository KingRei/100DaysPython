"""
100 Days of Python - shared diagram style.

Reproduces the visual language used since day 08:
  * transparent background
  * dark navy filled blocks / nodes with thin teal outlines
  * teal serif text with a soft cyan glow
  * teal arrows; red for collisions / failure paths
  * amber-orange for secondary annotations
  * dashed pale-cyan regions to group things

Usage
-----
    import sys; sys.path.append('../tools')
    from diagram_style import *

    fig, ax = canvas(10, 6)
    box(ax, 1, 1, 2, 1, label="key : 'a'\\nvalue: 5")
    node(ax, 6, 3, '1')
    arrow(ax, (3, 1.5), (5.4, 3))
    save(fig, 'imgs/example.png')
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, FancyArrowPatch,
                                FancyBboxPatch, PathPatch)
from matplotlib.path import Path
import matplotlib.patheffects as pe

# --------------------------------------------------------------------------
# palette (sampled straight out of day 08-11 png files)
# --------------------------------------------------------------------------
NAVY   = '#002639'   # block / node fill
TEAL   = '#0093a1'   # outlines, arrows, body text
TEAL_L = '#00aaaa'   # lighter teal, big labels
CYAN   = '#00ffff'   # glow colour only
GREY   = '#292c30'   # index cells / secondary blocks
GREY_L = '#818181'
RED    = '#ff0000'   # collision, failure, "don't do this"
AMBER  = '#ffc000'   # secondary annotation
YELLOW = '#ffff00'
WHITE  = '#ffffff'
PALE   = '#d6f2f4'   # dashed region fill

SERIF = ['Latin Modern Roman', 'Liberation Serif', 'DejaVu Serif']

plt.rcParams['font.family'] = SERIF
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['savefig.transparent'] = True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def glow(color=CYAN, layers=4, base=2.6, alpha=0.13):
    """Soft outer glow, the signature look of this series."""
    fx = [pe.withStroke(linewidth=base + 2.2 * i, foreground=color,
                        alpha=alpha) for i in reversed(range(layers))]
    return fx + [pe.Normal()]


def canvas(w=10.0, h=6.0, xlim=None, ylim=None, dpi=140):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.set_xlim(*(xlim or (0, w)))
    ax.set_ylim(*(ylim or (0, h)))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


GLOW_MAX_CHARS = 16   # longer than this and the neon glow hurts readability


def text(ax, x, y, s, color=TEAL, size=15, ha='center', va='center',
         weight='normal', glowing=None, style='normal', zorder=5, **kw):
    """Draw a label.

    ``glowing=None`` (the default) decides automatically: short labels get the
    signature cyan glow, longer sentences are drawn as plain text because the
    glow smears them and makes them hard to read. Pass True/False to force it.
    """
    if glowing is None:
        longest = max((len(line) for line in str(s).split('\n')), default=0)
        glowing = longest <= GLOW_MAX_CHARS
    return ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=zorder,
                   path_effects=glow() if glowing else None, **kw)


def box(ax, x, y, w, h, label=None, fill=NAVY, edge=TEAL, lw=1.6,
        color=WHITE, size=14, zorder=3, **kw):
    """Filled rectangle with optional centred label. (x, y) = lower-left."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=lw, zorder=zorder, **kw))
    if label is not None:
        text(ax, x + w / 2, y + h / 2, label, color=color, size=size,
             glowing=False, zorder=zorder + 1)


def cells(ax, x, y, n, w=1.0, h=0.8, labels=None, horizontal=True,
          fill=NAVY, index=False, index_fill=GREY, size=14, **kw):
    """A row (or column) of array cells. Returns list of (cx, cy) centres."""
    out = []
    for i in range(n):
        cx = x + i * w if horizontal else x
        cy = y if horizontal else y - i * h
        lab = None if labels is None else labels[i]
        box(ax, cx, cy, w, h, label=lab, fill=index_fill if index else fill,
            size=size, **kw)
        out.append((cx + w / 2, cy + h / 2))
    return out


def node(ax, x, y, label='', r=0.42, fill=NAVY, edge=TEAL, lw=1.8,
         color=WHITE, size=14, zorder=4):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                        linewidth=lw, zorder=zorder,
                        path_effects=glow(alpha=0.10)))
    if label != '':
        text(ax, x, y, str(label), color=color, size=size, glowing=False,
             zorder=zorder + 1)
    return (x, y)


def arrow(ax, p1, p2, color=TEAL, lw=1.8, rad=0.0, style='-|>', ms=12,
          shrinkA=2, shrinkB=2, ls='-', zorder=4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                 mutation_scale=ms, linewidth=lw,
                                 linestyle=ls, color=color,
                                 shrinkA=shrinkA, shrinkB=shrinkB,
                                 connectionstyle=f'arc3,rad={rad}',
                                 zorder=zorder))


def points_per_unit(ax):
    """How many typographic points one data unit is worth on this axes.

    ``FancyArrowPatch`` shrink values are in points, but node radii are in data
    units, so a directed edge needs this conversion - otherwise the arrowhead
    ends up underneath the node and disappears.
    """
    fig = ax.figure
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    return bbox.width / abs(x1 - x0) * 72.0 / fig.dpi


def edge(ax, p1, p2, color=TEAL, lw=1.8, r=0.42, directed=True, weight=None,
         rad=0.0, wsize=14, wcolor=TEAL_L):
    """Graph edge between two node centres, trimmed to the node radius."""
    gap = r * points_per_unit(ax)
    arrow(ax, p1, p2, color=color, lw=lw, rad=rad,
          style='-|>' if directed else '-',
          shrinkA=gap, shrinkB=gap)
    if weight is not None:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        off = 0.34
        text(ax, mx - dy / n * off, my + dx / n * off, str(weight),
             color=wcolor, size=wsize)


def region(ax, x, y, w, h, label=None, fill=PALE, edge_=TEAL_L, alpha=0.55,
           size=15, zorder=1):
    """Dashed pale-cyan grouping region (e.g. 'Hash Function')."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, alpha=alpha,
                           edgecolor=edge_, linewidth=1.4, linestyle='--',
                           zorder=zorder))
    if label:
        text(ax, x + w / 2, y + h + 0.28, label, color=TEAL_L, size=size)


def cross(ax, x, y, r=0.30, color=RED, label=None, lw=2.0):
    """The circled-X 'collision / failure' marker."""
    ax.add_patch(Circle((x, y), r, facecolor='none', edgecolor=color,
                        linewidth=lw, zorder=6))
    k = r * 0.72
    for s in (1, -1):
        ax.plot([x - k, x + k], [y - s * k, y + s * k], color=color,
                lw=lw, zorder=6, solid_capstyle='round')
    if label:
        text(ax, x, y - r - 0.42, label, color=color, size=17)


def brace(ax, x0, x1, y, height=0.28, color=AMBER, lw=1.8, down=False,
          label=None, size=15):
    """Curly brace spanning x0..x1 at height y."""
    s = -1 if down else 1
    mid = (x0 + x1) / 2
    h = height * s
    verts = [(x0, y), (x0, y + h * .6), (mid, y + h * .4), (mid, y + h),
             (mid, y + h * .4), (x1, y + h * .6), (x1, y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                           edgecolor=color, linewidth=lw, zorder=4))
    if label:
        text(ax, mid, y + h * 1.9, label, color=color, size=size)


def legend(ax, x, y, items, size=13, dy=0.42):
    """items = [(colour, 'meaning'), ...]"""
    for i, (c, s) in enumerate(items):
        yy = y - i * dy
        ax.add_patch(Rectangle((x, yy - 0.10), 0.26, 0.22, facecolor=c,
                               edgecolor=TEAL, linewidth=1.0, zorder=4))
        text(ax, x + 0.42, yy, s, color=TEAL, size=size, ha='left',
             glowing=False)


def save(fig, path, pad=0.15):
    fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)
    return path
