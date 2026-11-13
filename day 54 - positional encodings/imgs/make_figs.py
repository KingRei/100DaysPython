"""Day 54 figures.  Every number is computed here or by importing positional.py.
No CJK anywhere - the serif fonts in this container have no CJK glyphs."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle, Circle     # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import positional as P                               # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def liny(vals, y0, h, lo, hi):
    return [y0 + h * (v - lo) / (hi - lo) for v in vals]


# ---------------------------------------------------------------- figure 0
def fig0():
    """Where each scheme injects the position."""
    fig, ax = W(5.4)
    text(ax, 5.0, 5.55, 'attention is a set function: three places to put the order back',
         color=TEAL_L, size=13)
    cols = [
        ('sinusoidal / learned', AMBER, 'added to the input embedding',
         'x = embed(token) + PE[pos]', 'one vector per position,\nbefore anything else'),
        ('RoPE', TEAL_L, 'a rotation of q and k',
         'rot(q, pos), rot(k, pos)', 'inside attention,\nlengths unchanged'),
        ('ALiBi', PURPLE, 'a penalty on the scores',
         's = s - slope * (i - j)', 'after the scores,\nbefore the softmax'),
    ]
    for c, (name, col, what, formula, where) in enumerate(cols):
        x = 0.45 + c * 3.25
        box(ax, x, 3.70, 2.85, 0.60, label=name, edge=col, lw=1.8)
        text(ax, x + 1.42, 3.35, what, color=col, size=11.5, glowing=False)
        box(ax, x, 2.45, 2.85, 0.62, label=formula, edge=TEAL, lw=1.3, size=11)
        text(ax, x + 1.42, 1.85, where, color=GREY_L, size=11, glowing=False)
    text(ax, 5.0, 1.05, f'shuffle the tokens and the output shuffles with them to '
                        f'{P.permutation_test():.0e} - nothing in softmax(QK$^{{T}}$)V '
                        f'mentions an index', color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.52, 'so the choice is not whether to encode position, only where',
         color=CYAN, size=12.5)
    save(fig, 'day54_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """RoPE: one channel pair, rotating with position."""
    fig, ax = W(5.0)
    d, base = 64, 10000.0
    q = np.array([1.0, 0.0])
    for p, (cx, cy) in enumerate([(1.6, 3.0), (3.6, 3.0), (5.6, 3.0), (7.6, 3.0)]):
        pos = p * 3
        ang = pos / (base ** (0 / d))
        ax.add_patch(Circle((cx, cy), 0.85, facecolor='none', edgecolor=TEAL, lw=1.2,
                            zorder=2))
        ax.annotate('', xy=(cx + 0.85 * math.cos(ang), cy + 0.85 * math.sin(ang)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.2))
        text(ax, cx, cy - 1.25, f'position {pos}', color=TEAL_L, size=11.5)
    text(ax, 5.0, 4.55, 'RoPE rotates each (even, odd) channel pair by an angle '
                        'proportional to the position', color=TEAL_L, size=12.5)
    text(ax, 5.0, 1.78, 'shown for the fastest channel pair, which turns about one '
                        'radian per token', color=GREY_L, size=11, glowing=False)
    a, b, gap = P.rope_is_relative()
    text(ax, 5.0, 1.35, f'<RoPE(q, 3), RoPE(k, 11)> = {a:+.6f}     '
                        f'<RoPE(q, 100), RoPE(k, 108)> = {b:+.6f}',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.85, f'both pairs are 8 apart and the scores differ by {gap:.0e}: '
                        f'the dot product only knows the distance',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.35, 'a rotation changes direction, not length - and V is never touched',
         color=CYAN, size=12)
    save(fig, 'day54_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """ALiBi: one soft window per head."""
    fig, ax = W(5.0)
    rows = P.alibi_effective_window()
    X0, Y0, WD, HT = 1.45, 1.55, 7.0, 2.70
    axes2(ax, X0, Y0, WD, HT)
    top = max(r['window'] for r in rows) * 1.08
    bw = WD / len(rows) * 0.62
    for i, r in enumerate(rows):
        x = X0 + WD * (i + 0.5) / len(rows) - bw / 2
        hh = HT * r['window'] / top
        col = AMBER if i < 4 else TEAL
        ax.add_patch(Rectangle((x, Y0), bw, hh, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, Y0 + hh + 0.24, f"{r['window']:,}", color=col, size=11.5)
        text(ax, x + bw / 2, Y0 - 0.30, f"{r['slope']:.4f}", color=TEAL, size=10.5)
    text(ax, X0 + WD / 2, Y0 - 0.74, 'slope, one per head', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.30, 'tokens holding 99% of the weight, with no '
                                        'content signal', color=GREY_L, size=11,
         ha='left', glowing=False)
    text(ax, 5.0, 0.52, 'the penalty is a function of distance, so "further" always '
                        'means "weaker" - at any sequence length',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day54_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """How often a token beyond the trained length wins."""
    fig, ax = W(4.8)
    fw = P.far_token_wins()
    names = [('sinusoidal', AMBER), ('rope', TEAL_L), ('alibi', PURPLE)]
    base, maxh = 1.70, 2.30
    for i, (k, col) in enumerate(names):
        v = fw[k]
        x = 1.85 + i * 2.25
        hh = max(0.04, maxh * v)
        ax.add_patch(Rectangle((x, base), 1.30, hh, facecolor=col, edgecolor=col,
                               lw=1.2, alpha=.85, zorder=3))
        text(ax, x + 0.65, base + hh + 0.26, f'{v * 100:.1f}%', color=col, size=13)
        text(ax, x + 0.65, base - 0.32, k, color=TEAL, size=12)
    text(ax, 5.0, 4.55, 'how often a token past the trained length out-scores '
                        'everything inside it', color=TEAL_L, size=12.5)
    text(ax, 5.0, 0.95, 'trained length 32, distances up to 512, 200 random query-key '
                        'pairs, no training involved', color=GREY_L, size=11,
         glowing=False)
    text(ax, 5.0, 0.45, 'ALiBi cannot do it by construction; that is what "extrapolates '
                        'for free" means, and what it costs', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day54_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """What a block mask skips."""
    fig, ax = W(4.8)
    n, block, window = 1024, 128, 256
    nb = n // block
    qi = np.arange(n)[:, None]
    kj = np.arange(n)[None, :]
    masks = [('causal', P.mod_causal()(qi, kj), 3.20),
             (f'sliding {window}', P.mod_sliding(window)(qi, kj), 6.60)]
    cw = 0.30
    for name, allowed, x0 in masks:
        live = 0
        for a in range(nb):
            for b in range(nb):
                tile = allowed[a * block:(a + 1) * block, b * block:(b + 1) * block]
                on = tile.any()
                live += on
                full = tile.all()
                col = TEAL if full else (AMBER if on else GREY)
                ax.add_patch(Rectangle((x0 + b * cw, 3.10 - a * cw), cw * .88, cw * .88,
                                       facecolor=col if on else NAVY, edgecolor=col,
                                       lw=1.0, alpha=.9, zorder=3))
        text(ax, x0 + nb * cw / 2, 3.72, name, color=TEAL_L, size=12)
        text(ax, x0 + nb * cw / 2, 0.82, f'{live} of {nb * nb} tiles computed',
             color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 4.40, '1,024 tokens, 128-wide tiles: what a block mask lets the '
                        'kernel skip', color=TEAL_L, size=12.5)
    legend(ax, 0.45, 2.75, [(TEAL, 'every pair allowed'), (AMBER, 'partly allowed'),
                            (GREY, 'nothing allowed:\nnever computed')], size=11)
    text(ax, 5.0, 0.30, 'score_mod and mask_mod are functions, so the kernel stays '
                        'fused and the dead tiles are never built',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day54_4.png')


for f in (fig0, fig1, fig2, fig3, fig4):
    f()
    print(f.__name__, 'ok')
