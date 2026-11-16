"""Day 56 figures.  Every number is computed here or by importing kv_cache.py.
No CJK anywhere - the serif fonts in this container have no CJK glyphs."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import kv_cache as K                                 # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


# ---------------------------------------------------------------- figure 0
def fig0():
    """What gets recomputed, with and without a cache."""
    fig, ax = W(5.2)
    n, cw = 8, 0.40
    for side, x0, title, col in ((0, 0.95, 'no cache: every step redoes everything', RED_S),
                                 (1, 5.75, 'with a cache: one new row per step', TEAL)):
        for step in range(n):
            for j in range(step + 1):
                fresh = (side == 0) or (j == step)
                c = col if fresh else GREY
                ax.add_patch(Rectangle((x0 + j * cw, 4.05 - step * cw), cw * .88,
                                       cw * .88, facecolor=c if fresh else NAVY,
                                       edgecolor=c, lw=1.0, alpha=.9, zorder=3))
        text(ax, x0 + n * cw / 2, 4.75, title, color=col, size=12)
    text(ax, 0.95 + n * cw / 2, 0.76, 'work per step grows with the sequence',
         color=RED_S, size=11.5, glowing=False)
    text(ax, 5.75 + n * cw / 2, 0.76, 'work per step stays constant',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 5.35, 'each row is one generated token; each cell is a key and value '
                        'that had to be computed', color=TEAL_L, size=12)
    text(ax, 5.0, 0.28, 'the grey cells are read out of the cache instead of being '
                        'recomputed', color=TEAL, size=11.5, glowing=False)
    save(fig, 'day56_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """How much memory one token of cache costs."""
    fig, ax = W(5.0)
    rows = [('toy model here', 2, 4, 16, None, TEAL),
            ('32 layers, 32 kv heads', 32, 32, 128, None, RED_S),
            ('32 layers, GQA 8 kv heads', 32, 32, 128, 8, AMBER),
            ('80 layers, GQA 8 kv heads', 80, 64, 128, 8, TEAL_L)]
    vals = [K.kv_bytes_per_token(L, H, dh, 2, kv) for _, L, H, dh, kv, _ in rows]
    base, maxh, bw = 2.35, 2.00, 1.45
    for i, ((name, L, H, dh, kv, col), b) in enumerate(zip(rows, vals)):
        x = 0.85 + i * 2.25
        h = max(0.05, maxh * b / max(vals))
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, f'{b / 1024:,.0f} KB', color=col, size=12)
        text(ax, x + bw / 2, base - 0.30, f'{b * 131072 / 2**30:.1f} GB at 128k',
             color=TEAL, size=11)
        text(ax, x + bw / 2, base - 0.70, name, color=GREY_L, size=10.5, glowing=False)
    text(ax, 5.0, 5.00, 'KV cache per token, fp16 - and it is per sequence',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.95, 'two arrays, one entry per layer per kv head per token; the '
                        'queries are not kept because nothing reads them twice',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.45, 'sharing K and V across heads (GQA) is the cheapest way to make '
                        'this number smaller', color=CYAN, size=12)
    save(fig, 'day56_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Arithmetic intensity: prefill climbs, decode is pinned at 0.5."""
    fig, ax = W(5.0)
    X0, Y0, WD, HT = 1.55, 1.65, 6.9, 2.70
    axes2(ax, X0, Y0, WD, HT)
    Ts = [128, 512, 2048, 8192, 32768]
    pre = [K.phase_costs(32, 4096, 16384, T)['prefill_ai'] for T in Ts]
    dec = [K.phase_costs(32, 4096, 16384, T)['decode_ai'] for T in Ts]

    def ly(v):
        return Y0 + HT * (math.log10(max(v, 0.1)) + 1) / 6.0

    xs = [X0 + WD * i / (len(Ts) - 1) for i in range(len(Ts))]
    ax.plot(xs, [ly(v) for v in pre], color=TEAL_L, lw=2.6, zorder=4)
    ax.plot(xs, [ly(v) for v in dec], color=RED_S, lw=2.6, zorder=4)
    ax.plot([X0, X0 + WD], [ly(100)] * 2, color=AMBER, lw=1.3, ls='--', zorder=3)
    text(ax, X0 + 0.12, ly(100) + 0.24, 'roughly where a GPU stops being memory-bound',
         color=AMBER, size=11, ha='left')
    text(ax, xs[-1] - 0.1, ly(pre[-1]) + 0.30, f'prefill: {pre[-1]:,.0f} MACs/byte',
         color=TEAL_L, size=11.5, ha='right')
    text(ax, xs[-1] - 0.1, ly(dec[-1]) + 0.30, f'decode: {dec[-1]:.2f} MACs/byte, '
                                               f'at every context', color=RED_S,
         size=11.5, ha='right')
    for v, lab in ((0.1, '0.1'), (1, '1'), (100, '100'), (10000, '10,000')):
        text(ax, X0 - 0.20, ly(v), lab, color=TEAL, size=10.5, ha='right')
    for i, T in enumerate(Ts):
        text(ax, xs[i], Y0 - 0.30, f'{T:,}', color=TEAL, size=10.5)
    text(ax, X0 + WD / 2, Y0 - 0.72, 'tokens in the context', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.30, 'multiply-accumulates per byte moved',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.52, 'in decode every weight and every cached number is loaded to be '
                        'used exactly once: one MAC per two bytes',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day56_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """A cache that was not reset."""
    fig, ax = W(4.8)
    p, cfg = K.init_params()
    st = K.stale_cache(p, cfg)
    text(ax, 5.0, 4.55, 'starting a second sequence without clearing the cache',
         color=TEAL_L, size=13)
    boxes = [('sequence 1', 24, GREY, 'still in the cache'),
             ('sequence 2', 12, TEAL, 'the tokens you meant')]
    x = 1.15
    for name, n, col, note in boxes:
        w = n * 0.17
        ax.add_patch(Rectangle((x, 2.90), w, 0.52, facecolor=col, edgecolor=col,
                               lw=1.2, alpha=.8, zorder=3))
        text(ax, x + w / 2, 3.16, name, color=WHITE, size=11.5, glowing=False)
        text(ax, x + w / 2, 2.58, note, color=col if col != GREY else GREY_L, size=11,
             glowing=False)
        x += w + 0.15
    text(ax, 1.15, 1.95, f'what the new tokens could attend to: '
                         f'{st["stale"] * 100:.0f}% of it belonged to sequence 1',
         color=RED_S, size=11.5, ha='left', glowing=False)
    text(ax, 1.15, 1.45, f'largest change in the logits: {st["worst"]:.4f}  '
                         f'(mean {st["mean"]:.4f})', color=TEAL, size=11.5, ha='left',
         glowing=False)
    text(ax, 1.15, 0.95, f'the most likely next token changes on {st["flips"]} of '
                         f'{st["steps"]} steps', color=TEAL, size=11.5, ha='left',
         glowing=False)
    text(ax, 5.0, 0.40, 'nothing raises: the shapes are right and the softmax still '
                        'sums to one', color=CYAN, size=12)
    save(fig, 'day56_3.png')


for f in (fig0, fig1, fig2, fig3):
    f()
    print(f.__name__, 'ok')
