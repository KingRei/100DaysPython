"""Day 57 figures.  Every number is computed here or by importing radix_attention.py.
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
import radix_attention as R                          # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'

REQS = R.make_workload()
RES = R.run(REQS)
PLAIN = R.no_sharing(REQS)
ROWS, FULL = R.capacity_sweep(REQS)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


# ---------------------------------------------------------------- figure 0
def fig0():
    """The tree: one shared trunk, one branch per conversation."""
    fig, ax = W(5.2)
    text(ax, 5.0, 5.35, 'one radix tree over token ids: shared runs become shared cache',
         color=TEAL_L, size=12.5)
    box(ax, 0.55, 3.55, 2.30, 0.60, label='system prompt\n200 tokens', edge=AMBER, lw=1.8)
    box(ax, 3.05, 3.55, 2.30, 0.60, label='2 examples\n200 tokens', edge=AMBER, lw=1.8)
    arrow(ax, (2.90, 3.85), (3.00, 3.85))
    for i, y in enumerate((4.35, 3.05, 1.75)):
        lab = ('conversation 1', 'conversation 2', 'conversation 20')[i]
        box(ax, 5.85, y, 2.00, 0.52, label=lab, edge=TEAL, lw=1.5, size=12)
        ax.annotate('', xy=(5.80, y + 0.26), xytext=(5.40, 3.85),
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3, alpha=.8))
        for t in range(2):
            yy = y + (0.0 if t == 0 else -0.60)
            box(ax, 8.15, yy, 1.35, 0.50, label=f'turn {t + 1}', edge=TEAL_L, lw=1.2,
                size=12)
            ax.annotate('', xy=(8.10, yy + 0.25), xytext=(7.90, y + 0.26),
                        arrowprops=dict(arrowstyle='->', color=TEAL_L, lw=1.0,
                                        alpha=.7))
        if i == 1:
            text(ax, 6.85, y - 0.95, '. . .', color=GREY_L, size=13, glowing=False)
    text(ax, 2.0, 3.25, 'computed once, used by everything below',
         color=AMBER, size=11.5, glowing=False)
    text(ax, 5.0, 0.95, f'{len(REQS)} requests, {RES["total"]:,} tokens of prompt in '
                        f'total, of which {RES["reused"]:,} are a repeat of something '
                        f'already cached', color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.45, f'the tree holds {RES["slots"]:,} token slots instead of '
                        f'{PLAIN["slots"]:,}', color=CYAN, size=12.5)
    save(fig, 'day57_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """What the sharing saves."""
    fig, ax = W(4.8)
    new = RES['total'] - RES['reused']
    base, maxh, bw = 2.35, 2.10, 1.70
    vals = [('prefill with no sharing', PLAIN['total'], RED_S),
            ('actually new tokens', new, TEAL_L),
            ('slots held, no sharing', PLAIN['slots'], RED_S),
            ('slots held, one tree', RES['slots'], TEAL_L)]
    top = max(v for _, v, _ in vals)
    for i, (name, v, col) in enumerate(vals):
        x = 0.75 + i * 2.25
        h = max(0.05, maxh * v / top)
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, f'{v:,}', color=col, size=12)
        text(ax, x + bw / 2, base - 0.32, name, color=TEAL, size=10.5)
    text(ax, 5.0, 4.50, f'{RES["hit"] * 100:.1f}% of the prompt tokens in this workload '
                        f'had already been computed', color=TEAL_L, size=12.5)
    text(ax, 5.0, 1.05, 'prefill is the compute-bound phase, so not doing it is the '
                        'largest single saving available in serving',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.52, 'and the cache holds one copy of the shared trunk instead of '
                        'one per request', color=TEAL, size=11.5, glowing=False)
    save(fig, 'day57_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Hit rate against pool size: flat, then a cliff."""
    fig, ax = W(5.0)
    X0, Y0, WD, HT = 1.55, 1.65, 6.9, 2.70
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(ROWS) - 1) for i in range(len(ROWS))]
    ys = [Y0 + HT * r['hit'] for r in ROWS]
    ax.plot(xs, ys, color=TEAL_L, lw=2.6, zorder=4)
    for i, r in enumerate(ROWS):
        col = RED_S if r['hit'] < 0.5 else TEAL_L
        ax.add_patch(Rectangle((xs[i] - .07, ys[i] - .07), .14, .14, facecolor=col,
                               edgecolor=col, zorder=5))
        text(ax, xs[i], Y0 - 0.30, f"{r['cap']:,}", color=TEAL, size=10)
    for v, lab in ((0.0, '0'), (0.5, '50%'), (1.0, '100%')):
        text(ax, X0 - 0.20, Y0 + HT * v, lab, color=TEAL, size=10.5, ha='right')
    shared = len(R.SYSTEM) + 2 * len(R.EXAMPLES[0])
    text(ax, xs[3], ys[3] + 0.38, f"{ROWS[3]['hit'] * 100:.1f}% with "
                                  f"{ROWS[3]['frac'] * 100:.0f}% of the tree",
         color=TEAL_L, size=11.5)
    text(ax, xs[1] + 0.45, Y0 + 0.55, f'below {shared}-ish slots the cache cannot even '
                                      f'hold\nthe shared prefix, and every insert '
                                      f'evicts the last', color=RED_S, size=11,
         ha='left', glowing=False)
    text(ax, X0 + WD / 2, Y0 - 0.72, 'token slots in the pool', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.30, 'share of prompt tokens served from the cache',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.52, 'leaf-only LRU never evicts the shared trunk, so a pool a tenth '
                        'of the size gives the same hit rate', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day57_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """The two ways to get it wrong."""
    fig, ax = W(5.0)
    tp = R.prefix_on_text_not_tokens()
    ev = R.evict_internal_node(REQS)
    text(ax, 5.0, 5.00, 'two ways to share a prefix that should not be shared',
         color=TEAL_L, size=12.5)
    text(ax, 0.55, 4.25, 'matching text instead of tokens', color=AMBER, size=12,
         ha='left')
    text(ax, 0.75, 3.75, f'"{tp["a"]}"  ->  {tp["ta"]}', color=TEAL, size=12, ha='left',
         glowing=False)
    text(ax, 0.75, 3.35, f'"{tp["b"]}"  ->  {tp["tb"]}', color=TEAL, size=12, ha='left',
         glowing=False)
    text(ax, 0.75, 2.90, f'{tp["text_prefix"]} characters in common, '
                         f'{tp["token_prefix"]} tokens in common', color=RED_S,
         size=11.5, ha='left', glowing=False)
    text(ax, 0.75, 2.50, 'the tokeniser merged across the boundary, so the cached keys',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 0.75, 2.15, 'belong to a token the second request never contained',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.55, 4.25, 'evicting an internal node', color=AMBER, size=12, ha='left')
    text(ax, 5.75, 3.75, f'oldest node by LRU: {ev["victim_len"]} tokens',
         color=TEAL, size=11.5, ha='left', glowing=False)
    text(ax, 5.75, 3.35, f'children hanging off it: {ev["children"]}',
         color=TEAL, size=11.5, ha='left', glowing=False)
    text(ax, 5.75, 2.90, f'tokens orphaned with it: {ev["orphaned"]}', color=RED_S,
         size=11.5, ha='left', glowing=False)
    text(ax, 5.75, 2.50, 'the shared prefix has the oldest timestamp precisely',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.75, 2.15, 'because everyone matches through it, not onto it',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.0, 1.30, 'so: match on token ids, and evict leaves only',
         color=CYAN, size=13)
    text(ax, 5.0, 0.70, 'neither mistake raises anything - the shapes stay right and '
                        'the numbers stay finite', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day57_3.png')


for f in (fig0, fig1, fig2, fig3):
    f()
    print(f.__name__, 'ok')
