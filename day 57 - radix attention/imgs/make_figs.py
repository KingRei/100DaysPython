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
    """The algorithm: walk while the tokens agree, then split the edge."""
    fig, ax = W(6.4)
    text(ax, 5.0, 6.70, 'match(tokens): walk down while the tokens agree; '
                        'insert(): split the edge where they stop',
         color=TEAL_L, size=12.5)

    def chain(x0, y, runs, hi=-1, w=0.52):
        """Draw a node chain: each run is (label, n_tokens)."""
        x = x0
        for k, (lab, n) in enumerate(runs):
            ww = max(0.95, 0.34 * n)
            col = AMBER if k == hi else TEAL
            box(ax, x, y, ww, w, label=lab, edge=col, lw=1.7, size=11)
            if k < len(runs) - 1:
                ax.annotate('', xy=(x + ww + 0.22, y + w / 2), xytext=(x + ww + 0.02,
                            y + w / 2),
                            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))
            x += ww + 0.22
        return x

    # --- step 1: the tree already holds one request
    text(ax, 0.45, 5.95, '1.  the tree after one request', color=GREY_L, size=11.5,
         ha='left', glowing=False)
    text(ax, 0.45, 5.45, 'root', color=TEAL_L, size=11.5, ha='left')
    chain(1.35, 5.20, [('"S E A B"  (one edge)', 12)])

    # --- step 2: a second request walks down and diverges
    text(ax, 0.45, 4.45, '2.  a second request "S E A C" walks down', color=GREY_L,
         size=11.5, ha='left', glowing=False)
    text(ax, 0.45, 3.95, 'root', color=TEAL_L, size=11.5, ha='left')
    chain(1.35, 3.70, [('"S E A B"', 12)], hi=0)
    text(ax, 5.95, 3.96, 'matched "S E A", then B != C', color=AMBER, size=11.5,
         ha='left', glowing=False)

    # --- step 3: split
    text(ax, 0.45, 2.95, '3.  split the edge at the divergence', color=GREY_L,
         size=11.5, ha='left', glowing=False)
    text(ax, 0.45, 2.45, 'root', color=TEAL_L, size=11.5, ha='left')
    box(ax, 1.35, 2.20, 2.20, 0.52, label='"S E A"  shared', edge=AMBER, lw=1.8,
        size=11)
    box(ax, 4.35, 2.55, 1.30, 0.50, label='"B"', edge=TEAL, lw=1.5, size=11)
    box(ax, 4.35, 1.75, 1.30, 0.50, label='"C"', edge=TEAL, lw=1.5, size=11)
    ax.annotate('', xy=(4.30, 2.80), xytext=(3.60, 2.46),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))
    ax.annotate('', xy=(4.30, 2.00), xytext=(3.60, 2.46),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))
    text(ax, 5.95, 2.46, 'the shared head is now one node:\nthree tokens of cache, '
                         'computed once', color=AMBER, size=11.5, ha='left',
         glowing=False)

    text(ax, 5.0, 1.05, f'matching costs one comparison per token; a request of length '
                        f'n adds at most one split', color=TEAL, size=11.5,
         glowing=False)
    text(ax, 5.0, 0.55, f'on the workload in this file: {RES["hit"] * 100:.1f}% of '
                        f'{RES["total"]:,} prompt tokens were already in the tree',
         color=CYAN, size=12.5)
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


# ---------------------------------------------------------------- figure 4
def fig4():
    """Tiers: keep the branch somewhere slower instead of throwing it away."""
    fig, ax = W(5.4)
    rows, full, ideal, nreq = R.tier_table()
    text(ax, 5.0, 5.60, 'when one pool is not enough: demote instead of drop',
         color=TEAL_L, size=13)
    tiers = [('tier 1  GPU pool', '800 slots', AMBER, 4.35),
             ('tier 2  host DRAM', f"{rows[-1]['l2']:,} slots", TEAL, 3.45),
             ('tier 3  remote store', 'SSD / other machines', GREY_L, 2.55)]
    for name, note, col, y in tiers:
        box(ax, 0.75, y, 3.40, 0.62, label=name, edge=col, lw=1.7)
        text(ax, 4.35, y + 0.31, note, color=GREY_L, size=11, ha='left', glowing=False)
    for y0, y1 in ((4.35, 3.45), (3.45, 2.55)):
        ax.annotate('', xy=(2.10, y1 + 0.66), xytext=(2.10, y0 - 0.02),
                    arrowprops=dict(arrowstyle='->', color=RED_S, lw=1.5))
        ax.annotate('', xy=(2.80, y0 - 0.02), xytext=(2.80, y1 + 0.66),
                    arrowprops=dict(arrowstyle='->', color=TEAL_L, lw=1.5))
    text(ax, 1.62, 4.12, 'evict', color=RED_S, size=10.5, glowing=False)
    text(ax, 3.28, 4.12, 'fetch', color=TEAL_L, size=10.5, glowing=False)
    text(ax, 5.0, 1.85, f'{nreq} requests over 6 system prompts, a tree of {full:,} '
                        f'slots and a GPU pool of 800:', color=TEAL, size=11.5,
         glowing=False)
    text(ax, 5.0, 1.40, f"one pool: {rows[0]['hit'] * 100:.1f}% hit    "
                        f"with tier 2: {rows[-1]['hit'] * 100:.1f}%    "
                        f"(everything resident would be {ideal * 100:.1f}%)",
         color=TEAL_L, size=12)
    text(ax, 5.0, 0.95, f"{rows[-1]['from_l2']:,} tokens came back over the bus instead "
                        f"of being prefilled again", color=TEAL, size=11.5,
         glowing=False)
    text(ax, 5.0, 0.45, 'bandwidth instead of compute - which is the trade HiCache and '
                        'Mooncake are built on', color=CYAN, size=12)
    save(fig, 'day57_4.png')


for f in (fig0, fig1, fig2, fig3, fig4):
    f()
    print(f.__name__, 'ok')
