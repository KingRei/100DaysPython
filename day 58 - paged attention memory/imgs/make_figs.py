"""Day 58 figures.  Every number is computed here or by importing paged_memory.py.
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
import paged_memory as P                             # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'

REQS = P.make_requests()
FRAG = P.fragmentation(REQS)
WASTE = P.internal_waste(REQS)
SAMP = P.parallel_sampling()
PRE_R = P.preemption(policy='recompute')
PRE_S = P.preemption(policy='swap')


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


# ---------------------------------------------------------------- architecture
def fig_arch():
    """Block table per request, one shared pool, and who decides what happens."""
    fig, ax = W(6.4)
    text(ax, 5.0, 6.80, 'PagedAttention is a page table for the KV cache - and a '
                        'memory manager around it', color=TEAL_L, size=12.5)

    # requests with block tables
    text(ax, 0.40, 6.20, 'per request: a block table', color=GREY_L, size=11,
         ha='left', glowing=False)
    tables = [('request A', [0, 3, 7], 5.45, AMBER),
              ('request B', [1, 4], 4.55, TEAL),
              ('request C', [2, 5, 6], 3.65, PURPLE)]
    for name, blocks, y, col in tables:
        box(ax, 0.40, y, 1.65, 0.52, label=name, edge=col, lw=1.5, size=10.5)
        for k, b in enumerate(blocks):
            box(ax, 2.25 + k * 0.72, y, 0.62, 0.52, label=str(b), edge=col, lw=1.3,
                size=10.5)
    text(ax, 2.95, 3.25, 'the table holds block ids, not addresses', color=GREY_L,
         size=10.5, glowing=False)

    # the pool
    text(ax, 5.15, 6.20, 'one pool of fixed-size blocks', color=GREY_L, size=11,
         ha='left', glowing=False)
    owner = {0: AMBER, 3: AMBER, 7: AMBER, 1: TEAL, 4: TEAL, 2: PURPLE, 5: PURPLE,
             6: PURPLE}
    for b in range(12):
        col = owner.get(b, GREY)
        x = 5.15 + (b % 4) * 1.10
        y = 5.45 - (b // 4) * 0.80
        ax.add_patch(Rectangle((x, y), 0.95, 0.62, facecolor=col if b in owner else NAVY,
                               edgecolor=col, lw=1.4, alpha=.85, zorder=3))
        text(ax, x + 0.47, y + 0.31, str(b), color=WHITE if b in owner else GREY_L,
             size=10.5, glowing=False)
    text(ax, 7.35, 2.95, 'free blocks are interchangeable:\nno "wrong shape" of memory',
         color=TEAL, size=10.5, glowing=False)

    # the manager
    box(ax, 0.40, 1.65, 9.20, 0.62, label='the block manager: allocate on demand, '
                                          'refcount for sharing, evict or preempt when '
                                          'full', edge=CYAN, lw=1.7, size=11)
    for x in (2.6, 5.0, 8.6):
        ax.annotate('', xy=(x, 2.32), xytext=(x, 2.95),
                    arrowprops=dict(arrowstyle='->', color=GREY_L, lw=1.1, alpha=.8))
    text(ax, 5.0, 1.05, f'{FRAG["requests"]} requests, a 16,384-slot pool: contiguous '
                        f'refuses {FRAG["rejected_contiguous"]} (all of them with room '
                        f'free), paged refuses {FRAG["rejected_paged"]}',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.55, 'the kernel change is one indirection; everything else is '
                        'operating systems', color=CYAN, size=12.5)
    save(fig, 'day58_arch.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """External fragmentation: free memory in the wrong shape."""
    fig, ax = W(5.0)
    text(ax, 5.0, 5.05, 'the same free memory, two ways to hand it out',
         color=TEAL_L, size=12.5)
    # contiguous strip with holes
    text(ax, 0.45, 4.30, 'one contiguous range per request', color=GREY_L, size=11,
         ha='left', glowing=False)
    segs = [(0.0, 1.6, AMBER), (1.6, 0.5, None), (2.1, 2.2, TEAL), (4.3, 0.6, None),
            (4.9, 1.9, PURPLE), (6.8, 0.7, None), (7.5, 1.6, AMBER)]
    for x, w, col in segs:
        ax.add_patch(Rectangle((0.45 + x, 3.45), w - 0.05, 0.62,
                               facecolor=col or NAVY, edgecolor=col or GREY_L,
                               lw=1.3, alpha=.85, zorder=3))
    text(ax, 2.15, 3.15, 'free', color=GREY_L, size=10, glowing=False)
    text(ax, 5.05, 3.15, 'free', color=GREY_L, size=10, glowing=False)
    text(ax, 7.60, 3.15, 'free', color=GREY_L, size=10, glowing=False)
    text(ax, 5.0, 2.70, 'a request needing 1.8 of these units is refused, although '
                        '1.8 are free', color=RED_S, size=11.5, glowing=False)
    # paged strip
    text(ax, 0.45, 2.20, 'fixed-size blocks', color=GREY_L, size=11, ha='left',
         glowing=False)
    cols = [AMBER, AMBER, TEAL, None, PURPLE, AMBER, None, TEAL, PURPLE, None, AMBER,
            TEAL, None, PURPLE, AMBER, None]
    for i, col in enumerate(cols):
        ax.add_patch(Rectangle((0.45 + i * 0.57, 1.35), 0.50, 0.62,
                               facecolor=col or NAVY, edgecolor=col or GREY_L,
                               lw=1.3, alpha=.85, zorder=3))
    text(ax, 5.0, 0.95, 'the same four free blocks are usable by anyone, in any order',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.45, f'measured: {FRAG["rejected_fragmentation"]} of '
                        f'{FRAG["rejected_contiguous"]} contiguous refusals happened '
                        f'with enough free memory', color=CYAN, size=12)
    save(fig, 'day58_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Block size: waste against table size."""
    fig, ax = W(5.0)
    X0, Y0, WD, HT = 1.55, 1.70, 6.9, 2.60
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(WASTE) - 1) for i in range(len(WASTE))]
    top = max(r['waste'] for r in WASTE) * 1.12
    ax.plot(xs, [Y0 + HT * r['waste'] / top for r in WASTE], color=RED_S, lw=2.6,
            zorder=4)
    tmax = max(r['tables'] for r in WASTE)
    ax.plot(xs, [Y0 + HT * r['tables'] / tmax for r in WASTE], color=TEAL_L, lw=2.6,
            zorder=4)
    for i, r in enumerate(WASTE):
        text(ax, xs[i], Y0 - 0.30, str(r['block']), color=TEAL, size=10.5)
    text(ax, xs[-1] - 0.1, Y0 + HT * WASTE[-1]['waste'] / top + 0.30,
         f"wasted slots: {WASTE[-1]['waste'] * 100:.1f}%", color=RED_S, size=11.5,
         ha='right')
    text(ax, xs[0] + 0.15, Y0 + HT * 0.92,
         f"block table entries: {WASTE[0]['tables']:,}", color=TEAL_L, size=11.5,
         ha='left')
    text(ax, xs[2], Y0 + HT * WASTE[2]['waste'] / top - 0.34,
         f"16 tokens: {WASTE[2]['waste'] * 100:.1f}% wasted, "
         f"{WASTE[2]['tables']:,} entries", color=AMBER, size=11.5)
    text(ax, X0 + WD / 2, Y0 - 0.70, 'tokens per block', color=TEAL, size=11.5)
    text(ax, 5.0, 0.52, 'bigger blocks waste more and index less - and they coarsen '
                        'the granularity of prefix sharing', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day58_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Sharing and copy-on-write, and what preemption costs."""
    fig, ax = W(5.2)
    text(ax, 5.0, 5.30, 'sharing one prompt, and giving blocks back when the pool runs '
                        'out', color=TEAL_L, size=12.5)
    base, maxh, bw = 2.95, 1.55, 1.50
    vals = [('one copy per sample', SAMP['naive'], RED_S),
            ('shared + copy-on-write', SAMP['with_sharing'], TEAL_L)]
    top = max(v for _, v, _ in vals)
    for i, (name, v, col) in enumerate(vals):
        x = 1.05 + i * 2.05
        h = max(0.05, maxh * v / top)
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, f'{v} blocks', color=col, size=11.5)
        text(ax, x + bw / 2, base - 0.30, name, color=TEAL, size=10.5)
    text(ax, 2.35, 2.20, f'4 samples of a 900-token prompt: {SAMP["saved"] * 100:.0f}% '
                         f'fewer blocks,\nof which {SAMP["cow_blocks"]} are copies of '
                         f'the partly-full last block', color=GREY_L, size=10.5,
         glowing=False)
    text(ax, 7.00, 4.30, 'when the pool runs out', color=AMBER, size=12)
    rows = [('preemptions', f"{PRE_R['preemptions']} / {PRE_S['preemptions']}"),
            ('recompute policy', f"{PRE_R['recomputed_tokens']:,} tokens re-prefilled"),
            ('swap policy', f"{PRE_S['swapped_tokens']:,} tokens copied out and back"),
            ('both finish', f"all {PRE_R['finished']} requests")]
    for i, (k, v) in enumerate(rows):
        text(ax, 5.35, 3.75 - i * 0.45, k, color=GREY_L, size=11, ha='left',
             glowing=False)
        text(ax, 7.25, 3.75 - i * 0.45, v, color=TEAL, size=11, ha='left',
             glowing=False)
    text(ax, 5.0, 1.30, 'compute or bandwidth, the same trade as the tier below - '
                        'and far better than deadlocking', color=TEAL, size=11.5,
         glowing=False)
    text(ax, 5.0, 0.70, 'reference counts make sharing safe; a policy makes running '
                        'out survivable', color=CYAN, size=12)
    save(fig, 'day58_3.png')


for f in (fig_arch, fig1, fig2, fig3):
    f()
    print(f.__name__, 'ok')
