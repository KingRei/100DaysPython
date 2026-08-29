# -*- coding: utf-8 -*-
"""Figures for day 21 - selection and top-k.

Every number that appears on a figure is computed by importing selection.py,
never typed in by hand, so a change to the algorithm shows up in the diagrams.
"""
import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *                      # noqa: F401,F403
import selection as S                            # the day's actual code

# --------------------------------------------------------------- measurements
rng = random.Random(20260821)
N, K = 200_000, 8
data = [rng.random() for _ in range(N)]
st_sort, st_heap, st_qs = ({'cmp': 0} for _ in range(3))
S.heap_sort(list(data), st_sort)
S.heap_topk(data, K, st_heap)
S.quickselect(data, K, random.Random(7), st_qs)

r2 = random.Random(11)
ROW = [S.f32(r2.gauss(0, 2)) for _ in range(4000)]
TRACE = []
PIVOT, ABOVE, EQUAL = S.radix_select(ROW, 32, trace=TRACE)

r3 = random.Random(5)
WIDE = [S.f32(r3.gauss(0, 2)) for _ in range(20_000)]
FST = {}
S.filtered_topk(WIDE, 64, 'float32', stats=FST)


def fig1():
    """Selection recurses into one side; sorting orders things nobody asked for."""
    fig, ax = canvas(9, 6, xlim=(0, 9), ylim=(0, 6))
    x0, w0, h = 0.9, 6.6, 0.46

    text(ax, 4.5, 5.62, 'quickselect', size=19, color=TEAL_L)
    text(ax, 4.5, 5.15, 'partition, then recurse into ONE side',
         size=14, color=TEAL)

    y = 4.45
    labels = ['n', 'n/2', 'n/4', 'n/8', 'n/16']
    for i, lab in enumerate(labels):
        w = w0 / (2 ** i)
        # the half we keep hunting in, and the half we drop on the floor
        box(ax, x0, y, w / 2, h, fill=NAVY, edge=TEAL)
        box(ax, x0 + w / 2, y, w / 2, h, fill=GREY, edge=GREY_L, lw=1.2)
        text(ax, x0 + w + 0.42, y + h / 2, lab, size=14, color=TEAL,
             ha='left', glowing=False)
        if i == 0:
            text(ax, x0 + w0 * 0.75, y + h / 2, 'discarded', size=12,
                 color=GREY_L, glowing=False)
        y -= 0.72

    brace(ax, x0, x0 + w0, 1.42, height=0.26, down=True,
          label='n + n/2 + n/4 + ... = 2n')
    text(ax, 4.5, 0.62, 'measured on %s elements, k = %d:' % (f'{N:,}', K),
         size=13, color=TEAL, glowing=False)
    text(ax, 4.5, 0.22,
         'heap sort %s comparisons     quickselect %s'
         % (f'{st_sort["cmp"]:,}', f'{st_qs["cmp"]:,}'),
         size=13, color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day21_1.png'))


def fig2():
    """The order-preserving float key."""
    fig, ax = canvas(9.4, 5.6, xlim=(0, 9.4), ylim=(0, 5.6))
    vals = [0.0, 1.0, 3.5, float('inf'), -0.0, -1.0, -3.5, -float('inf')]
    show = ['0.0', '1.0', '3.5', 'inf', '-0.0', '-1.0', '-3.5', '-inf']

    text(ax, 4.7, 5.26, 'raw float32 bits, read as unsigned', size=15,
         color=TEAL, glowing=False)
    cw = 1.06
    top = cells(ax, 0.35, 4.05, 8, w=cw, h=0.72, labels=show, size=13)

    # sorted by the ordered key instead
    order = sorted(range(8), key=lambda i: S.to_ordered(vals[i]))
    bot_lab = [show[i] for i in order]
    bot = cells(ax, 0.35, 1.30, 8, w=cw, h=0.72, labels=bot_lab, size=13)
    text(ax, 4.7, 0.86, 'after to_ordered() - now they sort as numbers',
         size=15, color=TEAL, glowing=False)

    for src, i in enumerate(order):
        col = AMBER if vals[i] < 0 or (vals[i] == 0 and show[i] == '-0.0') else TEAL
        arrow(ax, (top[i][0], 4.02), (bot[src][0], 2.06), color=col,
              lw=1.4, rad=0.0, ms=10)

    text(ax, 1.15, 3.10, 'positive:\nflip the sign bit', size=13, color=TEAL,
         glowing=False)
    text(ax, 7.55, 3.10, 'negative:\ninvert every bit', size=13, color=AMBER,
         glowing=False)
    save(fig, os.path.join(HERE, 'day21_2.png'))


def _hist_panel(ax, x0, y0, w, hgt, hist, bucket, caption, sub):
    """One round of radix select drawn as its 256-bin histogram."""
    n = len(hist)
    bw = w / n
    top = max(hist) or 1
    # sqrt heights: the counts span three orders of magnitude and the whole
    # point of the picture is that the *small* bins are still there
    for b, c in enumerate(hist):
        if c == 0:
            continue
        bh = hgt * (c / top) ** 0.5
        col = AMBER if b == bucket else (TEAL if b > bucket else GREY)
        ax.add_patch(Rectangle((x0 + b * bw, y0), max(bw * 0.9, 0.012),
                               max(bh, 0.05), facecolor=col,
                               edgecolor='none', zorder=6 if b == bucket else 3))
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.4, zorder=4)
    # the straddling bucket
    ax.add_patch(Rectangle((x0 + bucket * bw - bw * 2.2, y0 - 0.10),
                           bw * 5.4, hgt + 0.20, facecolor='none',
                           edgecolor=AMBER, lw=1.8, zorder=5))
    text(ax, x0 - 0.18, y0 + hgt / 2, caption, size=13, color=TEAL,
         ha='right', glowing=False)
    text(ax, x0 + w + 0.16, y0 + hgt / 2, sub, size=12, color=TEAL_L,
         ha='left', glowing=False)


def fig3():
    """Radix select: count bins, keep one, repeat."""
    fig, ax = canvas(9.6, 6.2, xlim=(0, 9.6), ylim=(0, 6.2))
    text(ax, 4.8, 5.86, 'radix select', size=19, color=TEAL_L)
    text(ax, 4.8, 5.42, 'each round histograms 8 bits and keeps one bucket',
         size=13, color=TEAL, glowing=False)

    text(ax, 4.8, 5.08, 'a round moves 256 counters, not 4000 floats',
         size=13, color=AMBER, glowing=False)

    y = 4.05
    for t in TRACE[:3]:
        _hist_panel(ax, 2.05, y, 5.4, 0.80, t['hist'], t['bucket'],
                    'shift %2d' % t['shift'],
                    '%s left' % f'{t["candidates"]:,}')
        y -= 1.40

    legend(ax, 2.60, 0.86, [(TEAL, 'above the threshold: guaranteed winners'),
                            (AMBER, 'the straddling bucket - recurse in here'),
                            (GREY, 'below it: dropped, never looked at again')])
    save(fig, os.path.join(HERE, 'day21_3.png'))


def fig4():
    """One block per row: the filter pass and the v_lo / v_hi boundary."""
    fig, ax = canvas(9.4, 5.8, xlim=(0, 9.4), ylim=(0, 5.8))
    text(ax, 4.7, 5.48, 'one block owns one row', size=18, color=TEAL_L)

    box(ax, 1.15, 4.30, 3.5, 0.62, label='%s scores' % f'{len(WIDE):,}',
        fill=NAVY, size=14)
    text(ax, 2.90, 3.96, 'one streaming pass over global memory', size=12,
         color=TEAL, glowing=False)
    arrow(ax, (2.90, 3.76), (2.90, 3.28), color=TEAL)

    # the value axis, cut in three by the two boundaries
    ax.add_patch(Rectangle((2.05, 1.50), 1.70, 1.68, facecolor='none',
                           edgecolor=TEAL, lw=1.6, zorder=3))
    ax.add_patch(Rectangle((2.05, 2.62), 1.70, 0.56, facecolor=NAVY,
                           edgecolor=TEAL, lw=1.6, zorder=3))
    ax.add_patch(Rectangle((2.05, 2.06), 1.70, 0.56, facecolor='#0b3d4a',
                           edgecolor=AMBER, lw=1.8, zorder=3))
    text(ax, 2.90, 2.90, '%d in' % FST['winners'], size=13, color=WHITE,
         glowing=False)
    text(ax, 2.90, 2.34, '%d maybe' % FST['candidates'], size=13,
         color=AMBER, glowing=False)
    text(ax, 2.90, 1.78, 'out', size=13, color=GREY_L, glowing=False)

    for yy, lab, val in ((2.62, 'v_hi', FST['v_hi']), (2.06, 'v_lo', FST['v_lo'])):
        ax.plot([1.90, 3.90], [yy, yy], color=AMBER, lw=1.6, ls='--', zorder=4)
        text(ax, 1.76, yy, '%s = %.3f' % (lab, val), size=12, color=AMBER,
             ha='right', glowing=False)

    region(ax, 5.45, 1.72, 3.55, 1.62, label='shared memory')
    box(ax, 5.80, 2.44, 2.85, 0.55,
        label='%d candidates' % FST['candidates'], fill=NAVY, size=13)
    text(ax, 7.22, 2.08, 'sorted on chip - global\nmemory is never touched again',
         size=12, color=TEAL, glowing=False)
    arrow(ax, (3.82, 2.34), (5.72, 2.66), color=AMBER, rad=-0.18)

    text(ax, 4.70, 1.02,
         'two float compares against v_lo / v_hi replace a per-element',
         size=12, color=TEAL, glowing=False)
    text(ax, 4.70, 0.68,
         'float32 to float16 conversion in the innermost loop',
         size=12, color=TEAL, glowing=False)
    text(ax, 4.70, 0.28, '%.2f%% of the row survives the filter'
         % (100.0 * FST['candidates'] / len(WIDE)),
         size=13, color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day21_4.png'))


def _lanes(ax, x0, x1, ys, labels):
    for y, lab in zip(ys, labels):
        ax.plot([x0, x1], [y, y], color=GREY_L, lw=1.0, zorder=2)
        text(ax, x0 - 0.16, y, lab, size=13, color=TEAL, ha='right',
             glowing=False)


def fig5():
    """The grid-wide barrier, and why the histogram is triple buffered."""
    fig, ax = canvas(10.4, 6.0, xlim=(0, 10.4), ylim=(0, 6.0))
    text(ax, 4.8, 5.70, 'a barrier built from one counter', size=18,
         color=TEAL_L)

    ys = [4.85, 4.40, 3.95, 3.50]
    _lanes(ax, 1.30, 8.60, ys, ['CTA0', 'CTA1', 'CTA2', 'CTA3'])
    # three rounds; CTA3 is slow, so everyone waits on it
    spans = [(1.30, 2.90), (3.55, 5.15), (5.80, 7.40)]
    slow = [0.55, 0.70, 0.60]
    for r, (a, b) in enumerate(spans):
        for i, y in enumerate(ys):
            end = b + (slow[r] if i == 3 else 0.0)
            box(ax, a, y - 0.13, end - a, 0.26, fill=NAVY, edge=TEAL, lw=1.2)
            if i != 3:
                ax.plot([end, b + slow[r]], [y, y], color=AMBER, lw=1.4,
                        ls=':', zorder=4)
        bx = b + slow[r]
        ax.plot([bx, bx], [3.30, 5.05], color=AMBER, lw=1.8, zorder=5)
        text(ax, bx, 5.22, 'counter >= %d' % (4 * (r + 1)), size=12,
             color=AMBER, glowing=False)
    text(ax, 8.72, 3.50, 'the slow CTA -\neveryone waits\non the straggler',
         size=12, color=TEAL_L, ha='left', glowing=False)
    text(ax, 4.8, 3.06, 'the counter is never reset between rounds, and the '
                        'wait is  >=  not  ==', size=13, color=TEAL,
         glowing=False)

    # triple buffering
    text(ax, 4.8, 2.42, 'round r writes hist[r % 3]  -  one barrier, three buffers',
         size=13, color=TEAL_L, glowing=False)
    names = ['hist[0]', 'hist[1]', 'hist[2]']
    roles = [('accumulating', AMBER), ('being cleared', TEAL),
             ('slack: a straggler\nmay still be reading', GREY_L)]
    for i in range(3):
        x = 1.05 + i * 2.60
        box(ax, x, 1.30, 2.20, 0.62, label=names[i], fill=NAVY, size=13,
            edge=roles[i][1], lw=1.8)
        text(ax, x + 1.10, 0.92, roles[i][0], size=12, color=roles[i][1],
             glowing=False)
    text(ax, 4.8, 0.26, 'with one buffer this deadlocks; with two it races',
         size=13, color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day21_5.png'))


def fig6():
    """flashinfer issue #3610: who is allowed to reset the counter."""
    fig, ax = canvas(11.4, 6.0, xlim=(0, 11.4), ylim=(0, 6.0))
    text(ax, 5.2, 5.64, 'the wrong CTA resets the counter', size=18, color=RED)

    ys = [4.72, 4.34, 3.96, 3.58]
    _lanes(ax, 1.25, 8.60, ys, ['CTA0', 'CTA1', 'CTA2', 'CTA3'])
    ends = [3.60, 4.20, 4.55, 7.40]
    for y, e in zip(ys, ends):
        box(ax, 1.25, y - 0.11, e - 1.25, 0.22, fill=NAVY, edge=TEAL, lw=1.2)
    for y, e in zip(ys[1:3], ends[1:3]):
        ax.plot([e, 8.60], [y, y], color=RED, lw=1.4, ls=':', zorder=4)

    ax.plot([3.60, 3.60], [3.42, 4.90], color=RED, lw=1.8, zorder=5)
    text(ax, 3.60, 5.14, 'CTA0 finishes first and zeroes the counter',
         size=13, color=RED, glowing=False)
    # the CTAs that hang are the ones still waiting on the slow CTA3
    for y in ys[1:3]:
        cross(ax, 8.85, y, r=0.20)
    text(ax, 9.20, 4.15, 'stuck', size=13, color=RED, ha='left')
    text(ax, 5.20, 2.98,
         'CTA1 and CTA2 are still inside their final wait: they read 0,',
         size=13, color=RED, glowing=False)
    text(ax, 5.20, 2.64,
         'which is less than their target, and they spin forever',
         size=13, color=RED, glowing=False)

    region(ax, 1.25, 0.45, 7.90, 1.55,
           label='the fix: the exit barrier elects a leader')
    text(ax, 5.20, 1.64, 'every CTA increments once more; exactly one sees its own',
         size=13, color=TEAL, glowing=False)
    text(ax, 5.20, 1.24, 'increment land on the target, and that one is provably last -',
         size=13, color=TEAL, glowing=False)
    text(ax, 5.20, 0.84, 'so by then every peer has already stopped reading',
         size=13, color=TEAL, glowing=False)
    save(fig, os.path.join(HERE, 'day21_6.png'))


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print(f.__name__, f())
