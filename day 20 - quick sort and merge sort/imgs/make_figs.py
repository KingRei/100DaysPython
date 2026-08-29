"""Figures for Day 20 - merge sort, quick sort, and their parallel versions.

Every number printed on a figure is computed by importing quick_merge_sort,
never typed by hand.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *                      # noqa: F401,F403
import quick_merge_sort as Q


# ----------------------------------------------------------------- figure 1
def fig1():
    """Where the work happens: merge sort joins, quick sort splits."""
    fig, ax = canvas(9.6, 4.9, xlim=(0, 9.6), ylim=(0.9, 5.6))
    cw, ch = 0.52, 0.62

    text(ax, 0.05, 5.30, 'merge sort   split is free, the join does the work',
         color=TEAL_L, size=15, ha='left')
    y = 4.20
    cells(ax, 0.30, y, 4, w=cw, h=ch, labels=['1', '5', '6', '9'])
    text(ax, 2.55, y + ch / 2, '+', color=TEAL, size=17)
    cells(ax, 2.80, y, 3, w=cw, h=ch, labels=['2', '3', '7'])
    arrow(ax, (4.50, y + ch / 2), (5.35, y + ch / 2), lw=2.4)
    text(ax, 4.93, y - 0.32, 'merge  O(n)', color=AMBER, size=13)
    cells(ax, 5.50, y, 7, w=cw, h=ch,
          labels=['1', '2', '3', '5', '6', '7', '9'])

    text(ax, 0.05, 2.95, 'quick sort   the split does the work, '
         'the join is nothing', color=TEAL_L, size=15, ha='left')
    y = 1.85
    cells(ax, 0.30, y, 6, w=cw, h=ch, labels=['5', '2', '9', '1', '7', '3'])
    arrow(ax, (3.60, y + ch / 2), (4.45, y + ch / 2), lw=2.4)
    text(ax, 4.03, y - 0.32, 'partition  O(n)', color=AMBER, size=13)
    cells(ax, 4.62, y, 3, w=cw, h=ch, labels=['2', '1', '3'])
    box(ax, 6.28, y, cw, ch, label='5', edge=AMBER, lw=2.4)
    cells(ax, 6.96, y, 2, w=cw, h=ch, labels=['9', '7'])
    text(ax, 5.40, y - 0.32, '< 5', color=TEAL, size=13)
    text(ax, 6.54, y - 0.32, 'pivot', color=AMBER, size=13)
    text(ax, 7.48, y - 0.32, '> 5', color=TEAL, size=13)
    text(ax, 8.30, y + ch / 2,
         'sort each side,\nthen concatenate\n- nothing to do',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day20_1.png')


# ----------------------------------------------------------------- figure 2
def fig2():
    """The merge is a two-finger walk - and it counts inversions for free."""
    left = [1, 5, 6, 9]
    right = [2, 3, 7]
    inv = Q.count_inversions(left + right)
    take = len(left) - 1                              # emitting 2 while i = 1

    fig, ax = canvas(9.4, 5.1, xlim=(0, 9.4), ylim=(0.3, 5.1))
    cw, ch = 0.60, 0.66
    ly, ry = 4.05, 2.60

    cl = cells(ax, 1.30, ly, 4, w=cw, h=ch, labels=[str(v) for v in left])
    text(ax, 1.05, ly + ch / 2, 'left', color=TEAL, size=14, ha='right')
    cr = cells(ax, 1.30, ry, 3, w=cw, h=ch, labels=[str(v) for v in right])
    text(ax, 1.05, ry + ch / 2, 'right', color=TEAL, size=14, ha='right')

    box(ax, 1.30 + cw, ly, cw, ch, label='5', edge=AMBER, lw=2.6)
    box(ax, 1.30 + cw, ry, cw, ch, label='3', edge=AMBER, lw=2.6)
    text(ax, cl[1][0], ly + ch + 0.34, 'i', color=AMBER, size=15)
    text(ax, cr[1][0], ry - 0.34, 'j', color=AMBER, size=15)

    arrow(ax, (3.35, ry + ch / 2), (5.55, ly + ch / 2), lw=2.0, rad=-0.25)
    text(ax, 4.75, 4.92, '5 > 3, so emit 3', color=TEAL_L, size=14)
    cells(ax, 5.70, ly, 3, w=cw, h=ch, labels=['1', '2', '3'])
    text(ax, 7.60, ly + ch / 2, 'out', color=TEAL, size=14, ha='left')

    brace(ax, 1.30 + cw, 1.30 + 4 * cw, ly - 0.12, height=0.26, down=True,
          label='3 still unemitted', size=13)
    text(ax, 4.30, 2.72, 'every one of those 3 is bigger than 3,',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 4.30, 2.35, 'and sits in front of it - 3 inversions',
         color=AMBER, size=13, ha='left', glowing=False)

    text(ax, 4.70, 1.45, 'each emit from the right half adds '
         '(len(left) - i) inversions', color=TEAL, size=14)
    text(ax, 4.70, 1.02, 'here that is %d, so "how far is this list from '
         'sorted?" costs' % take, color=TEAL, size=14)
    text(ax, 4.70, 0.60, 'one addition per element  '
         '(left + right holds %d)' % inv, color=TEAL, size=14)
    save(fig, 'day20_2.png')


# ----------------------------------------------------------------- figure 3
def fig3():
    """Three-way partition, and the input it exists for."""
    arr = [7, 2, 9, 4, 7, 1, 7, 3]
    b = list(arr)
    lt, gt = Q.three_way_partition(b, 0, len(b) - 1, 7)

    n = 400
    s_lom = Q.new_stats(); Q.quicksort_lomuto([7] * n, s_lom, 'random',
                                              random.Random(7))
    s_3w = Q.new_stats(); Q.quicksort_3way([7] * n, s_3w, random.Random(7))

    fig, ax = canvas(9.4, 5.0, xlim=(0, 9.4), ylim=(0.15, 5.0))
    cw, ch = 0.66, 0.70
    y = 3.55
    x0 = 1.35
    for i, v in enumerate(b):
        eq = lt <= i <= gt
        box(ax, x0 + i * cw, y, cw, ch, label=str(v),
            edge=AMBER if eq else TEAL, lw=2.4 if eq else 1.6)
    text(ax, x0 - 0.20, y + ch / 2, 'pivot 7', color=TEAL, size=14, ha='right')

    brace(ax, x0, x0 + lt * cw, y + ch + 0.12, label='< 7', height=0.26)
    brace(ax, x0 + lt * cw, x0 + (gt + 1) * cw, y + ch + 0.12,
          label='== 7', height=0.26)
    brace(ax, x0 + (gt + 1) * cw, x0 + len(b) * cw, y + ch + 0.12,
          label='> 7', height=0.26)

    text(ax, x0 + lt * cw / 2, y - 0.45, 'recurse', color=TEAL, size=13)
    text(ax, x0 + (lt + gt + 1) * cw / 2, y - 0.45, 'done - never',
         color=AMBER, size=13)
    text(ax, x0 + (lt + gt + 1) * cw / 2, y - 0.80, 'touched again',
         color=AMBER, size=13)
    text(ax, x0 + (gt + 1 + len(b)) * cw / 2, y - 0.45, 'recurse',
         color=TEAL, size=13)

    text(ax, 4.70, 1.85, '%d equal values, two-way partition with a random '
         'pivot:' % n, color=TEAL, size=14)
    text(ax, 4.70, 1.42, '%s comparisons, recursion %d deep - the pivot is '
         'never in the middle' % (f'{s_lom["cmp"]:,}', s_lom['depth']),
         color=RED, size=14)
    text(ax, 4.70, 0.92, 'the same input, three-way partition:',
         color=TEAL, size=14)
    text(ax, 4.70, 0.49, '%d comparisons, depth %d - one pass and the array '
         'is sorted' % (s_3w['cmp'], s_3w['depth']), color=TEAL_L, size=14)
    save(fig, 'day20_3.png')


# ----------------------------------------------------------------- figure 4
def fig4():
    """Work and depth: the sequential merge is the bottleneck."""
    m = Q.WORKDEPTH_N
    arr = Q.workdepth_input(m)
    rows = []
    for name, mg, par in (('one core', Q.merge_seq, False),
                          ('parallel halves, plain merge', Q.merge_seq, True),
                          ('parallel halves, parallel merge', Q.merge_par,
                           True)):
        _, w, d = Q.msort_cost(arr, mg, par)
        rows.append((name, w, d, w / d))

    fig, ax = canvas(10.0, 5.4, xlim=(0, 10.0), ylim=(0.2, 5.6))
    text(ax, 5.0, 5.40, 'merge sort on n = %d, work and depth counted by the '
         'code' % m, color=TEAL_L, size=15)

    xs = [0.55, 4.30, 5.95, 7.60]
    ws = [3.60, 1.50, 1.50, 1.75]
    hdr = ['', 'work', 'depth', 'most usable cores']
    rh, gap = 0.70, 0.24
    top = 3.95
    for j, h in enumerate(hdr):
        if h:
            text(ax, xs[j] + ws[j] / 2, top + rh + 0.30, h, color=TEAL_L,
                 size=13)
    for i, (name, w, d, p) in enumerate(rows):
        yy = top - i * (rh + gap)
        vals = [name, f'{w:,}', f'{d:,}', '%.0fx' % p]
        for j, v in enumerate(vals):
            box(ax, xs[j], yy, ws[j], rh, label=v, size=13,
                fill=NAVY if j == 0 else GREY)
    # emphasis box - all four sides identical
    last = top - 2 * (rh + gap)
    ax.add_patch(Rectangle((xs[0] - 0.13, last - 0.13),
                           xs[3] + ws[3] - xs[0] + 0.26, rh + 0.26,
                           facecolor='none', edgecolor=AMBER, linewidth=2.0,
                           linestyle=(0, (5, 3)), zorder=6))

    text(ax, 5.0, 1.35, 'the middle row is the trap: the final merge of %d '
         'elements is one sequential loop,' % m, color=TEAL, size=14)
    text(ax, 5.0, 0.95, 'so the depth stays O(n) however many cores you own',
         color=TEAL, size=14)
    text(ax, 5.0, 0.45, 'a merge that is itself divide and conquer buys %.0fx '
         'the parallelism for %.0f%% more work'
         % (rows[2][3] / rows[1][3],
            100 * (rows[2][1] - rows[1][1]) / rows[1][1]),
         color=AMBER, size=14)
    save(fig, 'day20_4.png')


# ----------------------------------------------------------------- figure 5
def fig5():
    """Sample sort: split once, across machines - and mind the imbalance."""
    prng = random.Random(99)
    data = [prng.randint(0, 10 ** 6) for _ in range(20000)]
    p = 8
    runs = {o: Q.sample_sort(data, p, o, random.Random(5)) for o in (1, 32)}
    means = {}
    for o in (1, 32):
        imbs = [Q.sample_sort(data, p, o, random.Random(t))[2]
                for t in range(200)]
        means[o] = sum(imbs) / len(imbs)
    tallest = max(max(runs[o][1]) for o in runs)
    avg = len(data) / p

    fig, ax = canvas(9.6, 6.2, xlim=(0, 9.6), ylim=(0.1, 6.3))
    text(ax, 4.8, 6.05, 'sample sort across p = %d machines' % p,
         color=TEAL_L, size=15)
    text(ax, 4.8, 5.72, 'sample the data, sort the samples, keep p-1 '
         'splitters, exchange once - then sort locally',
         color=TEAL, size=13)

    bw, gap, bh = 0.80, 0.16, 1.55
    for over, y0 in ((1, 3.35), (32, 0.85)):
        _, sizes, imb = runs[over]
        worst = max(sizes)
        text(ax, 0.55, y0 + bh + 0.28, 'oversample %d' % over,
             color=TEAL_L, size=14, ha='left')
        text(ax, 2.35, y0 + bh + 0.28, 'slowest machine does %.2fx its share '
             '(%.2fx averaged over 200 samplings)' % (imb, means[over]),
             color=TEAL, size=13, ha='left', glowing=False)
        for i, sz in enumerate(sizes):
            x = 0.62 + i * (bw + gap)
            box(ax, x, y0, bw, max(bh * sz / tallest, 0.05),
                edge=RED if sz == worst else TEAL,
                lw=2.4 if sz == worst else 1.6)
            text(ax, x + bw / 2, y0 - 0.28, str(sz), color=TEAL, size=12)
        right = 0.62 + p * (bw + gap) - gap
        ax.plot([0.50, right + 0.08], [y0 + bh * avg / tallest] * 2,
                color=AMBER, lw=1.4, ls='--', zorder=6)
        text(ax, right + 0.20, y0 + bh * avg / tallest, 'fair share',
             color=AMBER, size=12, ha='left')
    save(fig, 'day20_5.png')


if __name__ == '__main__':
    os.chdir(HERE)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print('wrote', f.__name__)
