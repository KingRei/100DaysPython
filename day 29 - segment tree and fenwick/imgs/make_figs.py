"""Figures for day 29 - segment tree and Fenwick tree.

Every number on these figures is computed by importing range_structures,
so the pictures cannot drift away from the code.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *
import range_structures as B


def fig1():
    """The two naive answers, and the gap between them."""
    fig, ax = canvas(10.4, 6.4, xlim=(0, 10.4), ylim=(0, 6.4))
    text(ax, 5.2, 6.05, 'a[i] = v  and  sum(a[l:r])  -  we want both to be fast',
         color=TEAL_L, size=15, ha='center')

    # the prefix-sum array breaking under one update
    cw, x0, y = 0.62, 1.05, 4.55
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    pre = [0]
    for v in vals:
        pre.append(pre[-1] + v)
    text(ax, 0.85, y + 0.31, 'a', color=TEAL_L, size=13, ha='right')
    for i, v in enumerate(vals):
        box(ax, x0 + i * cw, y, cw - 0.04, 0.62, label=str(v),
            fill=NAVY, edge=AMBER if i == 2 else TEAL,
            lw=2.6 if i == 2 else 1.6)
    text(ax, x0 + 2 * cw + cw / 2, y - 0.28, 'a[2] = 4  ->  7',
         color=AMBER, size=11, ha='center', glowing=False)

    y2 = 3.05
    text(ax, 0.85, y2 + 0.31, 'prefix', color=TEAL_L, size=13, ha='right')
    for i, v in enumerate(pre[1:]):
        broken = i >= 2
        box(ax, x0 + i * cw, y2, cw - 0.04, 0.62, label=str(v),
            fill=NAVY, edge=RED if broken else TEAL, lw=2.4 if broken else 1.6)
    text(ax, x0 + 5 * cw, y2 - 0.30,
         'one write to a[2] invalidates every prefix after it: O(n)',
         color=RED, size=11, ha='center')

    box(ax, 0.35, 0.45, 9.70, 1.85, fill='none', edge=CYAN, lw=1.6)
    rows = [('', 'update', 'range sum'),
            ('plain array', 'O(1)', 'O(n)'),
            ('prefix sums', 'O(n)', 'O(1)'),
            ('segment tree / Fenwick', 'O(log n)', 'O(log n)')]
    cols = (1.05, 5.30, 7.90)
    for i, row in enumerate(rows):
        yy = 1.92 - i * 0.40
        for x, cell in zip(cols, row):
            if not cell:
                continue
            colour = TEAL_L if i == 0 else (CYAN if i == 3 else TEAL)
            text(ax, x, yy, cell, color=colour, size=12,
                 ha='left' if x == cols[0] else 'center', glowing=False)
    save(fig, os.path.join(HERE, 'day29_1.png'))


def _seg_positions():
    """x, y for nodes 1..15 of a 2n-layout tree over 8 leaves."""
    pos = {}
    for k in range(1, 16):
        lvl = k.bit_length() - 1          # 0 at the root
        first = 1 << lvl
        span = 8.8 / (1 << lvl)
        pos[k] = (0.85 + span * (k - first) + span / 2, 5.05 - lvl * 1.18)
    return pos


def _seg_range(k):
    """half-open leaf range covered by node k in a 2n layout over 8 leaves."""
    lo, hi = k, k + 1
    while lo < 8:
        lo, hi = lo * 2, hi * 2
    return lo - 8, hi - 8


def fig2():
    """One query, cut into whole subtrees."""
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    tree = B.SegmentTree(vals)
    cover = B.canonical_cover(8, 1, 7)
    pos = _seg_positions()

    fig, ax = canvas(10.4, 6.6, xlim=(0, 10.4), ylim=(0, 6.6))
    text(ax, 5.2, 6.25, 'query sum(a[1:7]) = %d  ->  %d whole subtrees, never a scan'
         % (tree.query(1, 7), len(cover)), color=TEAL_L, size=14, ha='center')

    for k in range(2, 16):
        x, y = pos[k]
        px, py = pos[k >> 1]
        on = k in cover
        edge(ax, (px, py - 0.24), (x, y + 0.24), directed=False, r=0.0,
             color=AMBER if on else GREY, lw=2.2 if on else 1.2)

    for k in range(1, 16):
        x, y = pos[k]
        on = k in cover
        node(ax, x, y, str(tree.t[k]), r=0.235,
             fill=NAVY, edge=AMBER if on else TEAL, lw=2.8 if on else 1.5)
        lo, hi = _seg_range(k)
        lab = 'a[%d:%d]' % (lo, hi) if hi - lo > 1 else 'a[%d]' % lo
        box(ax, x - 0.27, y - 0.53, 0.54, 0.24, label=lab, fill=NAVY,
            edge='none', lw=0, color=AMBER if on else GREY_L, size=8.2,
            zorder=6)

    text(ax, 5.2, 0.72,
         'the ranges of the %d chosen nodes are disjoint and their union is exactly a[1:7]'
         % len(cover), color=TEAL, size=11.5, ha='center', glowing=False)
    text(ax, 5.2, 0.36,
         'at most 2 per level  ->  at most 2*log2(n) nodes for any range',
         color=CYAN, size=12, ha='center', glowing=False)
    save(fig, os.path.join(HERE, 'day29_2.png'))


def fig3():
    """Lazy propagation: stop at the covered node, leave a note."""
    lazy_visits, eager_visits, pushes = B.lazy_vs_eager()

    fig, ax = canvas(10.4, 6.3, xlim=(0, 10.4), ylim=(0, 6.3))
    text(ax, 5.2, 5.98, 'range_add(a[2:6], +5)  -  where do we stop?',
         color=TEAL_L, size=15, ha='center')

    def tree(ax, x0, touched, note, title, colour):
        text(ax, x0 + 2.05, 5.35, title, color=colour, size=12.5, ha='center',
             glowing=False)
        p = {}
        for k in range(1, 16):
            lvl = k.bit_length() - 1
            span = 4.1 / (1 << lvl)
            p[k] = (x0 + span * (k - (1 << lvl)) + span / 2, 4.75 - lvl * 0.98)
        for k in range(2, 16):
            edge(ax, (p[k >> 1][0], p[k >> 1][1] - 0.17), (p[k][0], p[k][1] + 0.17),
                 directed=False, r=0.0, color=GREY, lw=1.1)
        for k in range(1, 16):
            x, y = p[k]
            on = k in touched
            node(ax, x, y, '', r=0.165, fill=AMBER if on else NAVY,
                 edge=AMBER if on else GREY, lw=2.0 if on else 1.2)
        for k, lab in note.items():
            x, y = p[k]
            box(ax, x - 0.30, y - 0.50, 0.60, 0.24, label=lab, fill=NAVY,
                edge=CYAN, lw=1.2, color=CYAN, size=8.5, zorder=6)

    # eager: every leaf of a[2:6] plus the path down
    eager = {1, 2, 3, 5, 6, 10, 11, 12, 13}
    tree(ax, 0.55, eager, {}, 'eager: walk down to all 4 leaves', RED)
    # lazy: stop at nodes 5 and 6, which are fully covered
    tree(ax, 5.75, {1, 2, 3, 5, 6}, {5: '+5 lazy', 6: '+5 lazy'},
         'lazy: stop at the 2 covered nodes', CYAN)

    box(ax, 0.55, 0.35, 9.30, 1.15, fill='none', edge=CYAN, lw=1.6)
    text(ax, 5.20, 1.16,
         'a fully covered node knows its own new sum: += delta * width. '
         'Nobody needs its children yet.',
         color=TEAL, size=11.5, ha='center', glowing=False)
    text(ax, 5.20, 0.62,
         '2000 random range ops on n = 100000:   eager %s node visits   vs   lazy %s   (%.0fx)'
         % ('{:,}'.format(eager_visits), '{:,}'.format(lazy_visits),
            eager_visits / lazy_visits),
         color=AMBER, size=10.5, ha='center', glowing=False)
    save(fig, os.path.join(HERE, 'day29_3.png'))


def fig4():
    """i & -i - the staircase that is the whole data structure."""
    f = B.Fenwick(list(range(1, 17)))
    fig, ax = canvas(10.4, 6.6, xlim=(0, 10.4), ylim=(0, 6.6))
    text(ax, 5.2, 6.28, 'lowbit(i) = i & -i   decides what t[i] is responsible for',
         color=TEAL_L, size=14.5, ha='center')

    x0, cw = 2.55, 0.47
    for j in range(16):
        text(ax, x0 + j * cw + cw / 2, 5.72, str(j + 1), color=GREY_L,
             size=9.5, ha='center', glowing=False)
    text(ax, 2.35, 5.72, 'a index', color=GREY_L, size=9.5, ha='right',
         glowing=False)

    for i in range(1, 17):
        y = 5.30 - (i - 1) * 0.285
        lo, hi = f.covered_by(i)
        w = hi - lo
        box(ax, x0 + lo * cw, y - 0.115, w * cw - 0.05, 0.235,
            fill=NAVY, edge=AMBER if w > 1 else TEAL, lw=1.7)
        text(ax, x0 + lo * cw + w * cw / 2, y, str(w),
             color=PALE, size=9, ha='center', glowing=False)
        text(ax, 1.02, y, 't[%2d]' % i, color=TEAL_L, size=10, ha='left',
             glowing=False)
        text(ax, 2.35, y, '%5s' % format(i, 'b'), color=GREY_L, size=9.5,
             ha='right', glowing=False)

    text(ax, 1.02, 0.56, 't[i] holds the sum of the lowbit(i) cells ending at i,',
         color=TEAL, size=11.5, ha='left', glowing=False)
    text(ax, 1.02, 0.22, 'so n counters cover every prefix - no pointers, no tree array.',
         color=CYAN, size=11.5, ha='left', glowing=False)
    save(fig, os.path.join(HERE, 'day29_4.png'))


def fig5():
    """The two walks: peel bits to read, add bits to write."""
    f = B.Fenwick(list(range(1, 17)))

    def walk_down(i):
        out = []
        while i > 0:
            out.append(i)
            i -= i & -i
        return out

    def walk_up(i, n=16):
        out = []
        while i <= n:
            out.append(i)
            i += i & -i
        return out

    down = walk_down(13)
    up = walk_up(5)

    fig, ax = canvas(10.4, 5.9, xlim=(0, 10.4), ylim=(0, 5.9))
    text(ax, 5.2, 5.58, 'both operations are just bit arithmetic on the index',
         color=TEAL_L, size=14.5, ha='center')

    def strip(y, seq, colour, head, tail):
        text(ax, 0.35, y + 0.72, head, color=colour, size=12.5, ha='left',
             glowing=False)
        x0, cw = 0.55, 0.60
        for i in range(1, 17):
            on = i in seq
            box(ax, x0 + (i - 1) * cw, y, cw - 0.06, 0.52, label=str(i),
                fill=NAVY, edge=colour if on else GREY,
                lw=2.6 if on else 1.2)
        for a, b in zip(seq, seq[1:]):
            xa = x0 + (a - 1) * cw + cw / 2 - 0.03
            xb = x0 + (b - 1) * cw + cw / 2 - 0.03
            dist = abs(xb - xa)
            mag = min(0.35, 0.70 / dist)
            arrow(ax, (xa, y - 0.10), (xb, y - 0.10), color=colour, lw=1.6,
                  rad=mag if b > a else -mag, ms=10)
        text(ax, 0.35, y - 0.95, tail, color=TEAL, size=11, ha='left',
             glowing=False)

    strip(3.75, down, AMBER,
          'prefix(13):  13 -> 12 -> 8,  peel the lowest set bit each time',
          '1101 -> 1100 -> 1000 -> 0     sum = t[13] + t[12] + t[8] = %d   (%d steps = popcount(13))'
          % (f.prefix(13), len(down)))

    strip(1.55, up, CYAN,
          'add(index 4, +v):  5 -> 6 -> 8 -> 16,  add the lowest set bit each time',
          '0101 -> 0110 -> 1000 -> 10000 -> past n     the %d counters that contain this cell'
          % len(up))

    text(ax, 5.2, 0.30,
         'update climbs, query descends, and the two walks meet at exactly the counters that overlap',
         color=CYAN, size=11.5, ha='center', glowing=False)
    save(fig, os.path.join(HERE, 'day29_5.png'))


def fig6():
    """Which one to reach for - and the operation a Fenwick tree cannot do."""
    t = B.timing()
    true_min, bit_min, seg_min = B.min_bit_failure()

    fig, ax = canvas(10.4, 6.2, xlim=(0, 10.4), ylim=(0, 6.2))
    text(ax, 5.2, 5.88, 'same complexity, different jobs', color=TEAL_L,
         size=15, ha='center')

    box(ax, 0.40, 3.10, 9.60, 2.35, fill='none', edge=TEAL, lw=1.6)
    rows = [
        ('', 'segment tree', 'Fenwick / BIT'),
        ('memory', '2n  (4n if lazy)', 'n'),
        ('range op', 'any associative op', 'needs an inverse: sum, xor'),
        ('range update', 'yes, with lazy tags', 'only via difference arrays'),
        ('code', '~30 lines', '~6 lines'),
    ]
    for i, row in enumerate(rows):
        yy = 5.10 - i * 0.42
        for x, cell, al in zip((0.70, 3.85, 6.75), row, ('left', 'left', 'left')):
            if not cell:
                continue
            text(ax, x, yy, cell, color=TEAL_L if i == 0 else TEAL, size=11.5,
                 ha=al, glowing=False)

    text(ax, 0.70, 2.72,
         'n = 20000, 2000 mixed update+query ops:  segment tree %.3f s  vs  fenwick %.3f s   (%.1fx)'
         % (t['segment tree'], t['fenwick'], t['segment tree'] / t['fenwick']),
         color=AMBER, size=11, ha='left', glowing=False)

    box(ax, 0.40, 0.28, 9.60, 2.10, fill='none', edge=RED, lw=1.8)
    text(ax, 0.70, 2.02, 'why not always Fenwick?', color=RED, size=12.5,
         ha='left')
    text(ax, 0.70, 1.62,
         'range(l, r) = prefix(r) - prefix(l) only works if the operation can be undone.',
         color=TEAL, size=11.5, ha='left', glowing=False)
    text(ax, 0.70, 1.26,
         'min has no inverse, so a prefix-min BIT silently returns a value from outside the range:',
         color=TEAL, size=11.5, ha='left', glowing=False)
    text(ax, 0.70, 0.86,
         'true min = %d      prefix-min BIT = %d   (wrong)      segment tree = %d'
         % (true_min, bit_min, seg_min),
         color=AMBER, size=11.5, ha='left', glowing=False)
    text(ax, 0.70, 0.60,
         'a segment tree only ever combines disjoint pieces, so it never needs to undo anything.',
         color=CYAN, size=11.5, ha='left', glowing=False)
    save(fig, os.path.join(HERE, 'day29_6.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print('figures written to', HERE)
