"""Figures for Day 23 - binary search.  Every number here is computed by importing
binary_search.py, so the pictures cannot drift from the code."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: F401,F403
import binary_search as B


# ---------------------------------------------------------------------------
def fig1():
    """The half-open invariant."""
    fig, ax = canvas(9.8, 5.0, xlim=(0, 9.8), ylim=(0, 5.0))
    n, w, h = 12, 0.60, 0.80
    x0, y0 = 1.3, 2.45
    lo_i, hi_i, mid_i = 4, 9, 6

    for i in range(n):
        x = x0 + i * w
        if i < lo_i:
            box(ax, x, y0, w, h, fill=GREY, edge=GREY_L, lw=1.4)
        elif i >= hi_i:
            box(ax, x, y0, w, h, fill=GREY, edge=GREY_L, lw=1.4)
        else:
            box(ax, x, y0, w, h, fill=NAVY, edge=TEAL, lw=1.6)
    # the probe
    box(ax, x0 + mid_i * w, y0, w, h, fill=NAVY, edge=AMBER, lw=2.6)
    text(ax, x0 + (mid_i + .5) * w, y0 + h + 0.30, 'mid', color=AMBER, size=15)

    lo_x, hi_x = x0 + lo_i * w, x0 + hi_i * w
    for xx, lab in ((lo_x, 'lo'), (hi_x, 'hi')):
        arrow(ax, (xx, y0 - 0.62), (xx, y0 - 0.08), color=TEAL_L, lw=1.8)
        text(ax, xx, y0 - 0.88, lab, color=TEAL_L, size=17)

    brace(ax, x0, lo_x, y0 + h + 0.70, height=0.24, label='known too small', size=14)
    brace(ax, lo_x, hi_x, y0 + h + 0.70, height=0.24,
          label='still possible', size=14)
    brace(ax, hi_x, x0 + n * w, y0 + h + 0.70, height=0.24,
          label='known big enough', size=14)

    text(ax, 4.9, 1.02, 'a[mid] < x   ->   lo = mid + 1', color=TEAL, size=15)
    text(ax, 4.9, 0.62, 'otherwise    ->   hi = mid', color=TEAL, size=15)
    text(ax, 4.9, 0.18, 'the window only shrinks, and when it is empty lo == hi is the answer',
         color=GREY_L, size=13)
    save(fig, os.path.join(HERE, 'day23_1.png'))


# ---------------------------------------------------------------------------
def fig2():
    """lower_bound and upper_bound on duplicates."""
    a = [2, 3, 3, 3, 5, 8, 8, 13]
    lb, ub = B.lower_bound(a, 3), B.upper_bound(a, 3)
    fig, ax = canvas(9.4, 5.0, xlim=(0, 9.4), ylim=(0, 5.0))
    w, h, x0, y0 = 0.86, 0.82, 1.2, 1.95
    for i, v in enumerate(a):
        box(ax, x0 + i * w, y0, w, h, label=str(v),
            edge=AMBER if v == 3 else TEAL, lw=2.4 if v == 3 else 1.6)
        text(ax, x0 + (i + .5) * w, y0 - 0.36, str(i), color=GREY_L, size=12)

    for idx, lab, top, ha in ((lb, f'lower_bound(3) = {lb}', 0.62, 'right'),
                              (ub, f'upper_bound(3) = {ub}', 1.42, 'left')):
        xx = x0 + idx * w
        arrow(ax, (xx, y0 + h + top), (xx, y0 + h + 0.12), color=TEAL_L, lw=1.8)
        text(ax, xx + (-0.14 if ha == 'right' else 0.14), y0 + h + top + 0.22,
             lab, color=TEAL_L, size=14, ha=ha)

    brace(ax, x0 + lb * w, x0 + ub * w, y0 - 0.50, height=0.24, down=True,
          label=f'count = {ub} - {lb} = {ub - lb}', size=15)
    text(ax, 4.7, 0.20, 'the only place the two boundaries differ is a value that exists',
         color=GREY_L, size=13)
    save(fig, os.path.join(HERE, 'day23_2.png'))


# ---------------------------------------------------------------------------
def fig3():
    """Binary search on the answer: the array that is never built."""
    piles, hours = [30, 11, 23, 4, 20], 6
    ans, probes = B.min_eating_speed(piles, hours)
    fig, ax = canvas(10.0, 5.4, xlim=(0, 10.0), ylim=(0, 5.4))
    speeds = list(range(14, 30))
    w, h, x0, y0 = 0.55, 0.62, 0.55, 2.20
    order = {s: k + 1 for k, (s, _) in enumerate(probes)}
    for i, s in enumerate(speeds):
        ok = B.koko_hours(piles, s) <= hours
        x = x0 + i * w
        box(ax, x, y0, w, h, label='T' if ok else '.',
            fill=NAVY if ok else GREY, edge=AMBER if s == ans else (TEAL if ok else GREY_L),
            lw=2.6 if s == ans else 1.5, color=WHITE, size=13)
        text(ax, x + w / 2, y0 - 0.34, str(s), color=GREY_L, size=11)
        if s in order:
            text(ax, x + w / 2, y0 + h + 0.26, str(order[s]), color=TEAL_L, size=13)

    text(ax, 5.0, 3.42, 'probe order', color=TEAL_L, size=14)
    text(ax, 5.0, 1.42, 'candidate eating speed (bananas / hour)', color=GREY_L, size=13)
    box(ax, 2.05, 4.45, 5.9, 0.74,
        label='ok(s)  =  hours(piles, s) <= h', edge=TEAL_L, size=16)
    arrow(ax, (5.0, 4.42), (5.0, 4.02), color=TEAL_L, lw=1.8)
    text(ax, 5.0, 3.82, 'a simulation, not a comparison', color=TEAL_L, size=13)
    text(ax, 5.0, 0.85, f'the first T is at {ans}, found in {len(probes)} probes '
         f'without ever building this row', color=TEAL, size=14)
    text(ax, 5.0, 0.42, 'all binary search needs is that the row is  . . . . T T T T',
         color=GREY_L, size=13)
    save(fig, os.path.join(HERE, 'day23_3.png'))


# ---------------------------------------------------------------------------
def fig4():
    """Galloping: doubling windows, then binary search inside the one that broke."""
    fig, ax = canvas(10.0, 4.4, xlim=(0, 10.0), ylim=(0, 4.4))
    unit, x0, y0, h = 0.28, 0.55, 1.55, 0.62
    total = 31
    p = 21                                   # shared prefix length used in the picture
    for i in range(total):
        x = x0 + i * unit
        box(ax, x, y0, unit, h, fill=NAVY if i < p else GREY,
            edge=TEAL if i < p else GREY_L, lw=1.2)
    cross(ax, x0 + (p + .5) * unit, y0 + h / 2, r=0.16, lw=1.6)
    text(ax, x0 + (p + .5) * unit, y0 - 0.36, 'first difference', color=RED, size=13)

    lo, step, yy = 0, 1, y0 + h + 0.34
    while lo < total:
        hi = min(lo + step, total)
        broke = lo <= p < hi
        col = RED if broke else TEAL_L
        ax.plot([x0 + lo * unit, x0 + hi * unit], [yy, yy], color=col, lw=2.2,
                solid_capstyle='butt')
        for e in (lo, hi):
            ax.plot([x0 + e * unit] * 2, [yy - 0.09, yy + 0.09], color=col, lw=2.2)
        text(ax, x0 + (lo + hi) / 2 * unit, yy + 0.26, str(hi - lo), color=col, size=12)
        if broke:
            text(ax, x0 + (lo + hi) / 2 * unit, yy + 0.72,
                 'this window breaks', color=RED, size=13)
            break
        lo, step = hi, step * 2
        yy += 0.0

    text(ax, 5.0, 3.62, 'gallop in doubling windows, then binary search inside the last one',
         color=TEAL, size=15)
    text(ax, 5.0, 0.92, 'windows of 1, 2, 4, 8, ... sum to about 2p, so the cost follows',
         color=GREY_L, size=13)
    text(ax, 5.0, 0.56, 'the length of the SHARED PREFIX, not the length of the arrays',
         color=GREY_L, size=13)
    text(ax, 5.0, 0.16, 'inside the broken window: hi - lo > 1, so lo = mid is safe',
         color=AMBER, size=13)
    save(fig, os.path.join(HERE, 'day23_4.png'))


# ---------------------------------------------------------------------------
def fig5():
    """Undoing a prefix sum with a boundary search."""
    seq_lens = [3, 5, 2]
    cu = [0]
    for L in seq_lens:
        cu.append(cu[-1] + L)
    total = cu[-1]
    fig, ax = canvas(9.6, 4.6, xlim=(0, 9.6), ylim=(0, 4.6))
    w, h, x0 = 0.72, 0.72, 1.05
    ytok, ycu = 1.30, 3.05

    for t in range(total):
        seq = B.bisect_right(cu, t) - 1
        box(ax, x0 + t * w, ytok, w, h, label=str(t),
            fill=NAVY, edge=AMBER if t in cu[:-1] else TEAL,
            lw=2.4 if t in cu[:-1] else 1.5, size=13)
    text(ax, x0 - 0.18, ytok + h / 2, 'tokens', color=TEAL, size=14, ha='right')

    for i, c in enumerate(cu):
        box(ax, x0 + c * w, ycu, w, h, label=str(c), fill=GREY, edge=TEAL_L,
            lw=1.6, size=13)
    text(ax, x0 - 0.18, ycu + h / 2, 'cu_seqlens', color=TEAL_L, size=14, ha='right')

    xt = x0 + 3 * w + w / 2
    arrow(ax, (xt, ytok + h + 0.08), (x0 + 3 * w + w / 2, ycu - 0.08),
          color=TEAL_L, lw=1.8)
    text(ax, xt + 0.10, (ytok + h + ycu) / 2 + 0.02, ' token 3', color=TEAL_L,
         size=13, ha='left')

    text(ax, 4.9, 0.86, 'bisect_right(cu, 3) - 1 = 1   ->   request 1   (correct)',
         color=TEAL, size=14)
    text(ax, 4.9, 0.46, 'bisect_left(cu, 3)  - 1 = 0   ->   request 0   (wrong)',
         color=RED, size=14)
    text(ax, 4.9, 0.08, 'off by one at exactly the first token of every request',
         color=GREY_L, size=13)
    text(ax, 4.9, 4.18, 'the scan built this table - a boundary search undoes it',
         color=TEAL_L, size=15)
    save(fig, os.path.join(HERE, 'day23_5.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('wrote day23_1..5.png')
