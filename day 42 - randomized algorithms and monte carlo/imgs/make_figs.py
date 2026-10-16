"""Day 42 figures.  Every number is computed by importing randomized.py - nothing typed by hand."""

import math
import os
import random
import sys
from itertools import accumulate, permutations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

from matplotlib.patches import Rectangle, Wedge   # noqa: E402
from diagram_style import *                       # noqa: F401,F403,E402
import randomized as rz                           # noqa: E402

os.chdir(HERE)


def W(h):
    """canvas(9, h) with a 0..10 x-range and equal aspect."""
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def cell(ax, x, y, w, h, label, edge_=TEAL, lw=1.4, fill=NAVY, color=WHITE, size=14, z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge_, linewidth=lw, zorder=z))
    if label is not None:
        text(ax, x + w / 2, y + h / 2, str(label), color=color, size=size, glowing=False, zorder=z + 2)


def fs(f):
    return rz.frac_str(f)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Darts in the unit square; the quarter circle holds pi/4 of them."""
    n = 2000
    rng = random.Random(42)
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    inside = sum(x * x + y * y < 1 for x, y in pts)
    fig, ax = W(5.0)
    s, x0, y0 = 4.6, 0.5, 0.35
    ax.add_patch(Rectangle((x0, y0), s, s, facecolor='none', edgecolor=TEAL, linewidth=1.8, zorder=2))
    ax.add_patch(Wedge((x0, y0), s, 0, 90, facecolor='none', edgecolor=AMBER, linewidth=1.8, zorder=4))
    xin = [x0 + x * s for x, y in pts if x * x + y * y < 1]
    yin = [y0 + y * s for x, y in pts if x * x + y * y < 1]
    xo = [x0 + x * s for x, y in pts if x * x + y * y >= 1]
    yo = [y0 + y * s for x, y in pts if x * x + y * y >= 1]
    ax.scatter(xin, yin, s=3.2, color=TEAL_L, zorder=3, linewidths=0)
    ax.scatter(xo, yo, s=3.2, color=GREY_L, zorder=3, linewidths=0)
    tx = 5.75
    text(ax, tx, 4.55, f'{n:,} random darts', color=TEAL_L, size=17, ha='left', glowing=False)
    text(ax, tx, 3.95, f'{inside:,} inside the quarter circle', color=AMBER, size=15, ha='left',
         glowing=False)
    text(ax, tx, 3.2, f'pi  ~  4 x {inside:,} / {n:,}', color=TEAL, size=16, ha='left', glowing=False)
    text(ax, tx, 2.6, f'    =  {4 * inside / n:.3f}', color=AMBER, size=18, ha='left', glowing=False)
    text(ax, tx, 1.75, f'one standard error at {n:,}:', color=TEAL, size=13, ha='left', glowing=False)
    text(ax, tx, 1.35, f'+- {rz.pi_std(n):.3f}', color=TEAL, size=13, ha='left', glowing=False)
    text(ax, tx, 0.75, f'at {1_000_000:,}: +- {rz.pi_std(1_000_000):.4f}', color=TEAL, size=13,
         ha='left', glowing=False)
    save(fig, 'day42_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """RMS error vs N on log-log axes, against the 1.642 / sqrt(N) law."""
    rng = random.Random(42)
    Ns = [100, 1_000, 10_000, 100_000]
    trials = {100: 400, 1_000: 200, 10_000: 100, 100_000: 40}
    rms = {}
    for n in Ns:
        errs = [rz.estimate_pi(n, rng) - math.pi for _ in range(trials[n])]
        rms[n] = math.sqrt(sum(e * e for e in errs) / len(errs))
    fig, ax = W(5.2)
    X0, X1, Y0, Y1 = 1.6, 9.4, 0.7, 5.2
    lx = lambda n: X0 + (math.log10(n) - 2) / 4 * (X1 - X0)          # 10^2 .. 10^6
    ly = lambda e: Y0 + (math.log10(e) + 3.5) / 3 * (Y1 - Y0)         # 10^-3.5 .. 10^-0.5
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for k in range(2, 7):
        ax.plot([lx(10 ** k)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, lx(10 ** k), Y0 - 0.35, f'{10 ** k:,}', color=TEAL, size=12, glowing=False)
    for e in (0.1, 0.01, 0.001):
        ax.plot([X0, X0 - 0.1], [ly(e)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.18, ly(e), f'{e:g}', color=TEAL, size=12, ha='right', glowing=False)
    text(ax, (X0 + X1) / 2, Y0 - 0.8, 'darts N', color=TEAL_L, size=14, glowing=False)
    text(ax, X0 - 1.1, Y1 - 0.1, 'error', color=TEAL_L, size=14, glowing=False)
    law = [(n, rz.pi_std(n)) for n in (100, 1_000_000)]
    ax.plot([lx(n) for n, _ in law], [ly(e) for _, e in law], color=AMBER, lw=1.8, zorder=3)
    for n in Ns:
        ax.scatter([lx(n)], [ly(rms[n])], s=60, color=TEAL_L, zorder=5, edgecolors=NAVY)
    n1, n2 = 1_000, 100_000
    text(ax, X0 + 0.3, 1.75, f'{n1:,} -> {n2:,} darts (100x):', color=TEAL_L, size=13, ha='left',
         glowing=False)
    text(ax, X0 + 0.3, 1.3, f'error {rz.pi_std(n1):.4f} -> {rz.pi_std(n2):.4f}  '
         f'({rz.pi_std(n1) / rz.pi_std(n2):.0f}x smaller)', color=TEAL_L, size=13, ha='left', glowing=False)
    legend(ax, 6.0, Y1 - 0.05, [(TEAL_L, 'measured (root-mean-square)'), (AMBER, '1.642 / sqrt(N)')], size=12)
    save(fig, 'day42_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """How many equally likely runs land on each order of [a, b, c]."""
    rows = [('swap with any slot', rz.shuffle_naive, RED),
            ('randrange(i) slip', rz.shuffle_off_by_one, RED),
            ('Fisher-Yates', rz.fisher_yates, TEAL_L)]
    perms = list(permutations(range(3)))
    fig, ax = W(5.6)
    bw, gap, x0 = 0.85, 0.25, 3.1
    for r, (name, fn, col) in enumerate(rows):
        cnt, runs = rz.shuffle_report(fn, 3)
        yb = 4.4 - r * 1.85
        text(ax, 2.85, yb + 0.55, name, color=col if col == RED else TEAL_L, size=14, ha='right',
             glowing=False)
        text(ax, 2.85, yb + 0.15, f'{runs} runs', color=GREY_L, size=12, ha='right', glowing=False)
        top = max(cnt.values())
        for i, p in enumerate(perms):
            x = x0 + i * (bw + gap)
            c = cnt.get(p, 0)
            h = 0.95 * c / top
            if c:
                hi = (col == RED and c == top and len(set(cnt.values())) > 1)
                ax.add_patch(Rectangle((x, yb), bw, h, facecolor=NAVY, edgecolor=AMBER if hi else TEAL,
                                       linewidth=2.4 if hi else 1.4, zorder=3))
                text(ax, x + bw / 2, yb + h + 0.2, f'{c}/{runs}', color=TEAL_L, size=11, glowing=False)
            else:
                text(ax, x + bw / 2, yb + 0.2, '0', color=RED, size=12, glowing=False)
            text(ax, x + bw / 2, yb - 0.22, ''.join('abc'[k] for k in p), color=GREY_L, size=12,
                 glowing=False)
        ax.plot([x0 - 0.1, x0 + 6 * (bw + gap) - gap + 0.1], [yb, yb], color=TEAL, lw=1.0)
    save(fig, 'day42_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Reservoir sampling on a b c d e: keep chance per step and final chance."""
    items = list('abcde')
    good = rz.exact_distribution(lambda rr: rz.reservoir_one(items, rr))
    bad = rz.exact_distribution(lambda rr: rz.reservoir_off_by_one(items, rr))
    fig, ax = W(5.2)
    w, gap, x0 = 1.15, 0.3, 3.0
    xs = [x0 + i * (w + gap) for i in range(5)]
    for i, x in enumerate(xs):
        text(ax, x + w / 2, 5.25, items[i], color=TEAL_L, size=18)
    blocks = [(4.05, 'randrange(i) == 0', 'i = 1, 2, 3 ...', [f'1/{i + 1}' for i in range(5)], good,
               TEAL_L),
              (1.35, 'randrange(i) == 0', 'i = 0, 1, 2 ...', ['keep'] + [f'1/{i}' for i in range(1, 5)], bad,
               RED)]
    for y, code, idx, keep, dist, col in blocks:
        text(ax, 2.75, y + 0.55, code, color=col, size=13, ha='right', glowing=False)
        text(ax, 2.75, y + 0.15, idx, color=GREY_L, size=12, ha='right', glowing=False)
        text(ax, 2.75, y - 0.65, 'final chance', color=GREY_L, size=12, ha='right', glowing=False)
        for i, x in enumerate(xs):
            cell(ax, x, y, w, 0.6, keep[i], size=13, color=WHITE if keep[i] != '1/1' else AMBER,
                 edge_=AMBER if keep[i] == '1/1' else TEAL, lw=2.4 if keep[i] == '1/1' else 1.4)
            p = dist.get(items[i], 0)
            text(ax, x + w / 2, y - 0.65, fs(p) if p else '0', color=RED if not p else col, size=15,
                 glowing=False)
    text(ax, xs[1] + w / 2, 1.35 + 0.95, 'always replaces a', color=AMBER, size=12, glowing=False)
    save(fig, 'day42_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """LeetCode 470: two rand7 calls = 49 cells; keep 40, reject 9."""
    one = rz.exact_distribution(rz.rand10_one_round)
    acc = sum(v for k, v in one.items() if k is not None)
    fig, ax = W(5.6)
    c, x0, y0 = 0.68, 0.9, 0.5
    for r in range(7):
        text(ax, x0 - 0.3, y0 + (6 - r) * c + c / 2, str(r + 1), color=GREY_L, size=12, glowing=False)
        text(ax, x0 + r * c + c / 2, y0 + 7 * c + 0.25, str(r + 1), color=GREY_L, size=12, glowing=False)
        for k in range(7):
            idx = r * 7 + k + 1
            x, y = x0 + k * c, y0 + (6 - r) * c
            if idx <= 40:
                cell(ax, x, y, c, c, (idx - 1) % 10 + 1, size=12)
            else:
                cell(ax, x, y, c, c, None, fill=GREY, edge_=RED, lw=1.4)
                cross(ax, x + c / 2, y + c / 2, r=0.2, lw=1.4)
    text(ax, x0 + 3.5 * c, y0 + 7 * c + 0.7, 'second rand7()', color=TEAL_L, size=13, glowing=False)
    text(ax, x0 - 0.65, y0 + 3.5 * c, 'first rand7()', color=TEAL_L, size=13, rotation=90, glowing=False)
    tx = 6.15
    text(ax, tx, 4.75, f'keep {acc.numerator} of {acc.denominator} cells', color=TEAL_L, size=16, ha='left',
         glowing=False)
    text(ax, tx, 4.25, f'each digit 1..10: {fs(one[1] / acc)}', color=TEAL_L, size=14, ha='left', glowing=False)
    text(ax, tx, 3.6, f'rejected {one[None].numerator} -> draw again', color=RED, size=14, ha='left',
         glowing=False)
    text(ax, tx, 2.8, f'expected calls: 2 x 49/40 = {2 * 49 / 40}', color=TEAL, size=13, ha='left',
         glowing=False)
    bias = rz.exact_distribution(lambda rr: rz.rand10_mod_bias(lambda: rr(7) + 1))
    text(ax, tx, 1.95, 'no rejection (idx % 10 + 1):', color=AMBER, size=13, ha='left', glowing=False)
    text(ax, tx, 1.5, f'1 gets {fs(bias[1])}, 2..10 get {fs(bias[2])}', color=AMBER, size=13, ha='left',
         glowing=False)
    save(fig, 'day42_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """LeetCode 528: prefix sums cut 0..total into slices; bisect finds the slice."""
    w = [1, 3]
    prefix = list(accumulate(w))
    good = rz.exact_distribution(lambda rr: rz.pick_index_exact(w, rr))
    bad = rz.exact_distribution(lambda rr: rz.pick_index_off_by_one(w, rr))
    fig, ax = W(4.9)
    u, x0 = 1.5, 2.4
    text(ax, 5.4, 5.0, f'w = {w}   prefix = {prefix}', color=TEAL_L, size=16, glowing=False)
    rows = [(3.35, f'randrange({prefix[-1]}) + bisect_right', range(prefix[-1]),
             lambda t: rz.bisect_right(prefix, t), good, TEAL_L),
            (1.15, f'randint(0, {prefix[-1]}) + bisect_left', range(prefix[-1] + 1),
             lambda t: rz.bisect_left(prefix, t), bad, RED)]
    for y, name, targets, f, dist, col in rows:
        text(ax, x0 - 0.15, y + 1.0, name, color=col, size=13, ha='left', glowing=False)
        for t in targets:
            i = f(t)
            extra = col == RED and t == 1
            cell(ax, x0 + t * u, y, u, 0.6, f'target {t}', fill=NAVY if i else PALE,
                 color=WHITE if i else NAVY,
                 edge_=AMBER if extra else TEAL, lw=2.6 if extra else 1.4, size=12)
            text(ax, x0 + t * u + u / 2, y - 0.3, f'-> index {i}', color=TEAL_L if i else AMBER, size=12,
                 glowing=False)
        text(ax, 1.0, y + 0.3, f'{fs(dist[0])} : {fs(dist[1])}', color=col, size=15, glowing=False)
    text(ax, 1.0, 3.35 + 1.0, 'P(0) : P(1)', color=GREY_L, size=12, glowing=False)
    save(fig, 'day42_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print('figures written')
