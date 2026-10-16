"""Day 41 figures.  Every number is imported from number_theory.py - nothing typed by hand."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

from matplotlib.patches import Rectangle   # noqa: E402
from diagram_style import *                # noqa: F401,F403,E402
import number_theory as nt                 # noqa: E402

os.chdir(HERE)


def W(h):
    """canvas(9, h) with a 0..10 x-range and equal aspect."""
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def cell(ax, x, y, w, h, label, edge_=TEAL, lw=1.4, fill=NAVY, color=WHITE, size=14, z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge_, linewidth=lw, zorder=z))
    if label is not None:
        text(ax, x + w / 2, y + h / 2, str(label), color=color, size=size, glowing=False, zorder=z + 2)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Square-and-multiply on 3^13 mod 100: one column per bit of 13."""
    a, e, m = 3, 13, 100
    rows = nt.power_trace(a, e, m)              # k = 0 .. 3
    n = len(rows)
    H = 5.4
    fig, ax = W(H)
    w, gap, x0 = 1.55, 0.35, 2.55
    xs = {r['k']: x0 + (n - 1 - r['k']) * (w + gap) for r in rows}   # bit 3 on the left
    y_bit, y_sq, y_res = 4.55, 3.15, 1.75
    text(ax, 2.25, y_bit + 0.3, 'bit of 13', color=TEAL_L, size=14, ha='right', glowing=False)
    text(ax, 2.25, y_sq + 0.3, f'{a}^(2^k) mod {m}', color=TEAL_L, size=14, ha='right', glowing=False)
    text(ax, 2.25, y_res + 0.3, 'result', color=TEAL_L, size=14, ha='right', glowing=False)
    for r in rows:
        x, on = xs[r['k']], r['bit']
        text(ax, x + w / 2, y_bit + 0.85, f'k = {r["k"]}', color=GREY_L, size=12, glowing=False)
        cell(ax, x, y_bit, w, 0.6, r['bit'], edge_=AMBER if on else TEAL, lw=2.6 if on else 1.4,
             color=WHITE if on else GREY_L)
        cell(ax, x, y_sq, w, 0.6, f'{a}^{2 ** r["k"]} = {r["square"]}', edge_=AMBER if on else TEAL,
             lw=2.6 if on else 1.4, size=13)
        cell(ax, x, y_res, w, 0.6, r['result'], edge_=AMBER if on else TEAL, lw=2.6 if on else 1.4)
        if on:
            arrow(ax, (x + w / 2, y_sq), (x + w / 2, y_res + 0.6), color=AMBER, lw=1.6)
            text(ax, x + w / 2 + 0.12, (y_sq + y_res + 0.6) / 2, 'x', color=AMBER, size=13, ha='left',
                 glowing=False)
    for k in range(n - 1):                       # squaring arrows, right to left
        arrow(ax, (xs[k], y_sq + 0.3), (xs[k + 1] + w, y_sq + 0.3), color=TEAL, lw=1.4)
        text(ax, (xs[k] + xs[k + 1] + w) / 2, y_sq + 0.52, 'sq', color=TEAL, size=11, glowing=False)
        arrow(ax, (xs[k], y_res + 0.3), (xs[k + 1] + w, y_res + 0.3), color=TEAL, lw=1.4)
    final = rows[-1]['result']
    text(ax, 5.0, 0.95, f'13 = 0b1101 = 8 + 4 + 1   so   3^13 = 3^8 x 3^4 x 3^1  ->  {final}   '
         f'(3^13 = {3 ** 13:,})', color=AMBER, size=13, glowing=False)
    naive, fast = nt.multiplications(10**18)
    text(ax, 5.0, 0.4, f'exponent 10^18: {fast} multiplications instead of {naive:,}',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day41_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Euclid as squares cut from a 252 x 105 rectangle."""
    A, B = 252, 105
    trace = nt.gcd_trace(A, B)
    s = 8.4 / A
    H = 5.2
    fig, ax = W(H)
    X0, Y0 = 0.8, 1.25
    # walk the cuts: keep a remaining rectangle (x, y, w, h) in data units
    x, y, w, h = 0.0, 0.0, A, B
    colours = [TEAL_L, AMBER, CYAN]
    for step, (a, b, q, r) in enumerate(trace):
        side = b
        for i in range(q):
            if w >= h:                            # cut squares from the left
                sx, sy = x + i * side, y
            else:                                 # cut squares from the top
                sx, sy = x, y + h - (i + 1) * side
            ax.add_patch(Rectangle((X0 + sx * s, Y0 + sy * s), side * s, side * s, facecolor=NAVY,
                                   edgecolor=colours[step], linewidth=2.2, zorder=3 + step))
            text(ax, X0 + (sx + side / 2) * s, Y0 + (sy + side / 2) * s, str(side), color=WHITE,
                 size=max(11, 18 - 3 * step), glowing=False, zorder=8)
        if w >= h:
            x, w = x + q * side, w - q * side
        else:
            h = h - q * side
    text(ax, X0 + A * s / 2, Y0 + B * s + 0.3, f'{A} wide', color=TEAL_L, size=13, glowing=False)
    text(ax, X0 - 0.15, Y0 + B * s / 2, f'{B}', color=TEAL_L, size=13, ha='right', glowing=False)
    lines = '     '.join(f'{a} = {q} x {b} + {r}' for a, b, q, r in trace)
    text(ax, 5.0, 0.75, lines, color=TEAL, size=13, glowing=False)
    text(ax, 5.0, 0.25, f'the last square that tiles what is left: gcd = {nt.gcd(A, B)}',
         color=AMBER, size=13, glowing=False)
    save(fig, 'day41_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Sieve of Eratosthenes on 0..49: who crossed each number out first."""
    N = 50
    first = {}
    for p, new in nt.sieve_trace(N):
        for q in new:
            first[q] = p
    primes = nt.sieve(N)
    H = 5.6
    fig, ax = W(H)
    w, h, x0, ytop = 0.78, 0.66, 0.6, 4.95
    fills = {2: GREY, 3: '#1d3a46', 5: '#3a2f12', 7: '#3b1616'}
    for i in range(N):
        r, c = divmod(i, 10)
        x, y = x0 + c * w, ytop - r * h
        if i < 2:
            cell(ax, x, y, w, h, i, edge_=GREY_L, fill=GREY, color=GREY_L, size=13)
        elif primes[i]:
            cell(ax, x, y, w, h, i, edge_=CYAN, lw=2.4, size=14)
        else:
            p = first[i]
            sq = i == p * p
            cell(ax, x, y, w, h, i, edge_=AMBER if sq else TEAL, lw=2.6 if sq else 1.0,
                 fill=fills[p], color=GREY_L, size=13, z=4 if sq else 3)
            text(ax, x + w - 0.06, y + 0.12, f'{p}', color=AMBER if sq else TEAL, size=9,
                 ha='right', glowing=False, zorder=7)
    lx = x0 + 10 * w + 0.35
    legend(ax, lx, ytop + 0.4, [(NAVY, 'prime'), (fills[2], 'crossed by 2'), (fills[3], 'crossed by 3'),
                                (fills[5], 'crossed by 5'), (fills[7], 'crossed by 7')], size=12, dy=0.42)
    text(ax, lx, ytop - 1.95, 'amber edge = p x p,', color=AMBER, size=12, ha='left', glowing=False)
    text(ax, lx, ytop - 2.3, 'where p starts', color=AMBER, size=12, ha='left', glowing=False)
    c_sq = nt.sieve_loop(10**6)[1]
    c_2p = nt.sieve_loop(10**6, start_at_square=False)[1]
    text(ax, 5.0, 1.75, '8 x 8 = 64 > 49: no prime past 7 has a multiple left to cross out', color=TEAL,
         size=13, glowing=False)
    text(ax, 5.0, 1.2, f'n = 1,000,000: {c_sq:,} crossings starting at p x p, {c_2p:,} starting at 2p',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day41_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Fermat's 'inverse' pow(a, m-2, m) on prime 11 and composite 10."""
    H = 5.6
    fig, ax = W(H)
    w, h = 0.72, 0.5
    blocks = [(11, 5.0), (10, 2.4)]
    for m, ytop in blocks:
        x0 = 2.75
        labels = ['a', f'pow(a, {m - 2}, {m})', f'a x that % {m}']
        for j, lab in enumerate(labels):
            text(ax, x0 - 0.15, ytop - j * h + h / 2, lab, color=TEAL_L, size=13, ha='right', glowing=False)
        for a in range(1, m):
            x = x0 + (a - 1) * w
            inv = nt.inverse_fermat(a, m)
            prod = a * inv % m
            invertible = math.gcd(a, m) == 1
            cell(ax, x, ytop, w, h, a, size=13)
            cell(ax, x, ytop - h, w, h, inv, size=13, color=WHITE if invertible else GREY_L)
            ok = prod == 1
            if not invertible:
                cell(ax, x, ytop - 2 * h, w, h, prod, fill=GREY, edge_=GREY_L, color=GREY_L, size=13)
            else:
                cell(ax, x, ytop - 2 * h, w, h, prod, edge_=CYAN if ok else RED, lw=2.4,
                     color=WHITE, size=13, z=4)
        if m == 11:
            note = '11 is prime: every row-3 cell is 1, Fermat really is the inverse'
            col = TEAL
        else:
            n_inv, wrong = nt.fermat_report(10)
            note = (f'10 = 2 x 5: {n_inv} numbers have an inverse, Fermat gets {wrong} of them wrong')
            col = RED
        text(ax, 5.0, ytop - 2 * h - 0.35, note, color=col, size=13, glowing=False)
    text(ax, 5.0, 0.55, f'(the inverse of 3 is {nt.inverse(3, 10)}, but pow(3, 8, 10) = {nt.inverse_fermat(3, 10)});'
         '  grey: no inverse exists, shares a factor with 10', color=GREY_L, size=12, glowing=False)
    save(fig, 'day41_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """C(n, 2) with // on factorials reduced mod 1e9+7: right until n = 13."""
    ns = list(range(9, 17))
    f = nt.factorials(20)
    H = 4.6
    fig, ax = W(H)
    w, h, x0, ytop = 0.98, 0.52, 2.05, 3.95
    labels = ['n', 'fact[n] = n! % p', 'true C(n, 2)', '// version']
    for j, lab in enumerate(labels):
        text(ax, x0 - 0.15, ytop - j * h + h / 2, lab, color=TEAL_L, size=13, ha='right', glowing=False)
    for i, n in enumerate(ns):
        x = x0 + i * w
        wrapped = math.factorial(n) >= nt.MOD
        cell(ax, x, ytop, w, h, n, size=13)
        cell(ax, x, ytop - h, w, h, f'{f[n]:,}', size=8 if f[n] > 10**8 else 9,
             edge_=AMBER if wrapped else TEAL, lw=2.2 if wrapped else 1.2, color=WHITE)
        cell(ax, x, ytop - 2 * h, w, h, math.comb(n, 2), size=13)
        got = nt.ncr_floor_div(n, 2, f)
        ok = got == math.comb(n, 2) % nt.MOD
        cell(ax, x, ytop - 3 * h, w, h, got, size=13, edge_=CYAN if ok else RED, lw=2.4, z=4)
    k = ns.index(13)
    xw = x0 + k * w
    ax.plot([xw, xw], [ytop - 3 * h - 0.2, ytop + h + 0.25], color=RED, lw=2.0, ls='--', zorder=7)
    text(ax, xw, ytop + h + 0.45, f'13! = {math.factorial(13):,} > p = {nt.MOD:,}', color=RED, size=13,
         glowing=False)
    text(ax, 5.0, 1.45, 'amber: fact[n] is now a remainder, not n!  - and // of remainders means nothing',
         color=AMBER, size=13, glowing=False)
    nf = nt.factorials(200)
    bad = sum(nt.ncr_floor_div(n, kk, nf) != math.comb(n, kk) % nt.MOD for n in range(31) for kk in range(n + 1))
    tot = sum(n + 1 for n in range(31))
    text(ax, 5.0, 0.9, f'every n <= 12 is right, so small tests pass;  n <= 30: {bad} of {tot} wrong',
         color=TEAL, size=13, glowing=False)
    text(ax, 5.0, 0.35, 'fix: multiply by the modular inverse of k! (n-k)!, never divide',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day41_5.png')


if __name__ == '__main__':
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn()
    print('ok')
