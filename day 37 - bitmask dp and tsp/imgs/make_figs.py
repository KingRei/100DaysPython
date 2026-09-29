"""Figures for Day 37 - state-compression DP and Held-Karp.

Every number on every figure comes from importing bitmask_dp, never typed in.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: E402,F403
import bitmask_dp as M               # noqa: E402

D = M.DIST
OPT, OPT_TOUR, _ = M.held_karp(D)
NN_COST, NN_TOUR = M.nearest_neighbour(D)
MASK_ONLY = M.tsp_mask_only(D)
MST = M.mst_weight(D)
BRUTE, BRUTE_TOUR, PERMS = M.tsp_brute(D)
N = len(D)


def place(x0, y0, w, h):
    """Map city coordinates into a (x0, y0, w, h) rectangle of the canvas."""
    xs = [c[1] for c in M.CITIES]
    ys = [c[2] for c in M.CITIES]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + (c[1] - lo_x) / (hi_x - lo_x) * w,
             y0 + (c[2] - lo_y) / (hi_y - lo_y) * h) for c in M.CITIES]


def draw_cities(ax, pts, r=0.30, size=11, fill=NAVY, edge=TEAL):
    for i, (x, y) in enumerate(pts):
        node(ax, x, y, M.SHORT[i], r=r, size=size, fill=fill, edge=edge)


def draw_tour(ax, pts, tour, color=TEAL, r=0.30, weights=True, lw=1.8,
              wsize=11, ls=None):
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]
        w = D[a][b] if weights else None
        edge(ax, pts[a], pts[b], color=color, r=r, directed=False,
             weight=w, lw=lw, wsize=wsize, wcolor=color)


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = canvas(9, 6.4, xlim=(0, 9), ylim=(0, 6.4))
    pts = place(1.9, 1.15, 4.6, 4.5)
    draw_tour(ax, pts, OPT_TOUR)
    draw_cities(ax, pts)
    text(ax, 4.5, 6.15, 'one closed tour, eight stops', color=TEAL_L, size=17)
    text(ax, 0.9, 3.7, f'{OPT} km', color=CYAN, size=26)
    text(ax, 0.9, 3.2, 'the shortest', color=TEAL, size=13)
    text(ax, 0.9, 2.85, 'of all of them', color=TEAL, size=13)
    text(ax, 7.9, 3.7, f'{PERMS:,}', color=AMBER, size=22)
    text(ax, 7.9, 3.25, 'orderings exist', color=AMBER, size=13)
    text(ax, 7.9, 2.75, f'{(1 << N) * N * N:,}', color=TEAL_L, size=22)
    text(ax, 7.9, 2.30, 'DP steps instead', color=TEAL_L, size=13)
    text(ax, 4.5, 0.3,
         'the answer is a cycle, so any city may be called the start',
         color=TEAL, size=12)
    save(fig, 'day37_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    fig, ax = canvas(9, 5.4, xlim=(0, 9), ylim=(0, 5.4))
    visited = [0, 1, 2, 3]
    mask = 0
    for i in visited:
        mask = M.add(mask, i)

    x0, w = 1.55, 0.72
    labs = ['1' if M.has(mask, i) else '0' for i in range(N)]
    cs = cells(ax, x0, 4.25, N, w=w, h=0.62, labels=labs,
               fill=NAVY, edge=TEAL, size=15)
    for i, (cx, _) in enumerate(cs):
        text(ax, cx, 5.08, M.SHORT[i], color=TEAL, size=11)
    text(ax, x0 - 0.22, 4.56, 'mask', color=TEAL_L, size=14, ha='right')
    text(ax, x0 + N * w + 0.25, 4.56, f'= {mask}', color=TEAL_L, size=14,
         ha='left')
    text(ax, 4.5, 3.62, 'the same four cities are behind me either way',
         color=TEAL, size=13)

    # two futures out of the same mask
    for col, last in enumerate((2, 3)):
        bx = 0.75 + col * 4.35
        box(ax, bx, 2.55, 3.35, 0.72,
            label=f'last = {M.SHORT[last]}', fill=GREY, edge=AMBER, lw=2.2,
            color=AMBER, size=15)
        nxt = 4
        text(ax, bx + 1.67, 2.15,
             f'next hop to {M.SHORT[nxt]} costs {D[last][nxt]} km',
             color=TEAL_L, size=13)
        far = 7
        text(ax, bx + 1.67, 1.72,
             f'next hop to {M.SHORT[far]} costs {D[last][far]} km',
             color=TEAL_L, size=13)
    text(ax, 4.5, 1.05,
         f'dp[mask] alone cannot tell these apart, so it cannot price the next edge',
         color=AMBER, size=13)
    text(ax, 4.5, 0.55, 'the state has to be (mask, last)', color=CYAN, size=18)
    save(fig, 'day37_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    fig, ax = canvas(9, 5.6, xlim=(0, 9), ylim=(0, 5.6))
    # left: what dp[mask] alone builds
    ptsL = place(0.75, 0.95, 3.05, 3.55)
    parent = {}
    seen = {0}
    while len(seen) < N:
        c, i, j = min((D[a][b], a, b) for a in seen for b in range(N)
                      if b not in seen)
        parent[j] = i
        seen.add(j)
    for j, i in parent.items():
        edge(ax, ptsL[i], ptsL[j], color=RED, r=0.26, directed=False, lw=1.8)
    draw_cities(ax, ptsL, r=0.26, size=9, edge=RED)
    text(ax, 2.25, 5.15, 'dp[mask] with no "last"', color=RED, size=15)
    text(ax, 2.25, 4.75, f'{MASK_ONLY} km', color=RED, size=20)
    text(ax, 2.25, 0.52, 'a tree: nobody ever comes home', color=RED, size=12)

    # right: the real tour
    ptsR = place(5.2, 0.95, 3.05, 3.55)
    draw_tour(ax, ptsR, OPT_TOUR, weights=False)
    draw_cities(ax, ptsR, r=0.26, size=9)
    text(ax, 6.7, 5.15, 'dp[mask][last]', color=TEAL_L, size=15)
    text(ax, 6.7, 4.75, f'{OPT} km', color=CYAN, size=20)
    text(ax, 6.7, 0.52, 'a cycle: every city has two neighbours',
         color=TEAL, size=12)

    text(ax, 4.5, 3.25, '<', color=AMBER, size=30)
    text(ax, 4.5, 2.62, 'smaller', color=AMBER, size=13)
    text(ax, 4.5, 2.30, 'than the', color=AMBER, size=13)
    text(ax, 4.5, 1.98, 'optimum', color=AMBER, size=13)
    save(fig, 'day37_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    rows = M.crossover(brute_upto=10, hk_upto=15)
    ns = [r[0] for r in rows]
    hk = [r[2] for r in rows]
    measured = [(r[0], r[1]) for r in rows if r[1] is not None]
    base_n, base_ms = measured[-1]
    extrap = [(n, base_ms * math.factorial(n - 1) / math.factorial(base_n - 1))
              for n in ns if n >= base_n]

    fig, ax = canvas(9, 5.6, xlim=(0, 9), ylim=(0, 5.6))
    X0, X1, Y0, Y1 = 1.45, 7.35, 1.05, 4.55
    LO, HI = 0.0, 8.2                                   # log10 milliseconds

    def px(n):
        return X0 + (n - ns[0]) / (ns[-1] - ns[0]) * (X1 - X0)

    def py(ms):
        return Y0 + (math.log10(ms) - LO) / (HI - LO) * (Y1 - Y0)

    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    for dec, lab in ((0, '1 ms'), (2, '0.1 s'), (4, '10 s'),
                     (6, '17 min'), (8, '28 h')):
        y = py(10 ** dec)
        ax.plot([X0 - 0.09, X0], [y, y], color=TEAL, lw=1.2)
        text(ax, X0 - 0.17, y, lab, color=TEAL, size=11, ha='right')
    for n in ns:
        ax.plot([px(n), px(n)], [Y0 - 0.08, Y0], color=TEAL, lw=1.2)
        text(ax, px(n), Y0 - 0.3, str(n), color=TEAL, size=11)
    text(ax, (X0 + X1) / 2, Y0 - 0.60, 'number of cities', color=TEAL, size=13)

    ax.plot([px(n) for n, _ in measured], [py(ms) for _, ms in measured],
            color=RED, lw=2.0, marker='o', ms=5)
    ax.plot([px(n) for n, _ in extrap], [py(ms) for _, ms in extrap],
            color=RED, lw=1.8, ls='--')
    ax.plot([px(n) for n in ns], [py(ms) for ms in hk],
            color=TEAL_L, lw=2.0, marker='o', ms=5)

    text(ax, px(10) + 0.12, py(measured[-1][1]) - 0.42,
         'brute force', color=RED, size=14, ha='left')
    text(ax, px(12), py(hk[4]) + 0.38, 'Held-Karp', color=TEAL_L, size=14)
    hours = extrap[-1][1] / 3600_000
    text(ax, X1 + 0.15, py(extrap[-1][1]), f'{hours:.0f} h', color=RED,
         size=15, ha='left')
    text(ax, X1 + 0.15, py(hk[-1]), f'{hk[-1] / 1000:.1f} s', color=CYAN,
         size=15, ha='left')
    text(ax, 4.5, 5.15, 'same exact answer, two exponentials',
         color=TEAL_L, size=17)
    text(ax, 4.5, 0.12,
         f'measured on this machine; dashed part is {PERMS:,}-permutation'
         f' timing scaled by (n-1)!',
         color=TEAL, size=11)
    save(fig, 'day37_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    fig, ax = canvas(9, 5.6, xlim=(0, 9), ylim=(0, 5.6))
    ptsL = place(0.55, 0.95, 2.70, 3.55)
    draw_tour(ax, ptsL, NN_TOUR, color=AMBER, r=0.26, weights=False)
    draw_cities(ax, ptsL, r=0.26, size=9, edge=AMBER)
    text(ax, 1.90, 5.15, 'nearest neighbour', color=AMBER, size=15)
    text(ax, 1.90, 4.75, f'{NN_COST} km', color=AMBER, size=20)
    bad_from = NN_TOUR[NN_TOUR.index(3)]
    bad_to = NN_TOUR[NN_TOUR.index(3) + 1]
    cross(ax, (ptsL[bad_from][0] + ptsL[bad_to][0]) / 2,
          (ptsL[bad_from][1] + ptsL[bad_to][1]) / 2, r=0.22)
    text(ax, 1.90, 0.52,
         f'{M.SHORT[bad_from]} to {M.SHORT[bad_to]} is the nearest hop,',
         color=RED, size=12)
    text(ax, 1.90, 0.17, 'and it strands the whole south', color=RED, size=12)

    ptsR = place(5.60, 0.95, 2.70, 3.55)
    draw_tour(ax, ptsR, OPT_TOUR, weights=False)
    draw_cities(ax, ptsR, r=0.26, size=9)
    text(ax, 6.95, 5.15, 'Held-Karp', color=TEAL_L, size=15)
    text(ax, 6.95, 4.75, f'{OPT} km', color=CYAN, size=20)
    text(ax, 6.95, 0.52, 'the exact optimum', color=TEAL, size=12)

    pct = 100 * (NN_COST - OPT) / OPT
    text(ax, 4.45, 3.35, f'{NN_COST - OPT} km', color=AMBER, size=20)
    text(ax, 4.45, 2.92, f'{pct:.1f}% over', color=AMBER, size=14)
    text(ax, 4.45, 2.40, 'no error,', color=TEAL, size=12)
    text(ax, 4.45, 2.08, 'no warning,', color=TEAL, size=12)
    text(ax, 4.45, 1.76, 'a real tour', color=TEAL, size=12)
    save(fig, 'day37_5.png')


if __name__ == '__main__':
    os.chdir(HERE)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures written')
