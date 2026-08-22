"""Figures for day 17 - Bellman-Ford, negative cycles, Floyd-Warshall, LC 787."""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
from diagram_style import *                       # noqa
import bellman_ford_floyd as BF                   # numbers come from the real code

NM = 'ABCDE'
P = {0: (1.6, 3.0), 1: (4.8, 4.9), 2: (8.0, 4.9), 3: (4.8, 1.1), 4: (8.0, 1.1)}
E = BF.EDGES
RAD = {(1, 2): 0.16, (2, 1): 0.16, (1, 4): 0.22, (3, 2): -0.22}
# weights whose straight-line midpoint would land on top of a node
WPOS = {(4, 0): (3.15, 2.95), (1, 2): (6.4, 4.34), (2, 1): (6.4, 5.46)}


def draw_graph(ax, edges, off=(0.0, 0.0), r=0.40, hot=(), weights=True, wsize=13):
    pos = {v: (P[v][0] + off[0], P[v][1] + off[1]) for v in P}
    for u, v, w in edges:
        col = AMBER if (u, v) in hot else TEAL
        moved = weights and (u, v) in WPOS
        if moved:
            text(ax, WPOS[(u, v)][0] + off[0], WPOS[(u, v)][1] + off[1], str(w),
                 color=(AMBER if (u, v) in hot else TEAL_L), size=wsize)
        edge(ax, pos[u], pos[v], weight=(None if (moved or not weights) else w), r=r,
             rad=RAD.get((u, v), 0.0), color=col,
             wcolor=(AMBER if (u, v) in hot else TEAL_L), wsize=wsize,
             lw=2.4 if (u, v) in hot else 1.8)
    for v in P:
        node(ax, pos[v][0], pos[v][1], NM[v], r=r)
    return pos


def cell_text(ax, x, y, w, h, s, color=TEAL_L, fill=NAVY, size=14, edgec=TEAL):
    box(ax, x, y, w, h, fill=fill, edge=edgec, lw=1.4)
    text(ax, x + w / 2, y + h / 2, s, color=color, size=size, glowing=False)


# ---------------------------------------------------------------- 1. rounds
def fig1():
    snaps = BF.rounds_of_bellman_ford(BF.N, E, 0)
    fig, ax = canvas(12, 6.2, xlim=(0, 19.2), ylim=(0, 7.0))
    draw_graph(ax, E, off=(0.0, 0.55))
    text(ax, 4.8, 6.75, 'relax every edge, V-1 times', color=TEAL, size=16)

    x0, y0, cw, ch = 10.8, 5.55, 1.28, 0.62
    cell_text(ax, x0, y0, 2.0, ch, 'round', fill=GREY, color=PALE)
    for j, c in enumerate(NM):
        cell_text(ax, x0 + 2.0 + j * cw, y0, cw, ch, c, fill=GREY, color=PALE)
    for i, d in enumerate(snaps):
        y = y0 - (i + 1) * ch
        cell_text(ax, x0, y, 2.0, ch, str(i), color=TEAL)
        for j, val in enumerate(d):
            s = 'inf' if val == BF.INF else str(val)
            changed = i > 0 and snaps[i - 1][j] != val
            cell_text(ax, x0 + 2.0 + j * cw, y, cw, ch, s,
                      color=AMBER if changed else TEAL_L,
                      edgec=AMBER if changed else TEAL)
    text(ax, 14.5, 1.55, 'round r settles every path of r edges', color=TEAL, size=14)
    text(ax, 14.5, 0.95, 'B drops 6 to 2: A-D-C-B is cheaper, and only', color=TEAL, size=14)
    text(ax, 14.5, 0.45, 'shows up after the negative edges are relaxed', color=TEAL, size=14)
    save(fig, os.path.join(HERE, 'day17_1.png'))


# ---------------------------------------------------------------- 2. negative cycle
def fig2():
    neg = [(u, v, 1 if (u, v) == (4, 2) else w) for u, v, w in E]
    cyc = BF.find_negative_cycle(BF.N, neg)
    hot = {(1, 4), (4, 2), (2, 1)}
    fig, ax = canvas(12, 6.0, xlim=(0, 17.4), ylim=(0, 6.6))
    draw_graph(ax, neg, off=(0.0, 0.35), hot=hot)
    text(ax, 4.8, 6.3, 'one weight changed: E to C is now 1', color=AMBER, size=15)
    text(ax, 4.8, 0.28, ' - '.join(NM[v] for v in cyc) +
         '  =  -4 + 1 - 2  =  -5', color=AMBER, size=15)

    bx, bw = 10.2, 6.9
    cx = bx + bw / 2
    box(ax, bx, 3.5, bw, 2.6, fill=NAVY, edge=TEAL, lw=1.6)
    text(ax, cx, 5.65, 'how it is detected', color=TEAL_L, size=15)
    text(ax, cx, 5.05, 'Bellman-Ford: run one extra round.', color=TEAL, size=13)
    text(ax, cx, 4.60, 'Anything that still improves is being', color=TEAL, size=13)
    text(ax, cx, 4.15, 'fed by a negative cycle.', color=TEAL, size=13)
    text(ax, cx, 3.70, 'SPFA: a vertex queued V times.', color=TEAL, size=13)

    box(ax, bx, 0.5, bw, 2.5, fill=NAVY, edge=RED, lw=1.6)
    cross(ax, bx + 0.75, 2.50, r=0.24)
    text(ax, cx + 0.5, 2.50, 'shortest path is undefined', color=RED, size=14)
    text(ax, cx, 1.85, 'go round the cycle once more and the cost', color=TEAL, size=13)
    text(ax, cx, 1.40, 'drops by another 5, forever - so the answer', color=TEAL, size=13)
    text(ax, cx, 0.95, 'is minus infinity, not a path.', color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day17_2.png'))


# ---------------------------------------------------------------- 3. Floyd
def fig3():
    fig, ax = canvas(12, 5.6, xlim=(0, 17.4), ylim=(0, 6.2))
    pi, pk, pj = (2.2, 1.6), (5.6, 4.7), (9.0, 1.6)
    edge(ax, pi, pk, weight='dist[i][k]', r=0.5, wsize=13)
    edge(ax, pk, pj, weight='dist[k][j]', r=0.5, wsize=13)
    edge(ax, pi, pj, weight='dist[i][j]', r=0.5, rad=-0.16, wsize=13)
    node(ax, pi[0], pi[1], 'i', r=0.5, size=16)
    node(ax, pk[0], pk[1], 'k', r=0.5, size=16)
    node(ax, pj[0], pj[1], 'j', r=0.5, size=16)
    text(ax, 5.6, 5.75, 'either the best i-to-j path goes through k, or it does not',
         color=TEAL, size=15)
    text(ax, 5.6, 0.55, 'dist[i][j] = min( dist[i][j],  dist[i][k] + dist[k][j] )',
         color=AMBER, size=16)

    bx, bw = 10.6, 6.4
    cx = bx + bw / 2
    box(ax, bx, 0.5, bw, 5.0, fill=NAVY, edge=TEAL, lw=1.6)
    text(ax, cx, 5.05, 'why k is the outer loop', color=TEAL_L, size=15)
    text(ax, cx, 4.45, 'after stage k, dist[i][j] is the', color=TEAL, size=13)
    text(ax, cx, 4.02, 'cheapest i-to-j path whose middle', color=TEAL, size=13)
    text(ax, cx, 3.59, 'vertices all come from {0..k}.', color=TEAL, size=13)
    text(ax, cx, 3.16, 'Stage k reads row k and column k,', color=TEAL, size=13)
    text(ax, cx, 2.73, 'so those must already be final.', color=TEAL, size=13)
    cross(ax, bx + 0.7, 2.05, r=0.22)
    text(ax, cx + 0.45, 2.05, 'k innermost still runs,', color=RED, size=13)
    text(ax, cx, 1.50, 'still fills the table, and quietly', color=TEAL, size=13)
    text(ax, cx, 1.07, 'leaves entries too large - nothing', color=TEAL, size=13)
    text(ax, cx, 0.64, 'crashes, the numbers are wrong.', color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day17_3.png'))


# ---------------------------------------------------------------- 4. LC 787
def fig4():
    flights = [(0, 1, 100), (1, 2, 100), (2, 0, 100), (1, 3, 600), (2, 3, 200)]
    LP = {0: (1.5, 4.0), 1: (4.6, 5.5), 2: (4.6, 2.5), 3: (7.7, 4.0)}
    fig, ax = canvas(12, 6.2, xlim=(0, 18.0), ylim=(0, 7.0))
    for u, v, w in flights:
        edge(ax, LP[u], LP[v], weight=w, r=0.42,
             rad=0.18 if (u, v) == (2, 0) else 0.0, wsize=13)
    for v in LP:
        node(ax, LP[v][0], LP[v][1], str(v), r=0.42)
    text(ax, 4.6, 6.55, 'src 0, dst 3, at most k = 1 stop', color=TEAL, size=15)
    text(ax, 4.6, 1.15, 'k stops = k+1 edges = k+1 rounds of Bellman-Ford',
         color=TEAL, size=14)
    text(ax, 4.6, 0.55, '0-1-3 costs 700; 0-1-2-3 costs 400 but uses 2 stops',
         color=TEAL, size=14)

    x0, cw, ch = 11.4, 1.3, 0.62

    def row(y, label, vals, color, edgec):
        text(ax, x0 - 0.35, y + ch / 2, label, color=color, size=13, ha='right',
             glowing=False)
        for j, s_ in enumerate(vals):
            cell_text(ax, x0 + j * cw, y, cw, ch, s_, color=color, edgec=edgec)

    text(ax, x0 + 2 * cw, 6.65, 'dist after each round', color=TEAL_L, size=15)
    for j, c in enumerate('0123'):
        cell_text(ax, x0 + j * cw, 5.85, cw, ch, c, fill=GREY, color=PALE)
    text(ax, x0 + 2 * cw, 5.40, 'relax from a copy of last round', color=TEAL, size=13)
    row(4.55, 'round 1', ['0', '100', 'inf', 'inf'], TEAL_L, TEAL)
    row(3.85, 'round 2', ['0', '100', '200', '700'], AMBER, AMBER)
    text(ax, x0 + 2 * cw, 3.35, 'relax in place - no copy', color=RED, size=13)
    row(2.50, 'round 1', ['0', '100', '200', '400'], RED, RED)
    text(ax, x0 + 2 * cw, 1.75, 'one round chained 0-1-2-3, so the answer',
         color=TEAL, size=13)
    text(ax, x0 + 2 * cw, 1.30, 'allows more stops than the problem asked for',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day17_4.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('done')
