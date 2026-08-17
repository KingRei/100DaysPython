"""Figures for day 14 - Union-Find (disjoint set union).

Run from this directory:  python3 make_figs.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from diagram_style import *


# ---------------------------------------------------------------- figure 1
# the forest and the parent array are the same thing
def fig1():
    fig, ax = canvas(9, 6.0, xlim=(0, 9), ylim=(0, 6))

    text(ax, 2.3, 5.55, 'a forest of disjoint sets', color=TEAL_L, size=16)

    # tree rooted at 0
    p0 = node(ax, 1.5, 4.55, '0', edge=AMBER, lw=2.4)
    p1 = node(ax, 0.7, 3.30, '1')
    p2 = node(ax, 2.3, 3.30, '2')
    p3 = node(ax, 2.3, 2.05, '3')
    for a, b in [(p1, p0), (p2, p0), (p3, p2)]:
        edge(ax, a, b, color=TEAL)

    # tree rooted at 4
    p4 = node(ax, 4.9, 4.55, '4', edge=AMBER, lw=2.4)
    p5 = node(ax, 4.9, 3.30, '5')
    edge(ax, p5, p4, color=TEAL)

    # singleton
    p6 = node(ax, 6.3, 4.55, '6', edge=AMBER, lw=2.4)

    text(ax, 0.78, 1.95, 'set {0,1,2,3}', color=TEAL, size=13, glowing=False)
    text(ax, 4.9, 2.35, 'set {4,5}', color=TEAL, size=13, glowing=False)
    text(ax, 6.3, 3.75, 'set {6}', color=TEAL, size=13, glowing=False)
    text(ax, 7.9, 5.05, 'amber ring\n= root\n(the set id)', color=AMBER, size=12,
         glowing=False)

    # the array
    text(ax, 4.5, 1.05, 'parent[ ]  - the whole data structure', color=TEAL_L, size=15)
    idx = ['0', '1', '2', '3', '4', '5', '6']
    par = ['0', '0', '0', '2', '4', '4', '6']
    x0 = 1.9
    cells(ax, x0, 0.05, 7, w=0.80, h=0.60, labels=par, size=14)
    for i, s in enumerate(idx):
        text(ax, x0 + i * 0.80 + 0.40, -0.35, s, color=GREY_L, size=12, glowing=False)
    text(ax, x0 - 0.30, 0.35, 'value', color=TEAL, size=12, ha='right', glowing=False)
    text(ax, x0 - 0.30, -0.35, 'index', color=GREY_L, size=12, ha='right', glowing=False)
    text(ax, 8.3, 0.35, 'parent[i] == i\nmeans i is a root', color=AMBER, size=12,
         ha='center', glowing=False)

    save(fig, 'day14_1.png')


# ---------------------------------------------------------------- figure 2
# union by rank: hang the shorter tree under the taller one
def fig2():
    fig, ax = canvas(9.4, 5.4, xlim=(0, 9.4), ylim=(0, 5.4))

    text(ax, 2.1, 5.0, 'union by rank / size', color=TEAL_L, size=16)
    text(ax, 7.2, 5.0, 'the other way round', color=RED, size=16)

    # left: short tree hangs under the tall one
    a = node(ax, 1.5, 4.05, 'a', edge=AMBER, lw=2.4)
    b = node(ax, 0.75, 2.90, 'b')
    c = node(ax, 2.25, 2.90, 'c')
    d = node(ax, 2.25, 1.75, 'd')
    for u, v in [(b, a), (c, a), (d, c)]:
        edge(ax, u, v, color=TEAL)
    x = node(ax, 3.7, 2.90, 'x')
    y = node(ax, 3.7, 1.75, 'y')
    edge(ax, y, x, color=TEAL)
    edge(ax, x, a, color=AMBER, lw=2.2)
    text(ax, 2.15, 0.85, 'height stays 3', color=TEAL, size=13, glowing=False)
    text(ax, 2.15, 0.35, 'rank 3 wins, rank 2 attaches', color=TEAL, size=12,
         glowing=False)

    # right: tall tree hangs under the short one
    x2 = node(ax, 7.35, 4.05, 'x', edge=AMBER, lw=2.4)
    y2 = node(ax, 6.55, 2.90, 'y')
    a2 = node(ax, 8.15, 2.90, 'a')
    b2 = node(ax, 7.45, 1.75, 'b')
    c2 = node(ax, 8.85, 1.75, 'c')
    d2 = node(ax, 8.85, 0.65, 'd')
    for u, v in [(y2, x2), (b2, a2), (c2, a2), (d2, c2)]:
        edge(ax, u, v, color=TEAL)
    edge(ax, a2, x2, color=RED, lw=2.2)
    cross(ax, 5.55, 2.60)
    text(ax, 7.2, -0.10, 'height grows to 4 - every later find pays for it',
         color=RED, size=12, glowing=False)

    save(fig, 'day14_2.png')


# ---------------------------------------------------------------- figure 3
# path compression flattens the chain
def fig3():
    fig, ax = canvas(9.4, 4.6, xlim=(0, 9.4), ylim=(0, 4.6))

    text(ax, 2.0, 4.25, 'before  find(3)', color=TEAL_L, size=16)
    text(ax, 7.0, 4.25, 'after  find(3)', color=TEAL_L, size=16)

    # chain 0 <- 1 <- 2 <- 3
    r = node(ax, 2.0, 3.30, '0', r=0.36, edge=AMBER, lw=2.4)
    n1 = node(ax, 2.0, 2.35, '1', r=0.36)
    n2 = node(ax, 2.0, 1.40, '2', r=0.36)
    n3 = node(ax, 2.0, 0.45, '3', r=0.36)
    for u, v in [(n1, r), (n2, n1), (n3, n2)]:
        edge(ax, u, v, color=TEAL, r=0.36)
    text(ax, 3.35, 1.90, 'walk up:\n3 - 2 - 1 - 0', color=AMBER, size=13,
         ha='left', glowing=False)

    arrow(ax, (4.45, 1.90), (5.15, 1.90), color=TEAL_L, lw=2.0)

    # flattened
    r2 = node(ax, 7.0, 3.30, '0', edge=AMBER, lw=2.4)
    a2 = node(ax, 5.9, 1.70, '1')
    b2 = node(ax, 7.0, 1.70, '2')
    c2 = node(ax, 8.1, 1.70, '3')
    for u in (a2, b2, c2):
        edge(ax, u, r2, color=AMBER, lw=2.0)
    text(ax, 7.0, 0.55, 'every node on the path now points\nstraight at the root',
         color=TEAL, size=13, glowing=False)
    text(ax, 7.0, -0.15, 'the next find is O(1)', color=AMBER, size=13, glowing=False)

    save(fig, 'day14_3.png')


# ---------------------------------------------------------------- figure 4
# LeetCode 547 - provinces
def fig4():
    fig, ax = canvas(9.4, 5.0, xlim=(0, 9.4), ylim=(0, 5.0))

    text(ax, 2.0, 4.65, 'isConnected', color=TEAL_L, size=16)
    M = [[1, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 0, 1]]
    x0, y0, w = 0.65, 3.55, 0.56
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            on = v == 1
            box(ax, x0 + j * w, y0 - i * w, w, w, label=str(v),
                fill=NAVY if on else GREY, edge=AMBER if on else TEAL,
                lw=1.8 if on else 1.0, size=12)
    for j in range(5):
        text(ax, x0 + j * w + w / 2, y0 + w + 0.28, str(j), color=GREY_L, size=11,
             glowing=False)
        text(ax, x0 - 0.22, y0 - j * w + w / 2, str(j), color=GREY_L, size=11,
             glowing=False)
    text(ax, 2.0, 0.20, 'only the upper triangle matters -\nthe matrix is symmetric',
         color=TEAL, size=12, glowing=False)

    arrow(ax, (4.15, 2.30), (5.05, 2.30), color=TEAL_L, lw=2.0)

    # the three components
    region(ax, 5.35, 2.75, 1.85, 1.45, label='province 1')
    p0 = node(ax, 5.85, 3.45, '0', r=0.34, size=12)
    p1 = node(ax, 6.70, 3.45, '1', r=0.34, size=12)
    edge(ax, p0, p1, color=AMBER, directed=False, r=0.34)

    region(ax, 7.45, 2.75, 1.85, 1.45, label='province 2')
    p2 = node(ax, 7.95, 3.45, '2', r=0.34, size=12)
    p3 = node(ax, 8.80, 3.45, '3', r=0.34, size=12)
    edge(ax, p2, p3, color=AMBER, directed=False, r=0.34)

    region(ax, 6.40, 0.78, 1.85, 1.30, label='province 3')
    node(ax, 7.32, 1.40, '4', r=0.34, size=12)

    text(ax, 7.32, 0.18, 'answer: 3 provinces', color=AMBER, size=15)

    save(fig, 'day14_4.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('wrote day14_1..4.png')
