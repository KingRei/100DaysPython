"""Figures for Day 18 - Tarjan: SCCs, bridges, articulation points.

All numbers are taken from tarjan.py so the pictures cannot drift from the code.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: F401,F403
import tarjan as T

# ---------------------------------------------------------------- shared layout
UP = {0: (1.1, 4.5), 1: (1.1, 1.7), 2: (3.3, 3.1),
      3: (5.5, 4.5), 4: (5.5, 1.7),
      5: (8.1, 1.7), 6: (10.1, 3.1), 7: (10.1, 0.4)}
TREE = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
BACK = [(2, 0), (4, 2), (7, 5)]
BRIDGE = (4, 5)

DP = {0: (1.0, 3.0), 1: (3.4, 4.6), 2: (6.0, 5.0), 3: (4.9, 3.2),
      4: (3.6, 1.2), 5: (6.0, 1.2)}


def stamp(ax, p, d, l, dy=-0.72, color=AMBER, size=12):
    text(ax, p[0], p[1] + dy, '%d / %d' % (d, l), color=color, size=size,
         glowing=False)


# ------------------------------------------------------------------- figure 1
def fig1():
    bridges, disc, low = T.find_bridges(T.UN_N, T.UN_EDGES)
    fig, ax = canvas(15.0, 6.9, xlim=(0, 17.6), ylim=(-1.3, 6.4))

    text(ax, 5.6, 6.0, 'one DFS, two numbers per vertex', color=TEAL_L, size=17)

    for u, v in TREE:
        col, lw = (RED, 2.6) if (u, v) == BRIDGE else (TEAL, 1.8)
        edge(ax, UP[u], UP[v], directed=False, color=col, lw=lw)
    for u, v in BACK:
        arrow(ax, UP[u], UP[v], color=AMBER, lw=1.5, ls='--', style='-',
              rad=0.22, shrinkA=18, shrinkB=18)
    for i, p in UP.items():
        node(ax, p[0], p[1], str(i))
        stamp(ax, p, disc[i], low[i])

    text(ax, 6.8, 2.25, 'bridge', color=RED, size=15)

    legend(ax, 0.4, -0.15, [(TEAL, 'tree edge, in DFS order'),
                          (AMBER, 'back edge to an ancestor'),
                          (RED, 'bridge:  low[v] > disc[u]')])

    box(ax, 11.5, 1.4, 5.9, 4.4, fill=NAVY)
    text(ax, 11.9, 5.35, 'disc / low', color=TEAL_L, size=16, ha='left')
    for i, s in enumerate([
            'disc: when the DFS first arrived',
            'low : the smallest disc this subtree',
            '      can reach with tree edges plus',
            '      at most one back edge',
            '',
            'low[4] = 2, because 4 has a back edge',
            'to 2. So 4 can escape upwards without',
            'using the tree edge above it.',
            'low[5] = 5: nothing under 5 escapes,',
            'so the edge 4-5 is the only way in.']):
        text(ax, 11.9, 4.85 - i * 0.36, s, color=TEAL, size=13, ha='left',
             glowing=False)

    text(ax, 11.5, 0.95, 'the label under each vertex is  disc / low',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 11.5, 0.45, 'bridges found: ' + str(bridges), color=TEAL,
         size=13, ha='left', glowing=False)
    save(fig, os.path.join(HERE, 'day18_1.png'))


# ------------------------------------------------------------------- figure 2
def fig2():
    fig, ax = canvas(15.0, 6.6, xlim=(0, 16.2), ylim=(-0.9, 6.6))
    text(ax, 8.1, 6.3, 'the same DFS, three readings of low', color=TEAL_L, size=17)

    panels = [
        (2.6, 'low[v] < disc[u]', TEAL,
         ['the subtree escapes above u,', 'so u is not needed at all',
          'not a bridge, not a cut vertex'], 'above'),
        (8.1, 'low[v] = disc[u]', AMBER,
         ['the subtree reaches u itself,', 'but no higher',
          'not a bridge - u IS a cut vertex'], 'onto'),
        (13.6, 'low[v] > disc[u]', RED,
         ['nothing under v gets back up', 'except through the edge itself',
          'bridge, and u is a cut vertex'], None),
    ]

    for cx, title, col, lines, back in panels:
        text(ax, cx, 5.7, title, color=col, size=16)
        a = (cx + 1.5, 4.8)
        u = (cx, 3.7)
        v = (cx, 2.3)
        node(ax, a[0], a[1], 'a', r=0.36, size=13)
        node(ax, u[0], u[1], 'u')
        node(ax, v[0], v[1], 'v')
        edge(ax, a, u, directed=False)
        edge(ax, u, v, directed=False,
             color=RED if back is None else TEAL,
             lw=2.6 if back is None else 1.8)
        region(ax, cx - 1.15, 0.55, 2.3, 1.35, fill=PALE, alpha=0.35)
        text(ax, cx, 1.2, 'subtree of v', color=TEAL, size=13, glowing=False)
        if back == 'above':
            arrow(ax, (cx + 1.05, 1.95), a, color=AMBER, lw=1.5, ls='--',
                  rad=0.38, shrinkA=4, shrinkB=14)
        elif back == 'onto':
            arrow(ax, (cx + 1.05, 1.95), u, color=AMBER, lw=1.5, ls='--',
                  rad=0.34, shrinkA=4, shrinkB=14)
        for i, s in enumerate(lines):
            text(ax, cx, 0.15 - i * 0.4, s,
                 color=col if i == 2 else TEAL, size=13, glowing=False)

    save(fig, os.path.join(HERE, 'day18_2.png'))


# ------------------------------------------------------------------- figure 3
def draw_directed(ax, note52=True, extra_edge_color=TEAL):
    for u, v in T.DI_EDGES:
        if (u, v) == (5, 2):
            arrow(ax, DP[5], DP[2], color=extra_edge_color, lw=1.8, rad=0.5,
                  shrinkA=14, shrinkB=14)
        elif (u, v) == (3, 1):
            edge(ax, DP[3], DP[1], rad=0.0)
        else:
            edge(ax, DP[u], DP[v])
    for i, p in DP.items():
        node(ax, p[0], p[1], str(i))


def fig3():
    comps, disc, low = T.tarjan_scc(T.DI_N, T.DI_EDGES)
    fig, ax = canvas(15.0, 6.6, xlim=(0, 17.0), ylim=(-0.9, 6.6))
    text(ax, 4.0, 6.3, 'directed: low[u] = disc[u] means "root of a component"',
         color=TEAL_L, size=16)

    region(ax, 2.7, 2.2, 4.1, 3.4, fill=PALE, alpha=0.30)
    region(ax, 2.9, 0.2, 3.8, 1.7, fill=PALE, alpha=0.30)
    draw_directed(ax)
    for i, p in DP.items():
        stamp(ax, p, disc[i], low[i], dy=-0.74)
    text(ax, 4.75, 5.75, '{1, 2, 3}', color=TEAL_L, size=14)
    text(ax, 4.8, -0.25, '{4, 5}', color=TEAL_L, size=14)
    text(ax, 1.0, 2.0, '{0}', color=TEAL_L, size=14)
    text(ax, 7.6, 3.2, '5 -> 2 is ignored:', color=AMBER, size=13,
         ha='left', glowing=False)
    text(ax, 7.6, 2.8, '2 left the stack long ago', color=AMBER, size=13,
         ha='left', glowing=False)

    box(ax, 10.8, 1.35, 6.0, 4.5, fill=NAVY)
    text(ax, 11.2, 5.35, 'the stack', color=TEAL_L, size=16, ha='left')
    for i, s in enumerate([
            'every visited vertex waits on a stack',
            'until its component is decided.',
            '',
            'low[1] = disc[1] = 1  ->  pop 3, 2, 1',
            'low[4] = disc[4] = 4  ->  pop 5, 4',
            'low[0] = disc[0] = 0  ->  pop 0',
            '',
            'a vertex may only lower its low against',
            'a vertex still on the stack - that is what',
            '"is my ancestor" means here.']):
        text(ax, 11.2, 4.85 - i * 0.36, s, color=TEAL, size=13, ha='left',
             glowing=False)
    text(ax, 10.8, 1.0, 'components, in the order Tarjan emits them:',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, 10.8, 0.5, str(comps), color=AMBER, size=13, ha='left',
         glowing=False)
    save(fig, os.path.join(HERE, 'day18_3.png'))


# ------------------------------------------------------------------- figure 4
def fig4():
    comps, _, _ = T.tarjan_scc(T.DI_N, T.DI_EDGES)
    bad = T.tarjan_scc_without_onstack(T.DI_N, T.DI_EDGES)
    fig, ax = canvas(15.0, 6.4, xlim=(0, 17.0), ylim=(-0.7, 6.6))

    text(ax, 4.0, 6.3, 'drop the on-stack test and components merge',
         color=RED, size=16)
    draw_directed(ax, extra_edge_color=RED)
    for i in (0, 4, 5):
        ax.add_patch(Circle(DP[i], 0.66, facecolor='none', edgecolor=RED,
                            linewidth=1.8, linestyle='--', zorder=6))
    text(ax, 4.0, 0.05, 'these three end up in one component: ' + str(bad[1]),
         color=RED, size=14, glowing=False)
    text(ax, 7.1, 2.4, 'disc[2] = 2 is smaller than disc[5],', color=RED, size=13,
         ha='left', glowing=False)
    text(ax, 7.1, 2.0, 'so low[5] and low[4] both drop to 2', color=RED, size=13,
         ha='left', glowing=False)
    text(ax, 7.1, 1.6, 'and 4 never becomes a root', color=RED, size=13,
         ha='left', glowing=False)

    # condensation
    text(ax, 13.0, 5.9, 'condensation', color=TEAL_L, size=16)
    b0 = box(ax, 11.9, 4.5, 2.3, 0.9, label='{0}')
    b1 = box(ax, 11.9, 2.9, 2.3, 0.9, label='{4, 5}')
    b2 = box(ax, 11.9, 1.3, 2.3, 0.9, label='{1, 2, 3}')
    arrow(ax, (13.05, 4.45), (13.05, 3.85), color=TEAL)
    arrow(ax, (13.05, 2.85), (13.05, 2.25), color=TEAL)
    arrow(ax, (14.3, 4.9), (14.3, 1.8), color=TEAL, rad=-0.55)
    for i, s in enumerate([
            'collapsing each component always',
            'leaves a DAG - a cycle between two',
            'components would have merged them.',
            'Tarjan emits them bottom-up, so the',
            'reversed list is a topological order.']):
        text(ax, 10.4, 0.95 - i * 0.38, s, color=TEAL, size=13, ha='left',
             glowing=False)
    save(fig, os.path.join(HERE, 'day18_4.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('wrote day18_1..4.png')
