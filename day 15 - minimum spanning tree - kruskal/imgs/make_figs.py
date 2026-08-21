"""Figures for day 15 - minimum spanning tree, Kruskal."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from diagram_style import *   # noqa

POS = {'A': (0.9, 4.5), 'B': (3.7, 5.3), 'C': (6.8, 4.6),
       'D': (2.3, 2.7), 'E': (5.3, 2.7), 'F': (2.4, 0.7), 'G': (7.2, 1.3)}
EDGES = [('A', 'B', 7), ('A', 'D', 5), ('B', 'C', 8), ('B', 'D', 9), ('B', 'E', 7),
         ('C', 'E', 5), ('D', 'E', 15), ('D', 'F', 6), ('E', 'F', 8),
         ('E', 'G', 9), ('F', 'G', 11)]
MST = {('A', 'D'), ('C', 'E'), ('D', 'F'), ('A', 'B'), ('B', 'E'), ('E', 'G')}


def draw_graph(ax, highlight=(), dim=(), r=0.42):
    for u, v, w in EDGES:
        key = (u, v)
        if key in highlight:
            col, lw, wc = AMBER, 3.0, AMBER
        elif key in dim:
            col, lw, wc = GREY_L, 1.2, GREY_L
        else:
            col, lw, wc = TEAL, 1.6, TEAL_L
        edge(ax, POS[u], POS[v], color=col, lw=lw, directed=False,
             weight=w, wcolor=wc, r=r)
    for n, (x, y) in POS.items():
        node(ax, x, y, n, r=r)


# ---------------------------------------------------------------- figure 1
fig, ax = canvas(9, 6.4, xlim=(0, 8.4), ylim=(0, 6.2))
draw_graph(ax, highlight=MST)
text(ax, 4.2, 6.0, 'total weight 39', color=AMBER, size=17)
legend(ax, 0.1, 1.4, [(AMBER, 'in the MST'), (TEAL, 'not needed')])
save(fig, 'day15_1.png')

# ---------------------------------------------------------------- figure 2
fig, ax = canvas(9, 4.2, xlim=(0, 11.6), ylim=(0, 4.0))
SCAN = [('A', 'D', 5, 'take'), ('C', 'E', 5, 'take'), ('D', 'F', 6, 'take'),
        ('A', 'B', 7, 'take'), ('B', 'E', 7, 'take'), ('B', 'C', 8, 'skip'),
        ('E', 'F', 8, 'skip'), ('B', 'D', 9, 'skip'), ('E', 'G', 9, 'take')]
x = 0.35
for u, v, w, what in SCAN:
    col = AMBER if what == 'take' else RED
    box(ax, x, 2.0, 1.05, 0.95, label='%s%s' % (u, v), edge=col,
        lw=2.4 if what == 'take' else 1.6)
    text(ax, x + 0.52, 1.62, str(w), color=TEAL_L, size=14)
    if what == 'skip':
        cross(ax, x + 0.52, 3.30, r=0.24)
    x += 1.25
arrow(ax, (0.35, 0.85), (11.25, 0.85), color=TEAL, lw=1.6)
text(ax, 5.8, 0.42, 'edges sorted by weight, scanned left to right',
     color=TEAL, size=15)
text(ax, 5.8, 3.85, 'a cross means both ends are already connected',
     color=RED, size=15)
save(fig, 'day15_2.png')

# ---------------------------------------------------------------- figure 3
fig, ax = canvas(9, 6.8, xlim=(0, 8.4), ylim=(0, 6.6))
region(ax, 0.15, 0.05, 4.35, 6.05)
text(ax, 1.15, 1.35, 'one side of the cut', color=TEAL_L, size=15)
CROSS = {('B', 'C'), ('B', 'E'), ('D', 'E'), ('E', 'F'), ('F', 'G')}
for u, v, w in EDGES:
    key = (u, v)
    if key == ('B', 'E'):
        col, lw = AMBER, 3.0
    elif key in CROSS:
        col, lw = TEAL_L, 2.0
    else:
        col, lw = GREY_L, 1.2
    edge(ax, POS[u], POS[v], color=col, lw=lw, directed=False,
         weight=w, wcolor=col, r=0.42)
for n, (x, y) in POS.items():
    node(ax, x, y, n)
text(ax, 4.2, 6.42, 'the cheapest edge crossing a cut is always safe',
     color=AMBER, size=16)
legend(ax, 5.75, 6.08, [(AMBER, 'cheapest crossing'), (TEAL_L, 'crosses the cut'),
                       (GREY_L, 'stays on one side')])
save(fig, 'day15_3.png')
print('done')
