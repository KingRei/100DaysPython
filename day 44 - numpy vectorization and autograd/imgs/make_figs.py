"""Day 44 figures.  Every number is computed by importing vectorize_autograd.py."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                # noqa: E402
from matplotlib.patches import Rectangle          # noqa: E402
from diagram_style import *                       # noqa: F401,F403,E402
import vectorize_autograd as va                   # noqa: E402

os.chdir(HERE)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def cell(ax, x, y, w, h, label, edge_=TEAL, lw=1.4, fill=NAVY, color=WHITE, size=12, z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge_, linewidth=lw, zorder=z))
    if label is not None:
        text(ax, x + w / 2, y + h / 2, str(label), color=color, size=size, glowing=False, zorder=z + 2)


# ---------------------------------------------------------------- figure 1
def fig1():
    """A list of pointers to boxed floats vs one buffer of raw doubles."""
    fig, ax = W(5.0)
    vals = [0.31, 0.77, 0.12, 0.95, 0.48]
    text(ax, 0.2, 5.1, 'Python list', color=TEAL_L, size=14, ha='left')
    for i, v in enumerate(vals):
        x = 0.3 + i * 0.9
        cell(ax, x, 4.0, 0.8, 0.6, '•', size=14)
        bx, by = 0.15 + i * 0.95, 2.75
        cell(ax, bx, by, 0.85, 0.75, f'float\n{v}', fill=GREY, size=10)
        arrow(ax, (x + 0.4, 4.0), (bx + 0.42, by + 0.75), lw=1.2)
    text(ax, 2.5, 2.35, 'each element: a pointer to a separate object\nwith its own type tag and reference count',
         color=TEAL, size=11, glowing=False)
    text(ax, 5.6, 5.1, 'NumPy array', color=TEAL_L, size=14, ha='left')
    for i, v in enumerate(vals):
        cell(ax, 5.6 + i * 0.8, 4.0, 0.8, 0.6, v, size=11)
    text(ax, 7.6, 3.6, 'one buffer, 8 bytes per value, one dtype', color=TEAL, size=11, glowing=False)
    a = np.zeros(1_000_000)
    text(ax, 7.6, 3.2, f'1,000,000 doubles = {a.nbytes:,} bytes', color=AMBER, size=11, glowing=False)
    box(ax, 0.3, 0.55, 4.2, 1.2, label='for x in xs: s += x * x\nthe interpreter runs every step\nand checks every type',
        fill=GREY, size=11)
    box(ax, 5.5, 0.55, 4.2, 1.2, label='np.dot(a, a)\none call; the loop runs in C\nover raw doubles', size=11)
    text(ax, 5.0, 1.15, 'vs', color=TEAL_L, size=14)
    save(fig, 'day44_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """One buffer, three views: shape and strides decide what you see."""
    fig, ax = W(5.6)
    m = np.arange(12).reshape(3, 4)
    text(ax, 0.2, 5.85, 'the buffer (int64, 8 bytes each)', color=TEAL_L, size=13, ha='left', glowing=False)
    for i in range(12):
        cell(ax, 0.2 + i * 0.8, 5.0, 0.75, 0.55, i, size=11)
    for name, v, x0 in (('m', m, 0.2), ('m.T', m.T, 4.2), ('m[:, ::2]', m[:, ::2], 7.6)):
        lay = va.layout(v, m)
        rows, cols = v.shape
        text(ax, x0, 4.35, name, color=TEAL_L, size=13, ha='left')
        for r in range(rows):
            for c in range(cols):
                cell(ax, x0 + c * 0.55, 3.75 - r * 0.5, 0.5, 0.45, v[r, c], size=10)
        text(ax, x0, 3.75 - rows * 0.5 + 0.05, f'shape {lay["shape"]}', color=TEAL, size=11, ha='left', glowing=False)
        text(ax, x0, 3.75 - rows * 0.5 - 0.35, f'strides {lay["strides"]}', color=AMBER, size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.35, 'all three share the same 12 numbers: transposing or slicing only changes the strides',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day44_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """pred of shape (5, 1) minus y of shape (5,) is a 5 x 5 table."""
    fig, ax = W(5.4)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = (y + 0.1).reshape(-1, 1)
    sq = (pred - y) ** 2
    x0, y0, s = 2.2, 0.9, 0.72
    for j in range(5):
        text(ax, x0 + j * s + s / 2, y0 + 5 * s + 0.3, f'{y[j]:.0f}', color=TEAL_L, size=12)
    text(ax, x0 + 2.5 * s, y0 + 5 * s + 0.75, 'y  (shape (5,))', color=TEAL_L, size=12, glowing=False)
    for i in range(5):
        text(ax, x0 - 0.3, y0 + (4 - i) * s + s / 2, f'{pred[i, 0]:.1f}', color=TEAL_L, size=12)
        for j in range(5):
            diag = i == j
            cell(ax, x0 + j * s, y0 + (4 - i) * s, s, s, f'{sq[i, j]:.2f}', size=10,
                 edge_=AMBER if diag else TEAL, lw=2.6 if diag else 1.2, z=6 if diag else 3)
    text(ax, x0 - 1.25, y0 + 2.5 * s, 'pred\n(5, 1)', color=TEAL_L, size=12, glowing=False)
    x = 6.6
    text(ax, x, 4.2, 'intended: the 5 amber cells', color=AMBER, size=12, ha='left', glowing=False)
    text(ax, x, 3.8, f'MSE = {va.mse(pred.ravel(), y):.4f}', color=AMBER, size=12, ha='left', glowing=False)
    text(ax, x, 2.9, 'computed: all 25 cells', color=RED, size=12, ha='left', glowing=False)
    text(ax, x, 2.5, f'MSE = {va.mse(pred, y):.4f}', color=RED, size=12, ha='left', glowing=False)
    text(ax, x, 1.6, 'no error, no warning', color=TEAL, size=12, ha='left', glowing=False)
    save(fig, 'day44_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Error of the difference quotient against the step h."""
    fig, ax = W(5.4)
    X0, X1, Y0, Y1 = 1.4, 9.0, 0.8, 5.4
    hs = [10.0 ** -(k / 2) for k in range(2, 27)]
    fe = [abs(va.forward_diff(math.sin, 1.0, h) - math.cos(1)) for h in hs]
    ce = [max(abs(va.central_diff(math.sin, 1.0, h) - math.cos(1)), 1e-17) for h in hs]
    lx = lambda h: X0 + (X1 - X0) * (-math.log10(h) - 1) / 12
    ly = lambda e: Y0 + (Y1 - Y0) * (math.log10(e) + 13) / 12
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for k in (1, 3, 5, 7, 9, 11, 13):
        ax.plot([lx(10.0 ** -k)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, lx(10.0 ** -k), Y0 - 0.32, f'1e-{k}', color=TEAL, size=11, glowing=False)
    for k in (-13, -10, -7, -4, -1):
        ax.plot([X0, X0 - 0.1], [ly(10.0 ** k)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.18, ly(10.0 ** k), f'1e{k}', color=TEAL, size=11, ha='right', glowing=False)
    text(ax, (X0 + X1) / 2, 0.1, 'step h (smaller to the right)', color=TEAL, size=12, glowing=False)
    text(ax, 0.1, Y1 + 0.35, 'error', color=TEAL, size=12, ha='left', glowing=False)
    ok = [(h, e) for h, e in zip(hs, fe) if e > 1e-13]
    ax.plot([lx(h) for h, _ in ok], [ly(e) for _, e in ok], color=GREY_L, lw=2.0, marker='o', ms=3)
    ok = [(h, e) for h, e in zip(hs, ce) if e > 1e-13]
    ax.plot([lx(h) for h, _ in ok], [ly(e) for _, e in ok], color=TEAL_L, lw=2.0, marker='o', ms=3)
    best = min(zip(ce, hs))
    text(ax, lx(best[1]) + 0.25, ly(best[0]) + 0.05, f'best central: h = {best[1]:.0e}, error {best[0]:.0e}',
         color=AMBER, size=11, ha='left', glowing=False)
    for k, (lab, col) in enumerate((('forward difference', GREY_L), ('central difference', TEAL_L))):
        yy = ly(10.0 ** -1.4) - k * 0.4
        ax.plot([lx(10.0 ** -4.6), lx(10.0 ** -5.2)], [yy, yy], color=col, lw=2.0)
        text(ax, lx(10.0 ** -5.4), yy, lab, color=col, size=11, ha='left', glowing=False)
    text(ax, lx(1e-11), ly(1e-4) + 0.6, 'too small: rounding\nin f(x+h) - f(x-h) takes over', color=RED,
         size=11, glowing=False)
    save(fig, 'day44_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """L = (a*b + c) * f: values forward, gradients backward."""
    fig, ax = W(5.6)
    res, order = va.micrograd_example()
    pos = {'a': (1.0, 4.6), 'b': (1.0, 2.8), 'c': (3.6, 1.45), 'e': (3.6, 3.75), 'd': (6.0, 2.65),
           'f': (6.0, 1.0), 'L': (8.6, 1.85)}
    ops = {'e': '*', 'd': '+', 'L': '*'}
    edges = [('a', 'e'), ('b', 'e'), ('e', 'd'), ('c', 'd'), ('d', 'L'), ('f', 'L')]
    for u, v in edges:
        edge(ax, pos[u], pos[v], r=0.5)
    for k, (x, y) in pos.items():
        node(ax, x, y, k, r=0.45, size=14)
        if k in ops:
            text(ax, x, y + 0.72, ops[k], color=TEAL_L, size=14)
        val, g = res[k]
        text(ax, x, y - 0.66, f'{k} = {val:g}', color=TEAL_L, size=11, glowing=False)
        text(ax, x, y - 0.98, f'grad {g:g}', color=AMBER, size=11, glowing=False)
    text(ax, 5.0, 6.0, 'teal: value (forward)    amber: dL/d(node) (backward, reverse topological order)',
         color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 5.6, 'topological order: ' + ' '.join(order) + '   ->   backward walks it from L to the leaves',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day44_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Two autograd bugs that still return a number."""
    fig, ax = W(5.0)
    x = va.Value(3.0)
    (x * x).backward()
    xa = va.AssignValue(3.0)
    (xa * xa).backward()
    text(ax, 2.4, 5.2, 'y = x * x at x = 3', color=TEAL_L, size=13, glowing=False)
    node(ax, 2.4, 1.6, 'x', r=0.45)
    node(ax, 2.4, 3.9, '*', r=0.45)
    arrow(ax, (2.1, 3.5), (2.1, 2.0), color=AMBER, rad=0.3, shrinkA=6, shrinkB=6)
    arrow(ax, (2.7, 3.5), (2.7, 2.0), color=AMBER, rad=-0.3, shrinkA=6, shrinkB=6)
    text(ax, 1.15, 2.75, '+3', color=AMBER, size=13)
    text(ax, 3.65, 2.75, '+3', color=AMBER, size=13)
    text(ax, 2.4, 0.8, f'grad += : {x.grad:g}', color=TEAL_L, size=12, glowing=False)
    text(ax, 2.4, 0.35, f'grad =  : {xa.grad:g}  (second write wins)', color=RED, size=12, glowing=False)
    x = va.Value(3.0)
    q = x * 2
    nodes = len((q * q).backward())
    good = x.grad
    x = va.Value(3.0)
    q = x * 2
    calls = (q * q).backward_naive()
    text(ax, 7.4, 5.2, 'y = (2x)^2 at x = 3', color=TEAL_L, size=13, glowing=False)
    node(ax, 7.4, 4.1, 'y', r=0.42)
    node(ax, 6.4, 2.75, 'q', r=0.42)
    node(ax, 8.4, 2.75, 'q', r=0.42)
    node(ax, 6.4, 1.45, 'x', r=0.42)
    node(ax, 8.4, 1.45, 'x', r=0.42, edge=RED)
    for p1, p2 in (((7.4, 4.1), (6.4, 2.75)), ((7.4, 4.1), (8.4, 2.75)), ((6.4, 2.75), (6.4, 1.45)), ((8.4, 2.75), (8.4, 1.45))):
        edge(ax, p1, p2, r=0.42)
    text(ax, 7.4, 2.1, 'naive recursion\nvisits q twice', color=RED, size=11, glowing=False)
    text(ax, 7.4, 0.8, f'topological order: {good:g}  ({nodes} node visits)', color=TEAL_L, size=12, glowing=False)
    text(ax, 7.4, 0.35, f'naive: {x.grad:g}  ({calls} visits, counted twice)', color=RED, size=12, glowing=False)
    save(fig, 'day44_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print('figures written')
