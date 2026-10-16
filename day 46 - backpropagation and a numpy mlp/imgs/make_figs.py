"""Day 46 figures.  Every number is computed by importing mlp.py."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.colors import ListedColormap         # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import mlp                                           # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)
CLASS_COLS = [TEAL_L, AMBER, '#ff5c5c']


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


# ---------------------------------------------------------------- figure 1
def fig1():
    """One forward pass, one backward pass, and the three rules in between."""
    fig, ax = W(5.0)
    net = mlp.MLP([2, 64, 3], np.random.default_rng(0))
    shapes = [tuple(p.shape) for p in net.params]
    cols = [(0.85, 'x\n(n, 2)'), (3.0, 'h = relu(z1)\n(n, 64)'), (5.15, 'z2\n(n, 3)'),
            (7.3, 'p = softmax\n(n, 3)'), (9.15, 'loss')]
    for x, lab in cols:
        box(ax, x - 0.75, 3.1, 1.5, 0.95, label=lab, size=11)
    for i in range(len(cols) - 1):
        arrow(ax, (cols[i][0] + 0.79, 3.78), (cols[i + 1][0] - 0.79, 3.78), lw=1.8)
        arrow(ax, (cols[i + 1][0] - 0.79, 3.0), (cols[i][0] + 0.79, 3.0), lw=1.8, color=AMBER)
    text(ax, 1.93, 4.08, f'W1 {shapes[0]}', color=TEAL, size=10, glowing=False)
    text(ax, 4.08, 4.08, f'W2 {shapes[2]}', color=TEAL, size=10, glowing=False)
    text(ax, 5.0, 4.95, 'forward: cache what the backward pass will need',
         color=TEAL_L, size=12, glowing=False)
    text(ax, 5.0, 2.35, 'backward: walk the same list in reverse', color=AMBER, size=12, glowing=False)
    text(ax, 8.25, 2.7, 'dz2 = (p − y)/n', color=AMBER, size=11, glowing=False)
    box(ax, 0.4, 0.45, 9.2, 1.45, label='', fill='none', edge=AMBER, lw=1.6)
    for i, (t, note) in enumerate((('dW = xᵀ @ dz', 'a transpose away'),
                                   ('db = dz.sum(axis=0)', 'undo the broadcast'),
                                   ('dx = dz @ Wᵀ', 'hand to the layer below'))):
        text(ax, 1.9 + i * 3.1, 1.48, t, color=AMBER, size=13, glowing=False)
        text(ax, 1.9 + i * 3.1, 0.95, note, color=TEAL, size=10, glowing=False)
    save(fig, 'day46_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """What the hidden layer buys: a curved boundary instead of straight lines."""
    X, y = mlp.spiral()
    net = mlp.MLP([2, 64, 3], np.random.default_rng(0))
    mlp.train(net, X, y)
    lin = mlp.MLP([2, 3], np.random.default_rng(0))
    mlp.train(lin, X, y)
    fig, ax = W(4.6)
    cmap = ListedColormap(['#0b3d44', '#3d3012', '#3d1414'])
    g = np.linspace(-1.25, 1.25, 220)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()]
    for k, (model, title, acc) in enumerate(
            ((net, 'one hidden layer, 64 ReLU units', mlp.accuracy(net, X, y)),
             (lin, 'no hidden layer (softmax regression)', mlp.accuracy(lin, X, y)))):
        x0 = 0.55 + k * 4.75
        pred = model.forward(grid).argmax(axis=1).reshape(gx.shape)
        ax.imshow(pred, extent=(x0, x0 + 3.9, 0.75, 4.65), origin='lower',
                  cmap=cmap, interpolation='nearest', zorder=1)
        sx = x0 + (X[:, 0] + 1.25) / 2.5 * 3.9
        sy = 0.75 + (X[:, 1] + 1.25) / 2.5 * 3.9
        for c in range(3):
            m = y == c
            ax.scatter(sx[m], sy[m], s=7, color=CLASS_COLS[c], zorder=3, linewidths=0)
        ax.add_patch(Rectangle((x0, 0.75), 3.9, 3.9, facecolor='none',
                               edgecolor=TEAL, linewidth=1.4, zorder=4))
        text(ax, x0 + 1.95, 4.95, title, color=TEAL_L, size=12, glowing=False)
        text(ax, x0 + 1.95, 0.42, f'training accuracy {acc:.3f}',
             color=AMBER if acc > 0.9 else RED, size=12, glowing=False)
    save(fig, 'day46_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """The gradient of softmax + cross entropy is the prediction minus the label."""
    logits = np.array([[2.0, 0.5, -1.0, 0.2]])
    label = 2
    p = mlp.softmax(logits)[0]
    y = np.zeros(4)
    y[label] = 1.0
    g = p - y
    num = mlp.numeric_grad(lambda: mlp.cross_entropy(logits, np.array([label])), logits)
    fig, ax = W(5.0)
    BASE, SC, BW = 3.15, 1.25, 0.46
    ax.plot([0.9, 9.3], [BASE, BASE], color=TEAL, lw=1.1, zorder=2)
    for i in range(4):
        gx = 1.5 + i * 2.0
        for k, (vals, col) in enumerate(((p, TEAL_L), (y, GREY_L), (g, AMBER))):
            x = gx + (k - 1) * (BW + 0.1) - BW / 2
            h = abs(vals[i]) * SC
            ax.add_patch(Rectangle((x, BASE if vals[i] >= 0 else BASE - h), BW, max(h, 0.012),
                                   facecolor=NAVY, edgecolor=col, linewidth=1.6, zorder=3))
            ty = BASE + h + 0.2 if vals[i] >= 0 else BASE - h - 0.22
            text(ax, x + BW / 2, ty, f'{vals[i]:+.2f}', color=col, size=10, glowing=False)
        text(ax, gx, BASE - 1.72, f'class {i}', color=TEAL_L if i == label else TEAL,
             size=11, glowing=False)
        if i == label:
            text(ax, gx, BASE - 2.05, 'the label', color=TEAL_L, size=10, glowing=False)
    for lx, col, lab in ((0.55, TEAL_L, 'p = softmax(logits)'), (4.05, GREY_L, 'y (one-hot label)'),
                         (7.35, AMBER, 'gradient  p − y')):
        legend(ax, lx, 4.95, [(col, lab)], size=11)
    text(ax, 5.0, 0.72, 'the softmax Jacobian and the log of the cross entropy cancel, leaving',
         color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 0.36, 'one subtraction - which is why the two are written as one function',
         color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 0.02, f'against central differences: largest difference '
                        f'{np.max(np.abs(g / 1.0 - num[0])):.1e}',
         color=AMBER, size=11, glowing=False)
    save(fig, 'day46_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Three stability traps, each returning a number that ends training."""
    rows = []
    big = np.array([[1000.0, 1001.0, 1002.0]])
    rows.append(('softmax([1000, 1001, 1002])', str(mlp.softmax_naive(big)[0][:1][0]),
                 f'{mlp.softmax(big)[0][0]:.6f}', 'exp(1002) = inf, then inf/inf'))
    lg = np.array([[-400.0, 0.0, 400.0]])
    rows.append(('cross entropy, true class 0', str(mlp.cross_entropy_naive(lg, np.array([0]))),
                 f'{mlp.cross_entropy(lg, np.array([0])):.1f}', 'p underflows to 0, then log(0)'))
    rows.append(('log(sigmoid(-800))', str(mlp.log_sigmoid_naive(np.array([-800.0]))[0]),
                 f'{mlp.log_sigmoid(np.array([-800.0]))[0]:.1f}', 'sigmoid rounds to 0, then log(0)'))
    fig, ax = W(3.6)
    text(ax, 2.3, 3.55, 'the textbook spelling', color=RED, size=12, glowing=False)
    text(ax, 5.3, 3.55, 'gives', color=RED, size=12, glowing=False)
    text(ax, 6.9, 3.55, 'the stable one', color=TEAL_L, size=12, glowing=False)
    for i, (what, bad, good, why) in enumerate(rows):
        y = 2.85 - i * 0.78
        text(ax, 0.3, y, what, color=TEAL_L, size=12, ha='left', glowing=False)
        text(ax, 5.3, y, bad, color=RED, size=13, glowing=False)
        text(ax, 6.9, y, good, color=TEAL_L, size=13, glowing=False)
        text(ax, 0.3, y - 0.3, why, color=TEAL, size=10, ha='left', glowing=False)
        ax.plot([0.25, 9.6], [y - 0.5, y - 0.5], color=TEAL, lw=0.7, alpha=0.4)
    text(ax, 5.0, 0.3, 'none of these raises: an inf or a nan propagates into every weight '
                       'on the next backward pass', color=AMBER, size=11, glowing=False)
    save(fig, 'day46_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """All-zero initialisation: 64 hidden units that stay the same unit forever."""
    X, y = mlp.spiral()
    zero = mlp.MLP([2, 64, 3], np.random.default_rng(0), zero=True)
    rand = mlp.MLP([2, 64, 3], np.random.default_rng(0))
    mlp.train(zero, X, y, steps=400)
    mlp.train(rand, X, y, steps=400)
    fig, ax = W(4.4)
    for k, (net, title) in enumerate(((zero, 'all weights start at 0'),
                                      (rand, 'random initialisation'))):
        h = mlp.hidden_activations(net, X[:6])
        x0 = 0.55 + k * 4.9
        text(ax, x0 + 1.95, 4.35, title, color=TEAL_L, size=12, glowing=False)
        hi = max(float(np.percentile(h, 92)), 1e-12)
        for r in range(6):
            for c in range(16):
                v = min(h[r, c] / hi, 1.0) ** 0.6
                ax.add_patch(Rectangle((x0 + c * 0.24, 3.75 - r * 0.26), 0.21, 0.23,
                                       facecolor=(0.02 + 0.05 * v, 0.14 + 0.76 * v,
                                                  0.22 + 0.65 * v),
                                       edgecolor='none', zorder=3))
        text(ax, x0 + 1.9, 2.0, 'first 16 of 64 hidden units, 6 examples',
             color=TEAL, size=10, glowing=False)
        uniq = len(np.unique(np.round(mlp.hidden_activations(net, X[:4]), 9)))
        text(ax, x0 + 1.9, 1.55, f'distinct activation values: {uniq}',
             color=RED if uniq < 5 else TEAL_L, size=12, glowing=False)
        text(ax, x0 + 1.9, 1.1, f'training accuracy {mlp.accuracy(net, X, y):.3f}',
             color=RED if uniq < 5 else TEAL_L, size=12, glowing=False)
    text(ax, 5.0, 0.5, 'with every weight equal, every hidden unit receives the same gradient,',
         color=AMBER, size=11, glowing=False)
    text(ax, 5.0, 0.12, 'so 64 units behave as one and no amount of training separates them',
         color=AMBER, size=11, glowing=False)
    save(fig, 'day46_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """A gradient check that fails while the code is right: the ReLU corner."""
    r_err, t_err, zeros, total = mlp.kink_demo()
    fig, ax = W(4.4)
    X0, Y0, S = 2.6, 1.5, 1.5
    xs = np.linspace(-1.3, 1.3, 200)
    ax.plot(X0 + xs * S, Y0 + np.maximum(xs, 0) * S, color=TEAL_L, lw=2.2, zorder=3)
    ax.plot([X0 - 1.5, X0 + 1.5], [Y0, Y0], color=TEAL, lw=1.0)
    ax.plot([X0, X0], [Y0 - 0.3, Y0 + 2.1], color=TEAL, lw=1.0)
    text(ax, X0 + 1.05, Y0 + 2.0, 'relu(x)', color=TEAL_L, size=12, glowing=False)
    node(ax, X0, Y0, '', r=0.09, fill=AMBER, edge=AMBER)
    h = 0.5
    for sgn, col in ((1, AMBER), (-1, AMBER)):
        ax.plot([X0 + sgn * h * S] * 2, [Y0, Y0 + max(sgn * h, 0) * S], color=col,
                lw=1.0, ls=(0, (3, 2)))
        text(ax, X0 + sgn * h * S, Y0 - 0.3, f'x {"+" if sgn > 0 else "−"} h',
             color=col, size=11, glowing=False)
    text(ax, X0, Y0 + 2.6, 'at x = 0 exactly, relu has no derivative',
         color=TEAL_L, size=12, glowing=False)
    x = 6.6
    text(ax, x, 3.95, 'backprop picks a side:', color=TEAL_L, size=12, ha='left', glowing=False)
    text(ax, x, 3.55, 'mask = x > 0  →  0', color=TEAL_L, size=12, ha='left', glowing=False)
    text(ax, x, 2.95, 'central differences average', color=AMBER, size=12, ha='left', glowing=False)
    text(ax, x, 2.55, 'both sides:  (h − 0) / 2h  =  0.5', color=AMBER, size=12, ha='left', glowing=False)
    text(ax, x, 1.85, f'{zeros} of {total} pre-activations land', color=RED, size=11, ha='left', glowing=False)
    text(ax, x, 1.5, 'exactly on the corner, so:', color=RED, size=11, ha='left', glowing=False)
    text(ax, x, 1.0, f'ReLU gradcheck  {r_err:.2f}', color=RED, size=13, ha='left', glowing=False)
    text(ax, x, 0.55, f'same net, tanh  {t_err:.0e}', color=TEAL_L, size=13, ha='left', glowing=False)
    text(ax, 5.0, 0.12, 'the code is correct - check gradients with a smooth activation',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day46_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print('figures written')
