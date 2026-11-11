"""Day 48 figures.  Every number is computed by importing trees.py."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import trees as TR                                   # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)
RED_S = '#ff5c5c'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def fit(P, x0, y0, w, h, lim=None, pad=0.06):
    P = np.asarray(P, dtype=float)
    lo = np.array(lim[0], dtype=float) if lim else P.min(axis=0)
    hi = np.array(lim[1], dtype=float) if lim else P.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    u = (P - lo) / span
    return np.column_stack([x0 + w * (pad + u[:, 0] * (1 - 2 * pad)),
                            y0 + h * (pad + u[:, 1] * (1 - 2 * pad))])


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def liny(vals, y0, h, lo, hi):
    return [y0 + h * (v - lo) / (hi - lo) for v in vals]


def bars(ax, x0, base, bw, gap, vals, cols, labs, maxh, top=None):
    hi = top or max(abs(v) for v in vals)
    for i, v in enumerate(vals):
        hh = max(.03, maxh * abs(v) / hi)
        x = x0 + i * (bw + gap)
        y = base if v >= 0 else base - hh          # negatives hang below the axis
        ax.add_patch(Rectangle((x, y), bw, hh, facecolor=cols[i], edgecolor=cols[i],
                               lw=1.2, alpha=.85, zorder=3))
        if labs and labs[i] is not None:
            text(ax, x + bw / 2, (base + hh + .22) if v >= 0 else (base - hh - .22),
                 labs[i], color=cols[i], size=11.5)


# ---------------------------------------------------------------- figure 1
def fig1():
    """One prediction is one root-to-leaf path."""
    fig, ax = W(5.4)
    X, y = TR.stripes(400, seed=0, noise=0.10)
    t = TR.DecisionTree(max_depth=3).fit(X, y)
    steps, val = t.path(X[0], names=['x', 'y'])

    # lay the tree out by depth; mark the nodes the query actually visits
    on_path = set()
    node = t.root
    row = X[0]
    while True:
        on_path.add(id(node))
        if node.is_leaf:
            break
        node = node.left if row[node.feat] <= node.thr else node.right
    pos = {}

    def place(nd, d, x0, x1):
        pos[id(nd)] = ((x0 + x1) / 2, 4.55 - d * 1.05)
        if not nd.is_leaf:
            place(nd.left, d + 1, x0, (x0 + x1) / 2)
            place(nd.right, d + 1, (x0 + x1) / 2, x1)

    place(t.root, 0, 0.45, 9.55)

    def draw(nd):
        px, py = pos[id(nd)]
        hot = id(nd) in on_path
        if not nd.is_leaf:
            for ch in (nd.left, nd.right):
                cx, cy = pos[id(ch)]
                both = hot and id(ch) in on_path
                ax.plot([px, py * 0 + px], [py - .20, py - .20], lw=0)
                ax.plot([px, cx], [py - .22, cy + .22],
                        color=AMBER if both else TEAL, lw=2.0 if both else 1.0,
                        alpha=1.0 if both else .55, zorder=2)
                draw(ch)
            box(ax, px - .80, py - .22, 1.60, .46,
                label=f'x{nd.feat} <= {nd.thr:.2f}'.replace('x0', 'x').replace('x1', 'y'),
                fill=NAVY, edge=AMBER if hot else TEAL, lw=2.0 if hot else 1.4, size=11)
        else:
            node_col = AMBER if hot else TEAL
            box(ax, px - .42, py - .20, .84, .40, label=f'{nd.value}', fill=NAVY,
                edge=node_col, lw=2.0 if hot else 1.2, size=12)

    draw(t.root)
    text(ax, 5.0, 5.72, 'a depth-3 tree: every internal node is one question',
         color=TEAL_L, size=13)
    text(ax, 5.0, 5.38,
         f'{t.n_nodes} nodes, {t.n_leaves} leaves, trained on 400 points with 10% of the labels flipped',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, .74,
         'the highlighted path is the model\'s entire answer for the point '
         f'({X[0][0]:.2f}, {X[0][1]:.2f}): class {val}',
         color=AMBER, size=11.5, glowing=False)
    text(ax, 5.0, .36, '  ->  '.join(steps), color=TEAL, size=11, glowing=False)
    save(fig, 'day48_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Training accuracy is monotone in depth and ends at exactly 1.000."""
    fig, ax = W(5.0)
    X, y = TR.stripes(400, seed=0, noise=0.10)
    perm = np.random.default_rng(0).permutation(400)
    tr, te = perm[:260], perm[260:]
    depths = [1, 2, 3, 5, 8, 12, None]
    a_tr, a_te, leaves = [], [], []
    for d in depths:
        t = TR.DecisionTree(max_depth=d).fit(X[tr], y[tr])
        a_tr.append(TR.accuracy(t.predict(X[tr]), y[tr]))
        a_te.append(TR.accuracy(t.predict(X[te]), y[te]))
        leaves.append(t.n_leaves)
    X0, Y0, WD, HT = 1.35, 1.35, 6.4, 2.70
    LO, HI = 0.60, 1.04
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(depths) - 1) for i in range(len(depths))]
    for x, d, lv in zip(xs, depths, leaves):
        text(ax, x, Y0 - .28, 'none' if d is None else str(d), color=TEAL, size=11)
        text(ax, x, Y0 - .62, str(lv), color=GREY_L, size=10)
    text(ax, X0 - .30, Y0 - .28, 'depth', color=TEAL, size=11, ha='right')
    text(ax, X0 - .30, Y0 - .62, 'leaves', color=GREY_L, size=10, ha='right')
    for v in (0.7, 0.8, 0.9, 1.0):
        text(ax, X0 - .22, liny([v], Y0, HT, LO, HI)[0], f'{v:.1f}', color=TEAL,
             size=11, ha='right')
    for vals, col, nm in [(a_tr, RED_S, 'training accuracy'), (a_te, TEAL_L, 'held-out accuracy')]:
        ys = liny(vals, Y0, HT, LO, HI)
        ax.plot(xs, ys, color=col, lw=2.3, zorder=4)
        ax.scatter(xs, ys, s=26, color=col, zorder=5, linewidths=0)
    best = int(np.argmax(a_te))
    ax.add_patch(Rectangle((xs[best] - .26, liny([a_te[best]], Y0, HT, LO, HI)[0] - .22),
                           .52, .44, facecolor='none', edgecolor=AMBER, lw=1.8, zorder=6))
    text(ax, xs[best], liny([a_te[best]], Y0, HT, LO, HI)[0] - .55,
         f'best {a_te[best]:.3f}', color=AMBER, size=11.5)
    text(ax, xs[-1] - .10, liny([1.0], Y0, HT, LO, HI)[0] + .28, '1.000',
         color=RED_S, size=12, ha='right')
    # label the curves where they are furthest apart instead of using a legend box
    text(ax, xs[4], liny([a_tr[4]], Y0, HT, LO, HI)[0] + .30, 'training accuracy',
         color=RED_S, size=11.5, glowing=False)
    text(ax, xs[5], liny([a_te[5]], Y0, HT, LO, HI)[0] - .32, 'held-out accuracy',
         color=TEAL_L, size=11.5, glowing=False)
    text(ax, 5.0, 5.18, 'the training score is monotone in depth and ends at exactly 1.000',
         color=TEAL_L, size=13)
    text(ax, 5.0, .42,
         'the tree keeps splitting until every leaf is pure, so the training score has nowhere to go but up',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day48_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """On XOR the correct split is worth nothing, so greedy never takes it."""
    fig, ax = W(5.0)
    X, y = TR.xor_data(400, seed=1)
    par = TR.gini(y, 2)
    gains = []
    for j in (0, 1):
        m = X[:, j] <= 0.0
        gains.append(par - (m.sum() * TR.gini(y[m], 2) + (~m).sum() * TR.gini(y[~m], 2)) / len(y))
    g_best, j_best, t_best = TR.best_split(X, y, 2, 'gini')
    lim = [[-1.05, -1.05], [1.05, 1.05]]
    for p, (title, thr, col) in enumerate([
            ('the split greedy picks', (j_best, t_best), RED_S),
            ('the split that solves it', (0, 0.0), TEAL_L)]):
        x0 = 0.60 + p * 4.55
        P = fit(X, x0, 1.25, 3.45, 3.45, lim)
        ax.add_patch(Rectangle((x0, 1.25), 3.45, 3.45, facecolor='none',
                               edgecolor=TEAL, lw=1.1, zorder=2))
        for c, cc in ((0, '#0093a1'), (1, AMBER)):
            m = y == c
            ax.scatter(P[m, 0], P[m, 1], s=7, color=cc, zorder=3, linewidths=0)
        j, t = thr
        a = fit(np.array([[t, -1.05]]), x0, 1.25, 3.45, 3.45, lim)[0] if j == 0 else \
            fit(np.array([[-1.05, t]]), x0, 1.25, 3.45, 3.45, lim)[0]
        b = fit(np.array([[t, 1.05]]), x0, 1.25, 3.45, 3.45, lim)[0] if j == 0 else \
            fit(np.array([[1.05, t]]), x0, 1.25, 3.45, 3.45, lim)[0]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=2.6, zorder=5)
        text(ax, x0 + 1.725, 4.98, title, color=col, size=12)
        g = g_best if p == 0 else gains[0]
        text(ax, x0 + 1.725, .88, f'gini gain {g:.6f}', color=col, size=12)
    text(ax, 5.0, 5.44, 'y = (x > 0) XOR (y > 0): neither column means anything on its own',
         color=TEAL_L, size=13)
    text(ax, 5.0, .46,
         'greedy takes the split worth 0.011 and learns nothing;',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, .14,
         'the split worth 0.000001 is the only one that leads anywhere',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day48_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Impurity importance measures opportunity; permutation importance measures use."""
    fig, ax = W(5.0)
    X, y, names = TR.cardinality_data(300, seed=2)
    r = np.random.default_rng(0).permutation(300)
    tr, te = r[:210], r[210:]
    t = TR.DecisionTree().fit(X[tr], y[tr])
    imp = t.importances(X.shape[1])
    pi = TR.permutation_importance(t, X[te], y[te], np.random.default_rng(7), repeats=20)
    short = ['real', 'cont.', 'b1', 'b2', 'b3']
    cols = [TEAL_L, RED_S, GREY_L, GREY_L, GREY_L]
    for p, (vals, title, fmt) in enumerate([
            (imp, 'impurity importance (what the tree says)', '{:.3f}'),
            (pi, 'permutation importance on held-out rows', '{:+.3f}')]):
        X0, WD = 0.75 + p * 4.85, 3.55
        base, maxh = 1.55, 2.30
        ax.plot([X0, X0 + WD], [base, base], color=TEAL, lw=1.2, zorder=2)
        top = max(max(imp), max(pi))
        bars(ax, X0 + .12, base, .52, .18, list(vals), cols,
             [fmt.format(v) for v in vals], maxh, top)
        for i, nm in enumerate(short):
            text(ax, X0 + .12 + i * .70 + .26, base - .88, nm, color=TEAL, size=11)
        text(ax, X0 + WD / 2, base + maxh + .80, title, color=TEAL_L, size=11.5,
             glowing=False)
    text(ax, 5.0, 5.30, 'the same tree, scored two ways', color=TEAL_L, size=13)
    text(ax, 5.0, .52,
         'the continuous noise column offers 209 candidate thresholds at the root; '
         'the real one offers 1',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, .18, 'real = the binary column that is the answer;  cont. = continuous noise;  '
         'b1-b3 = binary noise', color=GREY_L, size=11, glowing=False)
    save(fig, 'day48_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """Without feature subsampling every tree opens with the same question."""
    fig, ax = W(5.0)
    X, y = TR.dominant(900, seed=0)
    r = np.random.default_rng(1).permutation(900)
    tr, te = r[:600], r[600:]
    one = TR.accuracy(TR.DecisionTree().fit(X[tr], y[tr]).predict(X[te]), y[te])
    out = {}
    for nm, kw in [('bagging only', dict(max_features=None)),
                   ('random forest', dict(max_features='sqrt'))]:
        cnt = np.zeros(10, dtype=int)
        acc, dis = [], []
        for sd in range(3):
            f = TR.RandomForest(n_trees=30, seed=sd, **kw).fit(X[tr], y[tr])
            cnt += np.bincount([t.root.feat for t in f.trees], minlength=10)
            acc.append(TR.accuracy(f.predict(X[te]), y[te]))
            dis.append(f.disagreement(X[te]))
        out[nm] = (cnt, float(np.mean(acc)), float(np.mean(dis)))
    for p, nm in enumerate(out):
        cnt, acc, dis = out[nm]
        X0, WD = 0.80 + p * 4.80, 3.60
        base, maxh = 1.70, 2.20
        ax.plot([X0, X0 + WD], [base, base], color=TEAL, lw=1.2, zorder=2)
        cols = [AMBER if c == cnt.max() and c > 0 else TEAL_L for c in cnt]
        bars(ax, X0 + .10, base, .28, .06, list(cnt.astype(float)), cols,
             [str(c) if c else None for c in cnt], maxh, 90.0)
        for i in range(10):
            text(ax, X0 + .10 + i * .34 + .14, base - .34, str(i), color=GREY_L, size=9.5)
        text(ax, X0 + WD / 2, base - .72, 'root column', color=TEAL, size=11)
        text(ax, X0 + WD / 2, base + maxh + .72, nm, color=TEAL_L, size=12.5)
        text(ax, X0 + WD / 2, base + maxh + .38,
             f'test {acc:.3f}   disagreement {dis:.3f}', color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 5.30, 'which column each of 90 trees opens with', color=TEAL_L, size=13)
    text(ax, 5.0, .62, f'one tree on its own scores {one:.3f}',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, .26,
         'averaging trees that ask the same first question buys less than averaging trees that do not',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day48_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """A tree is a step function, and the last step runs flat forever."""
    fig, ax = W(5.0)
    Xw, yw = TR.wave(120, seed=3)
    t = TR.DecisionTree(task='reg', max_depth=6).fit(Xw, yw)
    f = TR.RandomForest(n_trees=40, task='reg', max_features=None, seed=0).fit(Xw, yw)
    grid = np.linspace(-3.0, 7.0, 500)[:, None]
    truth = np.sin(grid[:, 0]) + 0.35 * grid[:, 0]
    pt, pf = t.predict(grid), f.predict(grid)
    lim = [[-3.0, min(truth.min(), pt.min()) - .2], [7.0, max(truth.max(), pt.max()) + .2]]
    X0, Y0, WD, HT = 1.05, 1.35, 7.9, 3.00
    ax.add_patch(Rectangle((X0, Y0), WD, HT, facecolor='none', edgecolor=TEAL, lw=1.1, zorder=2))
    edge = fit(np.array([[3.0, lim[0][1]], [3.0, lim[1][1]]]), X0, Y0, WD, HT, lim)
    ax.plot(edge[:, 0], edge[:, 1], color=GREY_L, lw=1.4, ls='--', zorder=3)
    for vals, col, lw in [(truth, GREY_L, 1.6), (pf, TEAL_L, 1.8), (pt, AMBER, 2.0)]:
        P = fit(np.column_stack([grid[:, 0], vals]), X0, Y0, WD, HT, lim)
        ax.plot(P[:, 0], P[:, 1], color=col, lw=lw, zorder=4)
    D = fit(np.column_stack([Xw[:, 0], yw]), X0, Y0, WD, HT, lim)
    ax.scatter(D[:, 0], D[:, 1], s=7, color='#0093a1', zorder=5, linewidths=0)
    text(ax, edge[0][0], Y0 - .30, 'edge of the training data', color=GREY_L, size=10.5,
         glowing=False)
    legend(ax, X0 + .25, Y0 + HT - .25,
           [(AMBER, 'one depth-6 tree'), (TEAL_L, 'forest of 40 trees'),
            (GREY_L, 'the function being fitted')], size=11)
    far = float(t.predict(np.array([[100.0]]))[0])
    text(ax, 5.0, 5.30, 'what the model does past the last training point', color=TEAL_L, size=13)
    text(ax, 5.0, .62,
         f'at x = 100 the tree still answers {far:+.4f} and the forest still answers '
         f'{float(f.predict(np.array([[100.0]]))[0]):+.4f}',
         color=AMBER, size=11.5, glowing=False)
    text(ax, 5.0, .26,
         'the truth there is +34.4936 - a leaf holds a constant, so the last step runs flat forever',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day48_6.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    """The training loss is monotone; the held-out loss is not."""
    fig, ax = W(5.0)
    X, y = TR.tabular_reg(600, 6, seed=4)
    r = np.random.default_rng(2).permutation(600)
    tr, te = r[:420], r[420:]
    runs = {}
    for lr in (1.0, 0.1):
        runs[lr] = TR.GradientBoosting(n_trees=300, lr=lr, max_depth=2).fit(
            X[tr], y[tr], X[te], y[te])
    X0, Y0, WD, HT = 1.25, 1.35, 6.6, 2.85
    LO, HI = 0.0, 2.6
    axes2(ax, X0, Y0, WD, HT)
    n = 300
    xs = [X0 + WD * i / (n - 1) for i in range(n)]
    for m in (1, 50, 100, 150, 200, 250, 300):
        text(ax, X0 + WD * (m - 1) / (n - 1), Y0 - .28, str(m), color=TEAL, size=10.5)
    for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        text(ax, X0 - .22, liny([v], Y0, HT, LO, HI)[0], f'{v:.1f}', color=TEAL,
             size=10.5, ha='right')
    styles = {(1.0, 'train'): (RED_S, '--'), (1.0, 'test'): (RED_S, '-'),
              (0.1, 'train'): (TEAL_L, '--'), (0.1, 'test'): (TEAL_L, '-')}
    for lr, g in runs.items():
        for key, series in (('train', g.train_loss), ('test', g.val_loss)):
            col, ls = styles[(lr, key)]
            ax.plot(xs, liny(series, Y0, HT, LO, HI), color=col, lw=2.0, ls=ls, zorder=4)
    g1 = runs[1.0]
    b = int(np.argmin(g1.val_loss))
    bx, by = xs[b], liny([g1.val_loss[b]], Y0, HT, LO, HI)[0]
    ax.add_patch(Rectangle((bx - .20, by - .20), .40, .40, facecolor='none',
                           edgecolor=AMBER, lw=1.8, zorder=6))
    arrow(ax, (bx + 1.30, by + 1.05), (bx + .24, by + .26), color=AMBER, lw=1.5)
    text(ax, bx + 1.90, by + 1.22, f'best held-out loss, at {b + 1} trees',
         color=AMBER, size=11.5)
    legend(ax, X0 + WD * .74, Y0 + HT - .25,
           [(RED_S, 'lr = 1.0'), (TEAL_L, 'lr = 0.1')], size=11)
    text(ax, X0 + WD * .40, Y0 + HT - 1.15, 'dashed = training, solid = held out',
         color=GREY_L, size=10.5, ha='left', glowing=False)
    text(ax, X0 + WD + .22, Y0 - .02, 'trees', color=TEAL, size=10.5, ha='left')
    text(ax, 5.0, 5.30, 'mean squared error while boosting 300 depth-2 trees',
         color=TEAL_L, size=13)
    text(ax, 5.0, .42,
         f'at lr = 1.0 the training loss ends at {g1.train_loss[-1]:.4f} while the held-out loss '
         f'climbs back to {g1.val_loss[-1]:.4f}',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day48_7.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print(f.__name__, 'ok')
