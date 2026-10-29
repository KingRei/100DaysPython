"""Day 47 figures.  Every number is computed by importing clustering_knn.py."""

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
import clustering_knn as ck                          # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)
CLS = [TEAL_L, AMBER, '#b45cff']
RED_S = '#ff5c5c'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def fit(P, x0, y0, w, h, lim=None, pad=0.06):
    """Map data coordinates into a canvas rectangle, with a margin so that no
    point ends up sitting on the frame."""
    P = np.asarray(P, dtype=float)
    lo = np.array(lim[0], dtype=float) if lim else P.min(axis=0)
    hi = np.array(lim[1], dtype=float) if lim else P.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    u = (P - lo) / span
    return np.column_stack([x0 + w * (pad + u[:, 0] * (1 - 2 * pad)),
                            y0 + h * (pad + u[:, 1] * (1 - 2 * pad))])


def frame(ax, x0, y0, w, h, title=None, tsize=12, color=TEAL_L):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none', edgecolor=TEAL,
                           lw=1.2, zorder=2))
    if title:
        text(ax, x0 + w / 2, y0 + h + 0.28, title, color=color, size=tsize)


def axes(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def logx(vals, x0, w):
    lv = [math.log(v) for v in vals]
    return [x0 + w * (v - lv[0]) / (lv[-1] - lv[0]) for v in lv]


def liny(vals, y0, h, lo, hi):
    return [y0 + h * (v - lo) / (hi - lo) for v in vals]


def marker(ax, x, y, color=AMBER, s=120):
    ax.scatter([x], [y], s=s, marker='X', color=color, zorder=6,
               edgecolors=NAVY, linewidths=0.8)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Lloyd's algorithm is two half-steps, and each one lowers the same number."""
    fig, ax = W(4.6)
    X = ck.blobs([[0, 0], [6, 0.5], [3, 5]], 60, 0.9, seed=1)
    _, _, _, _, hist = ck.kmeans(X, 3, seed=0, init='kmeans++', trace=True)
    lim = [[X[:, 0].min(), X[:, 1].min()], [X[:, 0].max(), X[:, 1].max()]]
    picks = [0, 1, len(hist) - 1]
    labels = ['start', 'after one round', f'converged, {len(hist)} rounds']
    for p, (idx, lab) in enumerate(zip(picks, labels)):
        C, la, J = hist[idx]
        x0 = 0.30 + p * 3.22
        P = fit(X, x0, 0.95, 2.75, 2.45, lim)
        Q = fit(C, x0, 0.95, 2.75, 2.45, lim)
        for c in range(3):
            m = la == c
            ax.scatter(P[m, 0], P[m, 1], s=6, color=CLS[c], zorder=3, linewidths=0)
        for c in range(3):
            marker(ax, Q[c, 0], Q[c, 1])
        frame(ax, x0, 0.95, 2.75, 2.45, lab, tsize=11.5)
        text(ax, x0 + 1.375, 0.60, f'inertia {J:.0f}', color=AMBER, size=12)
    text(ax, 5.0, 4.72, 'assign to the nearest centroid, then move each centroid to the mean',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 5.0, 4.38,
         'both halves lower the same sum of squared distances, so the loop has to stop',
         color=TEAL, size=11.5, glowing=False)
    legend(ax, 0.35, 0.22, [(AMBER, 'centroid')], size=11)
    save(fig, 'day47_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """The objective falls for every extra centroid, with or without structure."""
    fig, ax = W(4.8)
    Xb = ck.blobs([[0, 0], [6, 0.5], [3, 5]], 60, 0.9, seed=1)
    Xu = np.random.default_rng(11).random((180, 2)) * 6.0
    ks = list(range(1, 9))
    Jb = [ck.kmeans(Xb, k, seed=0, init='kmeans++')[2] for k in ks]
    Ju = [ck.kmeans(Xu, k, seed=0, init='kmeans++')[2] for k in ks]
    X0, Y0, WD, HT = 1.30, 1.05, 6.6, 2.75
    axes(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(ks) - 1) for i in range(len(ks))]
    for x, k in zip(xs, ks):
        text(ax, x, Y0 - 0.28, str(k), color=TEAL, size=12)
    for frac, lab in [(0.0, '0'), (0.5, '50%'), (1.0, '100%')]:
        text(ax, X0 - 0.26, Y0 + HT * frac, lab, color=TEAL, size=11, ha='right')
    for J, col in [(Jb, TEAL_L), (Ju, '#b45cff')]:
        ys = [Y0 + HT * (j / J[0]) for j in J]
        ax.plot(xs, ys, color=col, lw=2.2, zorder=4)
        ax.scatter(xs, ys, s=26, color=col, zorder=5, linewidths=0)
    legend(ax, X0 + WD * 0.52, Y0 + HT - 0.18,
           [('#b45cff', 'uniform noise, no structure at all'),
            (TEAL_L, 'three real blobs')], size=11)
    xe, ye = xs[2], Y0 + HT * (Jb[2] / Jb[0])
    ax.add_patch(Rectangle((xe - 0.26, ye - 0.24), 0.52, 0.48, facecolor='none',
                           edgecolor=AMBER, lw=2.0, zorder=6))
    arrow(ax, (xe + 0.95, ye + 1.00), (xe + 0.32, ye + 0.30), color=AMBER, lw=1.6)
    text(ax, xe + 1.35, ye + 1.18, 'the elbow', color=AMBER, size=12)
    text(ax, X0 + WD + 0.30, Y0, 'k', color=TEAL, size=13)
    text(ax, 5.0, 4.92, 'inertia as a share of the k = 1 value', color=TEAL_L, size=13)
    d = [100 * (Jb[1] - Jb[2]) / Jb[1], 100 * (Jb[2] - Jb[3]) / Jb[2],
         100 * (Ju[1] - Ju[2]) / Ju[1], 100 * (Ju[2] - Ju[3]) / Ju[2]]
    text(ax, 5.0, 0.22,
         f'blobs drop {d[0]:.0f}% then {d[1]:.0f}%;   noise drops {d[2]:.0f}% then {d[3]:.0f}% '
         '- the same curve, with nothing to find',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day47_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Same 360 rows, three distance functions, three different answers."""
    fig, ax = W(4.8)
    Xr, yr = ck.requests()
    _, lr, _, _, _ = ck.kmeans(Xr, 3, seed=0, init='kmeans++')
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
    _, lz, _, _, _ = ck.kmeans((Xr - mu) / sd, 3, seed=0, init='kmeans++')
    _, lh, _, _, _ = ck.kmeans(Xr[:, 1:2], 3, seed=0, init='kmeans++')
    lim = [[Xr[:, 0].min(), 0.0], [Xr[:, 0].max(), 1.0]]
    panels = [(lr, 'raw units', ck.purity(lr, yr, 3)),
              (lz, 'standardised', ck.purity(lz, yr, 3)),
              (lh, 'hit rate only', ck.purity(lh, yr, 3))]
    for p, (la, name, pu) in enumerate(panels):
        x0 = 0.58 + p * 3.08
        P = fit(Xr, x0, 1.20, 2.50, 2.40, lim)
        for c in range(3):
            m = la == c
            ax.scatter(P[m, 0], P[m, 1], s=6, color=CLS[c], zorder=3, linewidths=0)
        frame(ax, x0, 1.20, 2.50, 2.40, name, tsize=12)
        col = AMBER if pu < 0.99 else TEAL_L
        text(ax, x0 + 1.25, 0.86, f'purity {pu:.3f}', color=col, size=12)
    text(ax, 0.40, 2.40, 'cache hit rate', color=TEAL, size=11, rotation=90, glowing=False)
    text(ax, 5.0, 0.44, 'prompt tokens  (200 - 4000)', color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 4.92, 'the three real groups differ only in cache hit rate',
         color=TEAL_L, size=13)
    text(ax, 5.0, 4.58,
         'the token axis is 4000x wider, so raw distance cuts vertical bands and never sees them',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day47_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """If the update does not minimise the distance you assigned with, the
    objective is free to go back up."""
    fig, ax = W(5.4)
    Xo = ck.outlier_blobs(seed=0)
    _, _, c_mean = ck.kmedians(Xo, 3, seed=6, update='mean')
    _, _, c_med = ck.kmedians(Xo, 3, seed=6, update='median')
    n = max(len(c_mean), len(c_med))
    up = next(i for i in range(len(c_mean) - 1) if c_mean[i + 1] > c_mean[i] + 1e-9)

    # left panel: the whole run
    X0, Y0, WD, HT = 1.15, 1.15, 4.25, 2.55
    lo, hi = min(c_mean + c_med), max(c_mean + c_med)
    axes(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (n - 1) for i in range(n)]
    for x, i in zip(xs, range(n)):
        text(ax, x, Y0 - 0.28, str(i), color=TEAL, size=10.5)
    for series, col in [(c_mean, RED_S), (c_med, TEAL_L)]:
        ys = liny(series, Y0, HT, lo, hi)
        ax.plot(xs[:len(series)], ys, color=col, lw=2.2, zorder=4)
        ax.scatter(xs[:len(series)], ys, s=22, color=col, zorder=5, linewidths=0)
    text(ax, X0 + WD - 0.15, Y0 + HT - 0.30, f'ends at {c_mean[-1]:.1f}',
         color=RED_S, size=11.5, ha='right')
    text(ax, X0 + WD - 0.15, Y0 + HT - 0.72, f'ends at {c_med[-1]:.1f}',
         color=TEAL_L, size=11.5, ha='right')
    bx = xs[up] - 0.22
    ax.add_patch(Rectangle((bx, liny([c_mean[up + 1]], Y0, HT, lo, hi)[0] - 0.22),
                           (xs[up + 1] - xs[up]) + 0.44, 0.50, facecolor='none',
                           edgecolor=AMBER, lw=1.8, zorder=6))
    text(ax, X0 + WD / 2, Y0 + HT + 0.26, 'the whole run', color=TEAL, size=11.5)
    text(ax, X0 + WD + 0.28, Y0 - 0.02, 'round', color=TEAL, size=10.5, ha='left')

    # right panel: the same two rounds, on their own scale
    Z0, ZW = 6.45, 2.95
    zvals = c_mean[up:up + 3] + c_med[up:up + 3]
    zpad = 0.14 * (max(zvals) - min(zvals))
    zlo, zhi = min(zvals) - zpad, max(zvals) + zpad
    axes(ax, Z0, Y0, ZW, HT)
    zx = [Z0 + ZW * i / 2 for i in range(3)]
    for x, i in zip(zx, range(up, up + 3)):
        text(ax, x, Y0 - 0.28, str(i), color=TEAL, size=10.5)
    for series, col, offs in [(c_mean, RED_S, [0.32, 0.32, -0.34]),
                              (c_med, TEAL_L, [-0.34, -0.34, 0.34])]:
        vals = series[up:up + 3]
        ys = liny(vals, Y0, HT, zlo, zhi)
        ax.plot(zx[:len(vals)], ys, color=col, lw=2.4, zorder=4)
        ax.scatter(zx[:len(vals)], ys, s=26, color=col, zorder=5, linewidths=0)
        for i_, (x, v, y) in enumerate(zip(zx, vals, ys)):
            ha = 'left' if i_ == 0 else ('right' if i_ == len(vals) - 1 else 'center')
            dx = 0.10 if i_ == 0 else (-0.10 if i_ == len(vals) - 1 else 0.0)
            text(ax, x + dx, y + offs[i_], f'{v:.1f}', color=col, size=11, ha=ha)
    text(ax, Z0 + ZW / 2, Y0 + HT + 0.26, f'rounds {up} to {up + 2}, zoomed',
         color=AMBER, size=11.5)
    text(ax, Z0 + ZW + 0.28, Y0 - 0.02, 'round', color=TEAL, size=10.5, ha='left')

    legend(ax, 2.55, 5.20, [(RED_S, 'L1 assignment + mean update'),
                            (TEAL_L, 'L1 assignment + median update')], size=11)
    text(ax, 5.0, 5.72, 'total L1 distance to the assigned centre, per round',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.18,
         f'the mean version rose from {c_mean[up]:.1f} to {c_mean[up + 1]:.1f} - '
         'a loop whose objective can rise has no termination proof left',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day47_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """k is the smoothing knob, and the training score points the wrong way."""
    fig, ax = W(5.0)
    Xk, yk = ck.moons(120, seed=12, noise=0.22)
    perm = np.random.default_rng(0).permutation(len(Xk))
    tr, te = perm[:160], perm[160:]
    ks = [1, 3, 5, 11, 25, 61, 160]
    names = ['train, self included', 'train, self excluded', 'held-out test']
    cols = {names[0]: RED_S, names[1]: AMBER, names[2]: TEAL_L}
    series = {n: [] for n in names}
    for k in ks:
        series[names[0]].append(ck.accuracy(ck.knn_predict(Xk[tr], yk[tr], Xk[tr], k), yk[tr]))
        series[names[1]].append(ck.accuracy(
            ck.knn_predict(Xk[tr], yk[tr], Xk[tr], k, exclude_self=True), yk[tr]))
        series[names[2]].append(ck.accuracy(ck.knn_predict(Xk[tr], yk[tr], Xk[te], k), yk[te]))
    X0, Y0, WD, HT = 1.35, 1.25, 6.3, 2.45
    LO, HI = 0.46, 1.02
    axes(ax, X0, Y0, WD, HT)
    xs = logx(ks, X0, WD)
    for x, k in zip(xs, ks):
        text(ax, x, Y0 - 0.28, str(k), color=TEAL, size=11)
    for v in (0.5, 0.75, 1.0):
        text(ax, X0 - 0.24, liny([v], Y0, HT, LO, HI)[0], f'{v:.2f}',
             color=TEAL, size=11, ha='right')
    for n in names:
        ys = liny(series[n], Y0, HT, LO, HI)
        ax.plot(xs, ys, color=cols[n], lw=2.2, zorder=4)
        ax.scatter(xs, ys, s=24, color=cols[n], zorder=5, linewidths=0)
    y1 = liny([1.0], Y0, HT, LO, HI)[0]
    ax.add_patch(Rectangle((xs[0] - 0.26, y1 - 0.20), 0.52, 0.40, facecolor='none',
                           edgecolor=RED_S, lw=1.8, zorder=6))
    arrow(ax, (2.55, 4.42), (xs[0] + 0.22, y1 + 0.26), color=RED_S, lw=1.6)
    text(ax, 3.05, 4.56, 'k = 1 scores 1.000 on its own training data, always',
         color=RED_S, size=11.5, ha='left', glowing=False)
    best = max(range(len(ks)), key=lambda i: series[names[2]][i])
    text(ax, xs[best], liny([series[names[2]][best]], Y0, HT, LO, HI)[0] + 0.32,
         f'best k = {ks[best]}', color=TEAL_L, size=11.5)
    legend(ax, X0 + 0.35, Y0 + 1.05, [(cols[n], n) for n in names], size=11)
    text(ax, X0 + WD + 0.28, Y0, 'k', color=TEAL, size=12, ha='left')
    text(ax, 5.0, 5.15, 'accuracy on 240 points from two overlapping moons, by k (log scale)',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'the nearest point to a training point is itself, so k = 1 cannot be wrong on it',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day47_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """In high dimensions every point is the same distance away."""
    fig, ax = W(4.9)
    dims = [2, 4, 8, 16, 64, 256, 1024]
    cons = [ck.contrast(500, d, seed=0) for d in dims]
    ndim = [2, 4, 10, 30, 100, 400]
    accs = []
    for d in ndim:
        Xn, yn = ck.noisy_features(400, d, seed=5)
        rn = np.random.default_rng(1).permutation(400)
        accs.append(ck.accuracy(ck.knn_predict(Xn[rn[:300]], yn[rn[:300]], Xn[rn[300:]], 11),
                                yn[rn[300:]]))
    Y0, HT = 1.45, 2.55
    for p, (xv, yv, name, col, fmt) in enumerate([
            (dims, cons, '(furthest - nearest) / nearest', TEAL_L, '{:.2f}'),
            (ndim, accs, 'accuracy, k = 11', AMBER, '{:.3f}')]):
        X0, WD = 0.95 + p * 4.85, 3.30
        axes(ax, X0, Y0, WD, HT)
        xs = logx(xv, X0, WD)
        lo, hi = min(yv), max(yv)
        ys = liny(yv, Y0 + 0.12, HT - 0.45, lo, hi)
        ax.plot(xs, ys, color=col, lw=2.2, zorder=4)
        ax.scatter(xs, ys, s=24, color=col, zorder=5, linewidths=0)
        for x, v in zip(xs, xv):
            text(ax, x, Y0 - 0.28, str(v), color=TEAL, size=9.5)
        text(ax, xs[0] + 0.16, ys[0] + 0.30, fmt.format(yv[0]), color=col, size=12, ha='left')
        text(ax, xs[-1] + 0.14, ys[-1] + 0.26, fmt.format(yv[-1]), color=col, size=12, ha='left')
        text(ax, X0 + WD / 2, Y0 + HT + 0.26, name, color=col, size=11.5, glowing=False)
        text(ax, X0 + WD + 0.22, Y0 - 0.02, 'dims', color=TEAL, size=10.5, ha='left')
    text(ax, 5.0, 5.05, 'adding columns does not add information, it adds averaging',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'left: 500 uniform random points.   right: the same two useful columns, buried in noise',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day47_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, 'ok')
