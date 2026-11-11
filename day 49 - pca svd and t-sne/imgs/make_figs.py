"""Day 49 figures.  Every number is computed here or by importing dimred.py."""

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
import dimred as DR                                  # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'


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


def equal_lim(P, w, h, pad=0.08):
    """Limits that make one data unit the same length on both axes, so that angles
    and perpendicularity in the figure are the angles in the data."""
    P = np.asarray(P, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    c = (lo + hi) / 2
    span = (hi - lo) * (1 + 2 * pad)
    if span[0] / span[1] < w / h:
        span[0] = span[1] * w / h
    else:
        span[1] = span[0] * h / w
    return [list(c - span / 2), list(c + span / 2)]


def clip_segment(a, b, box):
    """Clip the segment a-b to the rectangle box = (x0, y0, w, h)."""
    x0, y0, w, h = box
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for q, r in ((-d[0], a[0] - x0), (d[0], x0 + w - a[0]),
                 (-d[1], a[1] - y0), (d[1], y0 + h - a[1])):
        if q == 0:
            if r < 0:
                return None
            continue
        t = r / q
        if q < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return a + t0 * d, a + t1 * d


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
        ax.add_patch(Rectangle((x, base), bw, hh, facecolor=cols[i], edgecolor=cols[i],
                               lw=1.2, alpha=.85, zorder=3))
        if labs and labs[i] is not None:
            text(ax, x + bw / 2, base + hh + .22, labs[i], color=cols[i], size=11.5)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Without centring, PC1 points at where the data is, not at how it is shaped."""
    fig, ax = W(5.2)
    X = DR.anisotropic(300, seed=0)
    pc = DR.PCA(center_data=True).fit(X)
    pu = DR.PCA(center_data=False).fit(X)
    X0, Y0, WD, HT = 0.80, 0.95, 4.5, 3.8
    lim = equal_lim(np.vstack([X, [[0.0, 0.0]]]), WD, HT)
    box = (X0, Y0, WD, HT)
    ax.add_patch(Rectangle((X0, Y0), WD, HT, facecolor='none', edgecolor=TEAL,
                           lw=1.1, zorder=2))
    P = fit(X, X0, Y0, WD, HT, lim, pad=0.0)
    ax.scatter(P[:, 0], P[:, 1], s=6, color='#0093a1', zorder=3, linewidths=0)
    org = fit(np.array([[0.0, 0.0]]), X0, Y0, WD, HT, lim, pad=0.0)[0]
    mu = fit(np.array([X.mean(axis=0)]), X0, Y0, WD, HT, lim, pad=0.0)[0]
    ax.scatter([org[0]], [org[1]], s=48, marker='+', color=GREY_L, zorder=5)
    text(ax, org[0] + .42, org[1] - .02, 'origin', color=GREY_L, size=10.5, ha='left',
         glowing=False)
    for base, direction, col in [(X.mean(axis=0), pc.components_[0], TEAL_L),
                                 (np.zeros(2), pu.components_[0], RED_S)]:
        ends = fit(np.array([base - 30 * direction, base + 30 * direction]),
                   X0, Y0, WD, HT, lim, pad=0.0)
        seg = clip_segment(ends[0], ends[1], box)
        if seg is not None:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=col, lw=2.4,
                    zorder=6)
    ax.scatter([mu[0]], [mu[1]], s=58, marker='X', color=AMBER, zorder=7,
               edgecolors=NAVY, linewidths=.7)
    text(ax, mu[0] + .46, mu[1] - .30, 'the mean', color=AMBER, size=11, ha='left',
         glowing=False)
    ang = lambda a, b: math.degrees(math.acos(min(1.0, abs(float(a @ b)))))
    mdir = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    for i, (nm, col, line2, line3) in enumerate([
            ('centred', TEAL_L, f'explains {pc.explained_variance_ratio_[0]:.4f}',
             'measured about the mean'),
            ('not centred', RED_S, f'claims {pu.explained_variance_ratio_[0]:.4f}',
             'measured about the origin')]):
        yy = 4.15 - i * 1.30
        text(ax, 5.75, yy, 'PC1 ' + nm, color=col, size=13, ha='left', glowing=False)
        text(ax, 5.75, yy - .40, line2, color=col, size=12, ha='left', glowing=False)
        text(ax, 5.75, yy - .76, line3, color=GREY_L, size=10.5, ha='left', glowing=False)
    text(ax, 5.75, 1.42, f'the two differ by {ang(pu.components_[0], pc.components_[0]):.1f} degrees,',
         color=TEAL, size=12, ha='left', glowing=False)
    text(ax, 5.75, 1.04, f'and the red one sits {ang(pu.components_[0], mdir):.1f} degrees off',
         color=TEAL, size=12, ha='left', glowing=False)
    text(ax, 5.75, 0.66, 'the direction of the mean', color=TEAL, size=12, ha='left',
         glowing=False)
    text(ax, 5.0, 5.52, 'the same 300 points, with and without subtracting the mean',
         color=TEAL_L, size=13)
    text(ax, 5.0, .30, 'the broken version reports the better score', color=GREY_L,
         size=11.5, glowing=False)
    save(fig, 'day49_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """The widest direction and the useful direction are not the same direction."""
    fig, ax = W(5.0)
    H, hy = DR.hidden_signal(400, seed=1)
    p = DR.PCA().fit(H)
    Z = p.transform(H, 2)
    acc = [DR.best_threshold_accuracy(Z[:, j], hy) for j in range(2)]
    X0, Y0, WD, HT = 0.70, 1.15, 4.3, 3.3
    lim = equal_lim(H, WD, HT)
    ax.add_patch(Rectangle((X0, Y0), WD, HT, facecolor='none', edgecolor=TEAL, lw=1.1,
                           zorder=2))
    P = fit(H, X0, Y0, WD, HT, lim, pad=0.0)
    for c, col in ((0, TEAL_L), (1, PURPLE)):
        m = hy == c
        ax.scatter(P[m, 0], P[m, 1], s=6, color=col, zorder=3, linewidths=0)
    mu = fit(np.array([H.mean(axis=0)]), X0, Y0, WD, HT, lim, pad=0.0)[0]
    for j, (col, lab) in enumerate([(RED_S, 'PC1'), (AMBER, 'PC2')]):
        d = fit(np.array([H.mean(axis=0), H.mean(axis=0) + p.components_[j]]),
                X0, Y0, WD, HT, lim, pad=0.0)
        v = np.array(d[1]) - np.array(d[0])
        v = v / np.linalg.norm(v) * (1.9 if j == 0 else 1.1)
        ax.plot([mu[0] - v[0], mu[0] + v[0]], [mu[1] - v[1], mu[1] + v[1]],
                color=col, lw=2.4, zorder=6)
        text(ax, mu[0] + v[0] * 1.18, mu[1] + v[1] * 1.18, lab, color=col, size=12)
    # the two 1-D projections, as strip plots
    for j in range(2):
        bx, by = 5.65, 3.65 - j * 2.15
        lo, hi = Z[:, j].min(), Z[:, j].max()
        ax.plot([bx, bx + 3.6], [by, by], color=TEAL, lw=1.1, zorder=2)
        for c, col in ((0, TEAL_L), (1, PURPLE)):
            m = hy == c
            xs = bx + 3.6 * (Z[m, j] - lo) / (hi - lo)
            ax.scatter(xs, np.full(m.sum(), by + (0.16 if c else -0.16)), s=5,
                       color=col, zorder=3, linewidths=0)
        text(ax, bx, by + .92, f'projection on PC{j + 1}', color=[RED_S, AMBER][j],
             size=12, ha='left', glowing=False)
        text(ax, bx, by + .56,
             f'{100 * p.explained_variance_ratio_[j]:.2f}% of the variance',
             color=GREY_L, size=10.5, ha='left', glowing=False)
        text(ax, bx + 1.8, by - .58, f'best single cut: {acc[j]:.4f}',
             color=[RED_S, AMBER][j], size=12)
    text(ax, 5.0, 5.20, 'PC1 is the widest direction; PC2 is the one with the answer in it',
         color=TEAL_L, size=13)
    text(ax, 5.0, .42,
         f'the variance ratio between them is {p.explained_variance_[0] / p.explained_variance_[1]:.0f} to 1, '
         'and "keep 95% of the variance" drops PC2',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day49_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Squaring the condition number rounds small singular values out of existence."""
    fig, ax = W(5.0)
    eps_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    svd_v, eig_v, conds = [], [], []
    for eps in eps_list:
        Xi = DR.ill_conditioned(200, eps=eps, seed=0)
        sv = np.linalg.svd(Xi - Xi.mean(axis=0), full_matrices=False)[1]
        w, _ = DR.pca_via_covariance(Xi)
        svd_v.append(sv[-1])
        eig_v.append(float(np.sqrt(max(w[-1], 0.0) * (len(Xi) - 1))))
        conds.append(sv[0] / sv[-1])
    X0, Y0, WD, HT = 1.35, 1.35, 6.5, 2.85
    LO, HI = -11.0, -2.0                                    # log10 of the value
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(eps_list) - 1) for i in range(len(eps_list))]
    for x, e in zip(xs, eps_list):
        text(ax, x, Y0 - .30, f'{e:.0e}'.replace('e-0', 'e-'), color=TEAL, size=10.5)
    for v in (-10, -8, -6, -4, -2):
        text(ax, X0 - .22, liny([v], Y0, HT, LO, HI)[0], f'1e{v}', color=TEAL, size=10.5,
             ha='right')
    ys_svd = liny([math.log10(v) for v in svd_v], Y0, HT, LO, HI)
    ax.plot(xs, ys_svd, color=TEAL_L, lw=3.0, zorder=4)
    ax.scatter(xs, ys_svd, s=26, color=TEAL_L, zorder=5, linewidths=0)
    ys_eig, gone = [], []
    for i, v in enumerate(eig_v):
        if v > 0:
            ys_eig.append(liny([math.log10(v)], Y0, HT, LO, HI)[0])
        else:
            ys_eig.append(Y0 + .04)
            gone.append(i)
    ax.plot(xs, ys_eig, color=RED_S, lw=1.6, ls='--', zorder=6)
    ax.scatter(xs, ys_eig, s=20, color=RED_S, zorder=6, linewidths=0)
    for i in gone:
        ax.add_patch(Rectangle((xs[i] - .22, Y0 - .18), .44, .40, facecolor='none',
                               edgecolor=RED_S, lw=1.8, zorder=6))
    legend(ax, X0 + .30, Y0 + 1.20,
           [(TEAL_L, 'from the SVD of the centred data'),
            (RED_S, 'from eigh of the covariance matrix')], size=11)
    text(ax, 5.0, 5.20, 'the smallest singular value, recovered two ways',
         color=TEAL_L, size=13)
    text(ax, 5.0, .46, 'the boxed points came back as exactly 0 - a real direction, '
         'rounded out of existence', color=RED_S, size=11.5, glowing=False)
    text(ax, 5.0, .12, 'cond(cov) = cond(X) squared, so past cond(X) of 1e8 the small '
         'values fall off the end of float64', color=TEAL, size=11, glowing=False)
    save(fig, 'day49_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Eckart-Young: the truncation error is exactly the discarded tail."""
    fig, ax = W(5.0)
    rng = np.random.default_rng(5)
    A = rng.standard_normal((120, 40)) @ rng.standard_normal((40, 40))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    ks = list(range(1, 41))
    err = [float(np.linalg.norm(A - (U[:, :k] * s[:k]) @ Vt[:k])) for k in ks]
    pred = [float(math.sqrt((s[k:] ** 2).sum())) for k in ks]
    kept = [float((s[:k] ** 2).sum() / (s ** 2).sum()) for k in ks]
    X0, Y0, WD, HT = 1.30, 1.35, 6.6, 2.85
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * (k - 1) / 39 for k in ks]
    for k in (1, 10, 20, 30, 40):
        text(ax, X0 + WD * (k - 1) / 39, Y0 - .30, str(k), color=TEAL, size=11)
    hi = max(err)
    for v, lab in ((0.0, '0'), (hi / 2, f'{hi / 2:.0f}'), (hi, f'{hi:.0f}')):
        text(ax, X0 - .22, liny([v], Y0, HT, 0, hi)[0], lab, color=TEAL, size=10.5,
             ha='right')
    ax.plot(xs, liny(err, Y0, HT, 0, hi), color=TEAL_L, lw=3.2, zorder=4)
    ax.plot(xs, liny(pred, Y0, HT, 0, hi), color=AMBER, lw=1.4, ls='--', zorder=5)
    ax.plot(xs, liny([v * hi for v in kept], Y0, HT, 0, hi), color=PURPLE, lw=2.0,
            zorder=4)
    for v in (0.5, 0.9):
        y = liny([v * hi], Y0, HT, 0, hi)[0]
        ax.plot([X0, X0 + WD], [y, y], color='#2f5661', lw=.8, ls=':', zorder=2)
        text(ax, X0 + WD + .18, y, f'{int(v * 100)}%', color=PURPLE, size=10.5, ha='left')
    text(ax, xs[23], liny([0.68 * hi], Y0, HT, 0, hi)[0],
         'Frobenius error of the rank-k truncation', color=TEAL_L, size=11.5, ha='left',
         glowing=False)
    text(ax, xs[23], liny([0.59 * hi], Y0, HT, 0, hi)[0],
         '= sqrt of the discarded tail, exactly', color=AMBER, size=11,
         ha='left', glowing=False)
    text(ax, xs[2] + .10, liny([0.12 * hi], Y0, HT, 0, hi)[0],
         'fraction of the squared norm kept', color=PURPLE, size=11.5, ha='left',
         glowing=False)
    text(ax, X0 + WD + .20, Y0 - .30, 'k', color=TEAL, size=12, ha='left')
    text(ax, 5.0, 5.20, 'truncating the SVD is the best rank-k approximation there is',
         color=TEAL_L, size=13)
    text(ax, 5.0, .42,
         'the dashed line sits exactly on the solid one - on centred data that '
         'identity IS explained variance',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day49_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """A gaussian in the original space, a Student-t on the page."""
    fig, ax = W(5.0)
    X0, Y0, WD, HT = 1.25, 1.35, 6.6, 2.85
    axes2(ax, X0, Y0, WD, HT)
    d = np.linspace(0, 6, 400)
    g = np.exp(-d ** 2 / 2)
    t = 1.0 / (1.0 + d ** 2)
    xs = X0 + WD * d / 6
    ax.plot(xs, liny(g, Y0, HT, 0, 1.05), color=TEAL_L, lw=2.4, zorder=4)
    ax.plot(xs, liny(t, Y0, HT, 0, 1.05), color=AMBER, lw=2.4, zorder=4)
    for v, lab in ((0.0, '0'), (0.5, '0.5'), (1.0, '1')):
        text(ax, X0 - .22, liny([v], Y0, HT, 0, 1.05)[0], lab, color=TEAL, size=11,
             ha='right')
    for v in (0, 1, 2, 3, 4, 5, 6):
        text(ax, X0 + WD * v / 6, Y0 - .30, str(v), color=TEAL, size=11)
    d3 = 3.0
    g3, t3 = float(np.exp(-d3 ** 2 / 2)), float(1.0 / (1.0 + d3 ** 2))
    x3 = X0 + WD * d3 / 6
    ax.plot([x3, x3], [Y0, liny([t3], Y0, HT, 0, 1.05)[0]], color='#2f5661',
            lw=.9, ls=':', zorder=2)
    bx = X0 + WD * 3.15 / 6
    legend(ax, bx, liny([0.88], Y0, HT, 0, 1.05)[0],
           [(TEAL_L, 'gaussian, used for P in the data'),
            (AMBER, 'Student-t, used for Q on the page')], size=11)
    text(ax, bx, liny([0.58], Y0, HT, 0, 1.05)[0],
         f'at distance 3 the Student-t gives {t3:.3f}', color=AMBER, size=11.5,
         ha='left', glowing=False)
    text(ax, bx, liny([0.47], Y0, HT, 0, 1.05)[0],
         f'the gaussian gives {g3:.5f}: {t3 / g3:.0f} times smaller',
         color=TEAL_L, size=11.5, ha='left', glowing=False)
    text(ax, X0 + WD / 2, Y0 - .68, 'distance between two points', color=TEAL, size=11)
    text(ax, 5.0, 5.20, 'the two kernels, and why the heavy tail is the whole trick',
         color=TEAL_L, size=13)
    text(ax, 5.0, .42,
         'a moderately distant pair may sit far apart on the page, so not everything '
         'piles into the middle',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day49_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Cluster width and between-cluster distance do not survive the trip."""
    fig, ax = W(5.2)
    Xs, ys = DR.blobs([0.5, 5.0, 0.5], [[0, 0, 0, 0, 0], [20, 0, 0, 0, 0], [0, 60, 0, 0, 0]],
                      60, seed=1)
    Ys_, _ = DR.tsne(Xs, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    m = [ys == i for i in range(3)]
    o_sp = [DR.spread(Xs[mi]) for mi in m]
    e_sp = [DR.spread(Ys_[mi]) for mi in m]
    cols = [TEAL_L, AMBER, PURPLE]
    for p_, (vals, title, note) in enumerate([
            (o_sp, 'width of each blob in the data',
             f'widest / narrowest = {max(o_sp) / min(o_sp):.2f}'),
            (e_sp, 'width of each blob on the page',
             f'widest / narrowest = {max(e_sp) / min(e_sp):.2f}')]):
        X0, WD = 0.85 + p_ * 4.70, 3.1
        base, maxh = 2.45, 1.9
        ax.plot([X0, X0 + WD], [base, base], color=TEAL, lw=1.2, zorder=2)
        bars(ax, X0 + .35, base, .70, .35, list(vals), cols,
             [f'{v:.2f}' for v in vals], maxh, max(max(o_sp), max(e_sp)))
        for i in range(3):
            text(ax, X0 + .35 + i * 1.05 + .35, base - .34, f'blob {i}', color=GREY_L,
                 size=10.5)
        text(ax, X0 + WD / 2, base + maxh + .80, title, color=TEAL_L, size=12,
             glowing=False)
        text(ax, X0 + WD / 2, base - .78, note, color=AMBER if p_ else TEAL, size=12)
    rows = []
    for far in (30, 60, 120, 400):
        Xf, yf = DR.blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0],
                                            [0, far, 0, 0, 0]], 60, seed=1)
        Yf, _ = DR.tsne(Xf, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
        mf = [yf == i for i in range(3)]
        rows.append((DR.centre_distance(Xf, mf[0], mf[2]) / DR.centre_distance(Xf, mf[0], mf[1]),
                     DR.centre_distance(Yf, mf[0], mf[2]) / DR.centre_distance(Yf, mf[0], mf[1])))
    text(ax, 2.35, 1.28, 'distance ratio in the data', color=TEAL, size=11.5, ha='right',
         glowing=False)
    text(ax, 2.35, 0.90, 'the same ratio on the page', color=RED_S, size=11.5, ha='right',
         glowing=False)
    for i, (ro, re) in enumerate(rows):
        x = 3.15 + i * 1.55
        text(ax, x, 1.28, f'{ro:.2f}', color=TEAL, size=12)
        text(ax, x, 0.90, f'{re:.2f}', color=RED_S, size=12)
    text(ax, 5.0, 5.40, 'how much of the geometry survives t-SNE', color=TEAL_L, size=13)
    text(ax, 5.0, 0.36,
         f'{rows[-1][0]:.0f} times further away in the data, the same distance on the page',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day49_6.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    """The picture shows clusters that are not there, and they move every run."""
    fig, ax = W(5.0)
    Xn = DR.pure_noise(180, 10, seed=7)
    Xb, _ = DR.blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0], [0, 40, 0, 0, 0]],
                     60, seed=1)
    s_raw = DR.silhouette(Xn, DR.kmeans_small(Xn, 3, seed=0))
    pers = [5, 15, 30, 50]
    s_emb = []
    for per in pers:
        Yn, _ = DR.tsne(Xn, perplexity=float(per), n_iter=600, lr=50.0, seed=0)
        s_emb.append(DR.silhouette(Yn, DR.kmeans_small(Yn, 3, seed=0)))
    X0, WD = 0.90, 3.4
    base, maxh = 1.75, 2.3
    ax.plot([X0, X0 + WD], [base, base], color=TEAL, lw=1.2, zorder=2)
    vals = [s_raw] + s_emb
    cols = [TEAL_L] + [RED_S] * 4
    bars(ax, X0 + .25, base, .50, .22, vals, cols, [f'{v:.3f}' for v in vals], maxh, 0.45)
    labs = ['the data'] + [f'p={p_}' for p_ in pers]
    for i, nm in enumerate(labs):
        text(ax, X0 + .25 + i * .72 + .25, base - .34, nm, color=GREY_L, size=10.5)
    text(ax, X0 + WD / 2, base + maxh + .72, 'silhouette: is there any grouping here?',
         color=TEAL_L, size=12, glowing=False)
    text(ax, X0 + WD / 2, base - .76, '180 points of 10-D gaussian noise', color=GREY_L,
         size=11, glowing=False)
    runs = {}
    for nm, data in [('three real blobs', Xb), ('pure 10-D noise', Xn)]:
        Ys = [DR.tsne(data, perplexity=30.0, n_iter=600, lr=50.0, seed=sd)[0]
              for sd in range(3)]
        runs[nm] = float(np.mean([DR.pairwise_corr(Ys[a], Ys[b])
                                  for a, b in ((0, 1), (0, 2), (1, 2))]))
    X1, W1 = 5.55, 3.4
    ax.plot([X1, X1 + W1], [base, base], color=TEAL, lw=1.2, zorder=2)
    bars(ax, X1 + .70, base, 1.0, .70, list(runs.values()), [TEAL_L, RED_S],
         [f'{v:.3f}' for v in runs.values()], maxh, 1.0)
    for i, nm in enumerate(runs):
        text(ax, X1 + .70 + i * 1.70 + .50, base - .34, nm.split()[-1], color=GREY_L,
             size=10.5)
    text(ax, X1 + W1 / 2, base + maxh + .72, 'do three runs agree with each other?',
         color=TEAL_L, size=12, glowing=False)
    text(ax, X1 + W1 / 2, base - .76, 'correlation of the pairwise distances',
         color=GREY_L, size=11, glowing=False)
    text(ax, 5.0, 5.20, 'the method makes groups, so it makes groups', color=TEAL_L,
         size=13)
    _rr = sorted(v / s_raw for v in s_emb)
    text(ax, 5.0, 0.52,
         f'noise scores {_rr[0]:.1f} to {_rr[-1]:.1f} times the data at all four settings',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.16, 'real structure survives a different random seed; invented '
         'structure does not', color=GREY_L, size=11, glowing=False)
    save(fig, 'day49_7.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print(f.__name__, 'ok')
