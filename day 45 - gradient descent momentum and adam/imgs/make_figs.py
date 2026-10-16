"""Day 45 figures.  Every number is computed by importing optimizers.py."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                # noqa: E402
from diagram_style import *                       # noqa: F401,F403,E402
import optimizers as op                           # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


# ---------------------------------------------------------------- figure 1
def fig1():
    """An ill-conditioned bowl: gradient descent bounces across the ravine."""
    q = op.Quadratic()
    w0 = np.array([4.0, 1.0])
    lr = 0.09
    gd = op.run(op.SGD(lr=lr), q, w0, 25)
    fig, ax = W(4.4)
    S, CX, CY = 0.72, 5.2, 2.44
    X = lambda a: CX + a * S
    Y = lambda b: CY + b * S
    for f in (18.0, 9.0, 4.0, 1.4, 0.35):
        t = np.linspace(0, 2 * np.pi, 400)
        a0, a1 = math.sqrt(2 * f / q.curv[0]), math.sqrt(2 * f / q.curv[1])
        ax.plot(X(a0 * np.cos(t)), Y(a1 * np.sin(t)), color=TEAL, lw=1.0, alpha=0.55, zorder=2)
    ax.plot([X(p[0]) for p in gd], [Y(p[1]) for p in gd], color=AMBER, lw=1.8,
            marker='o', ms=3.2, zorder=4)
    node(ax, X(0), Y(0), '', r=0.08, fill=WHITE, edge=WHITE)
    text(ax, X(0), Y(0) - 1.35, 'minimum', color=TEAL_L, size=12)
    text(ax, X(4), Y(1) + 0.38, 'start', color=AMBER, size=12)
    text(ax, 5.0, 4.6, f'f(w) = 0.5 (w0² + 20 w1²): 20x more curved across than along',
         color=TEAL_L, size=12, glowing=False)
    text(ax, 5.0, 0.62, f'gradient descent at lr {lr}: the steep direction overshoots '
                        f'every step (multiplier {1 - lr * q.lmax:+.2f}),',
         color=AMBER, size=11, glowing=False)
    text(ax, 5.0, 0.24, f'while the flat direction only shrinks by {1 - lr * q.lmin:.2f} '
                        f'per step — {op.steps_to(op.SGD(lr=lr), q, w0)} steps in total',
         color=AMBER, size=11, glowing=False)
    save(fig, 'day45_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """The error shrinks by a fixed factor per step; momentum changes the factor."""
    q = op.Quadratic()
    w0 = np.array([1.0, 1.0])
    N = 150
    gd = op.run(op.SGD(lr=q.lr_best), q, w0, N)
    mo = op.run(op.SGD(lr=q.lr_momentum_best, momentum=q.beta_best), q, w0, N)
    start = float(np.linalg.norm(w0))
    fig, ax = W(4.8)
    X0, X1, Y0, Y1 = 1.5, 9.2, 0.9, 4.4
    px = lambda k: X0 + (X1 - X0) * k / N
    py = lambda e: Y0 + (Y1 - Y0) * (math.log10(max(e, 1e-8)) + 8) / 8.0
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for k in (0, 50, 100, 150):
        ax.plot([px(k)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, px(k), Y0 - 0.32, str(k), color=TEAL, size=11, glowing=False)
    for e in (1e-6, 1e-4, 1e-2, 1):
        ax.plot([X0, X0 - 0.1], [py(e)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.2, py(e), f'{e:g}', color=TEAL, size=11, ha='right', glowing=False)
    for traj, col, nm, rate in ((gd, AMBER, 'plain GD', q.rate_gd),
                                (mo, CYAN, f'momentum {q.beta_best:.2f}', q.rate_momentum)):
        errs = [float(np.linalg.norm(p)) / start for p in traj]
        keep = [(k, e) for k, e in enumerate(errs) if e > 3e-7]   # stop at the floor
        ax.plot([px(k) for k, _ in keep], [py(e) for _, e in keep],
                color=col, lw=1.9, zorder=3)
        ks = [k for k in range(N) if 1e-7 < rate ** k < 1]
        ax.plot([px(k) for k in ks], [py(rate ** k) for k in ks], color=col, lw=1.0,
                ls=(0, (3, 3)), zorder=2)
    ax.plot([X0, X1], [py(1e-6)] * 2, color=TEAL, lw=1.0, ls=(0, (4, 3)))
    text(ax, px(96), py(6e-2), f'plain GD — {op.steps_to(op.SGD(lr=q.lr_best), q, w0)} steps\n'
         f'rate {q.rate_gd:.3f} per step', color=AMBER, size=11, glowing=False)
    text(ax, px(62), py(8e-6), f'momentum — '
         f'{op.steps_to(op.SGD(lr=q.lr_momentum_best, momentum=q.beta_best), q, w0)} steps\n'
         f'rate {q.rate_momentum:.3f} per step', color=CYAN, size=11, glowing=False)
    text(ax, (X0 + X1) / 2, 0.2, 'steps', color=TEAL, size=12, glowing=False)
    text(ax, 0.15, Y1 + 0.38, 'distance to the minimum, as a fraction of the start '
                              '(dashed = theory)', color=TEAL_L, size=12, ha='left', glowing=False)
    save(fig, 'day45_2.png')


# ---------------------------------------------------------------- figure 2
def fig3():
    """Every coordinate is multiplied by |1 - lr*curvature|; the ceiling is 2/lmax."""
    q = op.Quadratic()
    fig, ax = W(5.0)
    X0, X1, Y0, Y1 = 1.3, 9.3, 0.9, 4.9
    lrs = np.linspace(0, 0.16, 400)
    px = lambda lr: X0 + (X1 - X0) * lr / 0.16
    py = lambda m: Y0 + (Y1 - Y0) * min(m, 2.4) / 2.4
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for lr in (0.0, 0.04, 0.08, 0.12, 0.16):
        ax.plot([px(lr)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, px(lr), Y0 - 0.32, f'{lr:g}', color=TEAL, size=11, glowing=False)
    for m in (0, 1, 2):
        ax.plot([X0, X0 - 0.1], [py(m)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.2, py(m), f'{m}', color=TEAL, size=11, ha='right', glowing=False)
    for curv, col, lab in ((1.0, GREY_L, 'along w0 (curvature 1)'),
                           (20.0, CYAN, 'along w1 (curvature 20)')):
        ax.plot([px(l) for l in lrs], [py(abs(1 - l * curv)) for l in lrs],
                color=col, lw=2.0, zorder=3)
    ax.plot([X0, X1], [py(1)] * 2, color=AMBER, lw=1.4, ls=(0, (4, 3)), zorder=2)
    text(ax, X1, py(1) + 0.22, 'multiplier 1: errors stop shrinking',
         color=AMBER, size=11, ha='right', glowing=False)
    ax.plot([px(q.lr_divergence)] * 2, [Y0, Y1], color=RED, lw=1.6, ls=(0, (4, 3)), zorder=3)
    text(ax, px(q.lr_divergence) + 0.12, Y1 - 0.25,
         f'lr = 2/20 = {q.lr_divergence:g}', color=RED, size=12, ha='left', glowing=False)
    text(ax, px(0.05), py(1.9), 'converges', color=TEAL_L, size=13, glowing=False)
    text(ax, px(0.135), py(1.9), 'diverges', color=RED, size=13, glowing=False)
    text(ax, px(0.152), py(0.95), 'along w0 (curvature 1)', color=GREY_L, size=11,
         ha='right', glowing=False)
    text(ax, px(0.022), py(0.28), 'along w1 (curvature 20)', color=CYAN, size=11,
         ha='left', glowing=False)
    text(ax, (X0 + X1) / 2, 0.18, 'learning rate', color=TEAL, size=12, glowing=False)
    text(ax, 0.15, Y1 + 0.3, 'error multiplier per step  |1 − lr × curvature|',
         color=TEAL_L, size=12, ha='left', glowing=False)
    save(fig, 'day45_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Steps to reach the same loss on a real least-squares problem, lr tuned per method."""
    ls = op.LeastSquares()
    z = np.zeros(3)
    res = [(name, *op.best_lr_loss(make, ls, z)) for name, make in op.FAMILIES]
    fig, ax = W(4.6)
    top = max(k for _, _, k in res)
    X0, BW = 3.4, 3.9
    for i, (name, lr, k) in enumerate(res):
        y = 4.1 - i * 0.62
        w = BW * math.log10(k) / math.log10(top)
        ax.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle(
            (X0, y - 0.19), w, 0.38, facecolor=NAVY,
            edgecolor=AMBER if k == min(x[2] for x in res) else TEAL, linewidth=1.8, zorder=3))
        text(ax, X0 - 0.15, y, name, color=TEAL_L, size=12, ha='right', glowing=False)
        text(ax, X0 + w + 0.18, y, f'{k} steps   (best lr {lr:.4f})', color=TEAL_L,
             size=11, ha='left', glowing=False)
    text(ax, 5.0, 4.75, f'least squares, {ls.n} rows, condition number {ls.kappa:.0f}:'
                        f'  full-batch steps until the excess loss < 1e-8',
         color=TEAL_L, size=12, glowing=False)
    text(ax, 5.0, 0.22, 'bar length is logarithmic; each optimiser got its own tuned '
                        'learning rate', color=TEAL, size=11, glowing=False)
    save(fig, 'day45_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """Adam without bias correction steps too far, and stays wrong for a long time."""
    fig, ax = W(4.6)
    X0, X1, Y0, Y1 = 1.4, 9.2, 0.9, 4.3
    ts = np.unique(np.round(np.logspace(0, 4, 300)).astype(int))
    px = lambda t: X0 + (X1 - X0) * math.log10(t) / 4
    py = lambda f: Y0 + (Y1 - Y0) * (f - 1) / 6.0
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for t in (1, 10, 100, 1000, 10000):
        ax.plot([px(t)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, px(t), Y0 - 0.32, f'{t:,}', color=TEAL, size=11, glowing=False)
    for f in (1, 3, 5, 7):
        ax.plot([X0, X0 - 0.1], [py(f)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.2, py(f), f'{f}x', color=TEAL, size=11, ha='right', glowing=False)
    ax.plot([px(t) for t in ts], [py(op.bias_factor(t)) for t in ts], color=CYAN, lw=2.2, zorder=3)
    ax.plot([X0, X1], [py(1)] * 2, color=TEAL, lw=1.2, ls=(0, (4, 3)))
    peak = max(range(1, 2001), key=op.bias_factor)
    node(ax, px(peak), py(op.bias_factor(peak)), '', r=0.08, fill=AMBER, edge=AMBER)
    text(ax, px(peak) + 0.15, py(op.bias_factor(peak)) + 0.05,
         f'worst at step {peak}: {op.bias_factor(peak):.2f}x too big',
         color=AMBER, size=12, ha='left', glowing=False)
    text(ax, px(900), py(2.3), f'still {op.bias_factor(1000):.2f}x at step 1,000',
         color=TEAL_L, size=11, ha='center', glowing=False)
    ax.plot([px(1000), px(1000)], [py(op.bias_factor(1000)) + 0.08, py(2.1)],
            color=TEAL_L, lw=0.9, zorder=2)
    text(ax, (X0 + X1) / 2, 0.2, 'step number t (log scale)', color=TEAL, size=12, glowing=False)
    text(ax, 0.15, Y1 + 0.42, 'un-corrected step ÷ corrected step  =  (1 − 0.9ᵗ) / √(1 − 0.999ᵗ)',
         color=TEAL_L, size=12, ha='left', glowing=False)
    save(fig, 'day45_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Adam + L2 divides the weight decay by sqrt(v); AdamW does not."""
    ls = op.LeastSquares()
    z = np.zeros(3)
    opt = op.Adam(lr=0.05)
    w = z.copy()
    for _ in range(400):
        w = opt.step(w, ls.grad(w))
    rv = np.sqrt(opt.v)                      # the denominator Adam divides by
    rel = (1.0 / rv) / np.mean(1.0 / rv)     # Adam + L2 decay strength, average = 1
    fig, ax = W(4.8)
    from matplotlib.patches import Rectangle
    X0, GH, UNIT = 3.5, 1.3, 2.3
    for j_, (scale, r) in enumerate(zip((1.0, 1.0, 8.0), rel)):
        y = 4.1 - j_ * GH
        text(ax, X0 - 0.2, y + 0.02, f'weight {j_}', color=TEAL_L, size=13,
             ha='right', glowing=False)
        text(ax, X0 - 0.2, y - 0.38, f'feature scale {scale:g},  √v = {rv[j_]:.2f}',
             color=TEAL, size=10, ha='right', glowing=False)
        ax.add_patch(Rectangle((X0, y + 0.06), UNIT, 0.3, facecolor=NAVY,
                               edgecolor=AMBER, linewidth=1.6, zorder=3))
        text(ax, X0 + UNIT + 0.15, y + 0.21, 'AdamW  1.00x', color=AMBER, size=11,
             ha='left', glowing=False)
        ax.add_patch(Rectangle((X0, y - 0.38), max(UNIT * r, 0.03), 0.3, facecolor=NAVY,
                               edgecolor=RED, linewidth=1.6, zorder=3))
        text(ax, X0 + max(UNIT * r, 0.03) + 0.15, y - 0.23,
             f'Adam + L2  {r:.2f}x', color=RED, size=11, ha='left', glowing=False)
    text(ax, 5.0, 5.0, 'how hard the weight decay actually pulls on each weight',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 5.0, 0.52, f'L2 puts wd·w into the gradient, so Adam divides it by √v: the '
                        f'weight with the', color=RED, size=11, glowing=False)
    text(ax, 5.0, 0.14, f'largest gradients is decayed {rel[0] / rel[2]:.0f}x less than '
                        f'weight 0 — backwards. AdamW subtracts lr·wd·w from w directly.',
         color=RED, size=11, glowing=False)
    save(fig, 'day45_6.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    """Small batches get further per epoch, then sit on a noise floor."""
    ls = op.LeastSquares()
    z = np.zeros(3)
    EP = 300
    series = [(1, '#ff5c5c'), (8, CYAN), (ls.n, TEAL_L)]
    hists = [(b, op.sgd_epochs(op.SGD(lr=0.002), ls, z, batch=b, epochs=EP)[1], c)
             for b, c in series]
    fig, ax = W(4.8)
    X0, X1, Y0, Y1 = 1.6, 8.6, 0.9, 4.4
    px = lambda e: X0 + (X1 - X0) * math.log10(max(e, 1)) / math.log10(EP)
    py = lambda v: Y0 + (Y1 - Y0) * (math.log10(max(v, 1e-5)) + 5) / 6.3
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for e in (1, 10, 100, 300):
        ax.plot([px(e)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, px(e), Y0 - 0.32, str(e), color=TEAL, size=11, glowing=False)
    for v in (1e-4, 1e-2, 1, 10):
        ax.plot([X0, X0 - 0.1], [py(v)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.2, py(v), f'{v:g}', color=TEAL, size=11, ha='right', glowing=False)
    for b, hist, col in hists:
        xs = [px(i + 1) for i in range(len(hist))]
        ax.plot(xs, [py(v) for v in hist], color=col, lw=0.8, alpha=0.3, zorder=2)
        med = [float(np.median(hist[max(0, i - 7):i + 8])) for i in range(len(hist))]
        ax.plot(xs, [py(v) for v in med], color=col, lw=2.0, zorder=3)
        lab = 'full batch' if b == ls.n else f'batch {b}'
        text(ax, xs[-1] + 0.15, py(med[-1]), lab, color=col, size=11, ha='left', glowing=False)
    floor = float(np.median(hists[1][1][150:]))
    ax.plot([px(12), X1], [py(floor)] * 2, color=CYAN, lw=1.0, ls=(0, (4, 3)), zorder=2)
    text(ax, px(26), py(0.22), f'batch 8 settles on a noise floor of {floor:.0e}',
         color=CYAN, size=11, glowing=False)
    ax.plot([px(26), px(26)], [py(0.13), py(floor) + 0.06], color=CYAN, lw=0.9, zorder=2)
    text(ax, (X0 + X1) / 2, 0.2, 'epochs (passes over the data, log scale)',
         color=TEAL, size=12, glowing=False)
    text(ax, 0.15, Y1 + 0.38, 'excess loss over the exact optimum,  SGD at lr 0.002   '
                              '(pale = raw, bold = 15-epoch median)',
         color=TEAL_L, size=12, ha='left', glowing=False)
    save(fig, 'day45_7.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print('figures written')
