"""Day 50 figures.  Every number is computed here or by importing evaluation.py."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import evaluation as EV                              # noqa: E402

np.seterr(all='ignore')
os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


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


# ---- the data every figure on this page is scored on, computed once -------
X, Y = EV.imbalanced(2000, 0.01, sep=1.6, d=6, seed=0)
FOLDS = EV.stratified_kfold_indices(Y, 5)
OOF = EV.cv_predictions(X, Y, FOLDS, lambda: EV.LogisticRegression())
MAJ, BASE = EV.majority_baseline(Y)
YB, BROAD, SHARP = EV.broad_and_sharp()


# ---------------------------------------------------------------- figure 1
def fig1():
    """Two tables that look the same; one of them came from a trained model."""
    fig, ax = W(5.0)
    pred = EV.predict_at(OOF, 0.5)
    allneg = np.zeros(len(Y), dtype=int)
    for col, (title, p) in enumerate((('always answer 0', allneg),
                                      ('logistic regression, cut at 0.5', pred))):
        tn, fp, fn, tp = EV.confusion(Y, p)
        x0 = 0.95 + col * 4.75
        y0 = 1.70
        cw, ch = 1.70, 0.95
        text(ax, x0 + cw, y0 + 2 * ch + 0.75, title, color=TEAL_L, size=12.5,
             glowing=False)
        text(ax, x0 + cw * 0.5, y0 + 2 * ch + 0.30, 'guessed 0', color=GREY_L, size=11)
        text(ax, x0 + cw * 1.5, y0 + 2 * ch + 0.30, 'guessed 1', color=GREY_L, size=11)
        text(ax, x0 - 0.14, y0 + ch * 1.5, 'really 0', color=GREY_L, size=11, ha='right')
        text(ax, x0 - 0.14, y0 + ch * 0.5, 'really 1', color=GREY_L, size=11, ha='right')
        grid = [[tn, fp], [fn, tp]]
        for r in range(2):
            for c in range(2):
                v = grid[r][c]
                hot = (r == 1 and c == 1)
                box(ax, x0 + c * cw, y0 + (1 - r) * ch, cw, ch, label=f'{v:,}',
                    edge=AMBER if hot else TEAL, lw=2.0 if hot else 1.4)
        text(ax, x0 + cw, y0 - 0.42, f'accuracy {EV.accuracy(Y, p):.4f}', color=TEAL,
             size=12.5)
        rec = EV.recall(Y, p)
        text(ax, x0 + cw, y0 - 0.86, f'recall {rec:.4f}', color=RED_S, size=12.5)
    text(ax, 5.0, 5.20, '2,000 rows, 20 of them positive', color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'the amber cell is the one the job depends on, and it is the one accuracy '
         'barely notices',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day50_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Precision and recall move in opposite directions as the cut moves."""
    fig, ax = W(5.0)
    cuts = [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01]
    prec, rec, f1s, accs = [], [], [], []
    for t in cuts:
        p = EV.predict_at(OOF, t)
        prec.append(0.0 if np.isnan(EV.precision(Y, p)) else EV.precision(Y, p))
        rec.append(EV.recall(Y, p))
        f1s.append(EV.f1(Y, p))
        accs.append(EV.accuracy(Y, p))
    X0, Y0, WD, HT = 1.25, 1.62, 6.5, 2.80
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(cuts) - 1) for i in range(len(cuts))]
    for i, t in enumerate(cuts):
        text(ax, xs[i], Y0 - 0.30, f'{t:g}', color=TEAL, size=10.5)
    for v, lab in ((0.0, '0'), (0.5, '0.5'), (1.0, '1')):
        text(ax, X0 - 0.22, liny([v], Y0, HT, 0, 1.05)[0], lab, color=TEAL, size=10.5,
             ha='right')
    for vals, col, nm in ((rec, TEAL_L, 'recall'), (prec, AMBER, 'precision'),
                          (f1s, PURPLE, 'F1'), (accs, GREY_L, 'accuracy')):
        ax.plot(xs, liny(vals, Y0, HT, 0, 1.05), color=col, lw=2.4, zorder=4)
    best = int(np.argmax(f1s))
    ax.plot([xs[best], xs[best]], [Y0, Y0 + HT], color='#2f5661', lw=.9, ls=':', zorder=2)
    text(ax, xs[best], Y0 + HT + 0.22, f'best F1 {f1s[best]:.4f}', color=PURPLE, size=11.5)
    legend(ax, X0 + 0.25, liny([0.92], Y0, HT, 0, 1.05)[0],
           [(TEAL_L, 'recall'), (AMBER, 'precision'), (PURPLE, 'F1'),
            (GREY_L, 'accuracy')], size=11)
    text(ax, X0 + WD / 2, Y0 - 0.76, 'the cut applied to the same scores', color=TEAL,
         size=11.5)
    text(ax, 5.0, 5.20, 'one model, one set of scores, nine decisions', color=TEAL_L,
         size=13)
    text(ax, 5.0, 0.33,
         f'accuracy is highest ({max(accs):.4f}) where recall is {rec[0]:.4f}, and falls '
         f'to {accs[best]:.4f} where F1 is best',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day50_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """The same two models, ranked in opposite orders by the two curves."""
    fig, ax = W(5.2)
    for col, (which, title) in enumerate(((0, 'the ROC curve'),
                                          (1, 'the precision-recall curve'))):
        X0, Y0, WD, HT = 0.95 + col * 4.75, 1.50, 3.55, 3.05
        axes2(ax, X0, Y0, WD, HT)
        ax.add_patch(Rectangle((X0, Y0), WD, HT, facecolor='none', edgecolor='#2f5661',
                               lw=.8, zorder=1))
        if which == 0:
            ax.plot([X0, X0 + WD], [Y0, Y0 + HT], color=GREY_L, lw=1.0, ls=':', zorder=2)
        for nm, sc, cc in (('broad', BROAD, TEAL_L), ('sharp', SHARP, AMBER)):
            if which == 0:
                xx, yy = EV.roc_curve(YB, sc)
                val = EV.roc_auc_rank(YB, sc)
            else:
                xx, yy = EV.pr_curve(YB, sc)
                val = EV.average_precision(YB, sc)
            ax.plot([X0 + WD * v for v in xx], [Y0 + HT * v for v in yy], color=cc,
                    lw=2.4, zorder=4)
            ly = (0.22, 0.10) if which == 0 else (0.80, 0.68)
            text(ax, X0 + WD * 0.97, Y0 + HT * (ly[0] if nm == 'broad' else ly[1]),
                 f'{nm} {val:.4f}', color=cc, size=12, ha='right', glowing=False)
        text(ax, X0 + WD / 2, Y0 + HT + 0.52, title, color=TEAL_L, size=12,
             glowing=False)
        text(ax, X0, Y0 + HT + 0.16, 'recall' if which == 0 else 'precision',
             color=GREY_L, size=10.5, ha='left', glowing=False)
        text(ax, X0 + WD / 2, Y0 - 0.34,
             'false positive rate' if which == 0 else 'recall', color=GREY_L, size=11,
             glowing=False)
    text(ax, 5.0, 5.60, 'ROC-AUC prefers broad; average precision prefers sharp',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.48,
         f'{int(YB.sum())} positives in {len(YB):,} rows: the ROC x-axis divides by '
         f'{int((1 - YB).sum()):,} negatives, precision divides by what you flagged',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day50_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """The question the two curves were standing in for: what is in the top k?"""
    fig, ax = W(4.8)
    ks = (10, 20, 50, 100)
    base, maxh, bw, gap = 1.45, 2.25, 0.52, 0.22
    hits_b = [EV.precision_at_k(YB, BROAD, k)[0] for k in ks]
    hits_s = [EV.precision_at_k(YB, SHARP, k)[0] for k in ks]
    top = max(hits_b + hits_s)
    for i, k in enumerate(ks):
        x = 1.35 + i * 1.85
        bars(ax, x, base, bw, gap, [hits_b[i], hits_s[i]], [TEAL_L, AMBER],
             [str(hits_b[i]), str(hits_s[i])], maxh, top)
        text(ax, x + bw + gap / 2, base - 0.36, f'top {k}', color=GREY_L, size=11.5)
    legend(ax, 1.30, base + maxh + 0.62,
           [(TEAL_L, f'broad  (ROC-AUC {EV.roc_auc_rank(YB, BROAD):.4f})'),
            (AMBER, f'sharp  (ROC-AUC {EV.roc_auc_rank(YB, SHARP):.4f})')], size=11.5)
    text(ax, 5.0, 5.15, 'how many of the 20 positives are in the k highest scores',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.52,
         f'review ten rows a day and sharp hands you {hits_s[0]} real cases, broad hands '
         f'you {hits_b[0]}',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.18, 'the model ROC-AUC calls much worse is the one worth deploying '
         'at this budget', color=GREY_L, size=11, glowing=False)
    save(fig, 'day50_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """Cut by row number and a fold can end up with no positives to score."""
    fig, ax = W(4.6)
    plain = [int(Y[f].sum()) for f in EV.kfold_indices(len(Y), 10, seed=0)]
    strat = [int(Y[f].sum()) for f in EV.stratified_kfold_indices(Y, 10)]
    cw, ch = 0.72, 0.78
    for row, (nm, counts) in enumerate((('shuffle, then cut', plain),
                                        ('cut each class', strat))):
        y0 = 2.95 - row * 1.35
        text(ax, 1.05, y0 + ch / 2, nm, color=TEAL_L, size=12, ha='right', glowing=False)
        for i, c in enumerate(counts):
            empty = (c == 0)
            box(ax, 1.25 + i * (cw + 0.10), y0, cw, ch, label=str(c),
                edge=RED_S if empty else TEAL, lw=2.2 if empty else 1.4)
    empty_runs = 0
    for seed in range(200):
        if min(int(Y[f].sum()) for f in EV.kfold_indices(len(Y), 10, seed=seed)) == 0:
            empty_runs += 1
    text(ax, 5.0, 4.55, 'positives in each of 10 folds, from the same 2,000 rows',
         color=TEAL_L, size=13)
    text(ax, 5.0, 1.15,
         f'a fold with 0 positives makes recall 0/0; over 200 shuffles {empty_runs} of them '
         f'({100 * empty_runs / 200:.0f}%) produce at least one',
         color=RED_S, size=11.5, glowing=False)
    text(ax, 5.0, 0.72, 'splitting each class separately costs one line, and works '
         'whenever there are at least k positives', color=TEAL, size=11.5, glowing=False)
    save(fig, 'day50_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Three ways the test rows can reach the model, and how much each one costs."""
    fig, ax = W(4.8)
    Xg, yg, g = EV.grouped_rows()
    n_g = len(yg)
    def knn_cv(fl):
        out = []
        for f in fl:
            tr = np.setdiff1d(np.arange(n_g), f)
            out.append(float(np.mean(EV.one_nearest_neighbour(Xg[tr], yg[tr], Xg[f]) == yg[f])))
        return float(np.mean(out))
    leak_a = knn_cv(EV.kfold_indices(n_g, 5, seed=0))
    honest_a = knn_cv(EV.group_kfold_indices(g, 5))
    out_s, in_s = [], []
    for seed in range(20):
        Xn, yn = EV.pure_noise_table(60, 400, seed=seed)
        fl = EV.stratified_kfold_indices(yn, 5)
        cols_all = EV.select_by_correlation(Xn, yn, 5)
        a_, b_ = [], []
        for f in fl:
            tr = np.setdiff1d(np.arange(60), f)
            a_.append(EV.nearest_centroid_accuracy(Xn[tr][:, cols_all], yn[tr],
                                                   Xn[f][:, cols_all], yn[f]))
            c2 = EV.select_by_correlation(Xn[tr], yn[tr], 5)
            b_.append(EV.nearest_centroid_accuracy(Xn[tr][:, c2], yn[tr],
                                                   Xn[f][:, c2], yn[f]))
        out_s.append(np.mean(a_))
        in_s.append(np.mean(b_))
    Xs, ys = EV.imbalanced(60, 0.3, sep=1.2, d=8, seed=5)
    fl = EV.stratified_kfold_indices(ys, 5)
    ins = EV.cv_predictions(Xs, ys, fl, lambda: EV.LogisticRegression(), scale_inside=True)
    outs = EV.cv_predictions(Xs, ys, fl, lambda: EV.LogisticRegression(), scale_inside=False)
    leak_c = EV.accuracy(ys, EV.predict_at(outs, 0.5))
    honest_c = EV.accuracy(ys, EV.predict_at(ins, 0.5))
    groups = [('the same customer\nin both halves', leak_a, honest_a),
              ('columns picked\nbefore the split', float(np.mean(out_s)), float(np.mean(in_s))),
              ('scaled using\nevery row', leak_c, honest_c)]
    base, maxh, bw, gap = 1.70, 1.95, 0.60, 0.26
    for i, (nm, bad_v, good_v) in enumerate(groups):
        x = 1.45 + i * 2.60
        bars(ax, x, base, bw, gap, [bad_v, good_v], [RED_S, TEAL_L],
             [f'{bad_v:.4f}', f'{good_v:.4f}'], maxh, 1.0)
        for j, line in enumerate(nm.split('\n')):
            text(ax, x + bw + gap / 2, base - 0.36 - j * 0.34, line, color=GREY_L,
                 size=11, glowing=False)
    y50 = base + maxh * 0.5
    ax.plot([1.25, 8.75], [y50, y50], color='#2f5661', lw=.9, ls=':', zorder=2)
    text(ax, 8.95, y50, '0.5', color=GREY_L, size=10.5, ha='left')
    legend(ax, 1.30, base + maxh + 1.08,
           [(RED_S, 'what the leaking pipeline reports'),
            (TEAL_L, 'what the same model is really worth')], size=11.5)
    text(ax, 5.0, 5.15, 'three kinds of leakage, same measurement both times',
         color=TEAL_L, size=13)
    gap = float(np.abs(ins - outs).max())
    text(ax, 5.0, 0.42,
         f'they are not equally bad: two of them turn noise into a result, the third '
         f'moves a score by at most {gap:.1e}',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day50_6.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    """Choose the best of m models on one set and the number you report goes up."""
    fig, ax = W(4.8)
    ms = (1, 10, 50, 200)
    picked, fresh = [], []
    for m in ms:
        a_, b_ = EV.best_of_m_on_one_set(m, n=200, reps=40, seed=1000)
        picked.append(a_)
        fresh.append(b_)
    X0, Y0, WD, HT = 1.45, 1.55, 6.2, 2.60
    axes2(ax, X0, Y0, WD, HT)
    lo, hi = 0.45, 0.63
    xs = [X0 + WD * i / (len(ms) - 1) for i in range(len(ms))]
    for i, m in enumerate(ms):
        text(ax, xs[i], Y0 - 0.32, str(m), color=TEAL, size=11.5)
    for v in (0.45, 0.50, 0.55, 0.60):
        yy = liny([v], Y0, HT, lo, hi)[0]
        ax.plot([X0, X0 + WD], [yy, yy], color='#2f5661', lw=.7, ls=':', zorder=1)
        text(ax, X0 - 0.22, yy, f'{v:.2f}', color=TEAL, size=10.5, ha='right')
    for vals, col in ((picked, RED_S), (fresh, TEAL_L)):
        ys_ = liny(vals, Y0, HT, lo, hi)
        ax.plot(xs, ys_, color=col, lw=2.6, zorder=4)
        ax.scatter(xs, ys_, s=26, color=col, zorder=5, linewidths=0)
    text(ax, xs[-1] - 0.10, liny([picked[-1]], Y0, HT, lo, hi)[0] + 0.30,
         f'{picked[-1]:.4f} on the set you chose on', color=RED_S, size=11.5, ha='right',
         glowing=False)
    text(ax, xs[-1] - 0.10, liny([fresh[-1]], Y0, HT, lo, hi)[0] - 0.32,
         f'{fresh[-1]:.4f} on a fresh set', color=TEAL_L, size=11.5, ha='right',
         glowing=False)
    text(ax, X0 + WD / 2, Y0 - 0.74, 'how many models you tried before picking one',
         color=TEAL, size=11.5)
    text(ax, 5.0, 5.15, 'every model here is a coin flip, and so is every label',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'nothing improved; the maximum of m noisy measurements just grows with m',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day50_7.png')


# ---------------------------------------------------------------- figure 8
def fig8():
    """Four ratios, four different denominators, on one basket of fruit."""
    fig, ax = W(5.2)
    y, p = EV.apples_and_oranges()
    tn, fp, fn, tp = EV.confusion(y, p)
    X0, Y0, CW, CH = 1.75, 1.95, 2.95, 1.05
    # the four cells: columns are what the model said, rows are what is true
    cells_ = [[(tp, 'apples it caught', AMBER), (fn, 'apples it missed', RED_S)],
              [(fp, 'oranges it grabbed', RED_S), (tn, 'oranges it left', TEAL)]]
    text(ax, X0 + CW * .5, Y0 + 2 * CH + .34, 'called APPLE', color=TEAL_L, size=12.5)
    text(ax, X0 + CW * 1.5, Y0 + 2 * CH + .34, 'called ORANGE', color=TEAL_L, size=12.5)
    text(ax, X0 - .18, Y0 + CH * 1.5, 'really\napple', color=TEAL_L, size=12, ha='right')
    text(ax, X0 - .18, Y0 + CH * .5, 'really\norange', color=TEAL_L, size=12, ha='right')
    for r in range(2):
        for c in range(2):
            n, lab, col = cells_[r][c]
            box(ax, X0 + c * CW, Y0 + (1 - r) * CH, CW, CH, label=f'{n}', edge=col, lw=2.0)
            text(ax, X0 + c * CW + CW / 2, Y0 + (1 - r) * CH + .22, lab, color=col,
                 size=10.5, glowing=False)
    # precision divides by the whole left column - everything called APPLE
    region(ax, X0 - .09, Y0 - .09, CW + .18, 2 * CH + .18, label=None)
    text(ax, X0 + CW / 2, Y0 + 2 * CH + .80,
         f'precision = {tp}/{tp + fp} = {EV.precision(y, p):.2f}', color=AMBER, size=12.5)
    text(ax, X0 + 2 * CW + .22, Y0 + CH * 1.5,
         f'recall = {tp}/{tp + fn} = {EV.recall(y, p):.2f}', color=CYAN, size=12.5, ha='left')
    text(ax, X0 + 2 * CW + .22, Y0 + CH * .5,
         f'FPR = {fp}/{fp + tn} = {EV.false_positive_rate(y, p):.3f}', color=RED_S,
         size=12.5, ha='left')
    text(ax, X0 + CW, Y0 - .42,
         f'accuracy = ({tp} + {tn})/100 = {EV.accuracy(y, p):.2f}', color=GREY_L, size=12.5)
    text(ax, 5.0, 5.55, '100 pieces of fruit, 10 of them apples, one apple detector',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.86, 'preci-SI-on asks about the SIDE you picked; rec-ALL asks about '
         'ALL the apples', color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.46, 'recall and FPR each divide inside one class; accuracy divides by '
         'the whole basket, which the data fixes', color=GREY_L, size=11, glowing=False)
    save(fig, 'day50_8.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
        print(f.__name__, 'ok')
