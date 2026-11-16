"""Figures for Day 59 - every number is computed by importing sampling.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import sampling as S                   # noqa: E402

M = S.NGram(S.make_corpus())


def bars(ax, x0, y0, probs, labels, w=0.46, gap=0.14, hmax=1.5, keep=None,
         color=TEAL, cut=None):
    """A little probability bar chart; `keep` marks the surviving tokens."""
    for i, (p, lab) in enumerate(zip(probs, labels)):
        x = x0 + i * (w + gap)
        h = max(p * hmax, 0.012)
        on = keep is None or keep[i]
        box(ax, x, y0, w, h, fill=NAVY if on else GREY,
            edge=TEAL if on else GREY_L, lw=1.5)
        text(ax, x + w / 2, y0 - 0.26, lab, size=12,
             color=TEAL if on else GREY_L, glowing=False)
    if cut is not None:
        ax.plot([x0 - 0.1, x0 + len(probs) * (w + gap)],
                [y0 + cut * hmax] * 2, color=AMBER, lw=1.6, ls='--', zorder=5)


def fig1():
    """The pipeline, with the distribution drawn at three points along it."""
    ctx = '\nthe '                      # a step where the model is genuinely unsure
    lg = M.logits(ctx)
    order = np.argsort(-lg)[:7]
    labs = [{' ': '_', '\n': 'NL'}.get(M.vocab[i], M.vocab[i]) for i in order]
    p0 = S.softmax(lg)
    p1 = S.softmax(S.temperature(lg, 1.6))
    p2 = S.top_p(p1, 0.9)

    fig, ax = canvas(10.4, 6.4, xlim=(0, 10.4), ylim=(0, 6.4))
    text(ax, 5.2, 6.1, 'one decode step: a vector of logits in, one token out',
         size=16)

    steps = ['logits', 'penalties', 'temperature', 'top-k', 'top-p', 'renormalise',
             'draw']
    for i, name in enumerate(steps):
        y = 5.2 - i * 0.72
        box(ax, 0.35, y - 0.26, 2.1, 0.52, label=name, size=13,
            fill=NAVY if i not in (6,) else GREY, edge=TEAL)
        if i:
            arrow(ax, (1.4, y + 0.46), (1.4, y + 0.27), lw=1.5)

    panels = [(p0, None, 'raw', 0.0), (p1, None, 'after / t = 1.6', 2.16),
              (p2, p2[order] > 0, 'after top-p 0.9', 4.32)]
    for pr, keep, title, dy in panels:
        y = 4.55 - dy
        text(ax, 6.6, y + 1.28, title, size=13, color=TEAL_L)
        bars(ax, 4.5, y, pr[order], labs, keep=keep, hmax=2.3, w=0.42, gap=0.17)
        kept = len(pr) if keep is None else int((pr > 0).sum())
        text(ax, 9.9, y + 0.5, f'{kept} live\ntokens', size=12, color=AMBER,
             glowing=False, ha='right')
        text(ax, 9.9, y + 1.28, f'entropy {S.entropy(pr):.2f} bits', size=12,
             color=AMBER, glowing=False, ha='right')
    save(fig, os.path.join(HERE, 'day59_1.png'))


def fig2():
    """Temperature buys variety and pays in correctness."""
    rows = S.temperature_table(M)
    fig, ax = canvas(9.8, 5.6, xlim=(0, 9.8), ylim=(0, 5.6))
    text(ax, 4.9, 5.35, 'temperature: variety bought, correctness paid', size=16)
    x0, gw, bw, hmax = 0.9, 1.22, 0.42, 2.75
    for i, r in enumerate(rows):
        x = x0 + i * gw
        box(ax, x, 1.1, bw, max(r['grammatical'] * hmax, 0.02), fill=NAVY, edge=TEAL)
        box(ax, x + bw + 0.1, 1.1, bw, max(r['distinct'] / 100 * hmax, 0.02),
            fill=AMBER, edge=AMBER)
        text(ax, x + bw, 0.78, f"t={r['t']:g}", size=12, glowing=False)
        text(ax, x + bw / 2, 1.15 + r['grammatical'] * hmax + 0.18,
             f"{r['grammatical']:.0%}", size=11, color=TEAL, glowing=False)
        text(ax, x + 1.5 * bw + 0.1, 1.15 + r['distinct'] / 100 * hmax + 0.18,
             f"{r['distinct']}", size=11, color=AMBER, glowing=False)
    legend(ax, 1.1, 4.85, [(TEAL, 'grammatical, % of 100 samples'),
                           (AMBER, 'distinct sentences, out of 1250 legal ones')])
    text(ax, 4.9, 0.35,
         'cold repeats itself, hot stops making sense, and the fall is sudden',
         size=12, color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day59_2.png'))


def fig3():
    """Same two rules, two shapes of step."""
    peaked_ctx, flat_ctx = 'the small cat sa', '\nthe '
    fig, ax = canvas(10.2, 5.2, xlim=(0, 10.2), ylim=(0, 5.2))
    text(ax, 5.1, 4.95, 'a count (top-k) and a mass (top-p) on two kinds of step',
         size=15)
    for col, (ctx, name) in enumerate([(peaked_ctx, 'model is certain'),
                                       (flat_ctx, 'model is unsure')]):
        p = S.softmax(M.logits(ctx))
        order = np.argsort(-p)[:7]
        labs = [M.vocab[i] if M.vocab[i] not in ' \n' else ('_' if M.vocab[i] == ' '
                else '\\n') for i in order]
        pr = p[order]
        nuc = S.nucleus_size(p, 0.9)
        x0 = 0.9 + col * 4.95
        text(ax, x0 + 1.7, 4.45, name, size=14, color=TEAL_L)
        bars(ax, x0, 2.5, pr, labs, keep=[i < nuc for i in range(len(pr))], hmax=1.7)
        text(ax, x0 + 1.8, 1.75, f'top-p 0.9 keeps {nuc}', size=13, color=CYAN)
        kept3 = float(np.sort(p)[::-1][:3].sum())
        text(ax, x0 + 1.8, 1.25, f'top-k 3 keeps 3, mass {kept3:.3f}', size=12,
             color=AMBER, glowing=False)
        mp = int((p >= 0.1 * p.max()).sum())
        text(ax, x0 + 1.8, 0.8, f'min-p 0.1 keeps {mp}', size=12, color=TEAL,
             glowing=False)
    legend(ax, 0.5, 0.35, [(TEAL, 'kept by the nucleus'), (GREY_L, 'cut')])
    save(fig, os.path.join(HERE, 'day59_3.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3()
    print('wrote day59_1.png day59_2.png day59_3.png')
