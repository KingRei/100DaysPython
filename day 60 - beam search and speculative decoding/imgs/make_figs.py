"""Figures for Day 60 - every number is computed by importing decoding.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import decoding as D                   # noqa: E402

M = D.NGram(D.make_corpus())


def fig1():
    """The search: one beam of width 2 keeping two prefixes alive."""
    ctx = '\nthe old robot s'
    fig, ax = canvas(10.4, 5.8, xlim=(0, 10.4), ylim=(0, 5.8))
    text(ax, 5.2, 5.5, 'beam search: keep the best w prefixes, not the best token',
         size=16)
    width = 2
    beams = [(0.0, ctx)]
    for step in range(3):
        x = 1.0 + step * 3.0
        cand = []
        for lp, t in beams:
            lg = np.log(M.probs(t))
            for i in np.argsort(-lg, kind='stable')[:3]:
                cand.append((lp + float(lg[i]), t + M.vocab[i], M.vocab[i], t))
        cand.sort(key=lambda c: -c[0])
        kept = cand[:width]
        for j, c in enumerate(cand[:4]):
            y = 4.4 - j * 1.0
            keep = c in kept
            box(ax, x, y, 2.1, 0.62,
                label=repr(c[1][-6:]).strip("'"), fill=NAVY if keep else GREY,
                edge=TEAL if keep else GREY_L, size=13)
            text(ax, x + 1.05, y - 0.22, f'{c[0]:.2f}', size=11,
                 color=TEAL if keep else GREY_L, glowing=False)
            if not keep:
                cross(ax, x + 2.35, y + 0.31, r=0.14)
        beams = [(c[0], c[1]) for c in kept]
        if step:
            arrow(ax, (x - 0.75, 4.7), (x - 0.1, 4.7), lw=1.5)
    text(ax, 1.0, 0.45, "context: 'the old robot s...'   kept: dark, pruned: grey",
         size=12, color=GREY_L, ha='left')
    save(fig, os.path.join(HERE, 'day60_1.png'))


def fig2():
    """Speculative decoding: one round of draft, verify, accept, repair."""
    fig, ax = canvas(10.4, 5.2, xlim=(0, 10.4), ylim=(0, 5.2))
    text(ax, 5.2, 4.95, 'speculative decoding: guess 4, verify in one pass', size=16)
    labs = ['i', 'n', ' ', 't']
    acc = [True, True, True, False]
    for i, (c, ok) in enumerate(zip(labs, acc)):
        x = 1.0 + i * 1.65
        box(ax, x, 3.6, 1.3, 0.7, label=f"'{c}'", fill=NAVY, edge=TEAL_L, size=14)
        box(ax, x, 1.9, 1.3, 0.7, label='keep' if ok else 'reject',
            fill=NAVY if ok else GREY, edge=TEAL if ok else RED, size=12)
        arrow(ax, (x + 0.65, 3.55), (x + 0.65, 2.65), lw=1.5,
              color=TEAL if ok else RED)
    text(ax, 0.9, 4.55, 'draft model: 4 cheap passes', size=13, color=TEAL_L, ha='left')
    text(ax, 0.9, 1.55, 'target model: ONE pass over all 4 positions', size=12,
         color=GREY_L, ha='left')
    region(ax, 7.55, 1.75, 2.4, 1.0, label=None)
    text(ax, 8.75, 2.25, 'resample from\n(p - q)+', size=12, color=RED)
    arrow(ax, (7.5, 2.25), (7.55 + 0.0, 2.25), lw=1.5, color=RED)
    text(ax, 5.2, 0.85,
         'emitted this round: 3 accepted + 1 repair = 4 tokens, 1 target pass',
         size=13, color=CYAN)
    save(fig, os.path.join(HERE, 'day60_2.png'))


def fig3():
    """What the two knobs actually buy."""
    rows = D.beam_table(M, widths=(1, 2, 4, 8, 16))
    d3 = D.NGram(D.make_corpus(), order=3)
    gam = D.gamma_table(M, d3, gammas=(1, 2, 4, 8), n=30)

    fig, ax = canvas(10.4, 5.2, xlim=(0, 10.4), ylim=(0, 5.2))
    text(ax, 2.5, 4.9, 'beam width -> rank of the sentence', size=14, color=TEAL_L)
    text(ax, 7.9, 4.9, 'gamma -> tokens per target pass', size=14, color=TEAL_L)
    hmax, base = 2.9, 1.1
    worst = max(r['rank'] for r in rows)
    for i, r in enumerate(rows):
        x = 0.55 + i * 0.95
        h = max(0.06, hmax * r['rank'] / worst)
        box(ax, x, base, 0.62, h, fill=NAVY, edge=TEAL)
        text(ax, x + 0.31, base + h + 0.2, str(r['rank']), size=12, color=TEAL,
             glowing=False)
        text(ax, x + 0.31, base - 0.28, str(r['width']), size=12, glowing=False)
    text(ax, 2.5, 0.45, 'beam width (lower rank is better)', size=12, color=GREY_L)
    top = max(r['per_round'] for r in gam)
    for i, r in enumerate(gam):
        x = 6.3 + i * 1.0
        h = max(0.06, hmax * r['per_round'] / top)
        box(ax, x, base, 0.62, h, fill=NAVY, edge=AMBER)
        text(ax, x + 0.31, base + h + 0.2, f'{r["per_round"]:.1f}', size=12,
             color=AMBER, glowing=False)
        text(ax, x + 0.31, base - 0.28, str(r['gamma']), size=12, glowing=False)
    text(ax, 7.9, 0.45, 'gamma (higher is better)', size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day60_3.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3()
    print('wrote day60_1.png day60_2.png day60_3.png')
