"""Figures for Day 65 - every number comes from importing embeddings.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import embeddings as E                 # noqa: E402

W = E.make_world()
EMB = E.train_embedding(W)
GOLD = E.truth(W)
BASE = E.recall_at_k(E.search(EMB['queries'], EMB['docs'], 10), GOLD)
ORACLE = E.recall_at_k(E.topk(E.unit(W['qmix']) @ E.unit(W['mix']).T, 10), GOLD)
CURVE = E.dimension_curve(W, EMB)
BEST = max(CURVE, key=lambda r: r['recall'])


def fig0():
    """The map: what a retrieval system decides, and what each decision costs."""
    fig, ax = canvas(12.4, 6.6, xlim=(0, 12.4), ylim=(0, 6.6))
    text(ax, 6.2, 6.25, 'one query, four decisions, and what each one costs', size=16)
    stages = [('query', 'a few words'), ('encode', 'one direction'),
              ('score every document', 'a matrix multiply'), ('top k', 'the answer')]
    for i, (name, sub) in enumerate(stages):
        x = 0.35 + i * 3.05
        box(ax, x, 4.55, 2.55, 1.05, label=name)
        text(ax, x + 1.27, 4.30, sub, size=11, color=GREY_L, glowing=False)
        if i < 3:
            arrow(ax, (x + 2.60, 5.07), (x + 3.00, 5.07), lw=1.7)
    rows = [('normalise the document vectors?',
             'cosine %.3f  vs  raw dot %.3f' % (BASE, E.recall_at_k(
                 E.search(EMB['queries'], EMB['docs'], 10, False), GOLD))),
            ('how many dimensions to keep?',
             '%d dims %.3f  vs  %d dims %.3f' % (CURVE[0]['dims'], CURVE[0]['recall'],
                                                 BEST['dims'], BEST['recall'])),
            ('how many bits per number?', 'int8 is free, one bit is not'),
            ('one vector per document, or several?', 'a mixed document needs several')]
    region(ax, 0.35, 0.55, 11.7, 3.35, label=None)
    for i, (q, a) in enumerate(rows):
        y = 3.35 - i * 0.72
        text(ax, 5.55, y, q, size=12, ha='right', color=TEAL_L, glowing=False)
        text(ax, 5.80, y, a, size=12, ha='left', color=GREY_L, glowing=False)
    text(ax, 6.2, 0.78, 'the answer key for this corpus tops out at %.3f recall@10 - '
         'that is the ceiling, not 1.0' % ORACLE, size=11.5, color=AMBER)
    save(fig, os.path.join(HERE, 'day65_0.png'))


def fig1():
    """Length bias: the dot product ranks by size."""
    lb = E.length_bias(W, EMB)
    fig, ax = canvas(11.0, 5.6, xlim=(0, 11.0), ylim=(0, 5.6))
    text(ax, 5.5, 5.25, 'a raw dot product ranks documents by how long they are',
         size=16)
    rows = list(lb['rows'].items())
    top = max(r[1]['mean_length_of_hits'] for r in rows)
    for i, (name, r) in enumerate(rows):
        y = 3.85 - i * 1.25
        text(ax, 3.05, y + 0.30, name, size=12, ha='right', color=TEAL_L, glowing=False)
        w = 5.0 * r['mean_length_of_hits'] / top
        box(ax, 3.20, y, w, 0.60, label=None, edge=RED if w > 2.5 else TEAL)
        text(ax, 3.32 + w, y + 0.30, '%.0f words in the top 10' % r['mean_length_of_hits'],
             size=11.5, ha='left', color=RED if w > 2.5 else TEAL, glowing=False)
        text(ax, 10.7, y + 0.30, 'recall@10 %.3f' % r['recall'], size=11.5, ha='right',
             color=GREY_L, glowing=False)
    text(ax, 0.35, 1.25, 'the median document is %.0f words long, and the vector length '
         'tracks it at r = %.2f' % (lb['median_length'], lb['norm_vs_length_corr']),
         size=11.5, ha='left', color=GREY_L, glowing=False)
    text(ax, 0.35, 0.80, 'cosine is the same dot product after dividing that length '
         'out - one line, five times the recall', size=11.5, ha='left', color=AMBER,
         glowing=False)
    save(fig, os.path.join(HERE, 'day65_1.png'))


def fig2():
    """Dimensions: recall against memory."""
    fig, ax = canvas(11.0, 6.0, xlim=(0, 11.0), ylim=(0, 6.0))
    text(ax, 5.5, 5.65, 'more dimensions is not more recall', size=16)
    x0, y0, w, h = 1.35, 1.35, 8.3, 3.5
    lo, hi = 0.3, 0.80
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = y0 + h * (v - lo) / (hi - lo)
        ax.plot([x0, x0 + w], [y, y], color=GREY, lw=0.8, zorder=1)
        text(ax, x0 - 0.20, y, '%.2f' % v, size=10, ha='right', color=TEAL,
             glowing=False)
    text(ax, x0 - 0.20, y0 + h + 0.30, 'recall@10', size=11, ha='right', color=TEAL,
         glowing=False)
    pts = []
    for i, r in enumerate(CURVE):
        px = x0 + w * i / (len(CURVE) - 1)
        py = y0 + h * (r['recall'] - lo) / (hi - lo)
        pts.append((px, py))
        text(ax, px, y0 - 0.32, '%d' % r['dims'], size=11, color=GREY_L, glowing=False)
        text(ax, px, y0 - 0.70, '%d B' % r['bytes_per_doc'], size=10, color=GREY_L,
             glowing=False)
    for a, b in zip(pts, pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=2.0, zorder=4)
    for (px, py), r in zip(pts, CURVE):
        hot = r['dims'] == BEST['dims']
        ax.plot([px], [py], 'o', color=CYAN if hot else TEAL, ms=7 if hot else 5,
                zorder=5)
        if hot:
            text(ax, px, py + 0.32, '%.3f' % r['recall'], size=11, color=CYAN)
    text(ax, x0 + w, y0 - 1.12, 'dimensions kept (and bytes per document)', size=11,
         ha='right', color=GREY_L, glowing=False)
    text(ax, 0.35, 0.62, 'the peak is at %d dims, %.3f against %.3f at %d - the tail '
         'of an SVD is noise, and noise votes'
         % (BEST['dims'], BEST['recall'], CURVE[0]['recall'], CURVE[0]['dims']),
         size=11, ha='left', color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day65_2.png'))


def fig3():
    """Bits per number: int8 is free, one bit is not."""
    trunc = {'queries': EMB['queries'][:, :BEST['dims']],
             'docs': EMB['docs'][:, :BEST['dims']]}
    rows = E.quantization_table(W, trunc)
    fig, ax = canvas(11.4, 5.6, xlim=(0, 11.4), ylim=(0, 5.6))
    text(ax, 5.7, 5.25, 'fewer bits per number, at %d dimensions' % BEST['dims'],
         size=16)
    top = max(r['bytes_per_doc'] for r in rows)
    for i, r in enumerate(rows):
        y = 3.95 - i * 0.95
        text(ax, 3.35, y + 0.28, r['name'], size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        wb = max(0.12, 2.3 * r['bytes_per_doc'] / top)
        box(ax, 3.50, y, wb, 0.56, label=None, edge=GREY_L)
        text(ax, 3.62 + wb, y + 0.28, '%.1f B/doc' % r['bytes_per_doc'], size=11,
             ha='left', color=GREY_L, glowing=False)
        wr = 3.4 * r['recall'] / max(x['recall'] for x in rows)
        good = r['recall'] > 0.8 * rows[0]['recall']
        box(ax, 6.95, y, wr, 0.56, label=None, edge=TEAL if good else RED)
        text(ax, 7.07 + wr, y + 0.28, 'recall %.3f' % r['recall'], size=11, ha='left',
             color=TEAL if good else RED, glowing=False)
    text(ax, 0.35, 0.95, 'int8 keeps the recall because one scale for the whole corpus '
         'leaves the dot product intact', size=11.5, ha='left', color=GREY_L,
         glowing=False)
    text(ax, 0.35, 0.50, 'one bit per dimension throws the magnitudes away - rescoring '
         'the shortlist buys most of it back', size=11.5, ha='left', color=AMBER,
         glowing=False)
    save(fig, os.path.join(HERE, 'day65_3.png'))


def fig4():
    """One vector per document is a bet."""
    rows = E.chunking_table(E.make_long_world())
    fig, ax = canvas(11.4, 5.4, xlim=(0, 11.4), ylim=(0, 5.4))
    text(ax, 5.7, 5.05, 'a document about five things has no single direction', size=16)
    for i, r in enumerate(rows):
        y = 3.35 - i * 1.45
        text(ax, 3.55, y + 0.60, r['name'], size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        n = r['vectors_per_doc']
        for j in range(n):
            box(ax, 3.70 + j * 0.52, y + 0.42, 0.44, 0.44, label=None,
                edge=CYAN if n > 1 else TEAL)
        wr = 3.2 * r['recall']
        box(ax, 6.60, y + 0.42, max(0.12, wr), 0.44, label=None,
            edge=TEAL if r['recall'] > 0.3 else RED)
        text(ax, 6.72 + max(0.12, wr), y + 0.64, 'recall %.3f' % r['recall'], size=11.5,
             ha='left', color=TEAL if r['recall'] > 0.3 else RED, glowing=False)
        text(ax, 11.1, y + 0.64, '%d B/doc' % r['bytes_per_doc'], size=11, ha='right',
             color=GREY_L, glowing=False)
    text(ax, 0.35, 0.95, '%.1fx the recall for %dx the vectors: the average of five '
         'directions points at none of them'
         % (rows[1]['recall'] / max(rows[0]['recall'], 1e-9), rows[1]['vectors_per_doc']),
         size=11.5, ha='left', color=AMBER, glowing=False)
    text(ax, 0.35, 0.50, 'this is why chunking exists, and what late-interaction models '
         'do one span at a time', size=11.5, ha='left', color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day65_4.png'))


if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4()
    print('wrote 5 figures')
