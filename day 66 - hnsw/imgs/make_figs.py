"""Figures for Day 66 - every number comes from importing hnsw.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import hnsw as H                       # noqa: E402

# the same world the module's main() uses, so every number matches the article
W = H.make_vectors(docs=3_000, queries=120)
GOLD = H.brute_force(W['points'], W['queries'], 10)
IX = H.HNSW(W['cfg']['dims'], M=12, ef_construction=60).build(W['points'])
CURVE = H.ef_curve(IX, W['points'], W['queries'], GOLD)
BF = H.brute_cost(W['points'], W['queries'])
AT50 = [r for r in CURVE if r['ef'] == 50][0]


def fig0():
    """The algorithm, drawn on a real two-dimensional index."""
    rng = np.random.default_rng(4)
    n = 140
    pts = rng.random((n, 2)).astype(np.float32)
    small = H.HNSW(2, M=4, ef_construction=40, seed=3, metric='l2').build(pts)
    q = np.array([0.62, 0.55], dtype=np.float32)

    path, ep = [], [small.entry]
    for l in range(small.levels[small.entry], 0, -1):
        nxt = small._search_layer(q, ep, 1, l)[0][1]
        path.append((l, ep[0], nxt))
        ep = [nxt]
    small.dist_calls = 0
    bottom = small._search_layer(q, ep, 16, 0)
    touched = small.dist_calls
    found = [i for _, i in sorted(bottom)[:5]]
    layers = small.layer_sizes()
    shown = [min(2, len(layers) - 1), 1, 0]                 # top -> bottom

    fig, ax = canvas(12.6, 6.6, xlim=(0, 12.6), ylim=(0, 6.6))
    text(ax, 6.3, 6.25, 'one index, three layers: the walk enters at the top, '
         'descends, and only the bottom layer holds every point', size=14.5)
    size = 3.15
    origins = [(0.55, 1.35), (4.70, 1.35), (8.85, 1.35)]
    place = lambda v, ox, oy: (ox + float(v[0]) * size, oy + float(v[1]) * size)

    for panel, (layer, (ox, oy)) in enumerate(zip(shown, origins)):
        nodes = list(small.graph[layer].keys())
        text(ax, ox + size / 2, oy + size + 0.45, 'layer %d  -  %d of %d points'
             % (layer, layers[layer], layers[0]), size=12, color=TEAL_L,
             glowing=False)
        for a in nodes:
            pa = place(pts[a], ox, oy)
            for b in small.graph[layer][a]:
                if b > a:
                    pb = place(pts[b], ox, oy)
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=TEAL, lw=0.8,
                            alpha=0.5, zorder=2)
        for a in nodes:
            pa = place(pts[a], ox, oy)
            hit = layer == 0 and a in found
            ax.plot([pa[0]], [pa[1]], 'o', ms=7 if hit else 3.5,
                    color=CYAN if hit else TEAL_L, zorder=4)
        qp = place(q, ox, oy)
        ax.plot([qp[0]], [qp[1]], '*', ms=16, color=RED, zorder=6)
        for (l, src, dst) in path:
            if l == layer:
                arrow(ax, place(pts[src], ox, oy), place(pts[dst], ox, oy), lw=1.8,
                      color=RED, ms=10)
        if layer == 0:
            for _, i in sorted(bottom)[:16]:
                pi = place(pts[i], ox, oy)
                ax.plot([pi[0]], [pi[1]], 'o', ms=5, color=AMBER, zorder=5,
                        alpha=0.9)
            for i in found:
                pi = place(pts[i], ox, oy)
                ax.plot([pi[0]], [pi[1]], 'o', ms=7, color=CYAN, zorder=6)

    captions = ['enter at the top, one greedy hop',
                'descend, still one hop at a time',
                'at the bottom, keep ef candidates']
    for (ox, oy), cap, col in zip(origins, captions, (RED, RED, AMBER)):
        text(ax, ox + size / 2, oy - 0.38, cap, size=11, color=col, glowing=False)
    text(ax, 0.35, 0.42, 'red star = the query   |   amber = the ef candidates kept '
         'alive   |   cyan = the answer   |   this walk computed %d of %d distances'
         % (touched, n), size=11, ha='left', color=GREY_L, glowing=False)
    text(ax, 6.3, 5.82, 'on the real %d-dimensional corpus this costs %.0f distance '
         'computations out of %d (%.0f%%) for recall %.2f'
         % (W['cfg']['dims'], AT50['dist_per_query'], len(W['points']),
            100 * AT50['fraction_seen'], AT50['recall']),
         size=11.5, color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day66_0.png'))


def fig1():
    """ef: the only query-time knob."""
    fig, ax = canvas(11.2, 6.2, xlim=(0, 11.2), ylim=(0, 6.2))
    text(ax, 5.6, 5.85, 'ef buys recall with distance computations', size=16)
    x0, y0, w, h = 1.45, 1.75, 8.1, 3.25
    lo, hi = 0.55, 1.02
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = y0 + h * (v - lo) / (hi - lo)
        ax.plot([x0, x0 + w], [y, y], color=GREY, lw=0.8, zorder=1)
        text(ax, x0 - 0.18, y, '%.2f' % v, size=10, ha='right', color=TEAL,
             glowing=False)
    text(ax, x0 - 0.18, y0 + h + 0.32, 'recall@10', size=11, ha='right', color=TEAL,
         glowing=False)
    xs = [r['dist_per_query'] for r in CURVE] + [BF['dist_per_query']]
    lox, hix = min(xs) * 0.8, max(xs) * 1.1
    px = lambda d: x0 + w * (np.log10(d) - np.log10(lox)) / (np.log10(hix) - np.log10(lox))
    py = lambda r: y0 + h * (r - lo) / (hi - lo)
    pts = [(px(r['dist_per_query']), py(r['recall'])) for r in CURVE]
    for a, b in zip(pts, pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=TEAL, lw=2.0, zorder=4)
    for (x, y), r in zip(pts, CURVE):
        hot = r['ef'] == 50
        ax.plot([x], [y], 'o', ms=8 if hot else 5, color=CYAN if hot else TEAL,
                zorder=5)
        text(ax, x, y + 0.28, 'ef %d' % r['ef'], size=10.5,
             color=CYAN if hot else GREY_L, glowing=False)
        text(ax, x, y0 - 0.32, '%.0f' % r['dist_per_query'], size=10, color=GREY_L,
             glowing=False)
    bx = px(BF['dist_per_query'])
    ax.plot([bx], [py(1.0)], 'o', ms=7, color=RED, zorder=5)
    text(ax, bx, py(1.0) - 0.30, 'brute force', size=10.5, color=RED,
         glowing=False)
    text(ax, bx, y0 - 0.32, '%.0f' % BF['dist_per_query'], size=10, color=RED,
         glowing=False)
    text(ax, x0, y0 - 0.95, 'distance computations per query (log scale)', size=11,
         ha='left', color=GREY_L, glowing=False)
    text(ax, 0.35, 0.32, 'at ef = %d the index reads %.0f%% of the corpus and keeps '
         '%.0f%% of the answers' % (AT50['ef'], 100 * AT50['fraction_seen'],
                                    100 * AT50['recall']),
         size=11.5, ha='left', color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day66_1.png'))


def fig2():
    """The hierarchy is not where the speed comes from."""
    rows = H.without_layers(IX, W['queries'], GOLD)
    sizes = IX.layer_sizes()
    fig, ax = canvas(11.6, 5.8, xlim=(0, 11.6), ylim=(0, 5.8))
    text(ax, 5.8, 5.45, 'removing the upper layers changes neither the recall nor '
         'the work', size=15)
    top = max(r['dist_per_query'] for r in rows)
    for i, r in enumerate(rows):
        y = 3.85 - i * 1.05
        text(ax, 4.15, y + 0.30, r['name'], size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        wbar = 3.4 * r['dist_per_query'] / top
        box(ax, 4.30, y, wbar, 0.60, label=None, edge=TEAL)
        text(ax, 4.42 + wbar, y + 0.30, '%.0f distances' % r['dist_per_query'],
             size=11, ha='left', color=TEAL, glowing=False)
        text(ax, 11.3, y + 0.30, 'recall %.3f' % r['recall'], size=11, ha='right',
             color=CYAN, glowing=False)
    text(ax, 0.35, 1.45, 'the layers still exist: %s nodes from the bottom up - but '
         'in high dimensions' % ' / '.join('%d' % s for s in sizes),
         size=11.5, ha='left', color=GREY_L, glowing=False)
    text(ax, 0.35, 1.00, 'the bottom graph already has hub nodes that act as the '
         'express lanes (the published', size=11.5, ha='left', color=GREY_L,
         glowing=False)
    text(ax, 0.35, 0.55, 'result: a flat graph matches HNSW and saves 38-39% of peak '
         'build memory at 100M vectors)', size=11.5, ha='left', color=AMBER,
         glowing=False)
    save(fig, os.path.join(HERE, 'day66_2.png'))


def fig3():
    """M: links are memory, and memory is not free."""
    small = H.make_vectors(seed=1, docs=1_000, queries=80)
    gold = H.brute_force(small['points'], small['queries'], 10)
    rows = H.m_sweep(small['points'], small['queries'], gold, Ms=(6, 12, 20))
    fig, ax = canvas(11.6, 5.8, xlim=(0, 11.6), ylim=(0, 5.8))
    text(ax, 5.8, 5.45, 'M is the memory knob and the build-time knob at once',
         size=16)
    for i, r in enumerate(rows):
        y = 3.95 - i * 1.15
        text(ax, 1.55, y + 0.30, 'M = %d' % r['M'], size=12, ha='right', color=TEAL_L,
             glowing=False)
        wv = 1.6
        box(ax, 1.70, y, wv, 0.60, label='vector', size=10, edge=GREY_L)
        wl = 1.6 * r['overhead']
        box(ax, 1.70 + wv, y, wl, 0.60, label='links' if wl > 0.8 else None, size=10,
            edge=AMBER)
        text(ax, 1.82 + wv + wl, y + 0.30, '%.0f + %.0f bytes/point'
             % (r['vector_bytes'], r['link_bytes']), size=11, ha='left',
             color=GREY_L, glowing=False)
        text(ax, 9.15, y + 0.30, 'build %.1f s' % r['build_seconds'], size=11,
             ha='right', color=GREY_L, glowing=False)
        text(ax, 11.3, y + 0.30, 'recall %.3f' % r['recall'], size=11, ha='right',
             color=CYAN, glowing=False)
    text(ax, 0.35, 0.90, 'at M = %d the links cost %.0f%% of what the vectors cost, '
         'and the build is %.1fx longer'
         % (rows[-1]['M'], 100 * rows[-1]['overhead'],
            rows[-1]['build_seconds'] / rows[0]['build_seconds']),
         size=11.5, ha='left', color=AMBER, glowing=False)
    text(ax, 0.35, 0.45, 'while the recall it buys saturates: the question is the '
         'smallest M that reaches your target', size=11.5, ha='left', color=GREY_L,
         glowing=False)
    save(fig, os.path.join(HERE, 'day66_3.png'))


def fig4():
    """Tombstones: the work stays, the answers shrink."""
    pts = H.make_vectors(seed=3, docs=1_500, queries=80)['points']
    rows = H.deletion_decay(pts, W['queries'][:80], fractions=(0.1, 0.3))
    fig, ax = canvas(11.6, 5.6, xlim=(0, 11.6), ylim=(0, 5.6))
    text(ax, 5.8, 5.25, 'a deletion is a tombstone: the point keeps being visited',
         size=16)
    top = max(r['dist_per_query'] for r in rows)
    for i, r in enumerate(rows):
        y = 3.75 - i * 1.15
        name = ('rebuilt after %.0f%%' % (100 * r['deleted']) if r.get('rebuilt')
                else '%.0f%% deleted' % (100 * r['deleted']))
        text(ax, 2.85, y + 0.30, name, size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        w = 4.2 * r['dist_per_query'] / top
        waste = r.get('wasted_share', 0.0)
        box(ax, 3.00, y, w * (1 - waste), 0.60, label=None, edge=TEAL)
        if waste:
            box(ax, 3.00 + w * (1 - waste), y, w * waste, 0.60, label=None, edge=RED)
        text(ax, 3.12 + w, y + 0.30, '%.0f distances/query' % r['dist_per_query'],
             size=11, ha='left', color=GREY_L, glowing=False)
        text(ax, 11.3, y + 0.30, 'recall %.3f' % r['recall'], size=11, ha='right',
             color=CYAN, glowing=False)
    text(ax, 0.35, 1.05, 'red = distance computations spent on points that can never '
         'be returned (%.0f%% at 30%% deleted)' % (100 * rows[1]['wasted_share']),
         size=11.5, ha='left', color=RED, glowing=False)
    text(ax, 0.35, 0.55, 'rebuilding the index on the survivors takes %.0f s here and '
         'brings the cost back to %.0f'
         % (rows[-1]['rebuild_seconds'], rows[-1]['dist_per_query']),
         size=11.5, ha='left', color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day66_4.png'))


def fig5():
    """What happens as the corpus grows."""
    rows = H.scaling(sizes=(500, 1_000, 2_000))
    rows.append({'docs': len(W['points']), 'recall': AT50['recall'],
                 'dist_per_query': AT50['dist_per_query'],
                 'fraction_seen': AT50['fraction_seen']})
    fig, ax = canvas(11.2, 6.0, xlim=(0, 11.2), ylim=(0, 6.0))
    text(ax, 5.6, 5.65, 'six times the data, twice the work', size=16)
    x0, y0, w, h = 1.45, 1.45, 8.2, 3.3
    hi = max(r['docs'] for r in rows) * 1.05
    for i, r in enumerate(rows):
        x = x0 + w * i / (len(rows) - 1)
        hb = h * r['dist_per_query'] / hi
        hf = h * r['docs'] / hi
        box(ax, x - 0.62, y0, 0.52, hf, label=None, edge=RED)
        box(ax, x - 0.04, y0, 0.52, hb, label=None, edge=TEAL)
        text(ax, x - 0.36, y0 + hf + 0.22, '%.0f' % r['docs'], size=10, color=RED,
             glowing=False)
        text(ax, x + 0.22, y0 + hb + 0.22, '%.0f' % r['dist_per_query'], size=10,
             color=TEAL, glowing=False)
        text(ax, x - 0.05, y0 - 0.32, '%s vectors' % f"{r['docs']:,}", size=10.5,
             color=GREY_L, glowing=False)
        text(ax, x - 0.05, y0 - 0.68, 'recall %.3f' % r['recall'], size=10,
             color=CYAN, glowing=False)
    text(ax, x0 - 0.20, y0 + h + 0.30, 'distance computations per query', size=11,
         ha='left', color=TEAL, glowing=False)
    text(ax, 0.35, 0.55, 'red = brute force (every vector)   teal = HNSW at ef 50   '
         '- the gap is the whole point of the index', size=11.5, ha='left',
         color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day66_5.png'))


if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4(); fig5()
    print('wrote 6 figures')
