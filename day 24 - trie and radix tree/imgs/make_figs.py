"""Figures for day 24 - trie, radix tree, and an LLM server's prefix cache.

Every number printed on a figure is computed by importing trie_radix.py, so the
pictures cannot drift away from the code.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: F401,F403
import trie_radix as T


# ---- numbers straight out of the module ----------------------------------
THREE = ["romane", "romanus", "romulus"]
_t3, _r3 = T.Trie(), T.RadixTree()
for w in THREE:
    _t3.insert(w)
    _r3.insert(w)
N_T3, N_R3 = _t3.count_nodes(), _r3.count_nodes()

_t7, _r7 = T.Trie(), T.RadixTree()
for w in T.ROMAN:
    _t7.insert(w)
    _r7.insert(w)
N_T7, N_R7 = _t7.count_nodes(), _r7.count_nodes()

_tw, _rw = T.Trie(), T.RadixTree()
for w in T.WORDS:
    _tw.insert(w)
    _rw.insert(w)
N_TW, N_RW = _tw.count_nodes(), _rw.count_nodes()


def dot(ax, x, y, label='', r=0.26, word=False, size=13, fill=None):
    """Small trie node; filled = a stored word ends here."""
    node(ax, x, y, label, r=r, size=size,
         fill=fill if fill else (TEAL if word else NAVY),
         color=WHITE if not word else '#00121a')
    return (x, y)


# --------------------------------------------------------------------------
def fig1():
    """One node per character, versus one node per branch point."""
    fig, ax = canvas(10.0, 7.6)

    region(ax, 0.30, 4.30, 9.40, 2.90)
    text(ax, 5.0, 7.32, 'trie: one node per character', color=TEAL_L, size=16)

    x0, dx, r = 0.85, 1.02, 0.26
    ys, ya, yb = 5.55, 6.20, 4.70

    # shared stem  root - r - o - m
    stem = [dot(ax, x0 + i * dx, ys, c, r=r) for i, c in enumerate(['', 'r', 'o', 'm'])]
    for a, b in zip(stem, stem[1:]):
        edge(ax, a, b, r=r)

    # branch A: a n {e | u s}
    a1 = dot(ax, x0 + 4 * dx, ya, 'a', r=r)
    a2 = dot(ax, x0 + 5 * dx, ya, 'n', r=r)
    a3 = dot(ax, x0 + 6 * dx, ya + 0.62, 'e', r=r, word=True)
    a4 = dot(ax, x0 + 6 * dx, ya - 0.62, 'u', r=r)
    a5 = dot(ax, x0 + 7 * dx, ya - 0.62, 's', r=r, word=True)
    for p, q in [(stem[3], a1), (a1, a2), (a2, a3), (a2, a4), (a4, a5)]:
        edge(ax, p, q, r=r)

    # branch B: u l u s
    b = [dot(ax, x0 + (4 + i) * dx, yb, c, r=r, word=(c == 's' and i == 3))
         for i, c in enumerate(['u', 'l', 'u', 's'])]
    edge(ax, stem[3], b[0], r=r)
    for p, q in zip(b, b[1:]):
        edge(ax, p, q, r=r)

    text(ax, x0 + 7.35 * dx, ya + 0.62, 'romane', color=TEAL, size=13, ha='left')
    text(ax, x0 + 7.35 * dx, ya - 0.62, 'romanus', color=TEAL, size=13, ha='left')
    text(ax, x0 + 7.35 * dx, yb, 'romulus', color=TEAL, size=13, ha='left')
    text(ax, 1.9, 4.55, '%d nodes' % N_T3, color=AMBER, size=15)

    # ---- radix panel -----------------------------------------------------
    region(ax, 0.30, 0.55, 9.40, 3.10)
    text(ax, 5.0, 3.80, 'radix tree: one node per branch point',
         color=TEAL_L, size=16)

    rr = 0.24
    n0 = dot(ax, 0.85, 2.20, '', r=rr)
    n1 = dot(ax, 3.05, 2.20, '', r=rr)
    n2 = dot(ax, 5.45, 2.85, '', r=rr)
    l1 = dot(ax, 7.70, 3.30, '', r=rr, word=True)
    l2 = dot(ax, 7.70, 2.40, '', r=rr, word=True)
    l3 = dot(ax, 5.45, 1.15, '', r=rr, word=True)
    for p, q, lab in [(n0, n1, 'rom'), (n1, n2, 'an'), (n2, l1, 'e'),
                      (n2, l2, 'us'), (n1, l3, 'ulus')]:
        edge(ax, p, q, r=rr, weight=lab, wsize=14)

    text(ax, 8.05, 3.30, 'romane', color=TEAL, size=13, ha='left')
    text(ax, 8.05, 2.40, 'romanus', color=TEAL, size=13, ha='left')
    text(ax, 5.85, 1.15, 'romulus', color=TEAL, size=13, ha='left')
    text(ax, 1.9, 0.80, '%d nodes' % N_R3, color=AMBER, size=15)

    text(ax, 5.0, 0.18,
         'the same words: %d nodes become %d.  On all seven Roman words, %d become %d; '
         'on a %d-word list, %d become %d.'
         % (N_T3, N_R3, N_T7, N_R7, len(T.WORDS), N_TW, N_RW),
         color=TEAL, size=13.5)
    save(fig, os.path.join(HERE, 'day24_1.png'))


# --------------------------------------------------------------------------
def fig2():
    """Insertion has to split an edge - the only hard operation."""
    fig, ax = canvas(10.0, 4.6)

    r = 0.24
    text(ax, 1.55, 4.10, 'before', color=TEAL_L, size=16)
    a0 = dot(ax, 0.60, 3.10, '', r=r)
    a1 = dot(ax, 2.60, 3.10, '', r=r, word=True)
    edge(ax, a0, a1, r=r, weight='romulus')
    text(ax, 1.60, 2.35, 'one word, one edge', color=TEAL, size=13)

    arrow(ax, (3.35, 3.10), (4.15, 3.10), color=AMBER, lw=2.0)
    text(ax, 3.75, 3.62, 'insert', color=AMBER, size=14)
    text(ax, 3.75, 2.62, '"romane"', color=AMBER, size=14)

    text(ax, 6.9, 4.10, 'after: the edge is split at the divergence',
         color=TEAL_L, size=16)
    b0 = dot(ax, 4.75, 3.10, '', r=r)
    bm = dot(ax, 6.55, 3.10, '', r=r, fill=AMBER)
    b1 = dot(ax, 8.70, 3.75, '', r=r, word=True)
    b2 = dot(ax, 8.70, 2.45, '', r=r, word=True)
    edge(ax, b0, bm, r=r, weight='rom')
    edge(ax, bm, b1, r=r, weight='ulus')
    edge(ax, bm, b2, r=r, weight='ane')
    text(ax, 9.05, 3.75, 'romulus', color=TEAL, size=12.5, ha='left')
    text(ax, 9.05, 2.45, 'romane', color=TEAL, size=12.5, ha='left')
    text(ax, 6.55, 2.35, 'new node', color=AMBER, size=13)

    text(ax, 5.0, 1.55,
         'The shared head becomes a node of its own; the two tails hang off it.',
         color=TEAL, size=14)
    text(ax, 5.0, 1.05,
         'An LLM server runs exactly this every time two requests share a prompt '
         'and then say different things.', color=TEAL, size=13.5)
    text(ax, 5.0, 0.50,
         'Nothing is copied twice - the split only moves a boundary.',
         color=TEAL_L, size=13.5)
    save(fig, os.path.join(HERE, 'day24_2.png'))


# --------------------------------------------------------------------------
def fig3():
    """Longest prefix match: the deepest marked node on the path wins."""
    fig, ax = canvas(10.0, 6.1)

    r = 0.30
    root = dot(ax, 0.75, 3.05, '', r=r, word=True)
    n10 = dot(ax, 3.15, 3.85, '', r=r, word=True)
    n20 = dot(ax, 5.55, 4.45, '', r=r, word=True)
    n30 = dot(ax, 7.95, 4.95, '', r=r, word=True)
    n192 = dot(ax, 3.15, 1.70, '', r=r, word=True)

    for p, q, lab, col in [(root, n10, '10.', AMBER), (n10, n20, '20.', AMBER),
                           (n20, n30, '30.', AMBER)]:
        edge(ax, p, q, r=r, weight=lab, color=col, lw=2.4, wcolor=AMBER)
    edge(ax, root, n192, r=r, weight='192.168.')

    text(ax, 0.75, 2.45, '/0', color=TEAL_L, size=13)
    text(ax, 0.75, 2.05, 'isp-default', color=TEAL, size=12.5)
    text(ax, 3.15, 3.25, '/8  corp-core', color=TEAL, size=12.5)
    text(ax, 5.55, 3.85, '/16  site-b', color=TEAL, size=12.5)
    text(ax, 8.55, 4.95, '/24  rack-7', color=TEAL, size=12.5, ha='left')
    text(ax, 3.15, 1.10, '/16  lab', color=TEAL, size=12.5)

    # the failed continuation
    arrow(ax, (5.85, 4.62), (6.70, 5.42), color=RED, lw=1.6, ls='--')
    cross(ax, 6.92, 5.60, r=0.20)
    text(ax, 7.42, 5.60, 'no ".99" child', color=RED, size=12.5, ha='left')

    text(ax, 5.0, 0.62,
         'Looking up 10.20.30.5 walks the amber path and answers rack-7; '
         '10.20.99.1 falls off after ".20"',
         color=TEAL, size=13.5)
    text(ax, 5.0, 0.22,
         'and answers site-b - the deepest marked node it did pass. '
         'Rule order never enters into it.',
         color=TEAL, size=13.5)
    save(fig, os.path.join(HERE, 'day24_3.png'))


# --------------------------------------------------------------------------
def _demo_cache():
    page = 4
    cache = T.TokenRadixCache(page_size=page)
    system = tuple(range(1000, 1040))
    reqs = [("req-A", system + tuple(range(1, 13))),
            ("req-B", system + tuple(range(50, 58))),
            ("req-C", system + tuple(range(1, 13)) + tuple(range(70, 78)))]
    rows = []
    for name, toks in reqs:
        reused, alloc = cache.insert(toks)
        rows.append((name, len(toks), reused, alloc))
    return cache, rows, sum(r[1] for r in rows)


def fig4():
    """Three requests, one system prompt, one shared subtree."""
    cache, rows, total = _demo_cache()
    fig, ax = canvas(10.0, 7.0)

    u, x0 = 0.108, 0.95
    text(ax, 5.0, 6.72, 'three requests, one shared system prompt',
         color=TEAL_L, size=16)
    for i, (name, n, reused, alloc) in enumerate(rows):
        y = 5.55 - i * 0.62
        text(ax, x0 - 0.15, y + 0.21, name, color=TEAL, size=13.5, ha='right')
        if reused:
            box(ax, x0, y, reused * u, 0.42, fill=TEAL, edge=TEAL_L,
                label='%d reused' % reused, size=12.5, color='#00121a')
        box(ax, x0 + reused * u, y, alloc * u, 0.42, fill=NAVY, edge=AMBER,
            lw=2.2, label='%d new' % alloc, size=12.5, color=AMBER)
    brace(ax, x0, x0 + 40 * u, 6.02, height=0.20,
          label='the 40-token system prompt', size=13)

    r = 0.40
    yv = 2.35
    root = dot(ax, 1.05, yv, '', r=0.20)
    a = node(ax, 2.55, yv, '40', r=r, size=13, fill=TEAL, color='#00121a')
    b = node(ax, 4.55, yv + 0.85, '12', r=r, size=13)
    c = node(ax, 4.55, yv - 0.85, '8', r=r, size=13)
    d = node(ax, 6.55, yv + 0.85, '8', r=r, size=13)
    edge(ax, root, a, r=r * 0.6)
    for p, q in [(a, b), (a, c), (b, d)]:
        edge(ax, p, q, r=r)
    text(ax, 2.55, yv - 0.72, 'shared by all three', color=TEAL_L, size=12.5)
    text(ax, 7.15, yv + 0.85, 'req-C tail', color=TEAL, size=12.5, ha='left')
    text(ax, 5.15, yv - 0.85, 'req-B tail', color=TEAL, size=12.5, ha='left')
    text(ax, 5.15, yv + 1.42, 'req-A tail, reused by req-C',
         color=TEAL, size=12.5, ha='left')

    text(ax, 5.0, 0.72,
         '%d tokens were asked for; only %d KV slots were ever allocated '
         '(%.0f%% saved).' % (total, cache.allocated,
                              100 * (1 - cache.allocated / total)),
         color=AMBER, size=14)
    text(ax, 5.0, 0.28,
         'The tree holds each distinct run exactly once, and every node is a '
         'prefix somebody can continue from.', color=TEAL, size=13.5)
    save(fig, os.path.join(HERE, 'day24_4.png'))


# --------------------------------------------------------------------------
def fig5():
    """A match is rounded down to a whole page."""
    page = 4
    c = T.TokenRadixCache(page_size=page)
    c.insert(tuple(range(100, 116)))
    reusable, _ = c.insert(tuple(range(100, 110)) + (999,))

    fig, ax = canvas(10.0, 4.5)
    w, h, x0, y = 0.62, 0.66, 1.20, 2.35
    for i in range(12):
        shared = i < 10
        box(ax, x0 + i * w, y, w, h,
            label=str(i), size=12,
            fill=TEAL if shared else GREY,
            edge=TEAL_L if shared else GREY_L,
            color='#00121a' if shared else WHITE)
    for p in range(4):
        ax.plot([x0 + p * 4 * w] * 2, [y - 0.30, y + h + 0.30],
                color=TEAL_L, lw=1.4, ls='--', zorder=2)
        if p < 3:
            text(ax, x0 + (p * 4 + 2) * w, y + h + 0.52, 'page %d' % p,
                 color=TEAL_L, size=12.5)

    box(ax, x0 - 0.06, y - 0.10, 8 * w + 0.12, h + 0.20, fill='none',
        edge=AMBER, lw=2.6, label=None, zorder=6)
    text(ax, x0 + 4 * w, y - 0.62, '%d tokens reusable' % reusable,
         color=AMBER, size=14)
    text(ax, x0 + 9 * w, y - 0.62, 'shared, but stranded',
         color=RED, size=12.5)
    arrow(ax, (x0 + 9 * w, y - 0.42), (x0 + 9 * w, y - 0.02), color=RED, lw=1.6)

    text(ax, 5.0, 3.95,
         'ten tokens are identical, but KV cache is allocated a page at a time',
         color=TEAL_L, size=15)
    text(ax, 5.0, 1.15,
         'Tokens 8 and 9 match and still cannot be reused: page 2 is only half '
         'written, and half a page', color=TEAL, size=13.5)
    text(ax, 5.0, 0.72,
         'is not a shareable unit. The match is rounded down - 10 becomes %d.'
         % reusable, color=TEAL, size=13.5)
    text(ax, 5.0, 0.26,
         'Bigger pages mean cheaper bookkeeping and more stranded tokens.',
         color=TEAL_L, size=13.5)
    save(fig, os.path.join(HERE, 'day24_5.png'))


# --------------------------------------------------------------------------
def fig6():
    """Eviction runs over leaves only."""
    fig, ax = canvas(10.0, 5.4)
    r = 0.38

    def tree(ox, evicted=()):
        root = dot(ax, ox + 0.00, 3.00, '', r=0.18)
        a = node(ax, ox + 1.05, 3.00, '40', r=r, size=13, fill=TEAL,
                 color='#00121a')
        pts = {'root': root, 'a': a}
        if 'c' not in evicted:
            pts['c'] = node(ax, ox + 2.35, 2.20, '8', r=r, size=13)
        if 'b' not in evicted:
            pts['b'] = node(ax, ox + 2.35, 3.80, '12', r=r, size=13)
        if 'd' not in evicted:
            pts['d'] = node(ax, ox + 3.65, 3.80, '8', r=r, size=13)
        edge(ax, root, a, r=r * 0.6)
        for p, q in [('a', 'b'), ('a', 'c'), ('b', 'd')]:
            if p in pts and q in pts:
                edge(ax, pts[p], pts[q], r=r)
        return pts

    p1 = tree(0.55)
    text(ax, 2.35, 4.90, 'before', color=TEAL_L, size=16)
    for k, lab in [('d', 'leaf'), ('c', 'leaf')]:
        node(ax, p1[k][0], p1[k][1], '', r=r + 0.10, fill='none', edge=AMBER,
             lw=2.2)
        text(ax, p1[k][0], p1[k][1] - 0.72, lab, color=AMBER, size=12.5)
    text(ax, 0.25, 2.42, 'not evictable:', color=TEAL_L, size=12.5,
         ha='left')
    text(ax, 0.25, 2.06, 'it still has children', color=TEAL_L,
         size=12.5, ha='left')

    arrow(ax, (5.05, 3.00), (5.85, 3.00), color=AMBER, lw=2.0)
    text(ax, 5.45, 3.50, 'evict', color=AMBER, size=13.5)

    p2 = tree(6.30, evicted=('d',))
    text(ax, 8.10, 4.90, 'after', color=TEAL_L, size=16)
    node(ax, p2['b'][0], p2['b'][1], '', r=r + 0.10, fill='none', edge=AMBER,
         lw=2.2)
    text(ax, p2['b'][0] + 0.60, p2['b'][1] + 0.68,
         'now a leaf itself', color=AMBER, size=12.5)

    text(ax, 5.0, 0.92,
         'Only leaves are candidates: freeing an interior node would orphan '
         'everything below it.', color=TEAL, size=13.5)
    text(ax, 5.0, 0.48,
         'So a cold branch peels away one node at a time, oldest first, while '
         'the hot shared prefix', color=TEAL, size=13.5)
    text(ax, 5.0, 0.10,
         'near the root outlives all of its children.', color=TEAL, size=13.5)
    save(fig, os.path.join(HERE, 'day24_6.png'))


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print(f.__name__, '->', f())
