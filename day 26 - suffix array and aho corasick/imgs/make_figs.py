"""Figures for day 26 - suffix array + LCP and Aho-Corasick.

Every number on these diagrams is computed by importing suffix_aho.py, so the
figures cannot drift away from the code.

    python3 make_figs.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: E402,F403
import suffix_aho as M               # noqa: E402

S = 'banana'
SA = M.suffix_array(S)
LCP = M.lcp_kasai(S, SA)


# ---------------------------------------------------------------- figure 1
def fig1():
    """The suffix array of 'banana' with its LCP column."""
    fig, ax = canvas(9.2, 6.0, xlim=(0, 9.2), ylim=(0, 6.0))
    x0, y0, h = 0.9, 4.9, 0.72
    text(ax, 0.9 + 0.35, y0 + 0.95, 'i', color=TEAL_L, size=15)
    text(ax, 1.9 + 0.4, y0 + 0.95, 'sa[i]', color=TEAL_L, size=15)
    text(ax, 3.0 + 0.4, y0 + 0.95, 'lcp[i]', color=TEAL_L, size=15)
    text(ax, 4.4, y0 + 0.95, 'sorted suffixes', color=TEAL_L, size=15, ha='left')

    best = max(range(len(LCP)), key=lambda i: LCP[i])
    for i, p in enumerate(SA):
        y = y0 - i * h
        box(ax, x0, y, 0.7, h * 0.86, label=str(i), fill=GREY, size=14)
        box(ax, 1.9, y, 0.8, h * 0.86, label=str(p), size=14)
        hot = (i == best)
        box(ax, 3.0, y, 0.8, h * 0.86, label=str(LCP[i]),
            edge=AMBER if hot else TEAL, lw=2.6 if hot else 1.6, size=14)
        text(ax, 4.4, y + h * 0.43, S[p:], color=TEAL_L, size=17, ha='left',
             glowing=False)

    # the two suffixes that share 'ana'
    a, b = best - 1, best
    ya, yb = y0 - a * h, y0 - b * h
    box(ax, 4.28, yb - 0.02, 1.2, h * 0.86 + h + 0.04, fill='none', edge=AMBER, lw=2.4)
    text(ax, 7.2, (ya + yb) / 2 + 0.3, "'ana' is a prefix of both",
         color=AMBER, size=15)
    text(ax, 7.2, (ya + yb) / 2 - 0.15, 'so it occurs twice', color=AMBER, size=15)

    text(ax, 4.6, 0.45, 'the longest repeat is the largest lcp entry',
         color=TEAL_L, size=16)
    save(fig, 'day26_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Prefix doubling: rank arrays after 1, 2 and 4 characters."""
    rounds = M.rank_rounds(S)
    fig, ax = canvas(9.2, 5.4, xlim=(0, 9.2), ylim=(0, 5.4))
    w, x0 = 0.86, 2.4
    for r, (k, rank) in enumerate(rounds):
        y = 4.2 - r * 1.35
        text(ax, x0 - 0.35, y + 0.3, 'k = %d' % k, color=TEAL_L, size=16, ha='right')
        for i, v in enumerate(rank):
            box(ax, x0 + i * w, y, w * 0.9, 0.62, label=str(v), size=14)
            if r == 0:
                text(ax, x0 + i * w + w * 0.45, y + 0.92, S[i], color=TEAL, size=15)
    # the doubling step
    y1, y2 = 4.2, 2.85
    arrow(ax, (x0 + 0.4, y1 - 0.06), (x0 + 0.4, y2 + 0.66), color=AMBER, lw=1.6)
    arrow(ax, (x0 + 1 * w + 0.4, y1 - 0.06), (x0 + 0.5, y2 + 0.66),
          color=AMBER, lw=1.6, rad=-0.25)
    text(ax, 9.0, 3.8, '(rank[i], rank[i+k])', color=AMBER, size=15, ha='right')
    text(ax, 4.6, 0.85, 'sorting by 2k characters = sorting a pair of ranks',
         color=TEAL_L, size=16)
    text(ax, 4.6, 0.35, 'log n rounds, and every comparison is O(1)',
         color=TEAL, size=15)
    save(fig, 'day26_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """A pattern query is a contiguous range of the suffix array."""
    lo, hi, probes = M.sa_range(S, SA, 'ana')
    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    x0, y0, h = 2.6, 4.4, 0.72
    for i, p in enumerate(SA):
        y = y0 - i * h
        inside = lo <= i < hi
        box(ax, x0, y, 2.3, h * 0.86, label=S[p:], size=15,
            edge=AMBER if inside else TEAL, lw=2.6 if inside else 1.6)
        text(ax, x0 - 0.35, y + h * 0.43, str(i), color=TEAL, size=14, ha='right')
    ytop = y0 - lo * h + h * 0.86
    ybot = y0 - (hi - 1) * h
    box(ax, x0 - 0.12, ybot - 0.1, 2.54, ytop - ybot + 0.2,
        fill='none', edge=AMBER, lw=2.4)
    text(ax, 6.4, (ytop + ybot) / 2 + 0.25, "every suffix starting", color=AMBER, size=15)
    text(ax, 6.4, (ytop + ybot) / 2 - 0.2, "with 'ana' is here", color=AMBER, size=15)
    arrow(ax, (0.9, y0 + 0.3), (x0 - 0.5, ytop - 0.1), color=TEAL_L)
    text(ax, 0.9, y0 + 0.55, 'lower bound', color=TEAL_L, size=14)
    arrow(ax, (0.9, ybot - 0.55), (x0 - 0.5, ybot), color=TEAL_L)
    text(ax, 0.9, ybot - 0.8, 'upper bound', color=TEAL_L, size=14)
    text(ax, 4.6, 0.35,
         'two binary searches, %d probes - the text is never re-read' % probes,
         color=TEAL_L, size=16)
    save(fig, 'day26_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Corpus dedup: the shared block found through the LCP array."""
    rep = M.dedup_report(M.DOCS, 20)
    fig, ax = canvas(9.2, 5.2, xlim=(0, 9.2), ylim=(0, 5.2))
    for d, x in enumerate((0.6, 5.0)):
        box(ax, x, 2.5, 3.6, 2.0, fill=NAVY, edge=TEAL)
        text(ax, x + 1.8, 4.72, 'document %d' % (d + 1), color=TEAL_L, size=15)
        box(ax, x + 0.25, 3.75, 3.1, 0.55, fill=GREY, edge=AMBER, lw=2.6,
            label='licence header', size=13)
        box(ax, x + 0.25, 3.05, 3.1, 0.55, fill=GREY, edge=AMBER, lw=2.6,
            label='same wrapper body', size=13)
        box(ax, x + 0.25, 2.66, 3.1, 0.34, fill=NAVY, edge=TEAL, lw=1.2,
            label='unique part', color=TEAL_L, size=11)
    arrow(ax, (2.4, 2.4), (4.6, 1.75), color=TEAL)
    arrow(ax, (6.8, 2.4), (4.6, 1.75), color=TEAL)
    box(ax, 2.9, 1.05, 3.4, 0.7, fill=NAVY, edge=TEAL_L,
        label='one suffix array + LCP', size=15)
    text(ax, 4.6, 0.5,
         '%d chars, %d repeated blocks, %.0f%% of the corpus is a copy'
         % (rep['chars'], len(rep['spans']), rep['pct']),
         color=AMBER, size=16)
    save(fig, 'day26_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """The automaton for {he, she, his, hers}: goto edges and fail links."""
    ac = M.AhoCorasick(['he', 'she', 'his', 'hers'])
    spell = [''] * ac.n_nodes()
    for v in range(ac.n_nodes()):
        for ch, u in ac.goto[v].items():
            spell[u] = spell[v] + ch
    pos = {'': (1.0, 3.2), 'h': (2.6, 4.3), 'he': (4.2, 4.3), 'her': (5.8, 4.3),
           'hers': (7.4, 4.3), 'hi': (4.2, 2.9), 'his': (5.8, 2.9),
           's': (2.6, 1.5), 'sh': (4.2, 1.5), 'she': (5.8, 1.5)}
    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    idx = {spell[v]: v for v in range(ac.n_nodes())}
    for v in range(ac.n_nodes()):
        for ch, u in ac.goto[v].items():
            edge(ax, pos[spell[v]], pos[spell[u]], r=0.36)
    for v in range(1, ac.n_nodes()):
        f = ac.fail[v]
        if f == 0:
            continue                      # fail links to the root are not drawn
        edge(ax, pos[spell[v]], pos[spell[f]], r=0.36, color=RED, lw=1.5, rad=0.25)
    for s, (x, y) in pos.items():
        v = idx[s]
        node(ax, x, y, s or 'root', r=0.36,
             fill=NAVY if not ac.out[v] else GREY,
             edge=AMBER if ac.out[v] else TEAL, size=12)
    text(ax, 7.4, 3.55, "out = ['hers','he']", color=AMBER, size=14)
    text(ax, 5.8, 0.95, "out = ['she','he']", color=AMBER, size=14)
    legend(ax, 0.4, 1.05, [(TEAL, 'goto: the character matched'),
                           (RED, 'fail: longest suffix still in the trie')])
    text(ax, 4.8, 0.28, 'every other state fails to the root, not drawn',
         color=GREY_L, size=13)
    save(fig, 'day26_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """k KMP passes over the text vs one Aho-Corasick pass."""
    text_s = 'the sheriff said he is his own usher here ' * 3
    pats = ['he', 'she', 'his', 'hers', 'usher']
    cmp = M.compare_scans(text_s, pats)
    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    for i, p in enumerate(pats):
        y = 4.9 - i * 0.42
        box(ax, 0.7, y, 6.4, 0.3, fill=GREY, edge=TEAL, lw=1.2)
        arrow(ax, (0.75, y + 0.15), (7.05, y + 0.15), color=TEAL, lw=1.3)
        text(ax, 7.35, y + 0.15, p, color=TEAL, size=13, ha='left')
    brace(ax, 0.7, 7.1, 5.35, height=0.3, label='%d passes, %d comparisons'
          % (cmp['kmp_passes'], cmp['kmp_cost']), color=AMBER, size=15)

    y = 2.1
    box(ax, 0.7, y, 6.4, 0.4, fill=NAVY, edge=TEAL_L, lw=1.6)
    arrow(ax, (0.75, y + 0.2), (7.05, y + 0.2), color=AMBER, lw=2.0)
    text(ax, 7.35, y + 0.2, 'all 5', color=AMBER, size=14, ha='left')
    text(ax, 3.9, 1.5, '1 pass, %d transitions, %d-node automaton'
         % (cmp['ac_cost'], cmp['ac_nodes']), color=TEAL_L, size=16)
    text(ax, 3.9, 0.85, 'the text is read once; k only grows the automaton',
         color=TEAL, size=15)
    text(ax, 3.9, 0.35, 'same %d hits' % len(cmp['hits']), color=TEAL, size=14)
    save(fig, 'day26_6.png')


if __name__ == '__main__':
    os.chdir(HERE)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print('wrote', f.__doc__.splitlines()[0])
