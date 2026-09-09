"""Figures for Day 27 - edit distance and similarity.

Every number on these diagrams is computed by importing edit_similarity, so a
change to the module shows up here instead of silently disagreeing with it.

    python3 make_figs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: F401,F403
import edit_similarity as E

A, B = 'kitten', 'sitting'
TABLE = E.edit_table(A, B)
OPS = E.edit_ops(A, B)


def path_cells():
    """The (i, j) cells the backtrace walks through, root to corner."""
    d, i, j, cells_ = TABLE, len(A), len(B), [(len(A), len(B))]
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if A[i - 1] == B[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                i, j = i - 1, j - 1
                cells_.append((i, j)); continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
        cells_.append((i, j))
    return list(reversed(cells_))


PATH = path_cells()


# --------------------------------------------------------------------------
# 1. the table, and the path through it
# --------------------------------------------------------------------------
def fig1():
    fig, ax = canvas(9.6, 6.6, xlim=(0, 9.6), ylim=(0, 6.6))
    w, h = 0.72, 0.60
    x0, ytop = 1.15, 5.05
    cx = lambda j: x0 + j * w
    cy = lambda i: ytop - i * h

    text(ax, 0.52, 5.72, 'kitten', color=TEAL_L, size=17)
    text(ax, x0 + 4 * w, 6.22, 'sitting', color=TEAL_L, size=17)

    for j, ch in enumerate(' ' + B):
        text(ax, cx(j) + w / 2, ytop + h + 0.18, ch, color=TEAL_L, size=15)
    for i, ch in enumerate(' ' + A):
        text(ax, x0 - 0.30, cy(i) + h / 2, ch, color=TEAL_L, size=15)

    pathset = set(PATH)
    for i in range(len(A) + 1):
        for j in range(len(B) + 1):
            on = (i, j) in pathset
            box(ax, cx(j), cy(i), w, h, label=str(TABLE[i][j]),
                fill=NAVY if (i and j) else GREY,
                edge=AMBER if on else TEAL, lw=2.4 if on else 1.1,
                size=14, color=AMBER if on else WHITE)

    for (i1, j1), (i2, j2) in zip(PATH, PATH[1:]):
        arrow(ax, (cx(j1) + w / 2, cy(i1) + h / 2),
              (cx(j2) + w / 2, cy(i2) + h / 2),
              color=AMBER, lw=1.6, ms=9, shrinkA=13, shrinkB=13, zorder=6)

    text(ax, 7.05, 5.05, 'the script', color=TEAL_L, size=15, ha='left')
    labs = [op for op in OPS if op[0] != 'match']
    for n, (op, pos, frm, to) in enumerate(labs):
        text(ax, 7.05, 4.58 - n * 0.46,
             '%s  %s -> %s' % (op, frm or '-', to or '-'),
             color=AMBER, size=14, ha='left', glowing=False)
    text(ax, 7.05, cy(6) + h / 2, 'distance = %d' % TABLE[-1][-1],
         color=CYAN, size=17, ha='left')

    text(ax, 4.8, 0.78,
         'every cell asks the same question: which of the three neighbours',
         color=TEAL, size=14)
    text(ax, 4.8, 0.42,
         'was cheapest to come from - the answer is the operation',
         color=TEAL, size=14)
    save(fig, 'day27_1.png')


# --------------------------------------------------------------------------
# 2. one cell: three ways in, and why one row is enough
# --------------------------------------------------------------------------
def fig2():
    fig, ax = canvas(9.6, 5.8, xlim=(0, 9.6), ylim=(0, 5.8))
    w, h, g = 1.30, 0.92, 0.55
    bx, by = 1.15, 2.85
    box(ax, bx, by + h + g, w, h, label='d[i-1][j-1]', fill=NAVY, size=13)
    box(ax, bx + w + g, by + h + g, w, h, label='d[i-1][j]', fill=NAVY, size=13)
    box(ax, bx, by, w, h, label='d[i][j-1]', fill=NAVY, size=13)
    box(ax, bx + w + g, by, w, h, label='d[i][j]', fill=NAVY, edge=AMBER,
        lw=2.6, color=AMBER, size=13)

    tx = bx + w + g
    arrow(ax, (tx + w / 2, by + h + g), (tx + w / 2, by + h),
          color=TEAL_L, lw=1.8, ms=11)                       # from above
    arrow(ax, (bx + w, by + h / 2), (tx, by + h / 2),
          color=TEAL_L, lw=1.8, ms=11)                       # from the left
    arrow(ax, (bx + w, by + h + g), (tx, by + h),
          color=TEAL_L, lw=1.8, ms=11)                       # diagonal

    rows = [('d[i-1][j-1] + 0 or 1', 'substitute, or a free match'),
            ('d[i-1][j]   + 1', 'delete a[i-1]'),
            ('d[i][j-1]   + 1', 'insert b[j-1]')]
    for n, (lhs, rhs) in enumerate(rows):
        y = 4.55 - n * 0.44
        text(ax, 4.55, y, lhs, color=AMBER, size=13, ha='left', glowing=False)
        text(ax, 6.85, y, rhs, color=TEAL, size=13, ha='left', glowing=False)
    text(ax, 6.35, 2.95, 'd[i][j] = min of the three', color=CYAN, size=16)

    # rolling row
    region(ax, 0.75, 0.72, 8.1, 1.32, label=None)
    text(ax, 4.8, 2.28, 'so only two rows are ever live', color=TEAL_L,
         size=15)
    prev = [3, 3, 2, 1, 2, 3, 4, 5]
    cur = [4, 4, 3, 2, 1, 2, 3, 4]
    for j, v in enumerate(prev):
        box(ax, 1.60 + j * 0.62, 1.42, 0.62, 0.48, label=str(v), fill=GREY,
            size=12)
    for j, v in enumerate(cur):
        box(ax, 1.60 + j * 0.62, 0.88, 0.62, 0.48, label=str(v), fill=NAVY,
            edge=AMBER if j == 4 else TEAL, lw=2.2 if j == 4 else 1.2,
            size=12, color=AMBER if j == 4 else WHITE)
    text(ax, 1.42, 1.66, 'prev', color=TEAL, size=13, ha='right',
         glowing=False)
    text(ax, 1.42, 1.12, 'cur', color=TEAL, size=13, ha='right',
         glowing=False)
    text(ax, 7.05, 1.66, 'O(min(n, m)) memory', color=TEAL, size=13,
         ha='left', glowing=False)
    text(ax, 7.05, 1.12, 'but no backtrace', color=RED, size=13, ha='left',
         glowing=False)
    text(ax, 4.8, 0.34,
         'the table is only needed if you also want the script back',
         color=TEAL, size=14)
    save(fig, 'day27_2.png')


# --------------------------------------------------------------------------
# 3. the band: if you only care whether d <= k
# --------------------------------------------------------------------------
def fig3():
    LONG_A = 'the quick brown fox jumps over the lazy dog ' * 3
    LONG_B = LONG_A.replace('quick', 'quack')
    dist, used = E.edit_distance_bounded(LONG_A, LONG_B, 3)
    full = E.full_cells(LONG_A, LONG_B)

    fig, ax = canvas(9.6, 5.8, xlim=(0, 9.6), ylim=(0, 5.8))
    n, k, s = 15, 2, 0.235
    x0, ytop = 1.20, 5.05
    for i in range(n):
        for j in range(n):
            inband = abs(i - j) <= k
            box(ax, x0 + j * s, ytop - i * s, s, s,
                fill=NAVY if inband else 'none',
                edge=AMBER if inband else GREY_L,
                lw=1.3 if inband else 0.6, zorder=3 if inband else 2)
    text(ax, x0 + n * s / 2, ytop + s + 0.32, 'b', color=TEAL_L, size=15)
    text(ax, x0 - 0.36, ytop - n * s / 2 + s, 'a', color=TEAL_L, size=15)

    arrow(ax, (x0 + n * s + 0.75, ytop - 0.95),
          (x0 + n * s + 0.10, ytop - 0.34), color=AMBER, lw=1.6)
    text(ax, x0 + n * s + 0.85, ytop - 1.02, 'width 2k+1', color=AMBER,
         size=15, ha='left')

    for n_, line in enumerate((
            'leaving the diagonal by one cell already',
            'costs one operation, so nothing outside',
            'the band can ever come in under k')):
        text(ax, 5.35, 3.55 - n_ * 0.38, line, color=TEAL, size=14,
             ha='left', glowing=False)

    box(ax, 0.85, 0.30, 7.9, 1.28, fill='none', edge=CYAN, lw=1.8)
    text(ax, 4.80, 1.24, '132 characters, 3 substitutions apart',
         color=TEAL_L, size=15)
    text(ax, 4.80, 0.86,
         'full table  %d cells          band, k = 3   %d cells  (%.1f%%)'
         % (full, used, 100.0 * used / full), color=CYAN, size=15,
         glowing=False)
    text(ax, 4.80, 0.50,
         'k = 1 stops after %d cells: the whole band was already above 1'
         % E.edit_distance_bounded(LONG_A, LONG_B, 1)[1],
         color=AMBER, size=14, glowing=False)
    save(fig, 'day27_3.png')


# --------------------------------------------------------------------------
# 4. from strings to sets: k-shingles and Jaccard
# --------------------------------------------------------------------------
def fig4():
    fig, ax = canvas(9.6, 6.2, xlim=(0, 9.6), ylim=(0, 6.2))
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    w, x0 = 0.95, 1.35
    for i, wd in enumerate(words):
        box(ax, x0 + i * w, 5.20, w, 0.62, label=wd, fill=NAVY, size=14)
    text(ax, 0.62, 5.51, 'doc', color=TEAL_L, size=15)
    text(ax, 7.30, 4.92, 'the set of 3-shingles', color=TEAL_L, size=15,
         ha='left')

    for n, st in enumerate((0, 1, 2, 3)):
        y = 4.30 - n * 0.62
        box(ax, x0 + st * w, y, w * 3, 0.50, fill='none', edge=AMBER, lw=1.8)
        text(ax, 7.30, y + 0.25, '"%s"' % ' '.join(words[st:st + 3]),
             color=AMBER, size=14, ha='left', glowing=False)

    text(ax, 0.55, 1.92, 'order above the shingle is discarded,',
         color=TEAL, size=14, ha='left', glowing=False)
    text(ax, 0.55, 1.58, 'a document is now just a set',
         color=TEAL, size=14, ha='left', glowing=False)
    text(ax, 0.55, 1.06, '%d docs -> %d pairs'
         % (len(E.DOCS), len(E.DOCS) * (len(E.DOCS) - 1) // 2),
         color=RED, size=15, ha='left')
    text(ax, 0.55, 0.66, '10M docs -> 5 * 10^13 pairs', color=RED, size=15,
         ha='left')

    s0 = E.shingles(E.DOCS[0])
    box(ax, 5.05, 0.35, 4.05, 1.85, fill='none', edge=CYAN, lw=1.8)
    text(ax, 7.07, 1.88, 'J = |A n B| / |A u B|', color=CYAN, size=16)
    rows = [('one phrase swapped', E.jaccard(s0, E.shingles(E.DOCS[1]))),
            ('a clause appended', E.jaccard(s0, E.shingles(E.DOCS[2]))),
            ('a different topic', E.jaccard(s0, E.shingles(E.DOCS[3])))]
    for n, (lab, v) in enumerate(rows):
        y = 1.42 - n * 0.36
        text(ax, 5.30, y, lab, color=TEAL, size=13, ha='left', glowing=False)
        text(ax, 8.85, y, '%.3f' % v, color=AMBER if v > .5 else GREY_L,
             size=13, ha='right', glowing=False)
    save(fig, 'day27_4.png')


# --------------------------------------------------------------------------
# 5. why the minimum hash agrees exactly as often as the Jaccard
# --------------------------------------------------------------------------
def fig5():
    fig, ax = canvas(9.6, 6.4, xlim=(0, 9.6), ylim=(0, 6.4))
    rows = [('s7', 1, 1), ('s2', 1, 0), ('s5', 0, 1), ('s1', 1, 1),
            ('s9', 0, 1), ('s3', 1, 0), ('s8', 1, 1), ('s4', 0, 1)]
    h, wc, ytop = 0.50, 0.95, 5.15
    text(ax, 1.38, ytop + 0.62, 'A u B', color=TEAL_L, size=15)
    text(ax, 2.83, ytop + 0.62, 'in A', color=TEAL_L, size=15)
    text(ax, 3.98, ytop + 0.62, 'in B', color=TEAL_L, size=15)
    arrow(ax, (0.62, ytop + 0.35), (0.62, ytop - 7 * h - 0.10),
          color=TEAL, lw=1.6)
    text(ax, 0.34, ytop - 3.5 * h + 0.20, 'hash value increases', color=TEAL,
         size=13, glowing=False, rotation=90)
    for i, (name, ina, inb) in enumerate(rows):
        y = ytop - i * h
        first = (i == 0)
        box(ax, 0.93, y, wc, h * 0.84, label=name, fill=NAVY,
            edge=AMBER if first else TEAL, lw=2.4 if first else 1.2, size=13)
        for col, flag in ((2.58, ina), (3.73, inb)):
            box(ax, col, y, 0.50, h * 0.84, label='*' if flag else '',
                fill=NAVY if flag else GREY,
                edge=AMBER if (first and flag) else TEAL,
                lw=2.2 if (first and flag) else 1.0, size=15, color=CYAN)
    text(ax, 4.95, ytop + 0.30, 'the smallest hash in the union',
         color=AMBER, size=14, ha='left', glowing=False)
    text(ax, 4.95, ytop - 0.06, 'is the min-hash of A, and of B',
         color=AMBER, size=14, ha='left', glowing=False)

    box(ax, 4.55, 2.10, 4.85, 2.05, fill='none', edge=CYAN, lw=1.8)
    for n, line in enumerate(('the signatures agree at this slot',
                              'exactly when that smallest element',
                              'happens to lie in A n B')):
        text(ax, 6.97, 3.82 - n * 0.36, line, color=TEAL, size=14,
             glowing=False)
    text(ax, 6.97, 2.52, 'P(equal) = |A n B| / |A u B| = J', color=CYAN,
         size=15)

    text(ax, 4.80, 1.32,
         'so K independent hashes are K coin flips with bias J:',
         color=TEAL, size=14, glowing=False)
    s0, s1 = E.shingles(E.DOCS[0]), E.shingles(E.DOCS[1])
    truth = E.jaccard(s0, s1)
    parts = ['K=%d: %.3f' % (K, E.estimate_jaccard(E.minhash_signature(s0, K),
                                                   E.minhash_signature(s1, K)))
             for K in (16, 64, 256, 1024)]
    text(ax, 4.80, 0.86, '    '.join(parts), color=AMBER, size=14,
         glowing=False)
    text(ax, 4.80, 0.40, 'true J = %.3f       error shrinks like 1/sqrt(K)'
         % truth, color=TEAL_L, size=14, glowing=False)
    save(fig, 'day27_5.png')


# --------------------------------------------------------------------------
# 6. LSH banding and the S-curve
# --------------------------------------------------------------------------
def fig6():
    fig, ax = canvas(9.6, 5.9, xlim=(0, 9.6), ylim=(0, 5.9))
    sigA = [[41, 7, 19, 88], [3, 62, 11, 5], [77, 2, 90, 14], [8, 31, 6, 55]]
    sigB = [[41, 7, 19, 88], [3, 62, 40, 5], [12, 2, 90, 14], [8, 31, 6, 21]]
    w, h = 0.46, 0.42
    for col, (sig, lab, x0) in enumerate(((sigA, 'doc i', 0.75),
                                          (sigB, 'doc j', 3.05))):
        text(ax, x0 + 2 * w, 5.42, lab, color=TEAL_L, size=15)
        for bi, band in enumerate(sig):
            same = sigA[bi] == sigB[bi]
            for r, v in enumerate(band):
                box(ax, x0 + r * w, 4.72 - bi * 0.64, w, h, label=str(v),
                    fill=NAVY, edge=AMBER if same else TEAL,
                    lw=2.2 if same else 1.0, size=11,
                    color=AMBER if same else WHITE)
            if col == 0:
                text(ax, x0 - 0.26, 4.72 - bi * 0.64 + h / 2, 'b%d' % bi,
                     color=TEAL, size=12, ha='right', glowing=False)
    text(ax, 2.55, 2.28, 'one identical band is enough:', color=CYAN,
         size=15)
    for n, line in enumerate(('the pair lands in the same bucket',
                              'the other three bands differ - they',
                              'never have to be looked at')):
        text(ax, 2.55, 1.90 - n * 0.36, line, color=TEAL, size=14,
             glowing=False)
    res = E.near_duplicates(E.DOCS, num_hashes=128, bands=32, rows=4,
                            threshold=0.5)
    text(ax, 2.55, 0.42, '%d candidate pairs instead of %d, same answer'
         % (res['candidates'], res['all_pairs']), color=AMBER, size=14,
         glowing=False)

    ox, oy, ow, oh = 5.85, 1.75, 3.35, 3.00
    box(ax, ox, oy, ow, oh, fill='none', edge=TEAL, lw=1.4)
    cfgs = (((32, 4), CYAN), ((16, 8), AMBER), ((8, 16), GREY_L))
    for cfg, colr in cfgs:
        xs = [i / 100.0 for i in range(101)]
        ys = [E.prob_candidate(x, *cfg) for x in xs]
        ax.plot([ox + x * ow for x in xs], [oy + y * oh for y in ys],
                color=colr, lw=1.9, zorder=5)
        t = E.lsh_threshold(*cfg)
        ax.plot([ox + t * ow], [oy + 0.5 * oh], marker='o', ms=4.5,
                color=colr, zorder=6)
    text(ax, ox + ow / 2, oy + oh + 0.30,
         'b bands x r rows sets the threshold', color=TEAL_L, size=14)
    text(ax, ox + ow / 2, oy - 0.32, 'true Jaccard  J', color=TEAL, size=14,
         glowing=False)
    text(ax, ox - 0.24, oy + oh / 2, 'P(candidate)', color=TEAL, size=14,
         rotation=90, glowing=False)
    for n, (cfg, colr) in enumerate(cfgs):
        text(ax, ox + 0.10, 1.02 - n * 0.35,
             'b = %-2d  r = %-2d   threshold ~ %.2f'
             % (cfg[0], cfg[1], E.lsh_threshold(*cfg)),
             color=colr, size=12, ha='left', glowing=False)
    save(fig, 'day27_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print(f.__name__, '->', f())
