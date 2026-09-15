"""Figures for Day 32 - Bloom filter, HyperLogLog, Count-Min sketch.

Every number on every figure is computed by importing probabilistic.py, so a
change in the code shows up in the pictures instead of silently disagreeing
with them.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *           # noqa: E402,F403
import probabilistic as P             # noqa: E402

OUT = HERE


# --------------------------------------------------------------------
# 1. what a Bloom filter actually stores: k bits, and a false positive
# --------------------------------------------------------------------

def fig1():
    bf = P.BloomFilter(m=18, k=3)
    members = ['cat', 'dog', 'fox']
    for w in members:
        bf.add(w)
    pool = ['owl', 'bat', 'elk', 'cow', 'ram', 'ape', 'eel', 'hen', 'jay',
            'koi', 'yak', 'emu', 'pig', 'rat', 'sow', 'ewe', 'auk', 'gnu',
            'asp', 'doe', 'kid', 'colt', 'lamb', 'mule', 'newt', 'toad',
            'crab', 'wasp', 'moth', 'hare', 'ibex', 'lynx', 'mink', 'orca',
            'puma', 'seal', 'swan', 'wolf', 'bear', 'deer', 'crow', 'dove']
    ghost = next((c for c in pool if c not in members and c in bf
                  and len(set(bf.indices(c))) == bf.k), None)
    assert ghost is not None
    bits = [(bf.bits[i >> 3] >> (i & 7)) & 1 for i in range(bf.m)]

    fig, ax = canvas(9.2, 7.2, xlim=(0, 9.2), ylim=(0, 7.2))
    cw, ay = 0.42, 4.10
    x0 = (9.2 - bf.m * cw) / 2
    centres = cells(ax, x0, ay, bf.m, w=cw, h=0.58,
                    labels=[str(b) for b in bits], size=12)
    for i in range(0, bf.m, 3):
        text(ax, x0 + (i + 0.5) * cw, ay - 0.30, str(i), color=GREY_L, size=10,
             glowing=False)

    text(ax, 4.6, 6.85, 'a Bloom filter keeps bits, never the words',
         color=TEAL_L, size=16, glowing=False)

    box(ax, 0.55, 5.60, 1.35, 0.62, label="add 'cat'", size=14)
    for i in bf.indices('cat'):
        arrow(ax, (1.22, 5.60), (centres[i][0], ay + 0.60), rad=-0.12, lw=1.5)
    text(ax, 2.35, 6.06, 'k = %d hash functions, so %d bits go to 1'
         % (bf.k, bf.k), color=TEAL, size=14, ha='left', glowing=False)
    text(ax, 2.35, 5.68, "'dog' and 'fox' set their own three, sharing freely",
         color=TEAL, size=13, ha='left', glowing=False)

    gi = bf.indices(ghost)
    box(ax, 0.55, 1.95, 1.65, 0.62, label="query '%s'" % ghost, size=14,
        edge=RED)
    for i in gi:
        arrow(ax, (1.38, 2.57), (centres[i][0], ay - 0.02), rad=0.14,
              color=RED, lw=1.5)
        box(ax, x0 + i * cw, ay, cw, 0.58, label='1', edge=RED, lw=2.4, size=12)
    cross(ax, 8.35, 2.26, r=0.28)
    text(ax, 4.6, 1.45, 'all %d of its bits are already 1, so the answer is YES'
         % bf.k, color=RED, size=15, glowing=False)
    text(ax, 4.6, 1.02, "but nobody added '%s' - those bits belong to the "
         "other three words" % ghost, color=RED, size=13, glowing=False)
    text(ax, 4.6, 0.35, 'a NO is proof of absence; a YES is only a hint',
         color=AMBER, size=15, glowing=False)
    save(fig, os.path.join(OUT, 'day32_1.png'))


# --------------------------------------------------------------------
# 2. the two knobs: bits per item, and how many hashes
# --------------------------------------------------------------------

def fig2():
    import math
    fig, ax = canvas(9.2, 5.2, xlim=(0, 9.2), ylim=(0, 5.2))

    px, py, pw, ph = 0.95, 1.35, 3.25, 3.05
    rows = []
    for bpi in range(2, 25, 2):
        n = 1000
        m = n * bpi
        bf = P.BloomFilter(m=m, k=P.optimal_hashes(m, n))
        bf.n = n
        rows.append((bpi, bf.expected_fpr()))
    ax.plot([px, px, px + pw], [py + ph, py, py], color=TEAL, lw=1.6)
    lo, hi = -6.0, 0.0
    xs = [px + pw * (b - 2) / 22 for b, _ in rows]
    ys = [py + ph * (math.log10(f) - lo) / (hi - lo) for _, f in rows]
    ax.plot(xs, ys, color=CYAN, lw=2.0, marker='o', ms=4)
    for tick in (0, -2, -4, -6):
        yy = py + ph * (tick - lo) / (hi - lo)
        text(ax, px - 0.14, yy, '10^%d' % tick, color=GREY_L, size=11,
             ha='right', glowing=False)
    for b in (2, 10, 18, 24):
        text(ax, px + pw * (b - 2) / 22, py - 0.26, str(b), color=GREY_L,
             size=11, glowing=False)
    text(ax, px + pw / 2, py - 0.62, 'bits per item', color=TEAL, size=13,
         glowing=False)
    text(ax, px + pw / 2, py + ph + 0.40, 'false positive rate',
         color=TEAL_L, size=14)
    ten = [f for b, f in rows if b == 10][0]
    tx = px + pw * 8 / 22
    ty = py + ph * (math.log10(ten) - lo) / (hi - lo)
    ax.plot([tx], [ty], marker='o', ms=7, color=AMBER)
    text(ax, tx + 0.18, ty + 0.36, '10 bits -> %.1f%%' % (100 * ten),
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, px + pw / 2, py - 1.06, '10x fewer mistakes = 4.8 more bits',
         color=TEAL, size=12, glowing=False)

    qx, qy, qw, qh = 5.35, 1.35, 3.25, 3.05
    sweep, best = P.k_sweep()
    ax.plot([qx, qx, qx + qw], [qy + qh, qy, qy], color=TEAL, lw=1.6)
    top = max(r['theory'] for r in sweep)
    kx = [qx + qw * (r['k'] - 1) / 11 for r in sweep]
    ky = [qy + qh * r['theory'] / top for r in sweep]
    my = [qy + qh * r['measured'] / top for r in sweep]
    ax.plot(kx, ky, color=CYAN, lw=2.0, marker='o', ms=4)
    ax.plot(kx, my, color=AMBER, lw=1.4, ls='--', marker='s', ms=3)
    bi = [r['k'] for r in sweep].index(best)
    ax.plot([kx[bi], kx[bi]], [qy, ky[bi]], color=AMBER, lw=1.4, ls=':')
    text(ax, kx[bi], ky[bi] + 0.95, 'k = %d' % best, color=AMBER, size=14)
    text(ax, kx[bi], ky[bi] + 0.55, '(m/n) ln 2', color=AMBER, size=12,
         glowing=False)
    for k in (1, 4, 7, 10, 12):
        text(ax, qx + qw * (k - 1) / 11, qy - 0.26, str(k), color=GREY_L,
             size=11, glowing=False)
    text(ax, qx + qw / 2, qy - 0.62, 'number of hash functions k', color=TEAL,
         size=13, glowing=False)
    text(ax, qx + qw / 2, qy + qh + 0.40, 'fpr at 10 bits per item',
         color=TEAL_L, size=14)
    text(ax, qx + qw / 2, qy - 1.06, 'too few wastes bits, too many fill them',
         color=TEAL, size=12, glowing=False)
    legend(ax, qx + 1.75, qy + qh - 0.30,
           [(CYAN, 'formula'), (AMBER, 'measured')], size=12)
    save(fig, os.path.join(OUT, 'day32_2.png'))


# --------------------------------------------------------------------
# 3. why there is no delete
# --------------------------------------------------------------------

def fig3():
    d = P.deletion_report()
    gone, other = d['deleted'], d['shared_with']
    words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
             'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november']
    bf = P.BloomFilter(m=64, k=3)
    gi, oi = sorted(set(bf.indices(gone))), sorted(set(bf.indices(other)))
    shown = sorted(set(gi) | set(oi))
    shared = sorted(set(gi) & set(oi))

    cb = P.CountingBloom(m=64, k=3)
    for w in words:
        cb.add(w)
    before_cnt = [cb.counters[b] for b in shown]
    cb.delete(gone)
    after_cnt = [cb.counters[b] for b in shown]

    fig, ax = canvas(9.2, 6.6, xlim=(0, 9.2), ylim=(0, 6.6))
    cw = 1.05
    x0 = (9.2 - len(shown) * cw) / 2

    def row(y, labels, edges):
        for i, (lab, ed) in enumerate(zip(labels, edges)):
            box(ax, x0 + i * cw, y, cw, 0.62, label=lab, edge=ed,
                lw=2.4 if ed in (AMBER, RED) else 1.6, size=15)
        for i, b in enumerate(shown):
            text(ax, x0 + (i + 0.5) * cw, y - 0.27, 'bit %d' % b,
                 color=GREY_L, size=10, glowing=False)

    text(ax, 4.6, 6.25, "%d words in a 64-bit filter; '%s' and '%s' share bit %s"
         % (len(words), gone, other, ', '.join(map(str, shared))),
         color=TEAL_L, size=15, glowing=False)

    text(ax, x0, 5.60, 'the bit array', color=TEAL, size=14, ha='left',
         glowing=False)
    row(4.85, ['1'] * len(shown),
        [AMBER if b in shared else TEAL for b in shown])

    text(ax, x0, 4.05, "after clearing the three bits of '%s'" % gone,
         color=TEAL, size=14, ha='left', glowing=False)
    row(3.30, ['0' if b in gi else '1' for b in shown],
        [RED if b in shared else TEAL for b in shown])
    text(ax, 4.6, 2.62, "'%s' was never deleted, but one of its bits is now 0"
         % other, color=RED, size=14, glowing=False)
    text(ax, 4.6, 2.22, 'the filter now answers NO for %d word(s) it holds: %s'
         % (len(d['false_negatives']), ', '.join(d['false_negatives'])),
         color=RED, size=13, glowing=False)

    text(ax, x0, 1.62, 'counting Bloom: 4-bit counters, same delete',
         color=TEAL, size=14, ha='left', glowing=False)
    row(0.85, ['%d' % c for c in after_cnt],
        [AMBER if b in shared else TEAL for b in shown])
    text(ax, 4.6, 0.20, 'the shared counter went %d -> %d, so nothing is lost '
         '(at 4x the memory)'
         % (before_cnt[shown.index(shared[0])],
            after_cnt[shown.index(shared[0])]),
         color=AMBER, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day32_3.png'))


# --------------------------------------------------------------------
# 4. HyperLogLog: one hash, two jobs
# --------------------------------------------------------------------

def fig4():
    p = 4
    h = P.HyperLogLog(p)
    for i in range(40):
        h.add('user-%d' % i)

    def split(it):
        x = P.hash64(it)
        idx = x & (h.m - 1)
        w = x >> p
        rho = 1
        while rho <= 64 - p and not (w >> (rho - 1)) & 1:
            rho += 1
        return x, idx, rho

    item = next(('user-%d' % i for i in range(40)
                 if 4 <= split('user-%d' % i)[2] <= 6), 'user-0')
    x, idx, rho = split(item)
    nbits = 20
    bits = [(x >> i) & 1 for i in range(nbits)]          # bits[0] = LSB

    fig, ax = canvas(9.2, 6.4, xlim=(0, 9.2), ylim=(0, 6.4))
    text(ax, 4.6, 6.05, 'one hash, split into two jobs', color=TEAL_L, size=16,
         glowing=False)

    cw = 0.40
    x0 = (9.2 - nbits * cw) / 2
    by = 4.35
    for i in range(nbits):                                # LSB on the right
        cx = x0 + (nbits - 1 - i) * cw
        idx_field = i < p
        run_field = p <= i < p + rho
        box(ax, cx, by, cw, 0.55, label=str(bits[i]),
            edge=AMBER if idx_field else (CYAN if run_field else TEAL),
            lw=2.4 if idx_field or run_field else 1.4, size=12)
    text(ax, x0 - 0.15, by + 0.27, '...', color=GREY_L, size=14, ha='right',
         glowing=False)
    box(ax, 0.40, 5.25, 2.05, 0.60, label="add '%s'" % item, size=14)
    arrow(ax, (1.43, 5.25), (x0 + nbits * cw * 0.45, by + 0.62), rad=-0.10)

    ix1 = x0 + nbits * cw
    ix0 = ix1 - p * cw
    brace(ax, ix0, ix1, by - 0.12, height=0.24, down=True, color=AMBER)
    text(ax, (ix0 + ix1) / 2, by - 0.78, 'register #%d' % idx, color=AMBER,
         size=13, glowing=False)
    rx1 = ix0
    rx0 = rx1 - rho * cw
    brace(ax, rx0, rx1, by - 0.12, height=0.24, down=True, color=CYAN)
    text(ax, (rx0 + rx1) / 2, by - 0.78, 'rho = %d' % rho, color=CYAN,
         size=13, glowing=False)
    text(ax, 4.6, by - 1.30, 'the low %d bits choose the register; the run of '
         'zeros above them is rho' % p, color=TEAL, size=13, glowing=False)

    ry = 1.45
    rcw = (9.2 - 1.2) / h.m
    x1 = 0.6
    for i in range(h.m):
        box(ax, x1 + i * rcw, ry, rcw, 0.58, label=str(h.reg[i]),
            edge=AMBER if i == idx else TEAL,
            lw=2.4 if i == idx else 1.5, size=13)
        text(ax, x1 + (i + 0.5) * rcw, ry - 0.26, str(i), color=GREY_L,
             size=9, glowing=False)
    text(ax, 4.6, 2.35, '%d registers, one byte each' % h.m, color=TEAL_L,
         size=14, glowing=False)
    text(ax, 4.6, 0.72, 'reg[%d] = max(reg[%d], rho) - a register keeps the '
         'longest run it has seen, not a list' % (idx, idx), color=TEAL,
         size=13, glowing=False)
    text(ax, 4.6, 0.25, 'long runs are rare, so a large maximum means many '
         'distinct items came through', color=AMBER, size=14, glowing=False)
    save(fig, os.path.join(OUT, 'day32_4.png'))


# --------------------------------------------------------------------
# 5. the memory line that does not move
# --------------------------------------------------------------------

def fig5():
    import math
    rows, theo = P.hll_report()
    fig, ax = canvas(9.2, 5.8, xlim=(0, 9.2), ylim=(0, 5.8))
    px, py, pw, ph = 1.55, 1.85, 6.2, 3.05
    ax.plot([px, px, px + pw], [py + ph, py, py], color=TEAL, lw=1.6)
    lo, hi = 3.6, 7.9
    xs = [px + pw * i / (len(rows) - 1) for i in range(len(rows))]
    ex = [py + ph * (math.log10(r['exact_bytes']) - lo) / (hi - lo)
          for r in rows]
    hl = [py + ph * (math.log10(r['hll_bytes']) - lo) / (hi - lo)
          for r in rows]
    ax.plot(xs, ex, color=CYAN, lw=2.2, marker='o', ms=5)
    ax.plot(xs, hl, color=AMBER, lw=2.2, marker='s', ms=5)
    for tick, lab in ((4, '10 KB'), (5, '100 KB'), (6, '1 MB'), (7, '10 MB')):
        yy = py + ph * (tick - lo) / (hi - lo)
        text(ax, px - 0.16, yy, lab, color=GREY_L, size=11, ha='right',
             glowing=False)
    for i, r in enumerate(rows):
        text(ax, xs[i], py - 0.30, '{:,}'.format(r['n']), color=GREY_L,
             size=11, glowing=False)
        text(ax, xs[i], py - 0.68, '%+.2f%%' % r['error_pct'], color=AMBER,
             size=12, glowing=False)
    text(ax, px + pw / 2, py - 1.12, 'distinct items counted (error underneath)',
         color=TEAL, size=13, glowing=False)
    text(ax, px - 0.30, py + ph + 0.36, 'memory', color=TEAL_L, size=15,
         ha='left')
    text(ax, xs[-1], ex[-1] + 0.36, 'an exact set', color=CYAN, size=14,
         ha='right', glowing=False)
    text(ax, xs[1], hl[1] + 0.34, 'HyperLogLog, p = 14: 16 KB, always',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 4.6, 0.38, 'the error stays near the theoretical '
         '1.04/sqrt(16384) = %.2f%% while the memory line never moves' % theo,
         color=TEAL, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day32_5.png'))


# --------------------------------------------------------------------
# 6. Count-Min sketch: d rows, take the minimum
# --------------------------------------------------------------------

def fig6():
    import random as _r
    cms = P.CountMinSketch(w=8, d=4)
    rnd = _r.Random(3)
    stream = (['the'] * 40 + ['of'] * 18 + ['cat'] * 9
              + ['w-%d' % rnd.randrange(400) for _ in range(120)])
    rnd.shuffle(stream)
    from collections import Counter as _C
    truth = _C(stream)
    for t in stream:
        cms.add(t)
    rare = next(k for k in truth
                if truth[k] == 1 and cms.query(k) > 1)
    heavy = 'the'

    fig, ax = canvas(9.2, 6.8, xlim=(0, 9.2), ylim=(0, 6.8))
    text(ax, 4.6, 6.50, '%d rows x %d counters; every row counts the whole '
         'stream' % (cms.d, cms.w), color=TEAL_L, size=15, glowing=False)
    cw, ch = 0.80, 0.62
    gx = (9.2 - cms.w * cw) / 2 + 0.4
    gy = 5.60
    hi = [cms._cols(heavy)[r] for r in range(cms.d)]
    ri = [cms._cols(rare)[r] for r in range(cms.d)]
    rmin = min(cms.rows[r][ri[r]] for r in range(cms.d))
    for r in range(cms.d):
        yy = gy - r * (ch + 0.12)
        for c in range(cms.w):
            hot = c == ri[r]
            box(ax, gx + c * cw, yy, cw, ch, label=str(cms.rows[r][c]),
                edge=RED if hot and cms.rows[r][c] > rmin else
                     (AMBER if hot else TEAL),
                lw=2.4 if hot else 1.3, size=12)
        text(ax, gx - 0.22, yy + ch / 2, 'h%d' % (r + 1), color=TEAL, size=13,
             ha='right', glowing=False)
    ylow = gy - (cms.d - 1) * (ch + 0.12)
    text(ax, 4.6, ylow - 0.58, "query '%s': the %d rows say %s"
         % (rare, cms.d, ', '.join(str(cms.rows[r][ri[r]])
                                   for r in range(cms.d))),
         color=TEAL, size=14, glowing=False)
    text(ax, 4.6, ylow - 1.04, 'take the MINIMUM = %d   (true count: %d)'
         % (rmin, truth[rare]), color=AMBER, size=15, glowing=False)
    text(ax, 4.6, ylow - 1.58, 'collisions can only add, never subtract, so '
         'every row over-counts', color=TEAL, size=13, glowing=False)
    text(ax, 4.6, ylow - 1.98, 'and the smallest row is the one that got '
         'away with the fewest collisions', color=TEAL, size=13,
         glowing=False)
    ha, hq = truth[heavy], cms.query(heavy)
    text(ax, 4.6, ylow - 2.50, "'%s' (true %d) reads back as %d; '%s' (true %d) "
         'reads back as %d' % (heavy, ha, hq, rare, truth[rare], rmin),
         color=CYAN, size=14, glowing=False)
    text(ax, 4.6, ylow - 2.94, 'nearly the same absolute error (+%d and +%d), '
         'but that is %+.0f%% and %+.0f%%'
         % (hq - ha, rmin - truth[rare], 100 * (hq - ha) / ha,
            100 * (rmin - truth[rare]) / truth[rare]),
         color=AMBER, size=14, glowing=False)
    save(fig, os.path.join(OUT, 'day32_6.png'))


# --------------------------------------------------------------------
# 7. which one, and which way does it lie
# --------------------------------------------------------------------

def fig7():
    table = P.comparison_table()
    fig, ax = canvas(9.4, 4.6, xlim=(0, 9.4), ylim=(0, 4.6))
    cols = [(0.20, 1.70), (1.95, 2.05), (4.05, 2.20), (6.30, 1.85),
            (8.20, 1.05)]
    head = ['structure', 'answers', 'how it lies', 'mergeable', 'bytes/1M']
    rh = 0.66
    top = 3.55
    for (cx, cwid), htxt in zip(cols, head):
        text(ax, cx + cwid / 2, top + 0.42, htxt, color=TEAL_L, size=14,
             glowing=False)
    for i, (name, q, err, mrg, b) in enumerate(table):
        y = top - i * rh
        vals = [name, q, err, mrg.replace(' (', '\n('),
                ('%.0f MB' % (b / 1e6)) if b > 1e6 else
                ('%d KB' % round(b / 1024))]
        for (cx, cwid), v in zip(cols, vals):
            box(ax, cx, y - rh + 0.10, cwid, rh - 0.12, label=v,
                edge=AMBER if i and cx == cols[2][0] else TEAL,
                lw=2.2 if i and cx == cols[2][0] else 1.4, size=11.5)
    text(ax, 4.7, 0.42, 'the size of the error is a knob; the direction is the '
         'design decision', color=AMBER, size=15, glowing=False)
    save(fig, os.path.join(OUT, 'day32_7.png'))


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print('ok', f.__name__)
