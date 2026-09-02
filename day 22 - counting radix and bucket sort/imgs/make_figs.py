"""Day 22 figures - counting / radix / bucket sort.

Every number drawn here is computed by importing counting_radix_bucket.py, so
the figures cannot drift away from the code.

    python3 make_figs.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *                                   # noqa: F401,F403
from counting_radix_bucket import (counting_sort, radix_sort, blelloch_scan,
                                   moe_align_block_size)


# ==========================================================================
# 1. counting sort: count -> scan -> scatter
# ==========================================================================
def fig1():
    keys = [4, 1, 3, 1, 0, 4, 1, 2]
    names = list('abcdefgh')
    tr = []
    out_keys, out_pay = counting_sort(keys, 5, payload=names, trace=tr)
    counts = dict(tr)['count']
    starts = dict(tr)['scan']

    fig, ax = canvas(11.6, 6.9, xlim=(0, 11.6), ylim=(0, 6.9))
    w, h = 0.82, 0.7
    x0 = 2.5

    # ---- input
    text(ax, x0 - 0.25, 5.85, 'input', color=TEAL_L, size=15, ha='right')
    cells(ax, x0, 5.5, 8, w, h, labels=[str(v) for v in keys])
    cells(ax, x0, 4.75, 8, w, h, labels=names, index=True, size=13)
    text(ax, x0 - 0.25, 5.1, 'payload', color=TEAL_L, size=15, ha='right')

    # ---- histogram
    text(ax, x0 - 0.25, 3.55, 'count', color=TEAL_L, size=15, ha='right')
    cells(ax, x0 + 1.5 * w, 3.2, 5, w, h, labels=[str(c) for c in counts])
    for i in range(5):
        text(ax, x0 + (1.5 + i) * w + w / 2, 2.98, f'key {i}',
             color=TEAL, size=11, glowing=False)

    # ---- scan
    text(ax, x0 - 0.25, 1.95, 'scan', color=TEAL_L, size=15, ha='right')
    cells(ax, x0 + 1.5 * w, 1.6, 5, w, h, labels=[str(s) for s in starts])

    # ---- output
    text(ax, x0 - 0.25, 0.65, 'output', color=TEAL_L, size=15, ha='right')
    cells(ax, x0, 0.3, 8, w, h, labels=out_pay, index=True, size=13)

    arrow(ax, (x0 + 4 * w, 4.65), (x0 + 4 * w, 3.98), lw=2.0)
    arrow(ax, (x0 + 0.6 * w, 3.15), (x0 + 0.6 * w, 2.4), lw=2.0)
    arrow(ax, (x0 + 4 * w, 1.5), (x0 + 4 * w, 1.06), lw=2.0)

    text(ax, x0 + 7.1 * w, 3.55, 'how many of each key',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, x0 + 7.1 * w, 1.95, 'how many come BEFORE each key\n= where its bucket starts',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 5.8, 6.6, 'no two elements are ever compared',
         color=TEAL_L, size=14, glowing=False)

    # the address the first '1' goes to
    ax.add_patch(Rectangle((x0 + 2.5 * w, 1.6), w, h, facecolor='none',
                           edgecolor=AMBER, linewidth=2.6, zorder=6))
    arrow(ax, (x0 + 3.0 * w, 1.6), (x0 + starts[1] * w + w / 2, 1.0),
          color=AMBER, lw=1.8, rad=-0.25)
    save(fig, os.path.join(HERE, 'day22_1.png'))


# ==========================================================================
# 2. LSD radix - stability is the correctness condition
# ==========================================================================
def fig2():
    vals = [0b1001, 0b0110, 0b1010, 0b0101, 0b0010, 0b1101]
    tr_s, tr_u = [], []
    ok = radix_sort(vals, bits=4, r=2, stable=True, trace=tr_s)
    bad = radix_sort(vals, bits=4, r=2, stable=False, trace=tr_u)
    assert ok == sorted(vals)
    assert bad != sorted(vals)

    def b4(v):
        return format(v, '04b')

    fig, ax = canvas(10.0, 6.7, xlim=(0, 10.0), ylim=(0, 6.7))
    w, h = 0.95, 0.66
    rows = [('input', vals, 5.4), ('pass 0  (bits 0-1)', tr_s[0]['result'], 4.3),
            ('pass 1  (bits 2-3)', tr_s[1]['result'], 3.2)]
    for label, data, y in rows:
        text(ax, 2.15, y + h / 2, label, color=TEAL_L, size=14, ha='right',
             glowing=False)
        cs = cells(ax, 2.4, y, 6, w, h, labels=[b4(v) for v in data], size=13)
        # underline the two bits this pass actually reads
        if label.startswith('pass'):
            dx = 0.24 if label.startswith('pass 0') else -0.24
            for cx, cy in cs:
                ax.plot([cx + dx - 0.19, cx + dx + 0.19], [cy - 0.22] * 2,
                        color=AMBER, lw=2.4, solid_capstyle='round', zorder=7)
    arrow(ax, (5.25, 5.3), (5.25, 5.02), lw=2.0)
    arrow(ax, (5.25, 4.2), (5.25, 3.92), lw=2.0)
    text(ax, 8.6, 4.63, 'sorted', color=TEAL_L, size=15)

    text(ax, 2.15, 1.85 + h / 2, 'same passes, unstable', color=RED, size=14,
         ha='right', glowing=False)
    cs = cells(ax, 2.4, 1.85, 6, w, h, labels=[b4(v) for v in bad], size=13,
               edge=RED)
    wrong = [i for i in range(5) if bad[i] > bad[i + 1]][0]
    for i in (wrong, wrong + 1):
        ax.add_patch(Rectangle((2.4 + i * w, 1.85), w, h, facecolor='none',
                               edgecolor=RED, linewidth=2.8, zorder=6))
    cross(ax, 8.85, 1.85 + h / 2, label='not sorted')

    text(ax, 5.0, 0.85, 'pass 1 reversed the ties that pass 0 had just ordered,\n'
                        'and no later pass can put them back',
         color=TEAL, size=14, glowing=False)
    text(ax, 5.0, 6.45, 'every pass is a counting sort on two bits, and the amber bits are the ones it reads',
         color=TEAL_L, size=14, glowing=False)
    save(fig, os.path.join(HERE, 'day22_2.png'))


# ==========================================================================
# 3. the scan in the middle - Blelloch up-sweep / down-sweep
# ==========================================================================
def fig3():
    hist = [3, 1, 7, 0, 4, 1, 6, 3]
    st = {}
    out, total = blelloch_scan(hist, st)

    fig, ax = canvas(10.0, 6.6, xlim=(0, 10.0), ylim=(0, 6.6))
    w, h = 0.9, 0.62
    x0 = 1.1

    # up-sweep: reproduce the partial sums level by level
    buf = list(hist)
    levels = [list(buf)]
    off = 1
    d = len(buf) >> 1
    while d > 0:
        for i in range(d):
            ai = off * (2 * i + 1) - 1
            bi = off * (2 * i + 2) - 1
            buf[bi] += buf[ai]
        levels.append(list(buf))
        off <<= 1
        d >>= 1

    ys = [5.3, 4.5, 3.7, 2.9]
    for li, (lv, y) in enumerate(zip(levels, ys)):
        cs = cells(ax, x0, y, 8, w, h, labels=[str(v) for v in lv], size=13,
                   fill=NAVY if li == 0 else GREY)
        if li > 0:
            step = 1 << li
            for i in range(0, 8, step):
                ax.add_patch(Rectangle((x0 + (i + step - 1) * w, y), w, h,
                                       facecolor='none', edgecolor=AMBER,
                                       linewidth=2.4, zorder=6))
    text(ax, x0 + 8 * w + 0.35, 5.3 + h / 2, 'histogram', color=TEAL_L,
         size=14, ha='left')
    text(ax, x0 + 8 * w + 0.35, 3.7 + h / 2, 'up-sweep', color=AMBER,
         size=15, ha='left')
    text(ax, x0 + 8 * w + 0.35, 2.9 + h / 2, 'total = %d' % total,
         color=AMBER, size=14, ha='left')

    text(ax, x0 + 4 * w, 2.35, 'down-sweep pushes the partial sums back down',
         color=TEAL, size=13, glowing=False)
    arrow(ax, (x0 + 7.5 * w, 2.75), (x0 + 7.5 * w, 1.95), lw=2.0)
    cells(ax, x0, 1.25, 8, w, h, labels=[str(v) for v in out], size=13)
    text(ax, x0 + 8 * w + 0.35, 1.25 + h / 2, 'exclusive scan', color=TEAL_L,
         size=14, ha='left')

    text(ax, x0 + 4 * w, 0.55,
         'work %d adds, depth %d rounds  -  the sequential loop needs %d adds '
         'but %d rounds' % (st['work'], st['depth'], len(hist) - 1, len(hist) - 1),
         color=TEAL, size=13, glowing=False)
    text(ax, x0 + 4 * w, 6.35, 'count and scatter parallelise on sight; this is the part that does not',
         color=TEAL_L, size=13, glowing=False)
    save(fig, os.path.join(HERE, 'day22_3.png'))


# ==========================================================================
# 4. moe_align_block_size - a counting sort with padded buckets
# ==========================================================================
def fig4():
    topk = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]
    block, ne = 4, 5
    sorted_ids, expert_ids, post = moe_align_block_size(topk, ne, block)
    flat = [e for row in topk for e in row]
    numel = len(flat)

    fig, ax = canvas(13.0, 6.9, xlim=(0, 13.0), ylim=(0, 6.9))
    w, h = 0.62, 0.6
    x0 = 1.35

    text(ax, x0 - 0.2, 5.5 + h / 2, 'token', color=TEAL_L, size=14, ha='right')
    cells(ax, x0, 5.5, numel, w, h, labels=[str(i) for i in range(numel)],
          index=True, size=12)
    text(ax, x0 - 0.2, 4.75 + h / 2, 'expert', color=TEAL_L, size=14, ha='right')
    cells(ax, x0, 4.75, numel, w, h, labels=[str(e) for e in flat], size=12)

    arrow(ax, (x0 + numel * w / 2, 4.65), (x0 + numel * w / 2, 3.95), lw=2.0)
    text(ax, x0 + numel * w / 2 + 0.25, 4.3, 'count, pad, scan, scatter',
         color=TEAL, size=13, ha='left', glowing=False)

    # the padded output, blocks of `block`
    y = 3.05
    labels = ['-' if v == numel else str(v) for v in sorted_ids]
    text(ax, x0 - 0.2, y + h / 2, 'sorted', color=TEAL_L, size=14, ha='right')
    for i, lab in enumerate(labels):
        pad = sorted_ids[i] == numel
        box(ax, x0 + i * w, y, w, h, label=lab, size=12,
            fill=GREY if pad else NAVY, edge=AMBER if pad else TEAL)
    for b, e in enumerate(expert_ids):
        bx = x0 + b * block * w
        ax.add_patch(Rectangle((bx, y - 0.06), block * w, h + 0.12,
                               facecolor='none', edgecolor=TEAL_L,
                               linewidth=1.6, linestyle='--', zorder=6))
        text(ax, bx + block * w / 2, y - 0.42, 'expert %d' % e, color=TEAL_L,
             size=12, glowing=False)
    text(ax, x0 + 15.0 * w, y + h + 0.62, 'one GEMM block',
         color=TEAL_L, size=13)
    arrow(ax, (x0 + 15.0 * w, y + h + 0.42), (x0 + 13.5 * w, y + h + 0.16),
          color=TEAL_L, lw=1.4, rad=0.2)

    text(ax, x0 - 0.2, 1.55 + h / 2, 'prefix', color=TEAL_L, size=14, ha='right')
    counts = [flat.count(e) for e in range(ne)]
    padded = [-(-c // block) * block for c in counts]
    pref, t = [], 0
    for c in padded:
        pref.append(t)
        t += c
    pref.append(t)
    cw = 1.0
    cells(ax, x0, 1.55, ne + 1, cw, h, labels=[str(v) for v in pref], size=13)
    text(ax, x0 + (ne + 1) * cw + 0.25, 1.55 + h / 2,
         'bucket starts, cursors,\nand the binary-search key for expert_ids',
         color=AMBER, size=13, ha='left', glowing=False)

    text(ax, 6.3, 0.75,
         'each bucket is rounded up to the block size BEFORE the scan;\n'
         'the holes hold token id %d, one past the last real token, and the GEMM masks them out'
         % numel, color=TEAL, size=13, glowing=False)
    text(ax, 6.3, 6.6, 'moe_align_block_size: sort %d routed tokens by expert id'
         % numel, color=TEAL_L, size=14, glowing=False)
    save(fig, os.path.join(HERE, 'day22_4.png'))


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('wrote day22_1..4.png')
