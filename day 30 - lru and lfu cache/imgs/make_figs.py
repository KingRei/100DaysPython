"""Day 30 figures - LRU / LFU caches.

Every number on these figures is computed by importing caches.py; nothing
is typed in by hand.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: E402
import caches as C                   # noqa: E402


# --------------------------------------------------------------------
# 1. The anatomy: a hash map for lookup, a linked list for order
# --------------------------------------------------------------------
def fig1():
    c = C.LRUCache(4)
    for k, v in ((10, 'a'), (20, 'b'), (30, 'c'), (40, 'd')):
        c.put(k, v)
    c.get(20)
    order = c.order()                       # most recent first

    fig, ax = canvas(9.6, 5.2, xlim=(0, 9.6), ylim=(0, 5.2))

    text(ax, 1.25, 4.75, 'hash map', color=TEAL_L, size=16, glowing=True)
    for i, k in enumerate(sorted(order)):
        y = 3.85 - i * 0.72
        box(ax, 0.35, y, 1.8, 0.58, label='%d  →' % k, size=13)
    text(ax, 1.25, 1.3, 'key → node, O(1)', color=TEAL, size=13)

    text(ax, 6.0, 4.75, 'doubly linked list', color=TEAL_L, size=16,
         glowing=True)
    xs = [2.9, 4.2, 5.5, 6.8, 8.1, 9.0]
    box(ax, xs[0] - 0.25, 2.55, 0.5, 0.95, label='H', fill=GREY, size=13)
    for i, k in enumerate(order):
        box(ax, xs[i + 1] - 0.45, 2.55, 0.9, 0.95,
            label='%d' % k, size=15)
    box(ax, 9.05, 2.55, 0.45, 0.95, label='T', fill=GREY, size=13)

    for i in range(5):
        a = xs[i] + (0.25 if i == 0 else 0.45)
        b = (xs[i + 1] - 0.45) if i < 4 else 9.05
        arrow(ax, (a, 3.22), (b, 3.22), lw=1.4)
        arrow(ax, (b, 2.86), (a, 2.86), lw=1.4)

    text(ax, 4.2, 1.95, 'most recently used', color=AMBER, size=13)
    text(ax, 8.1, 1.95, 'the victim', color=AMBER, size=13)
    arrow(ax, (8.1, 2.18), (8.1, 2.5), color=AMBER, lw=1.6)
    arrow(ax, (4.2, 2.18), (4.2, 2.5), color=AMBER, lw=1.6)
    text(ax, 5.9, 1.15,
         'both sentinels exist so that no operation has to ask'
         '\n"am I at the end of the list?" - every unlink and every'
         '\npush is four pointer writes, with no branches',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day30_1.png')


# --------------------------------------------------------------------
# 2. One missing line turns LRU into FIFO
# --------------------------------------------------------------------
def fig2():
    lru, fifo = C.LRUCache(3), C.FIFOCache(3)
    script = [('put', 1), ('put', 2), ('put', 3), ('get', 1), ('put', 4)]
    rows = {'lru': [], 'fifo': []}
    for op, k in script:
        for name, cache in (('lru', lru), ('fifo', fifo)):
            if op == 'put':
                cache.put(k, k)
            else:
                cache.get(k)
            rows[name].append(list(cache.order()))
    gone = {'lru': [k for k in rows['lru'][3] if k not in rows['lru'][4]][0],
            'fifo': [k for k in rows['fifo'][3] if k not in rows['fifo'][4]][0]}

    fig, ax = canvas(9.6, 5.8, xlim=(0, 9.6), ylim=(0, 5.8))
    labels = ['put 1', 'put 2', 'put 3', 'get 1', 'put 4']
    text(ax, 1.05, 4.9, 'LRU', color=TEAL_L, size=17, glowing=True)
    text(ax, 1.05, 4.45, 'refresh on get', color=TEAL, size=12)
    text(ax, 1.05, 2.8, 'FIFO', color=TEAL_L, size=17, glowing=True)
    text(ax, 1.05, 2.35, 'no refresh', color=RED, size=12)

    for r, name in enumerate(('lru', 'fifo')):
        base = 4.25 if r == 0 else 2.15
        for j, snap in enumerate(rows[name]):
            x = 2.3 + j * 1.42
            text(ax, x + 0.45, base + 0.75, labels[j], color=GREY_L, size=12)
            for i, k in enumerate(snap):
                hot = (j >= 3 and k == 1)
                box(ax, x, base - i * 0.46, 0.9, 0.4, label=str(k), size=13,
                    edge=AMBER if hot else TEAL, lw=2.4 if hot else 1.4)
        vi = rows[name][3].index(gone[name])
        cross(ax, 2.3 + 3 * 1.42 + 1.02, base - vi * 0.46 + 0.2, r=0.2)

    text(ax, 4.8, 1.05,
         'circled: the key put 4 is about to evict, most recent on top',
         color=GREY_L, size=13, glowing=False)
    text(ax, 4.8, 0.45,
         'get 1 moves key 1 back to the top on the left and does nothing on '
         'the right,\nso the same put 4 throws away a different key - and '
         'no exception is ever raised',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day30_2.png')


# --------------------------------------------------------------------
# 3. LFU: one bucket per frequency, and min_freq never searches
# --------------------------------------------------------------------
def fig3():
    f = C.LFUCache(6)
    for k in ('d', 'e', 'c', 'a', 'b'):
        f.put(k, k)
    for k in ('a', 'a', 'b', 'b', 'c'):
        f.get(k)
    snap = f.snapshot()
    mf = f.min_freq

    fig, ax = canvas(9.6, 4.3, xlim=(0, 9.6), ylim=(1.3, 5.6))
    text(ax, 3.6, 5.3, 'one bucket per use count', color=TEAL_L, size=16,
         glowing=False)
    ys = {}
    for i, freq in enumerate(sorted(snap, reverse=True)):
        y = 4.25 - i * 1.05
        ys[freq] = y
        box(ax, 1.15, y, 1.35, 0.72, label='count %d' % freq, size=14,
            fill=GREY)
        for j, k in enumerate(snap[freq]):
            box(ax, 2.85 + j * 0.95, y, 0.8, 0.72, label=str(k), size=16,
                edge=AMBER if (freq == mf and j == 0) else TEAL,
                lw=2.6 if (freq == mf and j == 0) else 1.6)

    lo = snap[mf]
    brace(ax, 2.85, 2.85 + len(lo) * 0.95 - 0.15, ys[mf] - 0.08,
          height=0.18, color=GREY_L, down=True)
    text(ax, 3.7, ys[mf] - 0.62, 'older → newer', color=GREY_L, size=12)

    arrow(ax, (0.35, ys[mf] + 0.36), (1.15, ys[mf] + 0.36), color=AMBER)
    text(ax, 0.75, ys[mf] + 0.9, 'min_freq', color=AMBER, size=13)
    text(ax, 5.05, ys[mf] + 0.5, '← the victim: the lowest count,',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 5.25, ys[mf] + 0.1, 'and the oldest use inside it',
         color=AMBER, size=13, ha='left', glowing=False)

    box(ax, 5.05, 3.15, 4.3, 1.9, fill=NAVY, edge=TEAL_L)
    text(ax, 7.2, 4.7, 'min_freq never searches', color=TEAL_L, size=15)
    text(ax, 7.2, 3.9,
         'a hit moves a key from bucket f to f+1;\n'
         'if that empties bucket f and f was the\n'
         'minimum, the new minimum is f+1.\n'
         'an insert sets min_freq = 1.',
         color=WHITE, size=13, glowing=False)
    save(fig, 'day30_3.png')


# --------------------------------------------------------------------
# 4. Scan pollution: LRU evicts exactly what it is about to ask for
# --------------------------------------------------------------------
def fig4():
    n = 5                                   # cache slots, drawn small
    loop = C.scan_trace(n_keys=n + 1, rounds=60)
    lru = C.run_trace(C.PolicyCache(n, C.POLICIES['lru']), loop)
    mru = C.run_trace(C.PolicyCache(n, C.POLICIES['mru']), loop)

    big = C.scan_trace(n_keys=33, rounds=120)
    lru_big = C.run_trace(C.PolicyCache(32, C.POLICIES['lru']), big)

    fig, ax = canvas(9.6, 4.3, xlim=(0, 9.6), ylim=(0.5, 4.8))
    text(ax, 4.8, 4.58,
         'a loop over %d keys through a %d-slot cache' % (n + 1, n),
         color=TEAL_L, size=16, glowing=False)

    xs = [0.9 + i * 1.35 for i in range(n + 1)]
    for i, x in enumerate(xs):
        box(ax, x, 3.2, 1.0, 0.85, label='k%d' % i, size=15,
            edge=RED if i == 0 else TEAL, lw=2.4 if i == 0 else 1.6)
    for i in range(n):
        arrow(ax, (xs[i] + 1.0, 3.62), (xs[i + 1], 3.62), lw=1.3)
    arrow(ax, (xs[-1] + 0.5, 3.1), (xs[0] + 0.5, 3.1), rad=-0.3, lw=1.3)
    text(ax, 4.8, 2.85, 'back to the start', color=GREY_L, size=12)

    text(ax, 4.8, 1.8,
         'k%d does not fit, so LRU evicts the oldest entry - k0 - and k0 is '
         'the very next request' % n,
         color=AMBER, size=13, glowing=False)
    text(ax, 4.8, 1.3,
         'measured hit rate: LRU %.1f%%, MRU %.1f%%   '
         '(32 slots, 33 keys: LRU %.1f%%)'
         % (100 * lru, 100 * mru, 100 * lru_big),
         color=TEAL, size=13, glowing=False)
    text(ax, 4.8, 0.8,
         'nothing in the cache is broken.  the policy is simply in '
         'lockstep with\nthe workload, and one step behind it',
         color=GREY_L, size=13, glowing=False)
    save(fig, 'day30_4.png')


# --------------------------------------------------------------------
# 5. One heap, five policies - the shape sglang actually ships
# --------------------------------------------------------------------
def fig5():
    fig, ax = canvas(9.6, 5.4, xlim=(0, 9.6), ylim=(0.2, 5.6))

    box(ax, 0.25, 3.15, 2.0, 1.5, fill=NAVY, edge=TEAL)
    text(ax, 1.25, 4.4, 'entry', color=TEAL_L, size=15)
    text(ax, 1.25, 3.85,
         'last_access\nhit_count\ncreated', color=WHITE, size=13,
         glowing=False)

    names = ['lru', 'lfu', 'fifo', 'mru', 'slru']
    keys = ['last_access',
            '(hit_count, last_access)',
            'created',
            '-last_access',
            '(protected, last_access)']
    region(ax, 2.75, 0.95, 4.55, 4.4)
    text(ax, 5.0, 5.15, 'get_priority(entry)', color=TEAL_L, size=15,
         glowing=True)
    for i, (n, k) in enumerate(zip(names, keys)):
        y = 4.35 - i * 0.78
        box(ax, 2.95, y, 0.95, 0.56, label=n, size=14)
        text(ax, 4.05, y + 0.28, '→  ' + k, color=TEAL, size=12,
             ha='left', glowing=False)
        arrow(ax, (2.25, 3.9), (2.95, y + 0.28), lw=1.0, color=GREY_L,
              rad=0.08)

    box(ax, 7.75, 2.55, 1.65, 1.3, fill=NAVY, edge=AMBER)
    text(ax, 8.57, 3.55, 'one heap', color=AMBER, size=14)
    text(ax, 8.57, 3.0, 'nsmallest(1,\nentries,\nkey=priority)',
         color=WHITE, size=12, glowing=False)
    arrow(ax, (7.3, 3.2), (7.75, 3.2), color=AMBER)
    arrow(ax, (8.57, 2.55), (8.57, 2.05), color=AMBER)
    text(ax, 8.57, 1.8, 'the victim', color=AMBER, size=13)

    text(ax, 4.8, 0.5,
         'sglang picks the key function with --radix-eviction-policy; '
         'the cache itself never changes',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day30_5.png')


# --------------------------------------------------------------------
# 6. Measured hit rates - no policy wins every workload
# --------------------------------------------------------------------
def fig6():
    order = ['opt', 'lru', 'lfu', 'slru']
    cols = {'opt': GREY_L, 'lru': CYAN, 'lfu': TEAL, 'slru': AMBER}
    data = {}
    for label, (trace, cap) in C.WORKLOADS.items():
        data[label] = C.compare_policies(trace, cap)

    fig, ax = canvas(9.6, 5.6, xlim=(0, 9.6), ylim=(0, 5.6))
    base, top = 1.35, 4.55
    span = top - base
    group_w = 2.1
    for gi, label in enumerate(data):
        x0 = 0.75 + gi * 2.25
        for bi, name in enumerate(order):
            h = max(span * data[label][name], 0.015)
            w = group_w / len(order) - 0.06
            box(ax, x0 + bi * (group_w / len(order)), base, w, h,
                fill=NAVY, edge=cols[name], lw=2.0)
            text(ax, x0 + bi * (group_w / len(order)) + w / 2, base + h + 0.16,
                 '%d' % round(100 * data[label][name]), color=cols[name],
                 size=11, glowing=False)
        short = label.split(' (')[0]
        text(ax, x0 + group_w / 2, 0.95, short, color=TEAL, size=12,
             glowing=False)
    arrow(ax, (0.55, base), (0.55, top + 0.25), lw=1.4)
    text(ax, 0.3, top + 0.1, '100', color=GREY_L, size=11)
    text(ax, 0.33, base, '0', color=GREY_L, size=11)
    text(ax, 4.8, 5.28, 'hit rate (%), capacity 32', color=TEAL_L, size=16,
         glowing=False)
    for i, n in enumerate(order):
        x = 3.0 + i * 1.45
        box(ax, x, 0.32, 0.3, 0.26, fill=NAVY, edge=cols[n], lw=2.0)
        text(ax, x + 0.42, 0.45, n, color=cols[n], size=12, ha='left',
             glowing=False)
    text(ax, 0.9, 0.45, 'opt = offline optimum', color=GREY_L, size=11,
         ha='center', glowing=False)
    save(fig, 'day30_6.png')


if __name__ == '__main__':
    os.chdir(HERE)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print('done')
