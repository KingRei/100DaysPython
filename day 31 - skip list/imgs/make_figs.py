"""Figures for Day 31 - Skip List.

Every number drawn here is computed by importing skip_list.py, so the figures
cannot drift away from the code.

    python3 make_figs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: F401,F403
import skip_list as SL


# ---------------------------------------------------------------- helpers
KEYS = list(range(1, 17))


def build_demo_list():
    sl = SL.SkipList(p=0.5, seed=5)
    for k in KEYS:
        sl.insert(k, k * k)
    return sl


def xk(k):
    """x centre of the column for key k."""
    return 0.95 + (k - 1) * 0.555


HEAD_X = 0.42
BW, BH = 0.44, 0.42


def draw_lanes(ax, levels, y0=0.8, dy=0.82, highlight=(), hot_edges=(),
               spans=None, head_label='H'):
    """Draw a skip list. levels[i] = sorted keys present on level i.

    highlight = {(level, key)} nodes we stood on; hot_edges = {(level, from, to)}
    pointers we actually followed, with None meaning the head.
    """
    top = len(levels) - 1
    box(ax, HEAD_X - 0.16, y0 - BH / 2, 0.32, top * dy + BH, label=None,
        fill=GREY, edge=TEAL)
    text(ax, HEAD_X, y0 + top * dy + 0.52, head_label, color=TEAL_L, size=13)
    pos = {}
    for i, row in enumerate(levels):
        y = y0 + i * dy
        prev, prev_key = (HEAD_X + 0.16, y), None
        for k in row:
            hot = (i, k) in highlight
            hot_e = (i, prev_key, k) in hot_edges
            box(ax, xk(k) - BW / 2, y - BH / 2, BW, BH, label=str(k),
                fill=NAVY, edge=AMBER if hot else TEAL,
                lw=2.6 if hot else 1.6, size=12)
            pos[(i, k)] = (xk(k), y)
            arrow(ax, prev, (xk(k) - BW / 2, y),
                  color=AMBER if hot_e else TEAL,
                  lw=2.2 if hot_e else 1.4, ms=10 if hot_e else 9)
            if spans is not None:
                sp = spans[i].get(k)
                if sp is not None:
                    text(ax, (prev[0] + xk(k) - BW / 2) / 2, y + 0.27,
                         str(sp), color=AMBER if hot_e else TEAL_L, size=11,
                         glowing=False)
            prev, prev_key = (xk(k) + BW / 2, y), k
        text(ax, 9.62, y, f'L{i}', color=TEAL_L, size=13, ha='left')
    return pos


def path_edges(path):
    out = set()
    for (l1, k1), (l2, k2) in zip(path, path[1:]):
        if l1 == l2:
            out.add((l1, k1, k2))
    return out


def span_map(sl):
    # label each hop with the span of the pointer that ARRIVES at that key
    out = []
    for row in sl.spans():
        keys = [k for k, _ in row]
        sizes = [s for _, s in row]
        out.append({keys[j]: sizes[j - 1] for j in range(1, len(keys))})
    return out


# ---------------------------------------------------------------- fig 1
def fig1(sl):
    fig, ax = canvas(9.9, 4.6, xlim=(0, 10), ylim=(0, 4.6))
    levels = sl.levels()
    path = sl.search_path(13)
    hot = {(i, k) for i, k in path if k is not None}
    draw_lanes(ax, levels, y0=0.75, dy=0.82, highlight=hot,
               hot_edges=path_edges(path))
    text(ax, 5.0, 4.25, 'search(13)', color=TEAL_L, size=17)
    sl.search(13)
    text(ax, 5.0, 0.22,
         f'{sl.steps} forward hops instead of 12 - the express lanes carry us '
         f'over the run 1..9', color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_1.png'))


# ---------------------------------------------------------------- fig 2
def fig2():
    fig, ax = canvas(9.6, 3.6, xlim=(0, 10), ylim=(0, 3.6))
    keys = [1, 2, 3, 4, 5, 6, 7, 8]
    y = 2.35
    for i, k in enumerate(keys):
        box(ax, 0.9 + i * 1.05, y, 0.72, 0.52, label=str(k), size=13)
        if i:
            arrow(ax, (0.9 + (i - 1) * 1.05 + 0.72, y + 0.26),
                  (0.9 + i * 1.05, y + 0.26), lw=1.4, ms=9)
    text(ax, 5.0, 3.25, 'sorted, but not addressable', color=TEAL_L, size=16)
    arrow(ax, (0.5, y + 0.26), (0.88, y + 0.26), lw=1.4, ms=9)
    arrow(ax, (4.6, 1.75), (4.6, y - 0.08), color=RED, lw=2.0, ms=11)
    cross(ax, 4.6, 1.48, label='jump to the middle')
    ll = SL.SortedLinkedList(list(range(0, 200, 2)))
    ll.search(198)
    _, probes = SL.binary_search_needs_random_access(list(range(0, 200, 2)), 198)
    text(ax, 5.0, 0.26,
         f'100 keys, searching the last one:  linked list {ll.steps} hops   '
         f'vs   array {probes} probes',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_2.png'))


# ---------------------------------------------------------------- fig 3
def fig3():
    fig, ax = canvas(9.6, 5.2, xlim=(0, 10), ylim=(0, 5.2))
    text(ax, 5.0, 4.88, 'the level is a coin, not a computation',
         color=TEAL_L, size=16)
    for i in range(4):
        x = 1.15 + i * 1.5
        node(ax, x, 4.05, 'H', r=0.28, size=12)
        if i < 3:
            arrow(ax, (x + 0.32, 4.05), (x + 1.18, 4.05), lw=1.4, ms=9)
            text(ax, x + 0.75, 4.38, 'p', color=AMBER, size=12)
    arrow(ax, (5.97, 4.05), (6.87, 4.05), color=RED, lw=1.4, ms=9)
    node(ax, 7.15, 4.05, 'T', r=0.28, size=12, edge=RED)
    text(ax, 7.15, 3.58, 'stop', color=RED, size=12)
    for i, lab in enumerate(('L1', 'L2', 'L3', 'L4')):
        text(ax, 1.15 + i * 1.5, 3.58, lab, color=TEAL_L, size=12)

    n = 20000
    sl, counts = SL.level_distribution(0.5, n, seed=0)
    rows = [['tower height', 'measured', 'theory  p^(k-1)']]
    for k in range(1, 6):
        rows.append([f'>= {k}',
                     f'{sum(v for lv, v in counts.items() if lv >= k) / n:.1%}',
                     f'{0.5 ** (k - 1):.1%}'])
    y0, h = 2.85, 0.44
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            box(ax, 1.4 + c * 2.45, y0 - r * h, 2.3, h - 0.02, label=val,
                fill=GREY if r == 0 else NAVY, size=12)
    text(ax, 5.0, 0.24,
         f'p = 0.5, n = {n}: the tower heights land on the geometric '
         f'distribution on their own',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_3.png'))


# ---------------------------------------------------------------- fig 4
def fig4(sl):
    fig, ax = canvas(9.9, 4.6, xlim=(0, 10), ylim=(0, 4.6))
    levels = sl.levels()
    sp = span_map(sl)
    sp[0] = {}          # every level-0 span is 1; the labels would be noise
    hot, hot_e, hops = set(), set(), []
    node_ = sl.head
    for i in range(sl.level - 1, -1, -1):
        while node_.forward[i] is not None and node_.forward[i].key <= 13:
            hops.append(node_.span[i])
            hot_e.add((i, node_.key, node_.forward[i].key))
            node_ = node_.forward[i]
            hot.add((i, node_.key))
    draw_lanes(ax, levels, y0=0.75, dy=0.82, highlight=hot, hot_edges=hot_e,
               spans=sp)
    text(ax, 5.0, 4.25, 'rank(13) = sum of the spans you jumped',
         color=TEAL_L, size=16)
    text(ax, 5.0, 0.22,
         ' + '.join(str(h) for h in hops) + f' = {sum(hops)}, so rank = '
         f'{sl.rank(13)} (0-based). Every level-0 span is 1. This is ZRANK.',
         color=AMBER, size=13)
    save(fig, os.path.join(HERE, 'day31_4.png'))


# ---------------------------------------------------------------- fig 5
def fig5():
    fig, ax = canvas(9.6, 4.7, xlim=(0, 10), ylim=(0, 4.7))
    text(ax, 5.0, 4.40, 'the span you forgot to bump', color=TEAL_L, size=16)
    ys = {0: 0.95, 1: 2.15, 2: 3.25}
    for lv, y in ys.items():
        text(ax, 9.5, y, f'L{lv}', color=TEAL_L, size=13, ha='left')
    box(ax, 0.5, 0.74, 0.32, 2.72, fill=GREY, edge=TEAL)
    xs = {'A': 2.0, 'NEW': 4.4, 'B': 6.8, 'C': 8.6}

    def cell(name, lv, hot=False):
        box(ax, xs[name] - 0.42, ys[lv] - 0.21, 0.84, 0.42, label=name,
            size=12, edge=AMBER if hot else TEAL, lw=2.6 if hot else 1.6)

    for name in ('A', 'NEW', 'B', 'C'):
        cell(name, 0, hot=(name == 'NEW'))
    for name in ('A', 'NEW', 'B'):
        cell(name, 1, hot=(name == 'NEW'))
    cell('A', 2)
    cell('C', 2)

    prev = (0.82, ys[0])
    for name in ('A', 'NEW', 'B', 'C'):
        arrow(ax, prev, (xs[name] - 0.42, ys[0]), lw=1.4, ms=9)
        prev = (xs[name] + 0.42, ys[0])
    arrow(ax, (0.82, ys[1]), (xs['A'] - 0.42, ys[1]), lw=1.4, ms=9)
    for a, b in (('A', 'NEW'), ('NEW', 'B')):
        arrow(ax, (xs[a] + 0.42, ys[1]), (xs[b] - 0.42, ys[1]),
              color=AMBER, lw=2.0, ms=10)
        text(ax, (xs[a] + xs[b]) / 2, ys[1] + 0.30, '1', color=AMBER, size=12)
    text(ax, 5.0, ys[1] - 0.66,
         'the levels NEW reaches: insert splits the old span, 2 = 1 + 1',
         color=AMBER, size=12)

    arrow(ax, (0.82, ys[2]), (xs['A'] - 0.42, ys[2]), lw=1.4, ms=9)
    arrow(ax, (xs['A'] + 0.42, ys[2]), (xs['C'] - 0.42, ys[2]),
          color=RED, lw=2.0, ms=10)
    text(ax, (xs['A'] + xs['C']) / 2, ys[2] + 0.30,
         'still 2, should be 3', color=RED, size=12)
    cross(ax, (xs['A'] + xs['C']) / 2, ys[2], r=0.22)

    rep = SL.span_bug_report()
    text(ax, 5.0, 0.28,
         f'keys, search and range stay correct; ranks do not - '
         f'{rep["wrong_rank_count"]}/200 wrong, the first at key '
         f'{rep["first_wrong_key"]} ({rep["buggy_rank"]} instead of '
         f'{rep["expected_rank"]})',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_5.png'))


# ---------------------------------------------------------------- fig 6
def fig6():
    fig, ax = canvas(9.0, 3.6, xlim=(0, 9), ylim=(0, 3.6))
    text(ax, 4.5, 3.35, 'choosing p', color=TEAL_L, size=16)
    rows = [['p', 'pointers / node', 'top level', 'hops / search']]
    for r in SL.p_tradeoff():
        rows.append([f'{r["p"]}', f'{r["ptr_per_node"]:.2f}',
                     f'{r["top_level"]}', f'{r["avg_steps"]:.1f}'])
    w, h = 2.0, 0.48
    x0, y0 = 0.5, 2.55
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            redis = (ri == 2)
            box(ax, x0 + ci * w, y0 - ri * h, w, h, label=val,
                fill=GREY if ri == 0 else NAVY,
                edge=AMBER if redis else TEAL,
                lw=2.6 if redis else 1.6, size=13)
    text(ax, 8.55, y0 - 2 * h + h / 2, 'Redis', color=AMBER, size=13, ha='left')
    text(ax, 4.5, 0.55, 'n = 20000, 2000 random searches',
         color=TEAL_L, size=13)
    text(ax, 4.5, 0.18,
         'p = 0.25 keeps a third of the optional pointers (0.34 vs 1.00 on top '
         'of the mandatory one) and pays 35% more hops',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_6.png'))


# ---------------------------------------------------------------- fig 7
def fig7():
    fig, ax = canvas(9.6, 5.0, xlim=(0, 9.6), ylim=(0, 5.0))
    text(ax, 4.8, 4.75, 'same layers, different meaning of "next"',
         color=TEAL_L, size=16)
    text(ax, 2.4, 4.30, 'skip list', color=TEAL_L, size=14)
    text(ax, 7.2, 4.30, 'HNSW', color=TEAL_L, size=14)
    bands = {2: 3.65, 1: 2.65, 0: 1.65}
    for lv, y in bands.items():
        region(ax, 0.35, y - 0.42, 4.1, 0.84)
        region(ax, 5.15, y - 0.42, 4.1, 0.84)
        text(ax, 0.12, y, f'L{lv}', color=TEAL_L, size=12, ha='left')

    # ---- left: keys on a line
    lanes = {0: [1, 2, 3, 4, 5, 6], 1: [2, 4, 6], 2: [4]}
    for lv, row in lanes.items():
        y, prev = bands[lv], None
        for k in row:
            x = 0.85 + (k - 1) * 0.62
            node(ax, x, y, str(k), r=0.21, size=11)
            if prev is not None:
                arrow(ax, (prev + 0.21, y), (x - 0.21, y), lw=1.3, ms=8)
            prev = x

    # ---- right: points in the plane
    rnd = __import__('random').Random(4)
    slots = [5.70, 6.95, 6.30, 7.55, 8.75, 8.15]
    px = {i: slots[i] for i in range(6)}
    members = {0: [0, 1, 2, 3, 4, 5], 1: [1, 3, 5], 2: [3]}
    links = {0: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)],
             1: [(1, 3), (3, 5)], 2: []}
    for lv, ids in members.items():
        y = bands[lv]
        jit = {i: y + (rnd.random() - 0.5) * 0.62 for i in ids}
        for a, b in links[lv]:
            arrow(ax, (px[a], jit[a]), (px[b], jit[b]), lw=1.2, ms=7,
                  style='-')
        for i in ids:
            node(ax, px[i], jit[i], '', r=0.14)
        if lv == 2:
            text(ax, px[3] + 0.40, y, 'entry', color=AMBER, size=11, ha='left')
    for lv in (2, 1):
        arrow(ax, (px[3], bands[lv] - 0.44), (px[3], bands[lv - 1] + 0.42),
              color=AMBER, lw=2.0, ms=11)

    text(ax, 4.8, 0.98,
         'the level of a point is the same geometric draw, written as '
         'floor(-ln(U) / ln M)', color=TEAL, size=13)
    h = SL.hnsw_report(n=8000, queries=100)
    text(ax, 4.8, 0.50,
         f'8000 points, recall@1 = {h["recall@1"]:.2f}: '
         f'{h["hnsw_dist_calls"]} distance computations instead of '
         f'{h["brute_dist_calls"]} ({h["speedup"]:.0f}x)',
         color=TEAL, size=13)
    save(fig, os.path.join(HERE, 'day31_7.png'))


def main():
    sl = build_demo_list()
    fig1(sl)
    fig2()
    fig3()
    fig4(sl)
    fig5()
    fig6()
    fig7()
    print('figures written')


if __name__ == '__main__':
    main()
