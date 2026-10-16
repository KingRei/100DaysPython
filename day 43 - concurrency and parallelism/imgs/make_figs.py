"""Day 43 figures.  Every number is computed by importing concurrency.py - nothing typed by hand."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

from matplotlib.patches import Rectangle          # noqa: E402
from diagram_style import *                       # noqa: F401,F403,E402
import concurrency as cc                          # noqa: E402

os.chdir(HERE)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def bar(ax, x0, x1, y, h=0.34, fill=NAVY, edge_=TEAL, lw=1.4, hatch=None, z=3):
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=fill, edgecolor=edge_,
                           linewidth=lw, hatch=hatch, zorder=z))


def wait(ax, x0, x1, y, color=AMBER):
    ax.plot([x0, x1], [y, y], color=color, lw=1.6, ls=(0, (3, 2)), zorder=3)


# ---------------------------------------------------------------- figure 1
def fig1():
    """Two CPU jobs on threads / processes, and two waits on threads."""
    fig, ax = W(5.2)
    X0, U = 2.3, 3.3            # x of t=0, width of one job
    rows = [('2 threads\nCPU work', 4.85), ('2 processes\nCPU work', 3.0), ('2 threads\nwaiting', 1.15)]
    for lab, y in rows:
        text(ax, 0.05, y, lab, color=TEAL_L, size=13, ha='left', glowing=False)
    # threads: one GIL, slices alternate, total 2U
    y = rows[0][1]
    k = 8
    for s in range(2 * k):
        lane = s % 2
        x0 = X0 + s * U / k
        bar(ax, x0, x0 + U / k, y + (0.25 if lane == 0 else -0.25), fill=NAVY if lane == 0 else GREY)
    text(ax, X0 + 2 * U + 0.15, y, 'one at a time:\n2 x the time', color=RED, size=12, ha='left', glowing=False)
    # processes: both lanes full
    y = rows[1][1]
    bar(ax, X0, X0 + U, y + 0.25)
    bar(ax, X0, X0 + U, y - 0.25, fill=GREY)
    text(ax, X0 + U + 0.15, y, 'two cores, both busy:\n1 x the time', color=TEAL_L, size=12, ha='left', glowing=False)
    # waits: tiny CPU then dashed waits that overlap
    y = rows[2][1]
    for lane, off in ((0, 0.0), (1, 0.12)):
        yy = y + (0.25 if lane == 0 else -0.25)
        bar(ax, X0 + off, X0 + off + 0.12, yy, fill=NAVY if lane == 0 else GREY)
        wait(ax, X0 + off + 0.12, X0 + off + U, yy)
    text(ax, X0 + U + 0.3, y, 'the GIL is released\nwhile a thread waits', color=TEAL_L, size=12, ha='left', glowing=False)
    ax.plot([X0, X0 + 2 * U], [0.55, 0.55], color=TEAL, lw=1.2)
    text(ax, X0 + 2 * U + 0.1, 0.55, 'time', color=TEAL, size=12, ha='left', glowing=False)
    legend(ax, 0.3, 0.05, [(NAVY, 'job A running')], size=11)
    legend(ax, 3.3, 0.05, [(GREY, 'job B running')], size=11)
    ax.plot([6.3, 6.56], [0.05, 0.05], color=AMBER, lw=1.6, ls=(0, (3, 2)))
    text(ax, 6.72, 0.05, 'waiting (I/O, sleep)', color=TEAL, size=11, ha='left', glowing=False)
    save(fig, 'day43_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """All six interleavings of two `x += 1`."""
    runs = list(cc.every_schedule([cc.incrementer(), cc.incrementer()], {'x': 0}))
    fig, ax = W(4.9)
    ops = ['read', 'write']
    text(ax, 5.0, 5.15, 'two threads, x = 0, each runs  r = x;  x = r + 1', color=TEAL_L, size=13, glowing=False)
    for row, (sched, out) in enumerate(runs):
        y = 4.45 - row * 0.72
        pos = [0, 0]
        for col, i in enumerate(sched):
            name = 'AB'[i] + '.' + ops[pos[i]]
            pos[i] += 1
            box(ax, 0.6 + col * 1.75, y - 0.27, 1.6, 0.54, label=name, fill=NAVY if i == 0 else GREY, size=12)
        x = cc.final_x(out)
        ok = x == 2
        text(ax, 8.25, y, f'x = {x}', color=TEAL_L if ok else RED, size=14, glowing=False)
        if not ok:
            cross(ax, 9.35, y, r=0.18)
    c = cc.xcount(cc.count_schedules([cc.incrementer()] * 3, {'x': 0}))
    tot = sum(c.values())
    text(ax, 5.0, 0.15, f'{sum(1 for _, o in runs if cc.final_x(o) == 1)} of {len(runs)} lose an update;'
         f'  with 3 threads only {c[3]} of {tot} are right', color=AMBER, size=13, glowing=False)
    save(fig, 'day43_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Five philosophers each holding their left fork."""
    fig, ax = W(5.6)
    cx, cy, R, RF = 2.9, 3.1, 2.0, 1.25
    n = 5
    pts = []
    for i in range(n):
        a = math.pi / 2 - 2 * math.pi * i / n
        pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    for i in range(n):
        a = math.pi / 2 - 2 * math.pi * (i + 0.5) / n          # fork i+1 sits between i and i+1
        fx, fy = cx + RF * math.cos(a), cy + RF * math.sin(a)
        node(ax, fx, fy, f'f{(i + 1) % n}', r=0.3, fill=GREY, size=11)
    for i, (x, y) in enumerate(pts):
        node(ax, x, y, f'P{i}', r=0.42)
    # P_i holds fork i (left) and waits for fork i+1 (right): arrows form a cycle
    for i in range(n):
        a_hold = math.pi / 2 - 2 * math.pi * (i - 0.5) / n
        a_want = math.pi / 2 - 2 * math.pi * (i + 0.5) / n
        hx, hy = cx + RF * math.cos(a_hold), cy + RF * math.sin(a_hold)
        wx, wy = cx + RF * math.cos(a_want), cy + RF * math.sin(a_want)
        ax.plot([pts[i][0], hx], [pts[i][1], hy], color=TEAL_L, lw=3.0, zorder=2)
        arrow(ax, pts[i], (wx, wy), color=RED, lw=1.4, ls='--', shrinkA=14, shrinkB=12)
    st, dead = cc.reachable_states(cc.philosophers(5), {})
    c = cc.count_schedules(cc.philosophers(5), {})
    tot = sum(c.values())
    st2, dead2 = cc.reachable_states(cc.philosophers(5, ordered=True), {})
    x = 5.6
    text(ax, x, 5.6, 'everyone took the left fork,', color=TEAL_L, size=13, ha='left', glowing=False)
    text(ax, x, 5.2, 'everyone waits for the right one', color=RED, size=13, ha='left', glowing=False)
    text(ax, x, 4.3, f'{st:,} reachable states,', color=TEAL, size=13, ha='left', glowing=False)
    text(ax, x, 3.9, f'{dead} of them stuck', color=TEAL, size=13, ha='left', glowing=False)
    text(ax, x, 3.1, f'{c[cc.DEADLOCK]} of {tot:,}', color=AMBER, size=13, ha='left', glowing=False)
    text(ax, x, 2.7, f'interleavings end here', color=AMBER, size=13, ha='left', glowing=False)
    text(ax, x, 2.3, f'(1 in {tot // c[cc.DEADLOCK]:,})', color=AMBER, size=13, ha='left', glowing=False)
    text(ax, x, 1.4, 'fix: lower-numbered fork first', color=TEAL_L, size=13, ha='left', glowing=False)
    text(ax, x, 1.0, f'{st2:,} states, {dead2} stuck', color=TEAL_L, size=13, ha='left', glowing=False)
    ax.plot([0.9, 1.3], [0.45, 0.45], color=TEAL_L, lw=3.0)
    text(ax, 1.45, 0.45, 'holds', color=TEAL, size=11, ha='left', glowing=False)
    arrow(ax, (2.4, 0.45), (2.85, 0.45), color=RED, lw=1.4, ls='--')
    text(ax, 3.0, 0.45, 'waits for', color=TEAL, size=11, ha='left', glowing=False)
    save(fig, 'day43_3.png')


# ---------------------------------------------------------------- figure 4
def dag_levels(g, outputs):
    """x position per addition node: spread nodes of the same level."""
    by = {}
    for k, lv in enumerate(g.level):
        by.setdefault(lv, []).append(k)
    return by


def draw_dag(ax, g, x0, x1, y_top, dy, n, title, color):
    xs_in = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
    for i, x in enumerate(xs_in):
        box(ax, x - 0.17, y_top - 0.17, 0.34, 0.34, label=str(i + 1), fill=GREY, size=10)
    pos = {}
    for k, (a, b) in enumerate(g.deps):
        def xy(v):
            return (xs_in[v[1]], y_top) if v[0] == 'in' else pos[v[1]]
        pa, pb = xy(a), xy(b)
        y = y_top - g.level[k] * dy
        x = pb[0] if g.level[k] > 0 else pb[0]
        x = (pa[0] + pb[0]) / 2 if title.startswith('tree') else pb[0]
        pos[k] = (x, y)
        for p in (pa, pb):
            ax.plot([p[0], x], [p[1] - 0.17, y + 0.16], color=TEAL, lw=1.0, zorder=2)
        node(ax, x, y, '+', r=0.16, size=10)
    return pos


def fig4():
    fig, ax = W(5.4)
    n = 8
    gc, _ = cc.sum_chain(n)
    gt, _ = cc.sum_tree(n)
    draw_dag(ax, gc, 0.2, 4.6, 5.55, 0.62, n, 'chain', TEAL)
    draw_dag(ax, gt, 5.4, 9.8, 5.55, 0.9, n, 'tree', TEAL)
    t = {p: len(gt.schedule(p)) for p in (1, 2, 4)}
    text(ax, 2.4, 0.55, f'loop: work {gc.work}, depth {gc.depth}', color=AMBER, size=13, glowing=False)
    text(ax, 2.4, 0.15, f'extra workers never help', color=AMBER, size=13, glowing=False)
    text(ax, 7.6, 1.75, f'tree: work {gt.work}, depth {gt.depth}', color=TEAL_L, size=13, glowing=False)
    text(ax, 7.6, 1.35, f'1 / 2 / 4 workers: {t[1]} / {t[2]} / {t[4]} steps', color=TEAL_L, size=13, glowing=False)
    save(fig, 'day43_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """Steps T_p against workers p for three scans of 1024 numbers (log-log)."""
    fig, ax = W(5.4)
    X0, X1, Y0, Y1 = 1.2, 9.0, 0.8, 5.5
    ps = [2 ** k for k in range(11)]
    curves = []
    for name, b, col in (('loop', cc.scan_serial, GREY_L), ('Hillis-Steele', cc.scan_hillis_steele, RED),
                         ('Blelloch', cc.scan_blelloch, TEAL_L)):
        g, _ = b(1024)
        curves.append((name, col, [(p, len(g.schedule(p))) for p in ps], g))
    lx = lambda p: X0 + (X1 - X0) * math.log2(p) / 10
    lo, hi = math.log10(8), math.log10(10000)
    ly = lambda t: Y0 + (Y1 - Y0) * (math.log10(t) - lo) / (hi - lo)
    ax.plot([X0, X1], [Y0, Y0], color=TEAL, lw=1.4)
    ax.plot([X0, X0], [Y0, Y1], color=TEAL, lw=1.4)
    for p in (1, 4, 16, 64, 256, 1024):
        ax.plot([lx(p)] * 2, [Y0, Y0 - 0.1], color=TEAL, lw=1.2)
        text(ax, lx(p), Y0 - 0.3, str(p), color=TEAL, size=11, glowing=False)
    for t in (10, 100, 1000, 10000):
        ax.plot([X0, X0 - 0.1], [ly(t)] * 2, color=TEAL, lw=1.2)
        text(ax, X0 - 0.18, ly(t), f'{t:,}', color=TEAL, size=11, ha='right', glowing=False)
    text(ax, (X0 + X1) / 2, 0.08, 'workers p', color=TEAL, size=12, glowing=False)
    text(ax, 0.1, Y1 + 0.35, 'steps', color=TEAL, size=12, ha='left', glowing=False)
    for name, col, pts, g in curves:
        ax.plot([lx(p) for p, _ in pts], [ly(t) for _, t in pts], color=col, lw=2.0, zorder=3,
                marker='o', ms=3.5)
        p, t = pts[-1]
        dy = {'Blelloch': 0.42, 'Hillis-Steele': 0.0}.get(name, 0)
        text(ax, lx(p) + 0.12, ly(t) + dy, f'{name}\nW={g.work:,} D={g.depth:,}', color=col, size=11,
             ha='left', glowing=False)
    hs, bl = curves[1][2], curves[2][2]
    t4h = dict(hs)[4]
    t4b = dict(bl)[4]
    text(ax, lx(4) + 0.2, ly(t4h) + 0.35, f'4 workers: {t4h:,} steps, slower than the loop', color=RED,
         size=11, ha='left', glowing=False)
    text(ax, lx(4) - 0.12, ly(t4b) - 0.12, f'{t4b} steps', color=TEAL_L, size=11, ha='right', glowing=False)
    save(fig, 'day43_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Ten 0.1 s fetches under asyncio: await, a blocking call, a Semaphore(3)."""
    fig, ax = W(5.2)
    X0, U = 2.6, 0.62            # x of t=0, width of 0.1 s
    for gi, (name, fetch, limit) in enumerate((('await asyncio.sleep', cc.fetch_await, None),
                                               ('time.sleep in async', cc.fetch_blocking, None),
                                               ('Semaphore(3)', cc.fetch_await, 3))):
        secs, _, log = cc.run_async(fetch, limit=limit)
        start = {i: round(t, 1) for i, k, t in log if k == 'start'}
        ytop = 5.3 - gi * 1.75
        text(ax, 0.05, ytop - 0.6, name, color=TEAL_L, size=12, ha='left', glowing=False)
        text(ax, 0.05, ytop - 0.95, f'{secs:.1f} s', color=RED if secs > 0.5 else AMBER, size=12,
             ha='left', glowing=False)
        for i in range(10):
            y = ytop - i * 0.14
            x = X0 + start[i] / 0.1 * U
            ax.plot([x, x + U], [y, y], color=RED if fetch is cc.fetch_blocking else TEAL_L,
                    lw=3.2, solid_capstyle='butt', zorder=3)
    ax.plot([X0, X0 + 10 * U], [0.2, 0.2], color=TEAL, lw=1.2)
    for k in range(0, 11, 2):
        ax.plot([X0 + k * U] * 2, [0.2, 0.1], color=TEAL, lw=1.0)
        text(ax, X0 + k * U, -0.1, f'{k / 10:.1f}', color=TEAL, size=10, glowing=False)
    text(ax, X0 + 10 * U + 0.1, 0.2, 's', color=TEAL, size=11, ha='left', glowing=False)
    save(fig, 'day43_6.png')


# ---------------------------------------------------------------- figure 7
def fig7():
    """Map, shuffle, reduce - and two mappers that disagree on the bucket."""
    fig, ax = W(5.0)
    docs = cc.make_corpus()
    truth = cc.Counter(w for d in docs for w in d)
    sim = cc.mapreduce(docs, salts=[b'mapper-0', b'mapper-1'])
    wrong = sorted((w for w in truth if sim[w] != truth[w]), key=lambda w: -truth[w])
    w = wrong[0]
    b0 = cc.stable_bucket(w, 4, b'mapper-0')
    b1 = cc.stable_bucket(w, 4, b'mapper-1')
    half0 = sum(cc.map_doc(d, 4, b'mapper-0')[b0][w] for d in docs[0::2])
    half1 = sum(cc.map_doc(d, 4, b'mapper-1')[b1][w] for d in docs[1::2])
    text(ax, 1.0, 5.45, 'map', color=TEAL_L, size=14)
    text(ax, 4.8, 5.45, 'shuffle by bucket', color=TEAL_L, size=14, glowing=False)
    text(ax, 8.6, 5.45, 'reduce', color=TEAL_L, size=14)
    box(ax, 0.2, 3.4, 1.6, 0.8, label='mapper 0\neven docs', size=11)
    box(ax, 0.2, 1.6, 1.6, 0.8, label='mapper 1\nodd docs', size=11, fill=GREY)
    for r in range(4):
        y = 4.4 - r * 1.05
        box(ax, 7.8, y - 0.3, 1.6, 0.6, label=f'reducer {r}', size=11)
    ry = lambda r: 4.4 - r * 1.05
    for r in range(4):
        arrow(ax, (1.8, 3.8), (7.8, ry(r)), color=TEAL, lw=0.8)
        arrow(ax, (1.8, 2.0), (7.8, ry(r)), color=TEAL, lw=0.8)
    arrow(ax, (1.8, 3.8), (7.8, ry(b0)), color=AMBER, lw=2.2)
    arrow(ax, (1.8, 2.0), (7.8, ry(b1)), color=RED, lw=2.2)
    m0 = (3.8 + ry(b0)) / 2
    m1 = (2.0 + ry(b1)) / 2
    text(ax, 4.8, m0 + (0.42 if ry(b0) >= 3.8 else -0.42), f'"{w}": {half0:,}  (bucket {b0})', color=AMBER, size=12, glowing=False)
    text(ax, 4.8, m1 + (0.42 if ry(b1) >= 2.0 else -0.42), f'"{w}": {half1:,}  (bucket {b1})', color=RED, size=12, glowing=False)
    text(ax, 5.0, 0.15, f'two reducers each count half of "{w}": output {sim[w]:,}, truth {truth[w]:,};'
         f' {len(wrong)} of {len(truth)} words wrong', color=RED, size=12, glowing=False)
    save(fig, 'day43_7.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print('figures written')
