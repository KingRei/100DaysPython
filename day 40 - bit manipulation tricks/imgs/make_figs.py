"""Day 40 figures.  Every number is imported from bit_tricks.py - nothing typed by hand."""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

from matplotlib.patches import Rectangle   # noqa: E402
from diagram_style import *                # noqa: F401,F403,E402
import bit_tricks as bt                    # noqa: E402

os.chdir(HERE)


def W(h):
    """canvas(9, h) with a 0..10 x-range and equal aspect."""
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def bitrow(ax, x0, y, value, n, w, h, digits=True, amber=(), red=(), size=13):
    """Low n bits of value, most significant on the left; returns {bit: centre x}."""
    xs = {}
    for i in range(n):
        k = n - 1 - i
        one = (value >> k) & 1
        fill = TEAL_L if (one and not digits) else NAVY
        edge_, lw = TEAL, 1.2
        if k in amber:
            edge_, lw = AMBER, 2.6
        if k in red:
            edge_, lw = RED, 2.6
        ax.add_patch(Rectangle((x0 + i * w, y), w, h, facecolor=fill if (one or digits) else NAVY,
                               edgecolor=edge_, linewidth=lw, zorder=4 if k in amber or k in red else 3))
        if digits:
            text(ax, x0 + i * w + w / 2, y + h / 2, str(one), color=WHITE if one else GREY_L,
                 size=size, glowing=False, zorder=6)
        xs[k] = x0 + i * w + w / 2
    return xs


# ---------------------------------------------------------------- figure 1
def fig1():
    """5 and -5 as endless bit strings, and what the 32-bit mask keeps."""
    H = 4.9
    fig, ax = W(H)
    n, w, h = 36, 0.225, 0.42
    x0 = 1.55
    rows = [('5', 5, f'bin(5) = {bin(5)}'),
            ('-5', -5, f"bin(-5) = '{bin(-5)}'   bit_count() = {(-5).bit_count()}"),
            ('-5 & M32', bt.u32(-5), f'u32(-5) = {bt.u32(-5):,}   bit_count() = {bt.popcount32(-5)}')]
    ytop = 4.3
    for i, (lab, v, note) in enumerate(rows):
        y = ytop - i * 1.25
        text(ax, 1.05, y + h / 2, lab, color=TEAL_L, size=15, ha='right')
        if i < 2:
            text(ax, x0 + 0.02, y + h / 2, '...', color=TEAL_L, size=13, ha='right', glowing=False)
        bitrow(ax, x0 + 0.08, y, v, n, w, h, digits=False)
        text(ax, x0 + 0.08, y - 0.28, note, color=TEAL, size=13, ha='left', glowing=False)
    # wall between bit 32 and bit 31
    xw = x0 + 0.08 + (n - 32) * w
    ax.plot([xw, xw], [0.95, ytop + h + 0.25], color=RED, lw=2.0, ls='--', zorder=7)
    text(ax, xw - 0.1, ytop + h + 0.42, 'bits 32 and up', color=RED, size=13, ha='right')
    text(ax, xw + 0.1, ytop + h + 0.42, 'a C int keeps bits 31 ... 0', color=AMBER, size=13, ha='left',
         glowing=False)
    text(ax, 5.0, 0.55, f'read bit 31 as -2**31:  s32({bt.u32(-5):,}) = {bt.s32(bt.u32(-5))}',
         color=AMBER, size=14, glowing=False)
    legend(ax, 8.3, 1.55, [(TEAL_L, 'bit = 1'), (NAVY, 'bit = 0')], size=12, dy=0.34)
    save(fig, 'day40_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """The three lowest-bit tricks on x = 180, each as x, helper, result."""
    x = 180
    groups = [('drop the lowest 1', [('x', x, ()), ('x - 1', x - 1, (0, 1, 2)),
                                     ('x & (x - 1)', bt.clear_lowest(x), (2,))]),
              ('keep only the lowest 1', [('x', x, ()), ('-x', -x, ()),
                                          ('x & -x', bt.lowbit(x), (2,))]),
              ('turn on the lowest 0', [('x', x, ()), ('x + 1', x + 1, (0,)),
                                        ('x | (x + 1)', bt.set_lowest_zero(x), (0,))])]
    H = 8.3
    fig, ax = W(H)
    w, h = 0.55, 0.5
    x0 = 3.2
    y = 8.75
    for title, rows in groups:
        text(ax, 0.15, y + 0.1, title, color=AMBER, size=15, ha='left', glowing=False)
        y -= 0.72
        for lab, v, amb in rows:
            text(ax, 2.95, y + h / 2, lab, color=TEAL_L, size=15, ha='right', glowing=False)
            bitrow(ax, x0 + 0.02, y, v, 8, w, h, amber=amb)
            note = f'{v}   (1s continue forever above)' if v < 0 else f'{v}'
            text(ax, x0 + 8 * w + 0.35, y + h / 2, note, color=TEAL, size=15, ha='left', glowing=False)
            y -= 0.6
        ax.plot([x0 - 0.1, x0 + 8 * w + 0.1], [y + 0.5 + 0.03, y + 0.5 + 0.03], color=TEAL, lw=0)  # spacer
        y -= 0.28
    save(fig, 'day40_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Kernighan on 180: one round per 1, versus the shift loop's one round per bit."""
    tr = bt.kernighan_trace(180)
    H = 4.6
    fig, ax = W(H)
    w, h = 0.55, 0.46
    x0 = 3.0
    y = 4.35
    for i, v in enumerate(tr):
        drop = ()
        if i + 1 < len(tr):
            drop = ((v & -v).bit_length() - 1,)
        lab = 'x' if i == 0 else f'round {i}'
        text(ax, 2.75, y + h / 2, lab, color=TEAL_L, size=15, ha='right', glowing=False)
        bitrow(ax, x0, y, v, 8, w, h, red=drop)
        text(ax, x0 + 8 * w + 0.3, y + h / 2, f'{v}', color=TEAL, size=15, ha='left', glowing=False)
        y -= 0.64
    text(ax, 5.0, 0.7, f'Kernighan: {len(tr) - 1} rounds (one per 1)      '
         f'shift loop: {(180).bit_length()} rounds (one per bit up to the top 1)',
         color=AMBER, size=13.5, glowing=False)
    text(ax, 8.65, 3.35, 'red = the 1\nthat x & (x - 1)\nremoves next', color=RED, size=12,
         ha='left', glowing=False)
    save(fig, 'day40_3.png')


# ---------------------------------------------------------------- figure 4
def measure():
    rng = random.Random(40)
    xs = [rng.getrandbits(32) for _ in range(200_000)]
    out = []
    for name, fn in bt.POPCOUNTS:
        out.append((name, bt.best_time(lambda: [fn(v) for v in xs], repeat=3) * 1000))
    return sorted(out, key=lambda t: t[1])


def fig4(times):
    """popcount timings: the builtin wins, the clever SWAR does not."""
    H = 4.8
    fig, ax = W(H)
    top = max(t for _, t in times)
    x0, span = 3.4, 5.2
    y = 4.6
    for name, t in times:
        wbar = max(0.04, span * t / top)
        hot = name == 'x.bit_count()'
        ax.add_patch(Rectangle((x0, y), wbar, 0.46, facecolor=NAVY,
                               edgecolor=AMBER if hot else TEAL, linewidth=2.4 if hot else 1.4, zorder=3))
        text(ax, x0 - 0.15, y + 0.23, name, color=AMBER if hot else TEAL_L, size=14, ha='right',
             glowing=False)
        text(ax, x0 + wbar + 0.12, y + 0.23, f'{t:.0f} ms', color=TEAL, size=13, ha='left', glowing=False)
        y -= 0.7
    text(ax, 5.0, 0.35, '200,000 random 32-bit ints, best of 3 (CPython 3.10)', color=TEAL,
         size=12.5, glowing=False)
    save(fig, 'day40_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """LeetCode 371 on (-1, 1): the carry marches left; with a mask it falls off at bit 32."""
    naive = bt.get_sum_trace(-1, 1, cap=40)
    masked = bt.get_sum_trace(-1, 1, cap=40, mask=True)
    n, w, h = 36, 0.2, 0.3
    x0 = 2.45
    H = 6.3
    fig, ax = W(H)
    rows = [('round 0', naive[0]), ('round 1', naive[1]), ('round 2', naive[2]),
            ('round 3', naive[3]), ('round 31', naive[31])]
    y = 6.2
    text(ax, x0 + n * w / 2, y + 0.55, 'a (top strip) and the carry b (bottom strip)',
         color=TEAL, size=13, glowing=False)
    ys = []
    for lab, (a, b) in rows:
        text(ax, x0 - 0.2, y + 0.02, lab, color=TEAL_L, size=13.5, ha='right', glowing=False)
        bitrow(ax, x0, y, a, n, w, h, digits=False)
        bitrow(ax, x0, y - h, b, n, w, h, digits=False, amber={b.bit_length() - 1})
        ys.append(y)
        y -= 0.9
    y -= 0.1
    a, b = masked[32]
    text(ax, x0 - 0.2, y + 0.02, 'round 32\nmasked', color=AMBER, size=13, ha='right', glowing=False)
    bitrow(ax, x0, y, a, n, w, h, digits=False)
    bitrow(ax, x0, y - h, b, n, w, h, digits=False)
    text(ax, x0 + n * w + 0.15, y, f'({a}, {b})\nstops', color=AMBER, size=13, ha='left', glowing=False)
    y -= 0.9
    a, b = naive[32]
    text(ax, x0 - 0.2, y + 0.02, 'round 32\nno mask', color=RED, size=13, ha='right', glowing=False)
    bitrow(ax, x0, y, a, n, w, h, digits=False)
    bitrow(ax, x0, y - h, b, n, w, h, digits=False, amber={b.bit_length() - 1})
    text(ax, x0 + n * w + 0.15, y, 'b = 2**32,\nnever 0', color=RED, size=13, ha='left', glowing=False)
    xw = x0 + (n - 32) * w
    ax.plot([xw, xw], [y - h - 0.15, ys[0] + h + 0.2], color=RED, lw=2.0, ls='--', zorder=7)
    text(ax, xw - 0.08, ys[0] + h + 0.22, 'bit 32', color=RED, size=12, ha='right')
    save(fig, 'day40_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """LeetCode 137 by columns on [-2, -2, -3, -2]: counts mod 3, then bit 31 is the sign."""
    nums = [-2, -2, -3, -2]
    raw = bt.single_number_ii_count_nosign(nums)
    n, w, h = 32, 0.24, 0.36
    x0 = 1.9
    H = 5.0
    fig, ax = W(H)
    y = 4.9
    for v in nums:
        text(ax, x0 - 0.2, y + h / 2, str(v), color=TEAL_L, size=14, ha='right', glowing=False)
        bitrow(ax, x0, y, v, n, w, h, digits=False)
        y -= 0.46
    counts = [sum((v >> k) & 1 for v in nums) for k in range(n)]
    y -= 0.1
    text(ax, x0 - 0.2, y + h / 2, 'count', color=TEAL, size=13, ha='right', glowing=False)
    for i in range(n):
        k = n - 1 - i
        text(ax, x0 + i * w + w / 2, y + h / 2, str(counts[k]), color=TEAL, size=10.5, glowing=False)
    y -= 0.5
    text(ax, x0 - 0.2, y + h / 2, '% 3', color=TEAL, size=13, ha='right', glowing=False)
    bitrow(ax, x0, y, raw, n, w, h, digits=False, amber={31})
    text(ax, x0, y - 0.3, 'bit 31: worth -2**31, not +2**31', color=AMBER, size=12.5, ha='left', glowing=False)
    text(ax, 5.0, 0.95, f'without s32: {raw:,}        with s32: {bt.s32(raw)}',
         color=AMBER, size=14.5, glowing=False)
    text(ax, 5.0, 0.45, 'the XOR state machine (ones / twos) needs no width and returns '
         f'{bt.Solution137().singleNumber(nums)} directly', color=TEAL, size=12.5, glowing=False)
    save(fig, 'day40_6.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3()
    if '--quick' in sys.argv:
        fig5(); fig6(); sys.exit()
    t = measure()
    print('popcount timings (ms):', [(n, round(v, 1)) for n, v in t])
    fig4(t); fig5(); fig6()
    print('wrote day40_1..6.png')
