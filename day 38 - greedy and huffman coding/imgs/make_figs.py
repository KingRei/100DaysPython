"""Day 38 figures.  Every number is imported from greedy.py - nothing typed by hand."""

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: F401,F403
import greedy as g

FREQ = Counter(g.TEXT)
CODES, MERGES = g.huffman(FREQ)
NOHEAP = g.huffman_no_heap(FREQ)

HUFF_BITS = g.encoded_bits(FREQ, CODES)
NAIVE_BITS = g.encoded_bits(FREQ, NOHEAP)
FIXED_W, FIXED_BITS = g.fixed_width_bits(FREQ)

UMAX = max(e for _, e in g.IV)


def xt(u):
    """Slot index -> x on a 1.0 .. 9.0 axis."""
    return 1.0 + u * (8.0 / UMAX)


def timeaxis(ax, y, step=4):
    ax.plot([xt(0), xt(UMAX)], [y, y], color=GREY_L, lw=1.2, zorder=1)
    for u in range(0, UMAX + 1, step):
        ax.plot([xt(u), xt(u)], [y, y - 0.10], color=GREY_L, lw=1.2, zorder=1)
        text(ax, xt(u), y - 0.32, g.clock(u), color=GREY_L, size=11)


def bar(ax, iv, y, h=0.34, fill=NAVY, edge=TEAL, lw=1.6, label=None, size=13,
        color=WHITE, outside=None):
    """Draw one interval.  A label that does not fit inside is moved to the
    right of the bar, because a clipped '10:30-10:45' reads as '0-10'."""
    s, e = iv
    w = xt(e) - xt(s)
    box(ax, xt(s), y - h / 2, w, h, fill=fill, edge=edge, lw=lw)
    if not label:
        return
    # 0.062 data units per point of font size is the measured width of one
    # ASCII glyph on this 10-unit-wide canvas; 0.30 is the padding.
    need = len(label) * size * 0.0062 + 0.30
    if outside is None:
        outside = need > w
    if outside:
        text(ax, xt(e) + 0.18, y, label, color=edge, size=size, ha='left',
             glowing=False, zorder=6)
    else:
        text(ax, (xt(s) + xt(e)) / 2, y, label, color=color, size=size,
             glowing=False, zorder=6)


# ---------------------------------------------------------------- figure 1
def fig1():
    chosen = set(g.select_by_end(g.IV))
    fig, ax = canvas(9, 6, xlim=(0, 10), ylim=(0, 6))

    text(ax, 4.6, 5.70, 'eight requests for one meeting room', color=TEAL_L,
         size=16)
    for i, iv in enumerate(g.IV):
        y = 5.15 - i * 0.50
        hit = iv in chosen
        text(ax, 0.55, y, g.NAME[iv], color=AMBER if hit else GREY_L, size=15)
        bar(ax, iv, y,
            fill=NAVY, edge=AMBER if hit else TEAL, lw=2.6 if hit else 1.6,
            label=g.span(iv), size=11, color=WHITE)
    timeaxis(ax, 1.00)

    text(ax, 4.6, 0.30,
         'keep the one that finishes earliest, drop everything it overlaps, '
         f'repeat  ->  {len(chosen)} meetings',
         color=TEAL, size=12)
    save(fig, 'day38_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    rows = [
        ('sort by finish time', g.select_by_end(g.IV), AMBER),
        ('sort by start time', g.select_by_start(g.IV), TEAL),
        ('sort by duration', g.select_by_duration(g.IV), TEAL),
    ]
    fig, ax = canvas(9, 5.2, xlim=(0, 10), ylim=(0, 5.2))
    text(ax, 5.0, 4.90, 'three sort keys, three legal schedules',
         color=TEAL_L, size=16)

    for i, (title, sel, col) in enumerate(rows):
        y = 4.05 - i * 1.35
        text(ax, 0.30, y + 0.45, title, color=col, size=14, ha='left',
             glowing=False)
        text(ax, 9.70, y + 0.45, f'{len(sel)} meetings', color=col, size=14,
             ha='right', glowing=False)
        ax.plot([xt(0), xt(UMAX)], [y, y], color=GREY, lw=1.0, zorder=1)
        for iv in sel:
            bar(ax, iv, y, h=0.40, fill=NAVY, edge=col,
                lw=2.6 if col is AMBER else 1.8, label=g.NAME[iv], size=14,
                outside=False)
    timeaxis(ax, 0.70)
    text(ax, 5.0, 0.12,
         'none of the three overlaps itself - the two losers are wrong, '
         'not broken',
         color=TEAL, size=13)
    save(fig, 'day38_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    first, unrestricted, forced = g.exchange_argument(g.IV)
    fig, ax = canvas(9, 3.5, xlim=(0, 10), ylim=(0, 3.5))
    text(ax, 5.0, 3.28,
         f'does keeping {g.NAME[first]} ({g.span(first)}) cost anything?',
         color=TEAL_L, size=16)

    # the forced row is the real thing: first + the best subset of what is left,
    # not the greedy's output standing in for it
    rest = [iv for iv in g.IV if g.compatible(iv, first) and iv != first]
    forced_sel = sorted([first] + g.select_bruteforce(rest))
    assert len(forced_sel) == forced
    for i, (title, sel, col) in enumerate([
            ('best over all 2^8 subsets', g.select_bruteforce(g.IV), TEAL),
            (f'best subset forced to contain {g.NAME[first]}', forced_sel, AMBER)]):
        y = 2.60 - i * 0.95
        text(ax, 0.30, y + 0.42, title, color=col, size=14, ha='left',
             glowing=False)
        ax.plot([xt(0), xt(UMAX)], [y, y], color=GREY, lw=1.0, zorder=1)
        for iv in sel:
            bar(ax, iv, y, h=0.40, fill=NAVY,
                edge=AMBER if iv == first else col,
                lw=2.6 if iv == first else 1.8, label=g.NAME[iv], size=14,
                outside=False)
    timeaxis(ax, 0.92)
    text(ax, 5.0, 0.16,
         f'{unrestricted} either way: the earliest finisher never loses, '
         'because it frees the most room',
         color=TEAL, size=12)
    save(fig, 'day38_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    order = sorted(FREQ.items(), key=lambda kv: (-kv[1], kv[0]))
    fig, ax = canvas(9, 5.6, xlim=(0, 10), ylim=(0, 5.6))
    text(ax, 5.0, 5.30, 'rare letters pay, common letters do not',
         color=TEAL_L, size=16)

    n = len(order)
    half = (n + 1) // 2
    for col in range(2):
        x0 = 0.55 + col * 4.9
        text(ax, x0 + 0.30, 4.72, 'char', color=GREY_L, size=12)
        text(ax, x0 + 1.15, 4.72, 'seen', color=GREY_L, size=12)
        text(ax, x0 + 2.55, 4.72, 'codeword', color=GREY_L, size=12)
        chunk = order[col * half:(col + 1) * half]
        for i, (ch, cnt) in enumerate(chunk):
            y = 4.32 - i * 0.44
            shown = "' '" if ch == ' ' else ch
            hot = i == 0 and col == 0
            c = AMBER if hot else TEAL
            box(ax, x0, y - 0.17, 0.60, 0.34, label=shown, fill=NAVY,
                edge=c, lw=2.4 if hot else 1.6, size=13)
            text(ax, x0 + 1.15, y, str(cnt), color=c, size=13)
            text(ax, x0 + 2.55, y, CODES[ch], color=c, size=13)
    text(ax, 5.0, 0.30,
         f'{len(FREQ)} distinct characters, {len(g.TEXT)} of them: '
         f'{FIXED_W} bits each = {FIXED_BITS} bits fixed width, '
         f'{HUFF_BITS} bits with Huffman',
         color=TEAL, size=13)
    save(fig, 'day38_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    items = [
        ('Huffman  (merge the two rarest, every time)', HUFF_BITS, TEAL),
        (f'fixed width  ({FIXED_W} bits per character)', FIXED_BITS, GREY_L),
        ('sorted once, merged left to right', NAIVE_BITS, RED),
    ]
    longest = max(len(c) for c in CODES.values())
    longest_naive = max(len(c) for c in NOHEAP.values())
    fig, ax = canvas(9, 4.2, xlim=(0, 10), ylim=(0, 4.2))
    text(ax, 5.0, 3.90, 'the broken tree still decodes perfectly',
         color=TEAL_L, size=16)

    scale = 8.2 / max(v for _, v, _ in items)
    for i, (lab, val, col) in enumerate(items):
        y = 3.10 - i * 0.85
        text(ax, 0.30, y + 0.30, lab, color=col, size=13, ha='left')
        box(ax, 0.30, y - 0.18, val * scale, 0.36, fill=NAVY, edge=col,
            lw=2.6 if col is RED else 1.8)
        text(ax, 0.30 + val * scale + 0.22, y, f'{val} bits', color=col,
             size=13, ha='left')
    text(ax, 5.0, 0.42,
         f'longest codeword: {longest} bits with Huffman, '
         f'{longest_naive} bits without re-sorting',
         color=GREY_L, size=13)
    text(ax, 5.0, 0.10,
         'both are prefix-free, both decode back to the original - '
         'one of them is worse than no compression',
         color=TEAL, size=13)
    save(fig, 'day38_5.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures written')
