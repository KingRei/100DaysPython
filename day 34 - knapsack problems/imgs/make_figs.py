"""Figures for Day 34 - knapsack problems. All numbers come from knapsack.py."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from diagram_style import *            # noqa
import knapsack as K                   # noqa

NAMES, W, V, CAP = K.NAMES, K.WEIGHTS, K.VALUES, K.CAP
DP = K.knap01_table(W, V, CAP)
BEST, CHOSEN = K.knap01_items(W, V, CAP)


def grid(ax, x0, y0, cw, ch, rows, row_tags, highlight=(), amber=(), ghost=()):
    """Draw the 2-D table; returns {(i, c): (cx, cy)}."""
    pos = {}
    for c in range(len(rows[0])):
        text(ax, x0 + c * cw + cw / 2, y0 + len(rows) * ch + 0.22,
             str(c), color=TEAL, size=12, glowing=False)
    for i, row in enumerate(rows):
        y = y0 + (len(rows) - 1 - i) * ch
        text(ax, x0 - 0.18, y + ch / 2, row_tags[i], color=TEAL_L, size=13,
             ha='right', glowing=False)
        for c, val in enumerate(row):
            fill = NAVY
            eg, lw = TEAL, 1.4
            if (i, c) in ghost:
                fill = GREY
            if (i, c) in highlight:
                fill = GREY
            if (i, c) in amber:
                eg, lw = AMBER, 2.6
            box(ax, x0 + c * cw, y, cw, ch, label=str(val), fill=fill,
                edge=eg, lw=lw, size=12)
            pos[(i, c)] = (x0 + c * cw + cw / 2, y + ch / 2)
    return pos


def fig1():
    """The recurrence: every cell reads two cells of the row above."""
    fig, ax = canvas(9.2, 5.8, xlim=(0, 9.2), ylim=(0, 5.8))
    tags = ['-'] + NAMES
    pos = grid(ax, 1.05, 1.55, 0.70, 0.62, DP, tags,
               amber=[(3, 8)], highlight=[(2, 8), (2, 3)])
    text(ax, 4.6, 5.45, 'dp[i][c] = best value, first i items, capacity c',
         color=TEAL_L, size=15, glowing=False)
    arrow(ax, pos[(2, 8)], pos[(3, 8)], color=TEAL_L, lw=1.8)
    arrow(ax, pos[(2, 3)], pos[(3, 8)], color=AMBER, lw=1.8, rad=-0.28)
    text(ax, 1.05, 1.05, 'leave pan: inherit dp[2][8] = 90',
         color=TEAL_L, size=13, ha='left', glowing=False)
    text(ax, 1.05, 0.68,
         'take pan:  dp[2][8-5] + 70 = 50 + 70 = 120  -> the bigger one wins',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 1.05, 0.28,
         'both readings live in row i-1, so the rows fill top to bottom',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day34_1.png')


def fig2():
    """One row, descending inner loop: dp[c-w] is still the previous row."""
    rows = K.knap_rolling_rows(W, V, CAP, descending=True)
    fig, ax = canvas(9.2, 4.9, xlim=(0, 9.2), ylim=(0, 4.9))
    x0, cw, ch = 1.05, 0.70, 0.66
    for c in range(CAP + 1):
        text(ax, x0 + c * cw + cw / 2, 3.95, str(c), color=TEAL, size=12,
             glowing=False)
    old, new = rows[2], rows[3]        # before and after 'pan' (w=5, v=70)
    for c, val in enumerate(old):
        box(ax, x0 + c * cw, 3.05, cw, ch, label=str(val), fill=NAVY,
            edge=TEAL, lw=1.4, size=12)
    for c, val in enumerate(new):
        eg = AMBER if c in (8, 3) else TEAL
        lw = 2.6 if c in (8, 3) else 1.4
        box(ax, x0 + c * cw, 1.55, cw, ch, label=str(val), fill=NAVY,
            edge=eg, lw=lw, size=12)
    text(ax, x0 - 0.18, 3.38, 'before', color=TEAL_L, size=13, ha='right',
         glowing=False)
    text(ax, x0 - 0.18, 1.88, 'after', color=TEAL_L, size=13, ha='right',
         glowing=False)
    arrow(ax, (x0 + 10 * cw + cw / 2, 1.10), (x0 + 5 * cw - 0.10, 1.10),
          color=AMBER, lw=2.0)
    text(ax, x0 + 7.6 * cw, 0.80, 'c runs downwards', color=AMBER, size=13,
         glowing=False)
    arrow(ax, (x0 + 3 * cw + cw / 2, 1.55 + ch + 0.04),
          (x0 + 8 * cw + cw / 2, 1.55 + ch + 0.04), color=TEAL_L, lw=1.7,
          rad=-0.26)
    text(ax, 4.60, 4.45,
         'writing dp[8] reads dp[3] - five cells to the LEFT, not yet rewritten',
         color=TEAL_L, size=14, glowing=False)
    text(ax, 1.05, 0.28,
         'so dp[3] is still "before": pan is offered exactly once',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day34_2.png')


def fig3():
    """The silent failure: ascending order lets one item be taken again."""
    up_rows = K.knap_rolling_rows(W, V, CAP, descending=False)
    tot, counts = K.unbounded_counts(W, V, CAP)
    fig, ax = canvas(9.2, 5.2, xlim=(0, 9.2), ylim=(0, 5.2))
    x0, cw, ch = 1.05, 0.70, 0.66
    for c in range(CAP + 1):
        text(ax, x0 + c * cw + cw / 2, 4.30, str(c), color=TEAL, size=12,
             glowing=False)
    before = up_rows[0]
    after = up_rows[1]                 # after 'rope' (w=3, v=50) going up
    for c, val in enumerate(before):
        box(ax, x0 + c * cw, 3.40, cw, ch, label=str(val), fill=NAVY,
            edge=TEAL, lw=1.4, size=12)
    for c, val in enumerate(after):
        eg = AMBER if c in (3, 6, 9) else TEAL
        lw = 2.6 if c in (3, 6, 9) else 1.4
        box(ax, x0 + c * cw, 1.95, cw, ch, label=str(val), fill=NAVY,
            edge=eg, lw=lw, size=12)
    text(ax, x0 - 0.18, 3.73, 'before', color=TEAL_L, size=13, ha='right',
         glowing=False)
    text(ax, x0 - 0.18, 2.28, 'after', color=TEAL_L, size=13, ha='right',
         glowing=False)
    arrow(ax, (x0 + 0.10, 1.52), (x0 + 10 * cw + cw, 1.52), color=RED, lw=2.0)
    text(ax, x0 + 3.2 * cw, 1.22, 'c runs upwards', color=RED, size=13,
         glowing=False)
    for a, b in ((3, 6), (6, 9)):
        arrow(ax, (x0 + a * cw + cw / 2, 1.95 + ch + 0.04),
              (x0 + b * cw + cw / 2, 1.95 + ch + 0.04), color=RED, lw=1.7,
              rad=-0.42)
    text(ax, 4.60, 4.82,
         'dp[6] reads dp[3], which this round already contains one rope',
         color=RED, size=14, glowing=False)
    text(ax, 1.05, 0.58, f'0/1 answer {K.knap01_rolling(W, V, CAP)}   '
         f'ascending answer {tot} = {counts[0]} x rope',
         color=AMBER, size=14, ha='left', glowing=False)
    text(ax, 1.05, 0.22,
         'one word - reversed - is the whole 0/1 constraint',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day34_3.png')


def fig4():
    """Binary splitting: k copies become log(k) items."""
    k = 13
    ok, parts = K.binary_split_reaches(k)
    assert ok
    fig, ax = canvas(9.2, 4.6, xlim=(0, 9.2), ylim=(0, 4.6))
    text(ax, 4.6, 4.22, f'{k} copies of one item -> {len(parts)} items',
         color=TEAL_L, size=16, glowing=False)
    cw = 0.44
    x = 0.85
    for i in range(k):
        box(ax, x + i * (cw + 0.08), 3.18, cw, 0.52, label='1', fill=GREY,
            edge=TEAL, lw=1.2, size=12)
    text(ax, 0.85, 2.88, f'{k} separate 0/1 items: O(cap x {k})',
         color=TEAL, size=13, ha='left', glowing=False)
    xs = [1.30, 2.55, 3.95, 5.60]
    ws = [0.70, 0.95, 1.20, 1.45]
    for i, (p, xx, ww) in enumerate(zip(parts, xs, ws)):
        box(ax, xx, 1.55, ww, 0.62, label=str(p), fill=NAVY, edge=AMBER,
            lw=2.2, size=15)
    text(ax, 0.85, 1.20, 'bundles 1, 2, 4 and the remainder 6',
         color=AMBER, size=13, ha='left', glowing=False)
    text(ax, 0.85, 0.78,
         'any count 0..13 is a subset of these: 5 = 1+4,  11 = 1+4+6,  13 = 1+2+4+6',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, 0.85, 0.34,
         'so a 0/1 choice over 4 bundles reproduces all 14 answers',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day34_4.png')


def fig5():
    """Greedy by ratio: exact when you may cut, quietly wrong when you may not."""
    frac, ftaken = K.fractional_greedy(W, V, CAP)
    g01, gtaken = K.greedy_01(W, V, CAP)
    fig, ax = canvas(9.2, 5.0, xlim=(0, 9.2), ylim=(0, 5.0))
    order = sorted(range(len(W)), key=lambda i: V[i] / W[i], reverse=True)
    text(ax, 4.6, 4.66, 'best value per unit of weight first',
         color=TEAL_L, size=15, glowing=False)
    x = 0.75
    for i in order:
        w = W[i] * 0.36
        box(ax, x, 3.70, w, 0.62, label=f'{NAMES[i]}', fill=NAVY, edge=TEAL,
            lw=1.5, size=13)
        text(ax, x + w / 2, 3.46, f'{V[i]}/{W[i]}={V[i] / W[i]:.1f}',
             color=TEAL, size=11, glowing=False)
        x += w + 0.26
    # fractional
    text(ax, 0.75, 2.92, f'cut allowed   -> {frac:.2f}', color=TEAL_L,
         size=14, ha='left', glowing=False)
    x = 0.75
    for i, t in ftaken:
        w = W[i] * 0.42 * t
        box(ax, x, 2.20, w, 0.55, label=NAMES[i] if t > 0.5 else '', fill=NAVY,
            edge=TEAL_L, lw=1.6, size=12)
        x += w
    box(ax, x, 2.20, CAP * 0.42 - (x - 0.75), 0.55, fill='none', edge=GREY,
        lw=1.2)
    text(ax, 0.75 + CAP * 0.42 + 0.30, 2.40,
         '= rope + pan + 1/3 of tent',
         color=TEAL_L, size=12, ha='left', glowing=False)
    # 0/1 greedy
    text(ax, 0.75, 1.62, f'no cutting    -> {g01}', color=RED, size=14,
         ha='left', glowing=False)
    x = 0.75
    for i in gtaken:
        w = W[i] * 0.42
        box(ax, x, 0.90, w, 0.55, label=NAMES[i], fill=NAVY, edge=RED, lw=1.6,
            size=12)
        x += w
    box(ax, x, 0.90, CAP * 0.42 - (x - 0.75), 0.55, fill='none', edge=GREY,
        lw=1.2)
    text(ax, x + 0.12, 1.17, 'wasted', color=GREY_L, size=12, ha='left',
         glowing=False)
    text(ax, 5.10, 1.17, f'DP finds {BEST} '
         f'({" + ".join(NAMES[i] for i in CHOSEN)})',
         color=AMBER, size=14, ha='left', glowing=False)
    text(ax, 0.75, 0.38,
         f'the greedy loses {BEST - g01} and reports nothing at all',
         color=RED, size=13, ha='left', glowing=False)
    save(fig, 'day34_5.png')


def fig6():
    """Pseudo-polynomial: the table grows with the value of W, not its size."""
    rows = K.scale_report(W, V, CAP)
    fig, ax = canvas(9.2, 4.6, xlim=(0, 9.2), ylim=(0, 4.6))
    text(ax, 4.6, 4.24, 'multiply every weight and the capacity by f',
         color=TEAL_L, size=15, glowing=False)
    heads = ['f', 'answer', 'table cells', 'input bits']
    xs = [1.05, 2.75, 4.75, 7.05]
    ws = [1.35, 1.65, 1.95, 1.60]
    for h, x, w in zip(heads, xs, ws):
        box(ax, x, 3.35, w, 0.58, label=h, fill=GREY, edge=TEAL, lw=1.4,
            size=13)
    for r, (f, best, cells_, bits) in enumerate(rows):
        y = 2.65 - r * 0.62
        vals = [str(f), str(best), f'{cells_:,}', str(bits)]
        for j, (val, x, w) in enumerate(zip(vals, xs, ws)):
            eg = AMBER if j == 1 else TEAL
            lw = 2.4 if j == 1 else 1.4
            box(ax, x, y, w, 0.58, label=val, fill=NAVY, edge=eg, lw=lw,
                size=13)
    text(ax, 1.05, 0.30,
         'the answer never moves; 4 extra bits of input buy 1000x the work',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day34_6.png')


if __name__ == '__main__':
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn()
        print('ok', fn.__name__)
