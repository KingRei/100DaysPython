"""Figures for Day 61 - every number is computed by importing quantization.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import quantization as Q               # noqa: E402

L = Q.make_layer()


def fig0():
    """The lead figure: the whole operation, and the one decision inside it."""
    fig, ax = canvas(10.6, 6.6, xlim=(0, 10.6), ylim=(0, 6.6))
    text(ax, 5.3, 6.35, 'quantization: one scale, one grid, one rounding', size=17)

    steps = ['float weights', 'scale = max|x| / qmax',
             'q = round(x / scale)\nstored in 8 or 4 bits',
             'x_hat = q * scale\nerror <= half a step']
    for i, t in enumerate(steps):
        x = 0.45 + i * 2.55
        box(ax, x, 4.55, 2.25, 0.95, label=t, size=11)
        if i:
            arrow(ax, (x - 0.26, 5.02), (x - 0.04, 5.02), lw=1.6)

    text(ax, 5.3, 3.95, 'the only real decision: which numbers share one scale',
         size=14, color=TEAL_L)
    schemes = [('per tensor: 1 scale', 1), ('per row: 1 per output', 2),
               ('group of 128: 4 per row', 4), ('block float: 1 per 32', 8)]
    for i, (name, n) in enumerate(schemes):
        x = 0.45 + i * 2.55
        per = 16 // n
        for j in range(16):
            col = NAVY if (j // per) % 2 == 0 else GREY
            box(ax, x + j * 0.14, 2.65, 0.12, 0.55, fill=col, edge=col, lw=0.4)
        text(ax, x + 1.12, 2.15, name, size=11, color=GREY_L)
    text(ax, 5.3, 1.35, 'more scales: smaller error, more memory and more work',
         size=12, color=GREY_L)
    text(ax, 5.3, 0.75, 'int4 with a scale per 128 weights beats int8 with one scale '
                        'for the whole tensor', size=12, color=CYAN)
    save(fig, os.path.join(HERE, 'day61_0.png'))


def fig_formats():
    """Every format as a bit layout, to scale."""
    rows = [r for r in Q.format_spec_table()]
    fig, ax = canvas(10.6, 7.0, xlim=(0, 10.6), ylim=(0, 7.0))
    text(ax, 5.3, 6.7, 'one bit layout per format, drawn to scale', size=16)
    text(ax, 5.3, 6.3, 'value = (-1)^sign x 2^(exponent - bias) x 1.mantissa',
         size=13, color=GREY_L)
    x0, unit = 2.70, 0.195                      # one bit wide
    for i, r in enumerate(rows):
        y = 5.65 - i * 0.48
        text(ax, x0 - 0.15, y + 0.17, f"{r['name']}  ({r['bits']}b)", size=12,
             ha='right', color=TEAL_L, glowing=False)
        box(ax, x0, y, unit, 0.34, fill=AMBER, edge=AMBER, lw=0.6)
        x = x0 + unit
        if r['e']:
            box(ax, x, y, unit * r['e'], 0.34, fill=NAVY, edge=TEAL, lw=1.0,
                label=f"E{r['e']}", size=10)
            x += unit * r['e']
        box(ax, x, y, unit * r['m'], 0.34, fill=GREY, edge=GREY_L, lw=1.0,
            label=(f"M{r['m']}" if r['e'] else f"{r['m']}b"), size=10)
        text(ax, 10.35, y + 0.17, f"max {r['max']:.3g}", size=11, ha='right',
             color=GREY_L, glowing=False)
    legend(ax, 0.55, 0.75, [(AMBER, 'sign'), (TEAL, 'exponent: range'),
                            (GREY_L, 'mantissa: precision')])
    text(ax, 6.6, 0.55, 'int8 and int4 have no exponent: the scale carries the range',
         size=12, color=CYAN)
    save(fig, os.path.join(HERE, 'day61_formats.png'))


def fig1():
    """Where the representable values sit: integers against floats."""
    fig, ax = canvas(10.4, 5.4, xlim=(0, 10.4), ylim=(0, 5.4))
    text(ax, 5.2, 5.1, 'the same 8 bits, two different grids', size=16)
    x0, w = 0.8, 8.8

    def axis_row(y, label, values, colour):
        ax.plot([x0, x0 + w], [y, y], color=TEAL, lw=1.4)
        text(ax, x0 - 0.1, y, label, size=12, ha='right', color=colour, glowing=False)
        for v in values:
            px = x0 + w * v
            ax.plot([px, px], [y - 0.14, y + 0.14], color=colour, lw=1.4)

    ints = np.linspace(0, 1, 33)
    axis_row(4.1, 'int8', ints, TEAL)
    text(ax, 5.2, 3.55, 'evenly spaced: the step is set by the largest value in the group',
         size=12, color=GREY_L)

    e = np.array([Q.float_round(v, 3, -6, 8) for v in np.linspace(0.001, 1, 4000)])
    fps = np.unique(np.round(e, 6))
    fps = fps[(fps > 0) & (fps <= 1)]
    axis_row(2.6, 'fp8 e4m3', fps, AMBER)
    text(ax, 5.2, 2.05, 'geometrically spaced: fine near zero, coarse near the maximum',
         size=12, color=GREY_L)

    rows = Q.format_table(L['W'])
    best = {r['name']: r['rel'] for r in rows}
    text(ax, 5.2, 1.2, f"on this layer's weights: int8 per tensor "
                       f"{best['int8, per tensor']:.4f}, fp8 e4m3 {best['fp8 e4m3']:.4f}, "
                       f"int8 per row {best['int8, per row']:.4f}", size=12, color=CYAN)
    text(ax, 5.2, 0.6, 'integers win on uniform data, floats win when the range is wide',
         size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day61_1.png'))


def fig2():
    """The outlier channels, and the three fixes."""
    X = L['X']
    mag = np.abs(X).max(axis=0)
    fig, ax = canvas(10.4, 6.0, xlim=(0, 10.4), ylim=(0, 6.0))
    text(ax, 5.2, 5.75, 'the hard part is the activations, not the weights', size=16)
    x0, w, base, h = 0.7, 8.9, 3.3, 1.5
    order = np.argsort(-mag)[:160]
    for i, c in enumerate(sorted(order)):
        px = x0 + w * i / len(order)
        hh = h * mag[c] / mag.max()
        ax.plot([px, px], [base, base + hh], color=RED if mag[c] > 5 * np.median(mag)
                else TEAL, lw=1.0)
    text(ax, x0, base + h + 0.35, 'per-channel maximum of the activations',
         size=12, ha='left', color=GREY_L)
    text(ax, x0 + w, base + h + 0.35, f'{len(L["outliers"])} channels in red',
         size=12, ha='right', color=RED)

    mp = Q.mixed_precision(L)
    sq = Q.smooth_quant(L, 0.5)
    rows = [('plain int8', mp['all_int8'], RED),
            ('keep 6 columns in 16 bit', mp['rel'], TEAL),
            ('SmoothQuant alpha=0.5', sq['smooth'], AMBER)]
    top = max(r[1] for r in rows)
    for i, (name, v, col) in enumerate(rows):
        y = 2.35 - i * 0.62
        ax.add_patch(plt.Rectangle((3.9, y - 0.17), 4.0 * v / top, 0.34,
                                   facecolor=col, edgecolor=col))
        text(ax, 3.75, y, name, size=12, ha='right', color=GREY_L, glowing=False)
        text(ax, 8.1, y, f'{v:.4f}', size=12, ha='left', color=col, glowing=False)
    text(ax, 5.2, 0.45, 'output error of the layer, int8 weights and activations',
         size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day61_2.png'))


def fig3():
    """Better rounding, and training through the rounding."""
    rt = Q.rounding_table(L, bits=4, group=128)
    q = Q.qat_demo()
    qo = Q.qat_demo(correlated=False)
    fig, ax = canvas(10.4, 5.2, xlim=(0, 10.4), ylim=(0, 5.2))
    text(ax, 2.7, 4.9, 'rounding better (4-bit weights)', size=14, color=TEAL_L)
    text(ax, 7.9, 4.9, 'training through the rounding', size=14, color=TEAL_L)

    rows = [('round to nearest', rt['rtn_out'], RED),
            ('GPTQ', rt['gptq_out'], TEAL),
            (f'AWQ (alpha {rt["awq_alpha"]:.2f})', rt['awq_out'], AMBER)]
    top = max(r[1] for r in rows)
    for i, (name, v, col) in enumerate(rows):
        y = 3.6 - i * 0.85
        ax.add_patch(plt.Rectangle((0.9, y - 0.18), 3.0 * v / top, 0.36,
                                   facecolor=col, edgecolor=col))
        text(ax, 0.85, y + 0.45, name, size=12, ha='left', color=GREY_L, glowing=False)
        text(ax, 4.05, y, f'{v:.3f}', size=12, ha='left', color=col, glowing=False)
    text(ax, 2.7, 0.75, 'output error, not weight error', size=12, color=GREY_L)

    pairs = [('correlated inputs', q['ptq'], q['qat']),
             ('orthogonal inputs', qo['ptq'], qo['qat'])]
    for i, (name, a, b) in enumerate(pairs):
        y = 3.4 - i * 1.5
        top2 = max(a, b)
        ax.add_patch(plt.Rectangle((6.2, y), 2.6 * a / top2, 0.34,
                                   facecolor=RED, edgecolor=RED))
        ax.add_patch(plt.Rectangle((6.2, y - 0.48), 2.6 * b / top2, 0.34,
                                   facecolor=TEAL, edgecolor=TEAL))
        text(ax, 6.2, y + 0.62, name, size=12, ha='left', color=GREY_L, glowing=False)
        text(ax, 9.0, y + 0.17, f'PTQ {a:.2f}', size=11, ha='left', color=RED,
             glowing=False)
        text(ax, 9.0, y - 0.31, f'QAT {b:.2f}', size=11, ha='left', color=TEAL,
             glowing=False)
    text(ax, 7.9, 0.75, 'the gain needs inputs that overlap', size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day61_3.png'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt       # noqa: E402
    globals()['plt'] = plt
    fig0(); fig_formats(); fig1(); fig2(); fig3()
    print('wrote day61_0, day61_formats, day61_1, day61_2, day61_3')
