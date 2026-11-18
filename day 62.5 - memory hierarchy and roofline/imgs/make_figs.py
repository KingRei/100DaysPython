"""Figures for Day 62.5 - every number is computed by importing memory_hierarchy.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import memory_hierarchy as M           # noqa: E402

ROWS, MACH = M.roofline_table()


def fig0():
    """The lead figure: the hierarchy, and the roofline that follows from it."""
    fig, ax = canvas(10.6, 7.2, xlim=(0, 10.6), ylim=(0, 7.2))
    text(ax, 5.3, 6.9, 'the memory hierarchy of an MI300X, and the line it draws',
         size=16)
    levels = M.hierarchy_table('MI300X')
    widths = [0.9, 1.4, 2.0, 2.7, 3.5, 4.3, 5.0]
    for i, (r, w) in enumerate(zip(levels, widths)):
        y = 6.1 - i * 0.74
        text(ax, 3.25, y, r['name'], size=11, ha='right', color=TEAL_L, glowing=False)
        box(ax, 3.4, y - 0.26, w, 0.52, label=None)
        text(ax, 9.25, y, f"{r['gbs']:,.0f} GB/s", size=11, ha='right', color=TEAL,
             glowing=False)
        text(ax, 10.4, y, f"{r['ns']:,.0f} ns", size=11, ha='right', color=GREY_L,
             glowing=False)
    text(ax, 3.4, 1.0, 'wider is bigger; every step down is slower and further away '
                        '(vendor specifications)', size=12, ha='center', color=GREY_L)
    text(ax, 5.3, 0.5,
         f'a kernel doing fewer than {MACH.balance:.0f} FLOPs per byte cannot be '
         f'compute-bound, however it is written', size=13, color=CYAN)
    save(fig, os.path.join(HERE, 'day62_5_0.png'))


def fig1():
    """The measured hierarchy: sequential against random."""
    seq = M.measure_sequential()
    rnd = M.measure_random()
    fig, ax = canvas(10.4, 5.6, xlim=(0, 10.4), ylim=(0, 5.6))
    text(ax, 5.2, 5.3, 'measured on this machine: order is everything', size=16)
    text(ax, 2.6, 4.75, 'reading in order (GB/s)', size=13, color=TEAL_L)
    top = max(r['gbs'] for r in seq)
    for i, r in enumerate(seq):
        y = 4.2 - i * 0.48
        ax.add_patch(plt.Rectangle((1.6, y - 0.15), 2.3 * r['gbs'] / top, 0.3,
                                   facecolor=TEAL, edgecolor=TEAL))
        lab = f'{r["kb"]} KB' if r['kb'] < 1024 else f'{r["kb"] // 1024} MB'
        text(ax, 1.5, y, lab, size=10, ha='right', color=GREY_L, glowing=False)
        text(ax, 4.0, y, f'{r["gbs"]:.0f}', size=10, ha='left', color=GREY_L,
             glowing=False)
    text(ax, 7.9, 4.75, 'reading at random (ns per access)', size=13, color=TEAL_L)
    worst = max(r['ns_per_access'] for r in rnd)
    for i, r in enumerate(rnd):
        y = 4.2 - i * 0.48
        ax.add_patch(plt.Rectangle((6.9, y - 0.15), 2.3 * r['ns_per_access'] / worst,
                                   0.3, facecolor=RED, edgecolor=RED))
        lab = f'{r["kb"]} KB' if r['kb'] < 1024 else f'{r["kb"] // 1024} MB'
        text(ax, 6.8, y, lab, size=10, ha='right', color=GREY_L, glowing=False)
        text(ax, 9.3, y, f'{r["ns_per_access"]:.1f}', size=10, ha='left',
             color=GREY_L, glowing=False)
    text(ax, 5.2, 0.5, 'the prefetcher hides the hierarchy from one of these and not '
                       'the other', size=12, color=CYAN)
    save(fig, os.path.join(HERE, 'day62_5_1.png'))


def fig2():
    """The roofline itself, log-log, with this series' kernels on it."""
    fig, ax = canvas(10.4, 6.3, xlim=(0, 10.4), ylim=(0, 6.3))
    text(ax, 5.2, 6.0, 'the roofline: arithmetic intensity decides the limit', size=16)
    x0, y0, w, h = 1.5, 1.25, 7.6, 3.6
    xlo, xhi = -1.0, 4.0                      # FLOPs/byte, log10
    ylo, yhi = 0.0, 3.3                       # TFLOP/s, log10
    px = lambda ai: x0 + w * (np.log10(max(ai, 10 ** xlo)) - xlo) / (xhi - xlo)
    py = lambda tf: y0 + h * (np.log10(max(tf, 10 ** ylo)) - ylo) / (yhi - ylo)
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.4)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.4)
    peak = MACH.flops / 1e12
    ridge = MACH.balance
    ais = np.logspace(xlo, xhi, 200)
    perf = np.minimum(peak, ais * MACH.bw / 1e12)
    ax.plot([px(a) for a in ais], [py(p) for p in perf], color=AMBER, lw=2.0)
    ax.plot([px(ridge), px(ridge)], [y0, py(peak)], color=GREY_L, lw=1.0, ls=':')
    text(ax, px(ridge) - 1.1, py(peak) + 0.32, f'ridge {ridge:.0f} FLOPs/byte',
         size=11, color=AMBER)
    for d in (-1, 0, 1, 2, 3, 4):
        text(ax, px(10 ** d), y0 - 0.28, f'1e{d}' if d else '1', size=10,
             color=GREY_L, glowing=False)
    text(ax, x0 + w / 2, y0 - 0.65, 'arithmetic intensity, FLOPs per byte', size=12,
         color=GREY_L)
    for v in (1, 10, 100, 1000):
        text(ax, x0 - 0.2, py(v), f'{v}', size=10, ha='right', color=GREY_L,
             glowing=False)
    text(ax, 0.45, y0 + h / 2, 'TFLOP/s', size=11, color=GREY_L)
    picks = [('residual add', 0.32), ('decode: the whole model', 0.32),
             ('prefill: attention (no tiling)', -0.45),
             ('prefill: one linear (GEMM)', -0.45)]
    for name, dy in picks:
        r = next(x for x in ROWS if x['name'] == name)
        x = px(r['intensity'])
        y = py(min(peak, r['intensity'] * MACH.bw / 1e12))
        node(ax, x, y, '', r=0.09, fill=CYAN, edge=CYAN)
        text(ax, x, y + dy, name.replace('prefill: ', '').replace('decode: ', ''),
             size=10, color=CYAN)
    text(ax, 5.2, 0.25, 'the same linear layer sits on both sides, depending only on '
                       'how many rows it has', size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day62_5_2.png'))


def fig3():
    """Tiling, and FlashAttention as the same argument."""
    sweep = M.tile_sweep()
    f = M.flash_io()
    fig, ax = canvas(10.4, 5.4, xlim=(0, 10.4), ylim=(0, 5.4))
    text(ax, 2.7, 5.1, 'matmul: bytes moved against tile size', size=13, color=TEAL_L)
    top = max(r['gb'] for r in sweep)
    for i, r in enumerate(sweep):
        y = 4.4 - i * 0.52
        col = TEAL if r['fits'] else GREY
        ax.add_patch(plt.Rectangle((1.5, y - 0.16), 3.0 * r['gb'] / top, 0.32,
                                   facecolor=col, edgecolor=col))
        text(ax, 1.4, y, f"T={r['tile']}", size=10, ha='right', color=GREY_L,
             glowing=False)
        text(ax, 4.6, y, f"{r['gb']:.1f} GB", size=10, ha='left',
             color=col if r['fits'] else GREY_L, glowing=False)
    text(ax, 2.7, 0.65, 'grey: too big for the cache, so the model stops holding',
         size=11, color=GREY_L)

    text(ax, 7.9, 5.1, 'attention at seq 4096', size=13, color=TEAL_L)
    bars = [('score matrix, resident', f['score_matrix_gb'], RED),
            ('standard, moved', f['standard_gb'], AMBER),
            ('tiled, every level', f['tiled_gb'], GREY_L),
            ('tiled, reaching HBM', f['tiled_hbm_gb'], TEAL)]
    top2 = max(b[1] for b in bars)
    for i, (name, v, col) in enumerate(bars):
        y = 4.2 - i * 0.62
        ax.add_patch(plt.Rectangle((6.4, y - 0.17), 2.6 * v / top2, 0.34,
                                   facecolor=col, edgecolor=col))
        text(ax, 6.4, y + 0.42, name, size=11, ha='left', color=GREY_L, glowing=False)
        text(ax, 9.2, y, f'{v:.2f} GB', size=10, ha='left', color=col, glowing=False)
    text(ax, 7.9, 1.25, f'the re-reads hit the Infinity Cache, not HBM;\n'
                        f'and the matrix is never allocated at all\n'
                        f'({f["memory_ratio"]:,.0f}x the LDS the kernel runs in)',
         size=11, color=CYAN)
    save(fig, os.path.join(HERE, 'day62_5_3.png'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    globals()['plt'] = plt
    fig0(); fig1(); fig2(); fig3()
    print('wrote day62_5_0 day62_5_1 day62_5_2 day62_5_3')
