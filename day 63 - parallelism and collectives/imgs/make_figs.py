"""Figures for Day 63 - every number is computed by importing parallelism.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import parallelism as P                # noqa: E402


def panel(ax, x, y, w, h, title, tag, foot):
    """One card of the map: a title, a tag, a drawing area, a footer line."""
    region(ax, x, y, w, h, label=None)
    text(ax, x + 0.14, y + h - 0.30, title, size=12.5, ha='left', color=TEAL_L,
         glowing=False)
    text(ax, x + w - 0.14, y + h - 0.30, tag, size=10.5, ha='right',
         color=AMBER if tag == 'training' else (CYAN if tag == 'inference' else GREY_L),
         glowing=False)
    text(ax, x + 0.14, y + 0.26, foot, size=10.5, ha='left', color=GREY_L,
         glowing=False)


def fig0():
    """The map: what each kind of parallelism actually cuts."""
    fig, ax = canvas(12.6, 8.0, xlim=(0, 12.6), ylim=(0, 8.0))
    text(ax, 6.3, 7.70, 'what each kind of parallelism cuts, and the collective it '
         'pays for', size=16)
    pw, ph = 4.00, 3.35
    xs = [0.25, 4.30, 8.35]
    ys = [3.95, 0.35]

    # --- DDP: the batch is cut, the weights are not
    px, py = xs[0], ys[0]
    panel(ax, px, py, pw, ph, 'DDP', 'training', 'cut: the batch - all-reduce(grad)')
    for i in range(4):
        box(ax, px + 0.45, py + 2.20 - i * 0.28, 1.30, 0.24, label=None)
        text(ax, px + 0.32, py + 2.32 - i * 0.28, 'r%d' % i, size=10, ha='right',
             color=GREY_L, glowing=False)
    text(ax, px + 1.10, py + 2.58, 'the batch', size=11, color=TEAL, glowing=False)
    for i in range(4):
        box(ax, px + 2.30 + (i % 2) * 0.70, py + 1.50 + (i // 2) * 0.56, 0.62, 0.50,
            label='W', size=11)
    text(ax, px + 2.96, py + 1.22, 'full copy each', size=11, color=TEAL,
         glowing=False)
    text(ax, px + 2.00, py + 0.78, 'nothing about the model is split', size=10.5,
         color=GREY_L, glowing=False)

    # --- ZeRO / FSDP: the parameters and the optimizer are cut
    px, py = xs[1], ys[0]
    panel(ax, px, py, pw, ph, 'ZeRO / FSDP', 'training',
          'cut: params + Adam - all-gather, reduce-scatter')
    for i in range(4):
        box(ax, px + 0.30 + i * 0.86, py + 1.70, 0.80, 0.68, label='r%d' % i, size=11)
    text(ax, px + 2.00, py + 2.58, 'weights, gradients, optimizer state', size=10.5,
         color=TEAL, glowing=False)
    text(ax, px + 2.00, py + 1.15, 'gathered one layer at a time, then dropped',
         size=10.5, color=GREY_L, glowing=False)

    # --- tensor parallel: one matrix is cut, twice
    px, py = xs[2], ys[0]
    panel(ax, px, py, pw, ph, 'tensor parallel', 'both',
          'cut: one matrix - all-reduce(activations)')
    for i in range(4):
        box(ax, px + 0.52 + i * 0.32, py + 1.50, 0.28, 0.86, label=None)
    text(ax, px + 1.10, py + 2.58, 'w1 by columns', size=11, color=TEAL, glowing=False)
    for i in range(4):
        box(ax, px + 2.35, py + 2.14 - i * 0.24, 1.20, 0.20, label=None)
    text(ax, px + 2.95, py + 2.58, 'w2 by rows', size=11, color=TEAL, glowing=False)
    text(ax, px + 2.00, py + 1.15, 'partial sums, closed by one all-reduce', size=10.5,
         color=GREY_L, glowing=False)

    # --- pipeline parallel: the layers are cut
    px, py = xs[0], ys[1]
    panel(ax, px, py, pw, ph, 'pipeline parallel', 'both',
          'cut: the layers - point-to-point')
    for i in range(8):
        box(ax, px + 0.32 + i * 0.43, py + 1.70, 0.38, 0.68, label=None,
            edge=TEAL if i % 2 == 0 else TEAL_L)
        if i % 2 == 0:
            text(ax, px + 0.51 + i * 0.43 + 0.21, py + 1.42, 'r%d' % (i // 2), size=10,
                 color=GREY_L, glowing=False)
    text(ax, px + 2.00, py + 2.58, '80 layers in 4 stages', size=11, color=TEAL,
         glowing=False)
    text(ax, px + 2.00, py + 1.05, 'only the boundary activation crosses', size=10.5,
         color=GREY_L, glowing=False)

    # --- expert parallel: the experts are cut
    px, py = xs[1], ys[1]
    panel(ax, px, py, pw, ph, 'expert parallel', 'both',
          'cut: the experts - all-to-all x2')
    for i in range(8):
        box(ax, px + 0.30 + i * 0.43, py + 1.70, 0.38, 0.68, label='e%d' % i, size=9.5)
        if i % 2 == 0:
            text(ax, px + 0.49 + i * 0.43 + 0.21, py + 1.42, 'r%d' % (i // 2), size=10,
                 color=GREY_L, glowing=False)
    text(ax, px + 2.00, py + 2.58, 'experts stay, tokens travel', size=11, color=TEAL,
         glowing=False)
    text(ax, px + 2.00, py + 1.05, 'one trip out, one trip back, every layer',
         size=10.5, color=GREY_L, glowing=False)

    # --- context parallel: the sequence is cut
    px, py = xs[2], ys[1]
    panel(ax, px, py, pw, ph, 'context parallel (DCP)', 'inference',
          'cut: the KV by position - all-gather(out, lse)')
    for i in range(4):
        box(ax, px + 0.30 + i * 0.86, py + 1.70, 0.80, 0.68,
            label='%d-%dK' % (i * 32, (i + 1) * 32), size=9.5)
        text(ax, px + 0.70 + i * 0.86, py + 1.42, 'r%d' % i, size=10, color=GREY_L,
             glowing=False)
    text(ax, px + 2.00, py + 2.58, 'the KV cache, split by position', size=11,
         color=TEAL, glowing=False)
    text(ax, px + 2.00, py + 1.05, 'for MLA, where TP cannot split by head',
         size=10.5, color=GREY_L, glowing=False)

    text(ax, 6.3, 0.10, 'DP attention is the sixth: cut the requests, not the model '
         '- inference only', size=11.5, color=AMBER)
    save(fig, os.path.join(HERE, 'day63_0.png'))


def fig1():
    """The collectives, drawn as rank-by-shard grids, before and after."""
    ZOO = [
        ('broadcast', [['A', '', '', '']] + [['', '', '', '']] * 3,
         [['A'] * 4] * 4, 'one rank has it, all need it'),
        ('all-gather', [['a', '', '', ''], ['', 'b', '', ''],
                        ['', '', 'c', ''], ['', '', '', 'd']],
         [['a', 'b', 'c', 'd']] * 4, 'each has 1/p, all need all'),
        ('reduce-scatter', [['1', '2', '3', '4']] * 4,
         [['4', '', '', ''], ['', '8', '', ''], ['', '', '12', ''], ['', '', '', '16']],
         'sum, then keep only 1/p'),
        ('all-to-all', [['a0', 'a1', 'a2', 'a3'], ['b0', 'b1', 'b2', 'b3'],
                        ['c0', 'c1', 'c2', 'c3'], ['d0', 'd1', 'd2', 'd3']],
         [['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1'],
          ['a2', 'b2', 'c2', 'd2'], ['a3', 'b3', 'c3', 'd3']],
         'transpose: rank r keeps column r'),
    ]
    fig, ax = canvas(12.6, 7.4, xlim=(0, 12.6), ylim=(0, 7.4))
    text(ax, 6.3, 7.05, 'the four collectives everything else is built from', size=16)
    cw, ch = 0.52, 0.42

    def grid(x, y, vals, filled):
        for r in range(4):
            for c in range(4):
                v = vals[r][c]
                box(ax, x + c * cw, y - r * ch, cw - 0.06, ch - 0.06,
                    label=v if v else None, size=9,
                    edge=(CYAN if filled else TEAL) if v else GREY,
                    fill=NAVY if v else '#141619', lw=1.6 if v else 1.0)

    for i, (name, before, after, what) in enumerate(ZOO):
        px = 0.45 + (i % 2) * 6.30
        py = 5.55 - (i // 2) * 3.15
        text(ax, px, py + 1.00, name, size=13, ha='left', color=TEAL_L, glowing=False)
        text(ax, px, py + 0.62, what, size=10.5, ha='left', color=GREY_L, glowing=False)
        grid(px, py, before, False)
        arrow(ax, (px + 2.35, py - 0.55), (px + 3.00, py - 0.55), lw=1.6, ms=10)
        grid(px + 3.15, py, after, True)
        for r in range(4):
            text(ax, px - 0.12, py - r * ch + 0.13, 'r%d' % r, size=9, ha='right',
                 color=GREY_L, glowing=False)

    text(ax, 6.3, 0.70, 'all-reduce is not on this list: it is a reduce-scatter '
         'followed by an all-gather,', size=12, color=AMBER)
    text(ax, 6.3, 0.32, 'which is why it costs twice as much - and why ZeRO gets its '
         'first two stages for free', size=12, color=AMBER)
    save(fig, os.path.join(HERE, 'day63_1.png'))

def fig2():
    """The ring all-reduce: reduce-scatter, then all-gather."""
    p = 4
    fig, ax = canvas(10.4, 6.4, xlim=(0, 10.4), ylim=(0, 6.4))
    text(ax, 5.2, 6.05, 'ring all-reduce: every rank ships 2(p-1)/p of the tensor',
         size=16)
    centres = [(2.35, 4.25), (4.55, 4.25), (6.75, 4.25), (8.95, 4.25)]
    for i, (cx, cy) in enumerate(centres):
        node(ax, cx, cy, f'r{i}', r=0.42)
        for j in range(p):
            box(ax, cx - 0.62 + j * 0.31, cy - 1.42, 0.27, 0.46,
                label=None, edge=TEAL if j == (i + 1) % p else GREY_L,
                lw=2.4 if j == (i + 1) % p else 1.1,
                fill=NAVY if j == (i + 1) % p else GREY)
        text(ax, cx, cy - 1.72, f'owns chunk {(i + 1) % p}', size=10, color=GREY_L,
             glowing=False)
        if i < p - 1:
            arrow(ax, (cx + 0.5, cy), (centres[i + 1][0] - 0.5, cy), lw=1.6, ms=10)
    arrow(ax, (8.95, 5.40), (2.35, 5.40), lw=1.6, ms=10)
    text(ax, 5.65, 5.62, 'r3 wraps back to r0', size=11, color=TEAL_L)

    wire = P.Wire()
    bufs = [np.ones(4096, dtype=np.float32) for _ in range(p)]
    P.ring_all_reduce(bufs, wire)
    n = bufs[0].nbytes
    lines = [
        f'step 1..{p - 1}   reduce-scatter: send chunk, add what arrives',
        f'step {p}..{2 * (p - 1)}   all-gather: pass the finished chunk around',
        f'total {wire.steps} steps, {wire.bytes_per_rank / n:.2f} N per rank '
        f'(naive: {2 * (p - 1)} N through one link)',
    ]
    for i, s in enumerate(lines):
        text(ax, 0.35, 1.85 - i * 0.52, s, size=12, ha='left',
             color=CYAN if i == 2 else TEAL, glowing=False)
    save(fig, os.path.join(HERE, 'day63_2.png'))


def fig3():
    """Bytes per axis: why TP and EP stay inside the node."""
    rows = P.placement_table()
    tag = {'tensor parallel': 'every layer, twice',
           'expert parallel': 'every layer, twice',
           'context parallel': 'every layer',
           'pipeline parallel': 'once per stage boundary',
           'data parallel': 'once per step, hideable'}
    fig, ax = canvas(12.2, 6.4, xlim=(0, 12.2), ylim=(0, 6.4))
    text(ax, 6.1, 6.05, 'bytes one rank ships per step, 8-way on each axis, '
         '8,192 tokens', size=15)
    gbs = [r['gb'] for r in rows]
    lo, hi = np.log10(min(gbs)), np.log10(max(gbs))
    for i, r in enumerate(rows):
        y = 5.05 - i * 0.84
        name = r['axis'].split(' (')[0]
        text(ax, 2.55, y, name, size=12, ha='right', color=TEAL_L, glowing=False)
        w = 0.4 + 4.6 * (np.log10(r['gb']) - lo) / (hi - lo)
        hot = r['gb'] > 10
        box(ax, 2.70, y - 0.23, w, 0.46, label=None,
            edge=AMBER if hot else TEAL, lw=2.2 if hot else 1.5)
        text(ax, 2.82 + w, y, f"{r['gb']:,.2f} GB", size=11.5, ha='left',
             color=CYAN if hot else TEAL, glowing=False)
        text(ax, 12.1, y, tag[name], size=11, ha='right', color=GREY_L,
             glowing=False)
    text(ax, 0.3, 0.88, 'log scale. amber = tens of GB. tensor and expert '
         'parallelism pay it in every layer, so they', size=11.5, ha='left',
         color=GREY_L, glowing=False)
    text(ax, 0.3, 0.48, 'need the short in-node links; data parallelism pays once '
         'per step and hides under the backward pass', size=11.5, ha='left',
         color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day63_3.png'))


def fig4():
    """The pipeline bubble."""
    stages, micro = 4, 4
    fig, ax = canvas(10.4, 6.3, xlim=(0, 10.4), ylim=(0, 6.3))
    frac = [r for r in P.pipeline_bubble(stages=stages, micro=(micro,))][0]
    text(ax, 5.2, 5.95, f'pipeline schedule: {stages} stages, {micro} micro-batches, '
         f"{frac['bubble']:.0%} of the slots are idle", size=15)
    cell_w, cell_h = 0.62, 0.62
    x0, y0 = 1.65, 5.05
    slots = micro + stages - 1
    for s in range(stages):
        text(ax, x0 - 0.18, y0 - s * cell_h - cell_h / 2, f'stage {s}', size=11,
             ha='right', color=TEAL_L, glowing=False)
        for t in range(slots):
            busy = s <= t < s + micro
            x = x0 + t * cell_w
            y = y0 - (s + 1) * cell_h
            box(ax, x, y, cell_w - 0.06, cell_h - 0.06,
                label=str(t - s) if busy else None,
                edge=TEAL if busy else GREY_L, lw=1.8 if busy else 1.0,
                fill=NAVY if busy else GREY)
    text(ax, x0 + slots * cell_w / 2, y0 - stages * cell_h - 0.40,
         'time ->', size=11, color=GREY_L, glowing=False)
    text(ax, 0.3, 1.32, 'numbered cells are micro-batches; empty cells are the bubble',
         size=12, ha='left', color=TEAL, glowing=False)
    text(ax, 0.3, 0.84, f"bubble = (stages - 1) / (micro-batches + stages - 1) = "
         f"{stages - 1} / {micro + stages - 1} = {frac['bubble']:.0%}",
         size=12, ha='left', color=CYAN, glowing=False)
    text(ax, 0.3, 0.36, 'more micro-batches shrink it; smaller micro-batches stop '
         'filling the GPU', size=12, ha='left', color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day63_4.png'))


def fig5():
    """What sharding the optimizer state buys."""
    rows = P.zero_table(p=8)
    fig, ax = canvas(12.0, 5.6, xlim=(0, 12.0), ylim=(0, 5.6))
    text(ax, 6.0, 5.25, 'memory per GPU for a 70B model with Adam, 8 GPUs', size=15)
    top = max(r['gb_per_rank'] for r in rows)
    for i, r in enumerate(rows):
        y = 4.3 - i * 0.92
        text(ax, 3.35, y, r['stage'], size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        w = 0.25 + 4.8 * r['gb_per_rank'] / top
        last = i == len(rows) - 1
        box(ax, 3.5, y - 0.24, w, 0.48, label=None,
            edge=AMBER if last else TEAL, lw=2.2 if last else 1.5)
        text(ax, 3.62 + w, y, f"{r['gb_per_rank']:,.0f} GB", size=11.5, ha='left',
             color=CYAN if last else TEAL, glowing=False)
        text(ax, 11.9, y, f"{r['comm_vs_ddp']:.2f}x wire", size=11, ha='right',
             color=GREY_L, glowing=False)
    text(ax, 0.3, 0.55, f"8x less memory for 1.5x the bytes: the same refetch-versus-"
         f"store trade as tiling", size=12, ha='left', color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day63_5.png'))

if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4(); fig5()
    print('wrote 6 figures')
