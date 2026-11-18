"""Figures for Day 63 - every number is computed by importing parallelism.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import parallelism as P                # noqa: E402


def fig0():
    """The map: six ways to split a model, and the collective each one calls."""
    fig, ax = canvas(12.2, 7.0, xlim=(0, 12.2), ylim=(0, 7.0))
    text(ax, 6.1, 6.68, 'six ways to split a model, and the collective each one calls',
         size=16)
    rows = [
        ('data parallel', 'same weights, different rows of the batch',
         'all-reduce(gradients)', 'once per step'),
        ('ZeRO / FSDP', 'the weights themselves are sharded',
         'all-gather(w) + reduce-scatter(g)', 'per layer'),
        ('tensor parallel', 'one matrix split across GPUs',
         'all-reduce(activations)', '2 per layer'),
        ('expert parallel', 'experts split, tokens travel',
         'all-to-all(tokens) x 2', '2 per layer'),
        ('context parallel', 'keys and values split by position',
         'all-gather(partial out + lse)', '1 per layer'),
        ('pipeline parallel', 'layers split into stages',
         'point-to-point(activation)', '1 per boundary'),
    ]
    for i, (name, what, coll, when) in enumerate(rows):
        y = 5.86 - i * 0.84
        box(ax, 0.25, y - 0.30, 2.20, 0.62, label=name, size=12)
        text(ax, 2.62, y, what, size=11, ha='left', color=TEAL_L, glowing=False)
        arrow(ax, (6.42, y), (6.82, y), lw=1.5, ms=10)
        box(ax, 6.92, y - 0.30, 3.45, 0.62, label=None, edge=TEAL_L)
        text(ax, 8.645, y, coll, size=10.5, color=CYAN, glowing=False)
        text(ax, 10.52, y, when, size=10.5, ha='left', color=GREY_L, glowing=False)
    text(ax, 6.1, 0.42, 'the strategies differ; the primitives underneath do not',
         size=13, color=AMBER)
    save(fig, os.path.join(HERE, 'day63_0.png'))


def fig1():
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
    save(fig, os.path.join(HERE, 'day63_1.png'))


def fig2():
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
    save(fig, os.path.join(HERE, 'day63_2.png'))


def fig3():
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
    save(fig, os.path.join(HERE, 'day63_3.png'))


def fig4():
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
    save(fig, os.path.join(HERE, 'day63_4.png'))


if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4()
    print('wrote 5 figures')
