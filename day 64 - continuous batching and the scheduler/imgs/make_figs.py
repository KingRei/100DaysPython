"""Figures for Day 64 - every number comes from importing scheduling.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import scheduling as S                 # noqa: E402

ROWS, W = S.batching_table()
ST8, ST32, CONT = ROWS


def fig0():
    """The loop: what the scheduler decides, and the knobs on each decision."""
    fig, ax = canvas(11.8, 6.6, xlim=(0, 11.8), ylim=(0, 6.6))
    text(ax, 5.9, 6.25, 'the serving loop: one decision, a few hundred times a second',
         size=16)
    box(ax, 0.30, 4.30, 2.05, 1.05, label='arrivals')
    box(ax, 2.75, 4.30, 2.35, 1.05, label='waiting queue')
    box(ax, 5.50, 4.30, 2.45, 1.05, label='admit what fits')
    box(ax, 8.35, 4.30, 2.30, 1.05, label='running set')
    for x0, x1 in ((2.35, 2.72), (5.10, 5.47), (7.95, 8.32)):
        arrow(ax, (x0, 4.82), (x1, 4.82), lw=1.7)
    box(ax, 8.35, 2.55, 2.30, 1.05, label='one step')
    arrow(ax, (9.50, 4.27), (9.50, 3.63), lw=1.7)
    box(ax, 5.50, 2.55, 2.45, 1.05, label='retire or preempt')
    arrow(ax, (8.32, 3.07), (7.98, 3.07), lw=1.7)
    arrow(ax, (5.50, 3.07), (3.95, 3.07), lw=1.7, rad=0.0)
    arrow(ax, (3.95, 3.07), (3.95, 4.27), lw=1.7)
    text(ax, 4.55, 2.32, 'preempted: back to the queue', size=11, color=AMBER,
         glowing=False)
    region(ax, 0.30, 0.35, 11.2, 1.75, label=None)
    knobs = [('KV pool', 'how many tokens of cache exist at all'),
             ('max running requests', 'the admission cap'),
             ('chunked-prefill budget', 'prompt tokens allowed into a decode step'),
             ('queue policy', 'who is admitted first')]
    for i, (k, v) in enumerate(knobs):
        y = 1.70 - i * 0.37
        text(ax, 2.95, y, k, size=11.5, ha='right', color=TEAL_L, glowing=False)
        text(ax, 3.10, y, v, size=11.5, ha='left', color=GREY_L, glowing=False)
    text(ax, 5.9, 2.48, 'the four knobs, and the only four', size=12, color=CYAN)
    save(fig, os.path.join(HERE, 'day64_0.png'))


def fig1():
    """Static against continuous, drawn as slots over time (schematic)."""
    fig, ax = canvas(11.8, 6.8, xlim=(0, 11.8), ylim=(0, 6.8))
    text(ax, 5.9, 6.45, 'two schedulers, the same arrivals', size=16)
    runs = [(4.90, 'static batching: the batch starts and ends together',
             [(0.0, 2.6, 1.0), (0.0, 1.2, 0.0), (0.0, 1.8, 0.0), (0.0, 0.8, 0.0)]),
            (1.95, 'continuous batching: slots are reused the step they free up',
             None)]
    x0, w = 1.95, 8.6
    # static: four slots, all held until the longest finishes
    y0 = runs[0][0]
    text(ax, 0.30, y0 + 1.20, runs[0][1], size=12, ha='left', color=TEAL_L,
         glowing=False)
    for i, (s, e, _) in enumerate(runs[0][2]):
        y = y0 + 0.55 - i * 0.42
        box(ax, x0 + s * 2.6, y, (e - s) * 2.6, 0.34, label=None)
        if e < 2.6:
            box(ax, x0 + e * 2.6, y, (2.6 - e) * 2.6, 0.34, label=None, fill=GREY,
                edge=GREY_L, lw=1.0)
        text(ax, 1.80, y + 0.17, 'slot %d' % i, size=10, ha='right', color=GREY_L,
             glowing=False)
    text(ax, x0 + 1.75 * 2.6, y0 - 1.30, 'grey = finished, still holding the slot',
         size=11, color=RED, glowing=False)
    arrow(ax, (x0 + 2.6 * 2.6, y0 + 1.00), (x0 + 2.6 * 2.6, y0 - 1.00), lw=1.4,
          style='-', color=AMBER)
    text(ax, x0 + 2.6 * 2.6 + 0.12, y0 - 1.00, 'the next batch may finally start',
         size=11, ha='left', color=AMBER, glowing=False)
    # continuous: slots refill
    y0 = runs[1][0]
    text(ax, 0.30, y0 + 1.20, runs[1][1], size=12, ha='left', color=TEAL_L,
         glowing=False)
    lanes = [[(0.0, 2.6)], [(0.0, 1.2), (1.2, 2.2)], [(0.0, 1.8), (1.8, 2.6)],
             [(0.0, 0.8), (0.8, 1.9), (1.9, 2.6)]]
    for i, lane in enumerate(lanes):
        y = y0 + 0.55 - i * 0.42
        for j, (s, e) in enumerate(lane):
            box(ax, x0 + s * 2.6 + 0.03, y, (e - s) * 2.6 - 0.06, 0.34, label=None,
                edge=TEAL if j == 0 else CYAN)
        text(ax, 1.80, y + 0.17, 'slot %d' % i, size=10, ha='right', color=GREY_L,
             glowing=False)
    text(ax, 5.9, 0.55, 'measured on the same 300 requests: '
         '%.0f against %.0f output tokens/s, and a median first token at '
         '%.2f s instead of %.1f s' % (CONT['throughput'], ST8['throughput'],
                                       CONT['ttft_p50'], ST8['ttft_p50']),
         size=11.5, color=CYAN, glowing=False)
    save(fig, os.path.join(HERE, 'day64_1.png'))


def fig2():
    """Throughput against latency as the offered load rises."""
    rows = S.rate_sweep()
    fig, ax = canvas(11.0, 6.4, xlim=(0, 11.0), ylim=(0, 6.4))
    text(ax, 5.5, 6.05, 'raising the load: throughput saturates, latency does not',
         size=16)
    x0, y0, w, h = 1.45, 1.25, 8.4, 4.0
    tps = [r['throughput'] for r in rows]
    tt = [r['ttft_p99'] for r in rows]
    hi_t, hi_l = max(tps) * 1.15, max(tt) * 1.1
    for i in range(5):
        y = y0 + i * h / 4
        ax.plot([x0, x0 + w], [y, y], color=GREY, lw=0.8, zorder=1)
        text(ax, x0 - 0.45, y, '%d' % (hi_t * i / 4), size=10, ha='right',
             color=TEAL, glowing=False)
    text(ax, x0 - 0.45, y0 + h + 0.35, 'tokens/s', size=11, ha='right', color=TEAL,
         glowing=False)
    text(ax, x0 + w + 0.15, y0 + h + 0.35, 'TTFT p99 (s)', size=11, ha='left',
         color=RED, glowing=False)
    xs = [x0 + w * i / (len(rows) - 1) for i in range(len(rows))]
    for i, r in enumerate(rows):
        bx = xs[i] - 0.30
        box(ax, bx, y0, 0.60, h * r['throughput'] / hi_t, label=None)
        text(ax, xs[i], y0 - 0.30, '%d/s' % r['rate'], size=10.5, color=GREY_L,
             glowing=False)
        text(ax, xs[i], y0 - 0.66, '%.0f%%' % (100 * r['goodput']), size=10.5,
             color=CYAN if r['goodput'] > 0.5 else RED, glowing=False)
    pts = [(xs[i], y0 + h * tt[i] / hi_l) for i in range(len(rows))]
    for a, b in zip(pts, pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=2.0, zorder=4)
    for (px, py), r in zip(pts, rows):
        ax.plot([px], [py], 'o', color=RED, ms=5, zorder=5)
        text(ax, px, py + 0.26, '%.1f' % r['ttft_p99'], size=10, color=RED,
             glowing=False)
    text(ax, x0, y0 - 1.05, 'bars: throughput    line: TTFT p99    bottom row: '
         'goodput (TTFT < 2 s and ITL p99 < 50 ms)', size=11, ha='left',
         color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day64_2.png'))


def fig3():
    """What the chunked-prefill budget trades."""
    rows = S.chunk_table(W)
    fig, ax = canvas(11.0, 5.8, xlim=(0, 11.0), ylim=(0, 5.8))
    text(ax, 5.5, 5.45, 'chunked prefill: the same work, cut into steps', size=16)
    top_itl = max(r['itl_p99'] for r in rows)
    top_ttft = max(r['ttft_p50'] for r in rows)
    for i, r in enumerate(rows):
        y = 4.35 - i * 1.02
        lbl = 'whole prompt' if not r['chunk'] else '%d tokens' % r['chunk']
        text(ax, 2.05, y + 0.26, lbl, size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        wl = 4.3 * r['itl_p99'] / top_itl
        box(ax, 2.20, y, wl, 0.52, label=None,
            edge=RED if r['itl_p99'] > 50 else TEAL)
        text(ax, 2.32 + wl, y + 0.26, 'ITL p99 %.0f ms' % r['itl_p99'], size=11,
             ha='left', color=RED if r['itl_p99'] > 50 else TEAL, glowing=False)
        wt = 2.2 * r['ttft_p50'] / top_ttft
        box(ax, 8.05, y, max(0.12, wt), 0.52, label=None, edge=AMBER)
        text(ax, 8.17 + max(0.12, wt), y + 0.26, 'TTFT %.2f s' % r['ttft_p50'],
             size=11, ha='left', color=AMBER, glowing=False)
    text(ax, 0.35, 0.95, 'a %s-token prompt is %.0f ms of arithmetic: run whole, every '
         'decoding request waits that long' % ('16,000', S.prefill_ms(16000)),
         size=11.5, ha='left', color=GREY_L, glowing=False)
    text(ax, 0.35, 0.50, 'for one of its tokens - cutting it into chunks moves the '
         'tail from the prompt size to the chunk size', size=11.5, ha='left',
         color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day64_3.png'))


def fig4():
    """Throughput is not the score: goodput is."""
    cfgs = [('continuous, no limits', S.run_continuous(W)),
            ('+ chunked prefill (512)', S.run_continuous(W, chunk=512)),
            ('+ admission cap (32)', S.run_continuous(W, chunk=512, admit_cap=32)),
            ('small KV pool, no cap', S.run_continuous(W, pool_tokens=60_000))]
    fig, ax = canvas(11.4, 5.8, xlim=(0, 11.4), ylim=(0, 5.8))
    text(ax, 5.7, 5.45, 'three of these move the same tokens per second', size=16)
    top = max(S.metrics(r)['throughput'] for _, r in cfgs)
    for i, (name, run) in enumerate(cfgs):
        m = S.metrics(run)
        g = S.goodput(run)
        y = 4.35 - i * 1.02
        text(ax, 2.75, y + 0.26, name, size=11.5, ha='right', color=TEAL_L,
             glowing=False)
        w = 3.0 * m['throughput'] / top
        box(ax, 2.90, y, w, 0.52, label=None)
        text(ax, 3.02 + w, y + 0.26, '%.0f tok/s' % m['throughput'], size=11,
             ha='left', color=TEAL, glowing=False)
        wg = 2.2 * g
        box(ax, 7.60, y, max(0.10, wg), 0.52, label=None,
            edge=CYAN if g > 0.9 else (AMBER if g > 0.3 else RED))
        text(ax, 7.72 + max(0.10, wg), y + 0.26, 'goodput %.0f%%' % (100 * g),
             size=11, ha='left',
             color=CYAN if g > 0.9 else (AMBER if g > 0.3 else RED), glowing=False)
    text(ax, 0.35, 0.80, 'goodput = requests that met the objective: first token '
         'under 2 s and ITL p99 under 50 ms', size=11.5, ha='left', color=GREY_L,
         glowing=False)
    text(ax, 0.35, 0.38, 'the left column cannot tell the first three apart; the '
         'right column is the whole decision', size=11.5, ha='left',
         color=AMBER, glowing=False)
    save(fig, os.path.join(HERE, 'day64_4.png'))


if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4()
    print('wrote 5 figures')
