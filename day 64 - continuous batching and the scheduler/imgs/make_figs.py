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


CELL = 0.46          # one square of the attention mask
CHIP = 0.40          # one token chip


def _chip(ax, x, y, label, cached, vertical=False):
    fill = GREY if cached else NAVY
    edge = GREY_L if cached else TEAL
    box(ax, x, y, CHIP, CHIP, label=None, fill=fill, edge=edge, lw=1.3)
    text(ax, x + CHIP / 2, y + CHIP / 2, label, size=8.5, color=WHITE,
         glowing=False, rotation=90 if vertical else 0)


def _pass(ax, x0, y0, cols, rows, mask, note, arrow_row=None, arrow_label=None):
    """One forward pass: column chips (keys), row chips (queries), the mask."""
    gx = x0 + 1.55                                   # where the grid starts
    for j, (lab, cached) in enumerate(cols):
        _chip(ax, gx + j * CELL + (CELL - CHIP) / 2, y0 + CHIP * 0.15, lab, cached,
              vertical=True)
    top = y0 - 0.12
    for i, (lab, cached) in enumerate(rows):
        y = top - (i + 1) * CELL
        _chip(ax, x0 + 1.05, y + (CELL - CHIP) / 2, lab, cached)
        for j in range(len(cols)):
            v = mask[i][j]
            if v == 0:
                box(ax, gx + j * CELL + 0.03, y + 0.03, CELL - 0.06, CELL - 0.06,
                    label=None, fill='none', edge=GREY_L, lw=0.9)
            else:
                box(ax, gx + j * CELL + 0.03, y + 0.03, CELL - 0.06, CELL - 0.06,
                    label=None, fill=NAVY if v == 1 else '#3a2a06',
                    edge=TEAL if v == 1 else AMBER, lw=1.6)
    if arrow_row is not None:
        y = top - (arrow_row + 1) * CELL + CELL / 2
        ax.annotate('', xy=(gx + len(cols) * CELL + 0.75, y),
                    xytext=(gx + len(cols) * CELL + 0.10, y),
                    arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.6))
        box(ax, gx + len(cols) * CELL + 0.80, y - CHIP / 2, 1.00, CHIP,
            label=arrow_label, size=9.5, edge=RED)
    nx = gx + len(cols) * CELL + 2.15
    for k, line in enumerate(note):
        text(ax, nx, top - 0.30 - k * 0.38, line, size=11, ha='left',
             color=TEAL_L if k == 0 else GREY_L, glowing=False)


def fig0():
    """One batch is a concatenation plus an attention mask."""
    fig, ax = canvas(13.2, 12.4, xlim=(0, 13.2), ylim=(0, 12.4))
    text(ax, 6.6, 12.05, 'continuous batching: one batch is a concatenation '
         'plus an attention mask', size=16)
    text(ax, 6.6, 11.62, 'prompts: "Why is decode slow", "The KV cache is read", '
         '"Each step"', size=11.5, color=GREY_L, glowing=False)

    p1 = ['<bos>', 'Why', 'is', 'dec', 'ode']
    p2a, p2b = ['<bos>', 'The', 'KV', 'ca'], ['che', 'is', 'read']
    p3 = ['<bos>', 'Each']

    # ---- pass 1: a plain prefill, 5 tokens = the whole token budget
    cols = [(t, False) for t in p1]
    rows = [(t, False) for t in p1]
    mask = [[1 if j <= i else 0 for j in range(5)] for i in range(5)]
    _pass(ax, 0.35, 10.95, cols, rows, mask,
          ['batch: 1 prompt, 5 tokens (prefill)',
           'the mask is the plain causal triangle',
           'the last row predicts the next token'],
          arrow_row=4, arrow_label='?')
    ax.plot([0.35, 12.85], [8.15, 8.15], color=RED, lw=1.2, ls=(0, (6, 4)))
    text(ax, 1.15, 8.32, 'forward pass', size=10.5, color=RED, glowing=False)

    # ---- pass 2: one decode token + the first chunk of the second prompt
    cols = [(t, True) for t in p1] + [('?', False)] + [(t, False) for t in p2a]
    rows = [('?', False)] + [(t, False) for t in p2a]
    mask = [[1] * 6 + [0] * 4]
    for i in range(4):
        mask.append([0] * 6 + [2 if j <= i else 0 for j in range(4)])
    _pass(ax, 0.35, 7.55, cols, rows, mask,
          ['batch: 1 prompt decoding (1 token)',
           '     + 1 prompt prefilling (4 tokens)',
           'two blocks, no padding, one forward pass'],
          arrow_row=0, arrow_label='<eos>')
    ax.plot([0.35, 12.85], [4.70, 4.70], color=RED, lw=1.2, ls=(0, (6, 4)))
    text(ax, 1.15, 4.87, 'forward pass', size=10.5, color=RED, glowing=False)

    # ---- pass 3: the rest of prompt 2, and the start of prompt 3
    cols = [(t, True) for t in p2a] + [(t, False) for t in p2b] + \
           [(t, False) for t in p3]
    rows = [(t, False) for t in p2b] + [(t, False) for t in p3]
    mask = []
    for i in range(3):
        mask.append([1] * 4 + [1 if j <= i else 0 for j in range(3)] + [0, 0])
    for i in range(2):
        mask.append([0] * 7 + [2 if j <= i else 0 for j in range(2)])
    _pass(ax, 0.35, 4.10, cols, rows, mask,
          ['batch: 1 prompt finishing its prefill (3 tokens)',
           '     + 1 new prompt starting (2 tokens)',
           'the finished request left; a waiting one took its place'],
          arrow_row=2, arrow_label='once')

    # ---- legend
    ly = 0.95
    box(ax, 0.60, ly, CHIP, CHIP, label=None, fill=NAVY, edge=TEAL, lw=1.3)
    text(ax, 1.15, ly + CHIP / 2, 'new token, computed this pass', size=11,
         ha='left', color=GREY_L, glowing=False)
    box(ax, 5.10, ly, CHIP, CHIP, label=None, fill=GREY, edge=GREY_L, lw=1.3)
    text(ax, 5.65, ly + CHIP / 2, 'old token, already in the KV cache', size=11,
         ha='left', color=GREY_L, glowing=False)
    box(ax, 0.60, ly - 0.60, CHIP, CHIP, label=None, fill='none', edge=GREY_L, lw=0.9)
    text(ax, 1.15, ly - 0.60 + CHIP / 2, 'False in the attention mask', size=11,
         ha='left', color=GREY_L, glowing=False)
    box(ax, 5.10, ly - 0.60, CHIP, CHIP, label=None, fill=NAVY, edge=TEAL, lw=1.6)
    box(ax, 5.60, ly - 0.60, CHIP, CHIP, label=None, fill='#3a2a06', edge=AMBER, lw=1.6)
    text(ax, 6.15, ly - 0.60 + CHIP / 2, 'True - one colour per request, which is '
         'what keeps them apart', size=11, ha='left', color=GREY_L, glowing=False)
    save(fig, os.path.join(HERE, 'day64_0.png'))


def fig1():
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
    save(fig, os.path.join(HERE, 'day64_1.png'))


def fig2():
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
    save(fig, os.path.join(HERE, 'day64_2.png'))


def fig3():
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
    save(fig, os.path.join(HERE, 'day64_3.png'))


def fig4():
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
    save(fig, os.path.join(HERE, 'day64_4.png'))


def fig5():
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
    save(fig, os.path.join(HERE, 'day64_5.png'))


if __name__ == '__main__':
    fig0(); fig1(); fig2(); fig3(); fig4(); fig5()
    print('wrote 6 figures')
