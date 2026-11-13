"""Day 52 figures.  Every number is computed here or by importing attention.py.
No CJK anywhere - the serif fonts in this container have no CJK glyphs."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import attention as A                                # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def liny(vals, y0, h, lo, hi):
    return [y0 + h * (v - lo) / (hi - lo) for v in vals]


def bars(ax, x0, base, bw, gap, vals, cols, labs, maxh, top=None):
    hi = top or max(abs(v) for v in vals)
    for i, v in enumerate(vals):
        hh = max(.03, maxh * abs(v) / hi)
        x = x0 + i * (bw + gap)
        ax.add_patch(Rectangle((x, base), bw, hh, facecolor=cols[i], edgecolor=cols[i],
                               lw=1.2, alpha=.85, zorder=3))
        if labs and labs[i] is not None:
            text(ax, x + bw / 2, base + hh + .22, labs[i], color=cols[i], size=11.5)


# ---------------------------------------------------------------- figure 0
def fig0():
    """How one output row is produced: scores, softmax, weighted average."""
    fig, ax = W(6.2)
    Q, K, V, out, w = A.toy_attention()
    d = Q.shape[1]
    scores = Q[0] @ K.T / math.sqrt(d)
    m = K.shape[0]
    cw, X0 = 1.18, 1.95
    text(ax, 5.0, 6.35, 'attention(Q, K, V)  =  softmax( Q K$^{T}$ / $\\sqrt{d}$ ) V',
         color=TEAL_L, size=16, glowing=False)
    text(ax, 5.0, 5.82, 'one output row, step by step: query 0 against 6 keys, head '
                        f'dimension {d}', color=GREY_L, size=11.5, glowing=False)
    box(ax, 0.35, 4.55, 1.35, 0.55, label='query 0', edge=AMBER, lw=1.7)
    for j in range(m):
        box(ax, X0 + j * cw, 4.55, cw * 0.86, 0.55, label=f'key {j}', edge=TEAL, lw=1.5)
    text(ax, 1.02, 4.12, 'dot product', color=AMBER, size=11, glowing=False)
    for j in range(m):
        ax.annotate('', xy=(X0 + j * cw + cw * 0.43, 3.92), xytext=(1.02, 4.40),
                    arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.0, alpha=.55))
    text(ax, 1.55, 3.62, 'score / $\\sqrt{d}$', color=TEAL, size=11.5, ha='right',
         glowing=False)
    for j in range(m):
        text(ax, X0 + j * cw + cw * 0.43, 3.62, f'{scores[j]:+.2f}', color=TEAL_L,
             size=12)
    text(ax, 1.55, 3.12, 'softmax', color=TEAL, size=11.5, ha='right', glowing=False)
    for j in range(m):
        ax.annotate('', xy=(X0 + j * cw + cw * 0.43, 3.24),
                    xytext=(X0 + j * cw + cw * 0.43, 3.42),
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.1, alpha=.6))
    base, maxh = 1.55, 1.35
    for j in range(m):
        hh = maxh * w[0, j] / w[0].max()
        x = X0 + j * cw + cw * 0.10
        ax.add_patch(Rectangle((x, base), cw * 0.66, hh, facecolor=TEAL,
                               edgecolor=TEAL, lw=1.2, alpha=.85, zorder=3))
        text(ax, x + cw * 0.33, base + hh + 0.24, f'{w[0, j]:.4f}', color=TEAL_L,
             size=11.5)
        text(ax, x + cw * 0.33, base - 0.26, f'v {j}', color=GREY_L, size=11,
             glowing=False)
    text(ax, 1.55, base + 0.55, 'weight', color=TEAL, size=11.5, ha='right',
         glowing=False)
    text(ax, 5.0, 0.92, f'the six weights sum to {w[0].sum():.2f}, so the output row is '
                        f'a weighted average of the six value rows',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.46, 'attention cannot invent a value - it only decides how much of '
                        'each one to use', color=CYAN, size=12.5)
    save(fig, 'day52_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """The weight matrix: one number per pair, every row a distribution."""
    fig, ax = W(5.0)
    n = m = 6
    msk, _, w, _ = A.causal_demo(n)
    cw = 0.60
    x0, y0 = 3.05, 1.25
    for i in range(n):
        for j in range(m):
            v = w[i, j]
            allowed = msk[i, j]
            col = AMBER if (allowed and v > 0.3) else (TEAL if allowed else '#2f3b40')
            box(ax, x0 + j * cw, y0 + (n - 1 - i) * cw, cw * 0.92, cw * 0.92,
                label=f'{v:.2f}' if allowed else '', edge=col, lw=1.4)
    text(ax, x0 + m * cw / 2, y0 - 0.34, 'keys  (j)', color=GREY_L, size=11.5)
    text(ax, x0 - 0.22, y0 + n * cw / 2, 'queries (i)', color=GREY_L, size=11.5, ha='right')
    for i in range(n):
        text(ax, x0 + m * cw + 0.40, y0 + (n - 1 - i) * cw + 0.28,
             f'{w[i].sum():.2f}', color=TEAL_L, size=11, ha='left')
    text(ax, x0 + m * cw + 0.72, y0 + n * cw + 0.30, 'row sum', color=TEAL_L, size=11,
         ha='center', glowing=False)
    text(ax, 5.0, 5.42, 'softmax(QK^T / sqrt(d)) with a causal mask',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.52,
         'the greyed cells are the future, given -inf before the softmax, so they get '
         'exactly zero and the row still sums to 1',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day52_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Without 1/sqrt(d) the softmax turns into an argmax as d grows."""
    fig, ax = W(4.8)
    rows = A.scale_table(ds=(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096))
    ds = [r['d'] for r in rows]
    unscaled = [r['raw'] for r in rows]
    scaled = [r['scaled'] for r in rows]
    X0, Y0, WD, HT = 1.35, 1.55, 6.5, 2.80
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(ds) - 1) for i in range(len(ds))]
    for vals, col in ((unscaled, RED_S), (scaled, TEAL_L)):
        ax.plot(xs, liny(vals, Y0, HT, 0, 1.05), color=col, lw=2.6, zorder=4)
    for v, lab in ((0.0, '0'), (0.5, '0.5'), (1.0, '1')):
        text(ax, X0 - 0.22, liny([v], Y0, HT, 0, 1.05)[0], lab, color=TEAL, size=10.5,
             ha='right')
    for i, dd in enumerate(ds):
        if dd in (4, 64, 1024, 4096):
            text(ax, xs[i], Y0 - 0.30, str(dd), color=TEAL, size=11)
    text(ax, xs[-1] - 0.15, liny([unscaled[-1]], Y0, HT, 0, 1.05)[0] - 0.34,
         f'no scaling: {unscaled[-1]:.4f}', color=RED_S, size=11.5, ha='right')
    text(ax, xs[-1] - 0.15, liny([scaled[-1]], Y0, HT, 0, 1.05)[0] + 0.34,
         f'divided by sqrt(d): {scaled[-1]:.4f}', color=TEAL_L, size=11.5, ha='right')
    text(ax, X0 + WD / 2, Y0 - 0.74, 'head dimension d', color=TEAL, size=11.5)
    text(ax, X0 + 0.30, Y0 + HT + 0.04, 'average largest weight in a row', color=GREY_L,
         size=11, ha='left', glowing=False)
    text(ax, 5.0, 5.15, 'the softmax collapses onto one key as the head gets wider',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'a maximum weight near 1 is an argmax: almost no gradient reaches the other keys',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day52_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """The score matrix you do not build."""
    fig, ax = W(4.8)
    ctxs = [1024, 4096, 16384, 131072]
    gbs = [A.score_matrix_bytes(c, 32) / 2**30 for c in ctxs]
    base, maxh, bw, gap = 1.60, 2.40, 0.95, 0.70
    for i, (c, g) in enumerate(zip(ctxs, gbs)):
        x = 1.60 + i * (bw + gap)
        h = maxh * math.log10(g + 1) / math.log10(gbs[-1] + 1)
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=RED_S, edgecolor=RED_S,
                               lw=1.2, alpha=.85, zorder=3))
        lab = f'{g:,.0f} GB' if g >= 1 else f'{g * 1024:.0f} MB'
        text(ax, x + bw / 2, base + h + 0.24, lab, color=RED_S, size=12)
        text(ax, x + bw / 2, base - 0.34, f'{c:,}', color=GREY_L, size=11.5)
    text(ax, 5.0, 5.15, 'the score matrix for one layer, 32 heads, fp16',
         color=TEAL_L, size=13)
    text(ax, 5.0, 1.05, 'context length  (log-scaled bars)', color=GREY_L, size=11.5)
    text(ax, 5.0, 0.56,
         'FlashAttention never allocates this - it holds one block at a time, so the '
         'memory is linear',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.20, 'while the arithmetic stays quadratic', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day52_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Paged KV: the waste is the tail of the last page, not the whole reservation."""
    fig, ax = W(4.8)
    lengths = [12, 300, 48, 900, 7, 2048, 64, 130]
    cap = 4096
    _, _, waste = A.contiguous_waste(lengths, cap)
    pages = [16, 64, 256]
    wastes = [A.paged_waste(lengths, p)[3] for p in pages]
    vals = [waste] + wastes
    labs = ['one block\nper sequence'] + [f'pages of\n{p}' for p in pages]
    cols = [RED_S, TEAL_L, TEAL, AMBER]
    base, maxh, bw = 1.75, 2.35, 1.10
    for i, v in enumerate(vals):
        x = 1.55 + i * 1.95
        h = max(.04, maxh * v)
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=cols[i], edgecolor=cols[i],
                               lw=1.2, alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, f'{100 * v:.1f}%', color=cols[i], size=13)
        for j, line in enumerate(labs[i].split('\n')):
            text(ax, x + bw / 2, base - 0.34 - j * 0.34, line, color=GREY_L, size=11,
                 glowing=False)
    text(ax, 5.0, 5.15, 'wasted KV slots, 8 sequences that could each reach 4,096 tokens',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.60,
         'a contiguous cache reserves for the longest the sequence might get; pages are '
         'handed out as it grows',
         color=TEAL, size=11, glowing=False)
    text(ax, 5.0, 0.24,
         'so the only waste left is the tail of the last page - at most page_size - 1 '
         'slots per sequence',
         color=GREY_L, size=11, glowing=False)
    save(fig, 'day52_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """What has to be cached, per token."""
    fig, ax = W(4.8)
    L, H, D = 32, 32, 128
    rows = [('MHA', A.kv_cache_bytes_per_token(L, H, D), TEAL),
            ('GQA', A.kv_cache_bytes_per_token(L, H, D, kv_heads=8), TEAL_L),
            ('MLA', A.kv_cache_bytes_per_token(L, H, D, latent=512, rope_dim=64), AMBER),
            ('MQA', A.kv_cache_bytes_per_token(L, H, D, kv_heads=1), PURPLE)]
    base, maxh, bw = 1.80, 2.30, 1.15
    top = rows[0][1]
    for i, (nm, b, col) in enumerate(rows):
        x = 1.45 + i * 1.95
        h = max(.05, maxh * b / top)
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, f'{131072 * b / 2**30:.1f} GB', color=col,
             size=12.5)
        text(ax, x + bw / 2, base - 0.34, nm, color=GREY_L, size=12.5)
        text(ax, x + bw / 2, base - 0.68, f'{b / 1024:.0f} KB/token', color=GREY_L,
             size=10.5, glowing=False)
    st = A.linear_state_bytes(L, H, D) / 2**20
    text(ax, 5.0, 5.15, 'KV cache at a 128k context: 32 layers, 32 heads, head dim 128, fp16',
         color=TEAL_L, size=12.5)
    text(ax, 5.0, 0.62,
         f'all four grow with the context; 32 layers of delta-rule state are {st:.0f} MB and do not',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.26,
         'MQA and GQA make heads share K and V; MLA caches one low-rank vector and '
         'rebuilds per-head K and V from it',
         color=GREY_L, size=10.5, glowing=False)
    save(fig, 'day52_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """A d x d state holds exactly d orthogonal associations."""
    fig, ax = W(4.8)
    ns = [4, 8, 12, 16, 20, 24, 28, 32]
    lost = [int((A.delta_rule_recall(d=16, n=n) > 0.5).sum()) for n in ns]
    X0, Y0, WD, HT = 1.45, 1.70, 6.5, 2.60
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * (n - ns[0]) / (ns[-1] - ns[0]) for n in ns]
    ax.plot(xs, liny(lost, Y0, HT, 0, 17), color=RED_S, lw=2.6, zorder=4)
    ax.scatter(xs, liny(lost, Y0, HT, 0, 17), s=28, color=RED_S, zorder=5, linewidths=0)
    x16 = X0 + WD * (16 - ns[0]) / (ns[-1] - ns[0])
    ax.plot([x16, x16], [Y0, Y0 + HT], color='#2f5661', lw=.9, ls=':', zorder=2)
    text(ax, x16, Y0 + HT + 0.22, 'd = 16', color=TEAL_L, size=11.5)
    for v in (0, 8, 16):
        text(ax, X0 - 0.22, liny([v], Y0, HT, 0, 17)[0], str(v), color=TEAL, size=10.5,
             ha='right')
    for n in (4, 16, 32):
        text(ax, X0 + WD * (n - ns[0]) / (ns[-1] - ns[0]), Y0 - 0.30, str(n),
             color=TEAL, size=11)
    text(ax, X0 + WD / 2, Y0 - 0.74, 'key-value pairs written into one 16 x 16 state',
         color=TEAL, size=11.5)
    text(ax, X0 + 0.30, Y0 + HT + 0.04, 'associations lost', color=GREY_L, size=11,
         ha='left', glowing=False)
    text(ax, 5.0, 5.15, 'constant memory, bounded recall', color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         'up to d orthogonal pairs fit exactly; past that, every extra write costs '
         'exactly one old association',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day52_6.png')


if __name__ == '__main__':
    for f in (fig0, fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, 'ok')
