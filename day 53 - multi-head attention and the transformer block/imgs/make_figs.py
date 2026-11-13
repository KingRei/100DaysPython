"""Day 53 figures.  Every number is computed here or by importing transformer_block.py.
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
import transformer_block as T                        # noqa: E402

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
    """The whole stack: one residual stream, N identical blocks writing into it."""
    fig, ax = W(8.8)
    SX, W2 = 3.45, 3.60                      # spine x, block width
    y = 0.60

    def stage(label, y, col=TEAL, h=0.56, w=W2, note=None, note_col=None):
        box(ax, SX - w / 2, y, w, h, label=label, edge=col, lw=1.7)
        if note:
            text(ax, SX + w / 2 + 0.25, y + h / 2, note, color=note_col or GREY_L,
                 size=11, ha='left', glowing=False)
        return y + h

    y = stage('token ids', y, note='integers from the tokeniser')
    ax.annotate('', xy=(SX, y + 0.34), xytext=(SX, y + 0.04),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6))
    y = stage('embedding  +  position', y + 0.38, col=AMBER,
              note='one vector per token: the stream starts here')
    ytop = y
    for i in range(3):
        ax.annotate('', xy=(SX, y + 0.30), xytext=(SX, y + 0.02),
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6))
        y += 0.34
        lab = ('block 1', 'block 2', 'block N')[i]
        box(ax, SX - W2 / 2, y, W2, 1.26, label=None, edge=TEAL_L, lw=1.5)
        box(ax, SX - W2 / 2 + 0.12, y + 0.12, W2 - 0.24, 0.46,
            label='norm -> attention -> add', edge=TEAL, lw=1.3, size=12)
        box(ax, SX - W2 / 2 + 0.12, y + 0.68, W2 - 0.24, 0.46,
            label='norm -> MLP -> add', edge=AMBER, lw=1.3, size=12)
        text(ax, SX - W2 / 2 - 0.30, y + 0.63, lab, color=TEAL_L, size=11.5, ha='right')
        y += 1.26
        if i == 1:
            text(ax, SX, y + 0.17, '. . .', color=GREY_L, size=13, glowing=False)
            y += 0.34
    ax.annotate('', xy=(SX, y + 0.30), xytext=(SX, y + 0.02),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6))
    y = stage('final LayerNorm', y + 0.34, col=AMBER,
              note='pre-LN leaves the stream unbounded; this is where it is bounded')
    ax.annotate('', xy=(SX, y + 0.34), xytext=(SX, y + 0.04),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6))
    y = stage('unembedding -> logits', y + 0.38,
              note='one score per vocabulary entry, per position')
    ax.annotate('', xy=(SX - W2 / 2 - 1.05, y - 0.10), xytext=(SX - W2 / 2 - 1.05, ytop + 0.10),
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=2.2))
    text(ax, SX - W2 / 2 - 1.15, (ytop + y) / 2, 'the residual stream:\nsame shape at\nevery point',
         color=CYAN, size=11.5, ha='right')
    text(ax, SX, y + 0.95, 'a decoder-only stack (GPT-style): no encoder, no cross-attention',
         color=GREY_L, size=11.5, glowing=False)
    p = T.block_params(4096)
    text(ax, 5.0, 0.18,
         f'every block is the same two edits to one stream: attention moves information '
         f'between positions, the MLP ({p["mlp_share"] * 100:.0f}% of the parameters) '
         f'reworks each position on its own',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day53_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    """Pre-LN and post-LN side by side: which path carries a normalisation."""
    fig, ax = W(5.4)

    def column(x0, title, pre):
        yy = 0.85
        text(ax, x0, yy, 'x', color=TEAL_L, size=12)
        stops = ([('LayerNorm', 'n'), ('multi-head attn', 'b'), ('+  residual', 'a'),
                  ('LayerNorm', 'n'), ('MLP  (4x)', 'b'), ('+  residual', 'a')] if pre else
                 [('multi-head attn', 'b'), ('+  residual', 'a'), ('LayerNorm', 'n'),
                  ('MLP  (4x)', 'b'), ('+  residual', 'a'), ('LayerNorm', 'n')])
        y = yy + 0.55
        for lab, kind in stops:
            col = AMBER if kind == 'n' else (TEAL if kind == 'b' else CYAN)
            box(ax, x0 - 1.30, y, 2.60, 0.52, label=lab, edge=col, lw=1.6)
            y += 0.72
        ax.plot([x0, x0], [yy + 0.18, y - 0.20], color=GREY_L, lw=1.0, ls=':', zorder=1)
        text(ax, x0, y + 0.10, title, color=TEAL_L, size=12.5)
        return y

    ytop = column(2.6, 'pre-LN', True)
    column(7.4, 'post-LN', False)
    ax.annotate('', xy=(0.95, ytop - 0.30), xytext=(0.95, 1.00),
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=2.0))
    text(ax, 0.95, ytop + 0.02, 'no norm on this path', color=CYAN, size=11)
    ax.annotate('', xy=(6.00, ytop - 0.30), xytext=(6.00, 1.00),
                arrowprops=dict(arrowstyle='->', color=RED_S, lw=2.0))
    text(ax, 5.85, ytop + 0.02, 'a norm on this one', color=RED_S, size=11, ha='right')
    text(ax, 5.0, 0.30, 'the residual stream is the same shape at every point, which is '
                        'why blocks stack', color=TEAL, size=11.5, glowing=False)
    save(fig, 'day53_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """Splitting the heads: features per token, or whole tokens."""
    fig, ax = W(5.0)
    X, good, bad = T.head_split_demo()
    n, d = X.shape
    cw, ch = 0.53, 0.52

    def grid(x0, y0, pick, title):
        for i in range(n):
            for j in range(d):
                head = pick(i, j)
                col = AMBER if head == 0 else TEAL
                box(ax, x0 + j * cw, y0 + (n - 1 - i) * ch, cw * .94, ch * .9,
                    label=f'{X[i, j]:+.1f}', edge=col, lw=1.3)
        text(ax, x0 + d * cw / 2, y0 + n * ch + 0.22, title, color=TEAL_L, size=12)

    grid(0.45, 2.45, lambda i, j: 0 if j < d // 2 else 1,
         'split_heads: head 0 takes features 0-3')
    grid(5.45, 2.45, lambda i, j: 0 if i < n // 2 else 1,
         'reshape(2, 4, 4): head 0 takes tokens 0-1')
    text(ax, 2.57, 2.05, 'every token, half its features', color=AMBER, size=11.5)
    text(ax, 7.57, 2.05, 'half the tokens, all their features', color=RED_S, size=11.5)
    mx, rel, _ = T.split_output_gap()
    text(ax, 5.0, 1.30, f'same shape, no exception, and the layer output differs by '
                        f'{rel * 100:.0f}% of its own norm',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.80, 'and the positions it then attends over are half-tokens, so the '
                        'output rows stop being tokens',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day53_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Two patterns at once: what a second head is for."""
    fig, ax = W(5.0)
    V, prev, first, target = T.two_pattern_target()
    A, M, single = T.best_single_head()
    two, zero = T.two_head_error_many(), T.target_scale()
    cw = 0.34
    n = prev.shape[0]

    def mat(x0, y0, M, title, col):
        hi = max(0.05, float(np.max(M)))
        for i in range(n):
            for j in range(n):
                x, y = x0 + j * cw, y0 + (n - 1 - i) * cw
                ax.add_patch(Rectangle((x, y), cw * .9, cw * .9, facecolor=NAVY,
                                       edgecolor=TEAL, lw=0.8, zorder=3))
                v = float(M[i, j]) / hi
                if v > 0.02:
                    ax.add_patch(Rectangle((x, y), cw * .9, cw * .9, facecolor=col,
                                           edgecolor='none', alpha=min(1.0, v), zorder=4))
        text(ax, x0 + n * cw / 2, y0 + n * cw + 0.24, title, color=TEAL_L, size=11.5)

    mat(0.75, 2.10, prev, 'head 1: previous token', TEAL)
    mat(3.85, 2.10, first, 'head 2: first token', AMBER)
    mat(6.95, 2.10, A, 'one head: the compromise', RED_S)
    text(ax, 5.0, 1.45, f'over 12 inputs: two heads are exact (MSE {two:.1f}); one head, '
                        f'with its own channel matrix, stops at {single:.4f}',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.95, f'predicting zero everywhere would be {zero:.4f}, so the single '
                        f'head gets part of the way and no further',
         color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.45, 'one head, one attention matrix for every channel',
         color=CYAN, size=12)
    save(fig, 'day53_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """How wrong E[x^2] - E[x]^2 gets as the mean offset grows, in float32."""
    fig, ax = W(5.2)
    rows = T.cancellation_table()
    X0, Y0, WD, HT = 1.55, 1.70, 6.6, 2.60
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * i / (len(rows) - 1) for i in range(len(rows))]
    LO, HI = -4.0, 4.0                      # log10 of the relative error

    def ly(v):
        return Y0 + HT * (max(LO, math.log10(max(v, 1e-4))) - LO) / (HI - LO)

    ys = [ly(r['rel_err']) for r in rows]
    ax.plot(xs, ys, color=RED_S, lw=2.6, zorder=4)
    for i, r in enumerate(rows):
        if r['one_pass'] <= 0:
            cross(ax, xs[i], ys[i])
        else:
            ax.add_patch(Rectangle((xs[i] - .06, ys[i] - .06), .12, .12,
                                   facecolor=RED_S, edgecolor=RED_S, zorder=5))
    ax.plot([X0, X0 + WD], [ly(1.0), ly(1.0)], color=TEAL, lw=1.2, ls='--', zorder=2)
    text(ax, X0 + 0.12, ly(1.0) + 0.24, 'as wrong as the answer itself',
         color=TEAL, size=11, ha='left')
    for i, r in enumerate(rows):
        text(ax, xs[i], Y0 - 0.30, f"{r['offset']:,.0f}", color=TEAL, size=10.5)
    for v, lab in ((1e-4, 'exact'), (1e-2, '1%'), (1.0, '100%'), (1e2, '1e4%')):
        text(ax, X0 - 0.20, ly(v), lab, color=TEAL, size=10.5, ha='right')
    bad = next(r for r in rows if r['one_pass'] < 0)
    k = [r['offset'] for r in rows].index(bad['offset'])
    text(ax, xs[k] - 0.30, ys[k] + 0.44, f"one pass says {bad['one_pass']:.0f}",
         color=RED_S, size=11.5, ha='right')
    text(ax, X0 + WD / 2, Y0 - 0.76, 'constant added to every value (float32)',
         color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.10,
         f"error in the one-pass variance; two passes stay at about "
         f"{rows[0]['two_pass']:.4f} throughout",
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.52, 'the crossed points are negative or zero variances, and the '
                        'square root of that is nan', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day53_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """What the residual stream norm does with depth."""
    fig, ax = W(5.0)
    tr = T.depth_signal(layers=32)
    g = T.residual_growth(layers=32)
    gs = T.residual_growth(layers=32, branch_scale=1 / math.sqrt(64))
    start, plain, scaled = g['start'], g['norms'], gs['norms']
    X0, Y0, WD, HT = 1.35, 1.50, 7.0, 2.80
    axes2(ax, X0, Y0, WD, HT)
    L = len(plain)
    xs = [X0 + WD * i / (L - 1) for i in range(L)]
    top = 210.0
    series = ((plain, AMBER, f'branches unscaled: {plain[-1]:.0f}'),
              (scaled, TEAL_L, f'scaled by 1/sqrt(2L): {scaled[-1]:.0f}'),
              ([t['norm'] for t in tr['post']], PURPLE,
               f"post-LN: {tr['post'][-1]['norm']:.0f} every block"))
    for vals, col, lab in series:
        ax.plot(xs, liny(vals, Y0, HT, 0, top), color=col, lw=2.6, zorder=4)
    legend(ax, 1.75, Y0 + HT - 0.15, [(c, lab) for _, c, lab in series],
           size=11.5, dy=0.38)
    for v in (0, 100, 200):
        text(ax, X0 - 0.20, liny([v], Y0, HT, 0, top)[0], str(v), color=TEAL, size=10.5,
             ha='right')
    for i in (0, 7, 15, 31):
        text(ax, xs[i], Y0 - 0.30, str(i + 1), color=TEAL, size=10.5)
    text(ax, X0 + WD / 2, Y0 - 0.74, 'blocks stacked', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.30, 'norm of the residual stream', color=GREY_L,
         size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.30, f'summing 32 blocks into one stream grows it from {start:.0f} to '
                        f'{plain[-1]:.0f} = {plain[-1] / start:.1f}x, the sqrt(32) = '
                        f'{math.sqrt(32):.1f} of a running sum',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day53_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Where the parameters are, and when the quadratic term takes over."""
    fig, ax = W(5.0)
    p = T.block_params(4096)
    bw, base, maxh = 1.00, 2.40, 2.00
    vals = [p['attn'], p['mlp']]
    cols = [TEAL, AMBER]
    labs = [f"attention\n{p['attn'] / 1e6:.0f}M", f"MLP\n{p['mlp'] / 1e6:.0f}M"]
    for i, v in enumerate(vals):
        hh = maxh * v / max(vals)
        x = 1.15 + i * 1.55
        ax.add_patch(Rectangle((x, base), bw, hh, facecolor=cols[i], edgecolor=cols[i],
                               lw=1.2, alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + hh + 0.30, labs[i], color=cols[i], size=11.5)
    text(ax, 2.30, base - 0.34, f"one block at d_model 4096: "
                                f"{p['mlp_share'] * 100:.0f}% of it is the MLP",
         color=TEAL_L, size=11.5)
    X0, Y0, WD, HT = 5.60, 2.40, 3.60, 2.00
    axes2(ax, X0, Y0, WD, HT)
    ns = [2 ** k for k in range(8, 18)]
    sh = [T.block_flops(n, 4096)['quad_share'] for n in ns]
    xs = [X0 + WD * i / (len(ns) - 1) for i in range(len(ns))]
    ax.plot(xs, liny(sh, Y0, HT, 0, 1.0), color=CYAN, lw=2.6, zorder=4)
    cx = T.flop_crossover(4096)
    ci = (math.log2(cx) - 8) / (17 - 8)
    ax.plot([X0 + WD * ci, X0 + WD * ci], [Y0, Y0 + HT], color=RED_S, lw=1.4, ls='--',
            zorder=3)
    text(ax, X0 + WD * ci - 0.12, Y0 + 0.16, f'crossover {cx:,.0f}',
         color=RED_S, size=11, ha='right')
    for v, lab in ((0.0, '0'), (0.5, '50%'), (1.0, '100%')):
        text(ax, X0 - 0.18, liny([v], Y0, HT, 0, 1.0)[0], lab, color=TEAL, size=10.5,
             ha='right')
    for i, n in enumerate(ns):
        if n in (256, 4096, 131072):
            text(ax, xs[i], Y0 - 0.30, f'{n:,}', color=TEAL, size=10.5)
    text(ax, X0 + WD / 2, Y0 - 0.74, 'context length', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.06, 'share of the block that is quadratic',
         color=GREY_L, size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.95, 'with d_ff = 4 d_model the crossover is always 6 d_model, '
                        'whatever the head count', color=TEAL, size=11.5, glowing=False)
    text(ax, 5.0, 0.45, 'h splits the same four matrices and feeds the same two matmuls',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day53_6.png')


for f in (fig0, fig1, fig2, fig3, fig4, fig5, fig6):
    f()
    print(f.__name__, 'ok')
