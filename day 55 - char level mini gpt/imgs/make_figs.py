"""Day 55 figures.  Trains the model once (about a minute) and draws from that run.
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
import mini_gpt as M                                 # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'

TEXT = M.make_corpus()
TOK = M.CharTokenizer(TEXT)
DATA = TOK.encode(TEXT)
ENT = M.source_entropy()[0]
P, CFG = M.init_params(TOK.vocab_size)
HIST = M.train(P, CFG, DATA, steps=500)
P2, CFG2 = M.init_params(TOK.vocab_size, seed=0)
HIST2 = M.train(P2, CFG2, DATA, steps=250, causal=False)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def smooth(v, k=20):
    return [float(np.mean(v[max(0, i - k):i + 1])) for i in range(len(v))]


# ---------------------------------------------------------------- figure 0
def fig0():
    fig, ax = W(5.0)
    steps = [('characters', '"the cat sat"', TEAL),
             ('ids', '19, 11, 8, ...', TEAL),
             ('embed + position', f'{CFG["d_model"]} numbers per character', AMBER),
             (f'{CFG["n_layer"]} blocks', 'attention, then MLP', TEAL_L),
             ('final norm', '', AMBER),
             ('logits', f'{CFG["vocab"]} scores for the next character', TEAL)]
    y = 4.05
    for name, note, col in steps:
        box(ax, 1.00, y, 3.20, 0.52, label=name, edge=col, lw=1.6)
        if note:
            text(ax, 4.45, y + 0.26, note, color=GREY_L, size=11.5, ha='left',
                 glowing=False)
        if y > 0.9:
            ax.annotate('', xy=(2.60, y - 0.12), xytext=(2.60, y - 0.02),
                        arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.4))
        y -= 0.64
    text(ax, 5.0, 5.05, f'a {M.param_count(P):,}-parameter character-level GPT, '
                        f'start to finish', color=TEAL_L, size=13)
    text(ax, 5.0, 0.42, 'sample a character from the last row of logits, append it, '
                        'and run the whole thing again', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day55_0.png')


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = W(5.0)
    X0, Y0, WD, HT = 1.45, 1.60, 7.0, 2.70
    axes2(ax, X0, Y0, WD, HT)
    top = math.log(TOK.vocab_size) * 1.05
    a, b = smooth(HIST), smooth(HIST2)
    xs = [X0 + WD * i / (len(a) - 1) for i in range(len(a))]
    ax.plot(xs, [Y0 + HT * v / top for v in a], color=TEAL_L, lw=2.4, zorder=4)
    xs2 = [X0 + WD * i / (len(a) - 1) for i in range(len(b))]
    ax.plot(xs2, [Y0 + HT * v / top for v in b], color=RED_S, lw=2.4, zorder=4)
    floor = ENT * math.log(2)
    ax.plot([X0, X0 + WD], [Y0 + HT * floor / top] * 2, color=AMBER, lw=1.4, ls='--',
            zorder=3)
    text(ax, X0 + 0.12, Y0 + HT * floor / top - 0.28,
         f'the entropy of the generator: {ENT:.4f} bits/char', color=AMBER, size=11.5,
         ha='left')
    text(ax, X0 + WD - 0.1, Y0 + HT * a[-1] / top + 0.52,
         f'trained properly: {a[-1] / math.log(2):.3f} bits/char', color=TEAL_L,
         size=11.5, ha='right')
    text(ax, xs2[-1] + 0.20, Y0 + 0.62,
         f'no causal mask: {b[-1] / math.log(2):.3f} bits/char', color=RED_S, size=11.5,
         ha='left')
    for v, lab in ((0.0, '0'), (math.log(2), '1 bit'), (math.log(TOK.vocab_size), f'ln({TOK.vocab_size})')):
        text(ax, X0 - 0.18, Y0 + HT * v / top, lab, color=TEAL, size=10.5, ha='right')
    for i in (0, 250, 499):
        text(ax, xs[i], Y0 - 0.30, str(i + 1), color=TEAL, size=10.5)
    text(ax, X0 + WD / 2, Y0 - 0.72, 'training step', color=TEAL, size=11.5)
    text(ax, X0 + 0.10, Y0 + HT + 0.30, 'cross-entropy loss (nats)', color=GREY_L,
         size=11, ha='left', glowing=False)
    text(ax, 5.0, 0.52, 'the lower curve is the one that cheats: every position can see '
                        'the character it must predict', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day55_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    fig, ax = W(5.0)
    rows = []
    for temp, top_k in ((1.0, None), (0.8, 5), (0.2, None)):
        out = M.sample(P, CFG, TOK, n=240, temperature=temp, top_k=top_k, seed=7)
        frac, n = M.grammatical_fraction(out)
        line = [l for l in out.split('\n')[1:-1] if l]
        rows.append((temp, top_k, frac, n, line[0] if line else ''))
    base, maxh = 2.55, 1.90
    for i, (temp, top_k, frac, n, line) in enumerate(rows):
        x = 1.35 + i * 2.55
        h = max(0.05, maxh * frac)
        col = TEAL_L if frac >= 0.9 else (AMBER if frac >= 0.6 else RED_S)
        ax.add_patch(Rectangle((x, base), 1.45, h, facecolor=col, edgecolor=col,
                               lw=1.2, alpha=.85, zorder=3))
        text(ax, x + 0.72, base + h + 0.26, f'{frac * 100:.0f}%', color=col, size=13)
        lab = f'temp {temp}' + (f', top-k {top_k}' if top_k else '')
        text(ax, x + 0.72, base - 0.30, lab, color=TEAL, size=11.5)
    text(ax, 5.0, 5.00, 'how many generated lines the grammar could have produced',
         color=TEAL_L, size=12.5)
    text(ax, 5.0, 1.75, 'one line from each sample:', color=GREY_L, size=11,
         glowing=False)
    for i, (temp, top_k, frac, n, line) in enumerate(rows):
        text(ax, 5.0, 1.35 - i * 0.40, f'{line[:58]}', color=TEAL, size=11,
             glowing=False)
    save(fig, 'day55_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    fig, ax = W(6.4)
    s = 'the small cat sat on the mat.\nth'[:CFG['ctx']]
    ids = TOK.encode(s)[None, :]
    _, cache = M.forward(P, CFG, ids)
    w = cache['layers'][1]['w'][0, 0]           # last block, first head
    n = len(s)
    cw = 4.40 / n
    x0, y0 = 3.15, 1.35
    hi = float(w.max())
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            v = float(w[i, j]) / hi
            if v > 0.03:
                ax.add_patch(Rectangle((x0 + j * cw, y0 + (n - 1 - i) * cw), cw * .92,
                                       cw * .92, facecolor=TEAL, edgecolor='none',
                                       alpha=min(1.0, v), zorder=3))
    ax.add_patch(Rectangle((x0, y0), n * cw, n * cw, facecolor='none', edgecolor=TEAL,
                           lw=1.2, zorder=4))
    for i in range(n):
        ch = s[i].replace('\n', '|')
        text(ax, x0 - 0.16, y0 + (n - 1 - i) * cw + cw / 2, ch, color=GREY_L, size=8.5,
             ha='right', glowing=False)
        text(ax, x0 + i * cw + cw / 2, y0 - 0.18, ch, color=GREY_L, size=8.5,
             glowing=False)
    text(ax, 5.0, 6.45, 'one head of the second block, reading "the small cat sat on '
                        'the mat."', color=TEAL_L, size=12.5)
    text(ax, 5.0, 0.60, 'each row is one position and shows where it looked; the empty '
                        'upper triangle is the causal mask', color=TEAL, size=11.5,
         glowing=False)
    save(fig, 'day55_3.png')


for f in (fig0, fig1, fig2, fig3):
    f()
    print(f.__name__, 'ok')
