"""Day 51 figures.  Every number is computed here or by importing bpe.py."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
from diagram_style import *                          # noqa: F401,F403,E402
import bpe as B                                      # noqa: E402

os.chdir(HERE)
RED_S = '#ff5c5c'
PURPLE = '#b45cff'

MERGES, VOCAB, TRACE = B.train(B.CORPUS, 400, verbose_until=8)


def W(h):
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def axes2(ax, x0, y0, w, h):
    ax.plot([x0, x0 + w], [y0, y0], color=TEAL, lw=1.2, zorder=2)
    ax.plot([x0, x0], [y0, y0 + h], color=TEAL, lw=1.2, zorder=2)


def liny(vals, y0, h, lo, hi):
    return [y0 + h * (v - lo) / (hi - lo) for v in vals]


def piece(i):
    return VOCAB[i].decode('utf-8', 'replace')


def strip(ax, x0, y, ids, cw=0.78, ch=0.62, edge=None, labels=True):
    """A row of token boxes, each as wide as it needs to be."""
    x = x0
    for i in ids:
        t = piece(i).replace(' ', '␣')            # show the space
        w = max(cw, 0.30 + 0.26 * len(t))
        box(ax, x, y, w, ch, label=t, edge=edge or TEAL, lw=1.6)
        x += w + 0.10
    return x


# ---------------------------------------------------------------- figure 1
def fig1():
    """The same five letters, one token or three, depending on a space."""
    fig, ax = W(4.4)
    rows = [(' tokens', AMBER), ('tokens', RED_S)]
    for r, (word, col) in enumerate(rows):
        ids = B.encode(word, MERGES)
        y = 3.05 - r * 1.15
        text(ax, 1.55, y + 0.31, repr(word), color=col, size=13, ha='right')
        end = strip(ax, 1.75, y, ids, edge=col)
        text(ax, end + 0.25, y + 0.31, f'{len(ids)} token' + ('s' if len(ids) > 1 else ''),
             color=col, size=13, ha='left')
    text(ax, 5.0, 4.55, 'one leading space, one token instead of three',
         color=TEAL_L, size=13)
    text(ax, 5.0, 1.00,
         'a leading space is part of the word, because that is how the word appears '
         'in running text',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day51_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """One sentence, ten words, twelve tokens."""
    fig, ax = W(4.2)
    s = 'the cost of a model call is counted in tokens'
    ids = B.encode(s, MERGES)
    widths = [0.30 + 0.24 * len(piece(i).replace(' ', '␣')) for i in ids]
    total = sum(widths) + 0.08 * (len(ids) - 1)
    scale = min(1.0, 9.0 / total)
    x = (10 - total * scale) / 2
    for i, w0 in zip(ids, widths):
        t = piece(i).replace(' ', '␣')
        w = w0 * scale
        hot = piece(i).strip() in ('ca', 'll')
        box(ax, x, 2.35, w, 0.66, label=t, edge=AMBER if hot else TEAL,
            lw=2.0 if hot else 1.6, )
        text(ax, x + w / 2, 2.08, str(i), color=GREY_L, size=9.5, glowing=False)
        x += w + 0.08 * scale
    text(ax, 5.0, 4.35, f'"{s}"', color=TEAL_L, size=13)
    text(ax, 5.0, 3.40,
         f'{len(s.split())} words, {len(s)} characters, {len(s.encode("utf-8"))} bytes, '
         f'{len(ids)} tokens',
         color=TEAL, size=12.5)
    text(ax, 5.0, 1.30, 'the amber pair is the word "call", which this corpus never '
         'learned to keep together', color=AMBER, size=11.5, glowing=False)
    text(ax, 5.0, 0.92, 'the numbers underneath are what the model actually receives',
         color=GREY_L, size=11, glowing=False)
    save(fig, 'day51_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Encoding replays the merges in the order they were learned."""
    fig, ax = W(5.2)
    word = ' counted'
    ids = tuple(B.to_bytes(word))
    rank = {p: k for k, (p, _) in enumerate(MERGES)}
    new_of = {p: n for p, n in MERGES}
    rows = [(None, ids)]
    while len(ids) >= 2:
        cand = [p for p in set(zip(ids, ids[1:])) if p in rank]
        if not cand:
            break
        best = min(cand, key=lambda p: rank[p])
        ids = B.merge_word(ids, best, new_of[best])
        rows.append((rank[best] + 1, ids))
    for r, (mk, row) in enumerate(rows):
        y = 4.85 - r * 0.62
        lab = 'bytes' if mk is None else f'merge #{mk}'
        text(ax, 2.05, y + 0.24, lab, color=GREY_L if mk is None else AMBER, size=11,
             ha='right', glowing=False)
        strip(ax, 2.25, y, row, cw=0.60, ch=0.52,
              edge=TEAL_L if r == len(rows) - 1 else TEAL)
    text(ax, 5.0, 5.55, "encoding ' counted': apply the earliest merge that still fits",
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.52,
         'the rank decides, not the length - longest-match is a different algorithm',
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day51_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """Ties: the most frequent pair is usually not unique."""
    fig, ax = W(4.6)
    steps, ties, first = B.tie_report(B.CORPUS, 400)
    base, maxh, bw = 1.55, 2.45, 1.30
    for i, (lab, val, col) in enumerate([
            ('merge steps', steps, TEAL), ('steps with a tie', ties, RED_S)]):
        x = 2.60 + i * 2.60
        h = maxh * val / steps
        ax.add_patch(Rectangle((x, base), bw, h, facecolor=col, edgecolor=col, lw=1.2,
                               alpha=.85, zorder=3))
        text(ax, x + bw / 2, base + h + 0.26, str(val), color=col, size=13)
        text(ax, x + bw / 2, base - 0.32, lab, color=GREY_L, size=11.5, glowing=False)
    st, cnt, examples, howmany = first
    shown = ', '.join(repr(B.decode([a], VOCAB, 'replace') + B.decode([b], VOCAB, 'replace'))
                      for a, b in examples)
    text(ax, 5.0, 4.95, f'{100 * ties / steps:.0f}% of the merge steps had a tie at the top',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.84,
         f'the first is at step {st}: {howmany} pairs all occur {cnt} times - {shown}',
         color=AMBER, size=11.5, glowing=False)
    text(ax, 5.0, 0.44,
         'both choices compress to within a few tokens of each other and disagree '
         'about what a token is',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day51_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """A three-byte character split across tokens: join first, decode once."""
    fig, ax = W(4.6)
    zh = '€5 café €9'
    ids = B.encode(zh, MERGES)
    x = 0.70
    for i in ids:
        ok = B._valid_utf8(VOCAB[i])
        w = 0.50 if len(VOCAB[i]) == 1 else 0.82
        lab = VOCAB[i].decode('utf-8').replace(' ', '␣') if ok else f'{VOCAB[i][0]:02x}'
        box(ax, x, 3.05, w, 0.62, label=lab, edge=TEAL_L if ok else RED_S,
            lw=2.0 if ok else 1.5)
        x += w + 0.08
    bad = sum(1 for i in ids if not B._valid_utf8(VOCAB[i]))
    text(ax, 5.0, 4.85, f'{zh!r} is {len(zh.encode("utf-8"))} bytes and {len(ids)} tokens',
         color=TEAL_L, size=13)
    text(ax, 5.0, 2.70, f'{bad} of the {len(ids)} tokens are not valid UTF-8 on their own',
         color=RED_S, size=12)
    for r, (lab, val, col) in enumerate([
            ('join the bytes, decode once', B.decode(ids, VOCAB), TEAL_L),
            ('decode each token, then join',
             ''.join(VOCAB[i].decode('utf-8', 'replace') for i in ids), RED_S)]):
        y = 1.95 - r * 0.60
        text(ax, 4.75, y, lab, color=GREY_L, size=11.5, ha='right', glowing=False)
        text(ax, 5.00, y, val, color=col, size=13, ha='left')
    text(ax, 5.0, 0.52,
         "errors='replace' is why the second line does not raise",
         color=TEAL, size=11.5, glowing=False)
    save(fig, 'day51_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Vocabulary size against tokens per byte."""
    fig, ax = W(4.8)
    sizes, ratios, nmerges = [], [], []
    for size in range(256, 401, 8):
        mm, vv, _ = B.train(B.CORPUS, size)
        n, b, r = B.tokens_per_byte(B.CORPUS, mm, vv)
        sizes.append(len(vv))
        ratios.append(r)
        nmerges.append(len(mm))
    X0, Y0, WD, HT = 1.35, 1.55, 6.6, 2.85
    axes2(ax, X0, Y0, WD, HT)
    xs = [X0 + WD * (s - sizes[0]) / (sizes[-1] - sizes[0]) for s in sizes]
    ax.plot(xs, liny(ratios, Y0, HT, 0, 1.05), color=TEAL_L, lw=2.6, zorder=4)
    for v, lab in ((0.0, '0'), (0.5, '0.5'), (1.0, '1')):
        text(ax, X0 - 0.22, liny([v], Y0, HT, 0, 1.05)[0], lab, color=TEAL, size=10.5,
             ha='right')
    for s in (256, 300, 350, sizes[-1]):
        xx = X0 + WD * (s - sizes[0]) / (sizes[-1] - sizes[0])
        text(ax, xx, Y0 - 0.30, str(s), color=TEAL, size=11)
    text(ax, xs[0] + 0.45, liny([ratios[0]], Y0, HT, 0, 1.05)[0] + 0.26,
         f'{ratios[0]:.4f}', color=TEAL_L, size=12, ha='left')
    text(ax, xs[-1] - 0.15, liny([ratios[-1]], Y0, HT, 0, 1.05)[0] + 0.34,
         f'{ratios[-1]:.4f}', color=TEAL_L, size=12, ha='right')
    text(ax, X0 + WD / 2, Y0 - 0.74, 'vocabulary size', color=TEAL, size=11.5)
    text(ax, X0 + WD - 0.10, Y0 + HT + 0.04, 'tokens per byte', color=GREY_L, size=11,
         ha='right', glowing=False)
    text(ax, 5.0, 5.15, 'every merge spends one id to make the text a little shorter',
         color=TEAL_L, size=13)
    text(ax, 5.0, 0.42,
         f'{nmerges[-1]} merges take it from {ratios[0]:.4f} to {ratios[-1]:.4f} tokens '
         f'per byte; greedy spends the big savings first',
         color=TEAL, size=11, glowing=False)
    save(fig, 'day51_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, 'ok')
