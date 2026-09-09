"""Figures for day 25 - KMP, Rabin-Karp, and the string matching inside a server.

Every number printed on a figure is computed by importing string_matching.py, so
the pictures cannot drift away from the code.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: F401,F403
import string_matching as S


# ---- numbers straight out of the module ----------------------------------
TEXT = "a" * 40 + "aaab"
PAT = "aaab"
N_CMP = S.naive_search(TEXT, PAT)[1]
K_CMP = S.kmp_search(TEXT, PAT)[1]

P2 = "ababaca"
FAIL2 = S.build_failure(P2)

VOCAB = {1: "The", 2: " answer", 3: " is", 4: " 42]", 5: "]", 6: " so",
         7: " far", 8: " and", 9: " more"}
TOK = S.ToyTokenizer(VOCAB)

COL_A, COL_B = S.find_collision(mod=101, length=3)


def _h(s, base=256, mod=101):
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h


# --------------------------------------------------------------------------
def fig1():
    """The naive matcher re-reads the text; KMP never does."""
    fig, ax = canvas(10.4, 7.2, xlim=(0, 10.4), ylim=(0, 7.2))

    txt = "a a a a a a a b".split()
    w, x0 = 0.78, 1.55

    text(ax, 5.2, 6.95, 'naive: every alignment starts over', color=TEAL_L, size=16)
    row = cells(ax, x0, 6.10, len(txt), w=w, h=0.62, labels=txt, size=14)
    text(ax, x0 - 0.18, 6.41, 'text', color=TEAL, size=14, ha='right')

    # three failing alignments
    for k, y in enumerate([5.20, 4.42, 3.64]):
        labs = list(PAT)
        cells(ax, x0 + k * w, y, 4, w=w, h=0.62, labels=labs, size=14,
              fill=GREY)
        # the mismatch is always the last character
        box(ax, x0 + (k + 3) * w, y, w, 0.62, label='b', fill=GREY,
            edge=RED, lw=2.4, size=14)
        cross(ax, x0 + (k + 4) * w + 0.28, y + 0.31, r=0.20)
        if k < 2:
            arrow(ax, (x0 + (k + 3.5) * w, y - 0.08), (x0 + (k + 1.4) * w, y - 0.40),
                  color=RED, lw=1.6, rad=0.30)
    text(ax, x0 + 4.3 * w, 3.30, 'three characters compared, then the\n'
         'text pointer walks back and does it again',
         color=RED, size=13, ha='left')

    text(ax, 4.6, 2.72, 'KMP: one pass, only the pattern index moves back',
         color=TEAL_L, size=16)
    row2 = cells(ax, x0, 1.85, len(txt), w=w, h=0.62, labels=txt, size=14)
    arrow(ax, (x0 + 0.05, 1.45), (x0 + len(txt) * w - 0.05, 1.45),
          color=TEAL, lw=2.0)
    text(ax, x0 + len(txt) * w / 2, 1.14, 'i only ever increases', color=TEAL, size=14)
    text(ax, x0 + len(txt) * w + 0.25, 2.16,
         'only the pattern index j\nfalls back, to fail[j-1]',
         color=AMBER, size=13, ha='left')

    text(ax, 5.2, 0.45, "on text = 'a'*40 + 'aaab', pattern 'aaab':  "
         "naive %d comparisons, KMP %d" % (N_CMP, K_CMP),
         color=AMBER, size=15)
    return save(fig, os.path.join(HERE, 'day25_1.png'))


# --------------------------------------------------------------------------
def fig2():
    """The failure function is a table of borders."""
    fig, ax = canvas(10.0, 7.0, xlim=(0, 10.0), ylim=(0, 7.0))

    w, x0, y = 1.05, 1.45, 4.30
    cells(ax, x0, y, len(P2), w=w, h=0.72, labels=list(P2), size=16)
    cells(ax, x0, y - 0.72, len(P2), w=w, h=0.72,
          labels=[str(v) for v in FAIL2], size=15, index=True)
    text(ax, x0 - 0.18, y + 0.36, 'pattern', color=TEAL, size=14, ha='right')
    text(ax, x0 - 0.18, y - 0.36, 'fail', color=TEAL, size=14, ha='right')

    text(ax, 5.0, 6.70, "fail[i] = longest border of pat[:i+1]",
         color=TEAL_L, size=16)

    # the border of pat[:5] = 'ababa' is 'aba'
    brace(ax, x0 + 2 * w, x0 + 5 * w, 5.75, height=0.24, label="suffix 'aba'", size=14)
    brace(ax, x0, x0 + 3 * w, 5.15, height=0.24, label="prefix 'aba'", size=14)
    text(ax, 5.0, 3.15, "pat[:5] = 'ababa', prefix 'aba' == suffix 'aba', so fail[4] = 3",
         color=AMBER, size=15)

    text(ax, 5.0, 2.15, "the text ended with pat[:j], so the only alignments still alive\n"
         "are the borders of pat[:j] - everything in between is a guaranteed\n"
         "mismatch, and is skipped without a single comparison",
         color=TEAL, size=15)

    box(ax, 1.30, 0.25, 7.40, 0.95,
        label="n - fail[n-1]  is the smallest period of the pattern\n"
              "(LC 459 is exactly that, in one line)",
        fill=NAVY, edge=AMBER, lw=2.0, size=15)
    return save(fig, os.path.join(HERE, 'day25_2.png'))


# --------------------------------------------------------------------------
def fig3():
    """A stop string arriving in pieces, and the tail that must be withheld."""
    fig, ax = canvas(10.8, 7.0, xlim=(0, 10.8), ylim=(0, 7.0))

    out = "The answer is 42.</s>"
    stop_at = len(out) - 4                       # where '</s>' begins
    w, x0, y = 0.46, 0.55, 4.30
    for k, ch in enumerate(out):
        inside = k >= stop_at
        box(ax, x0 + k * w, y, w - 0.03, 0.70,
            label=('_' if ch == ' ' else ch),
            fill=GREY if inside else NAVY,
            edge=AMBER if inside else TEAL, lw=2.2 if inside else 1.5, size=14)

    text(ax, 5.4, 6.35, 'the stop string is split across three streamed chunks',
         color=TEAL_L, size=16)

    bounds = [(0, 15, 0.22), (15, 18, 0.22), (18, 20, 0.22), (20, 21, 0.52)]
    for n, (a, b, hh) in enumerate(bounds):
        brace(ax, x0 + a * w + 0.04, x0 + b * w - 0.06, y + 0.80,
              height=hh, label='chunk %d' % (n + 1), size=13)

    brace(ax, x0, x0 + 17 * w - 0.06, y - 0.14, height=0.24, down=True,
          label="emitted: 'The answer is 42.'", size=14)
    brace(ax, x0 + 17 * w, x0 + 18 * w - 0.06, y - 0.14, height=0.24, down=True,
          label="held", size=14)

    text(ax, 5.40, 3.15,
         "the '<' is a 1-character match of '</s>', so it is not streamed "
         "- at that moment nobody knows yet",
         color=AMBER, size=14)

    box(ax, 0.55, 1.55, 9.70, 1.30,
        label="hold back k characters, where k = the longest suffix of the output that is\n"
              "also a prefix of a stop string - a border across two strings, which the\n"
              "failure function of  stop + sep + tail  answers in one pass",
        fill=NAVY, edge=TEAL, size=14)
    text(ax, 5.40, 0.85, 'the matcher state survives between chunks, so text that has '
         'already been\nstreamed is never looked at again', color=AMBER, size=14)
    return save(fig, os.path.join(HERE, 'day25_3.png'))


# --------------------------------------------------------------------------
def fig4():
    """Tokens are not characters: the window that speculative decoding breaks."""
    fig, ax = canvas(10.8, 6.6, xlim=(0, 10.8), ylim=(0, 6.6))

    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    w, x0, y = 1.18, 0.75, 4.90
    for k, i in enumerate(ids):
        box(ax, x0 + k * w, y, w - 0.06, 0.72, label=repr(VOCAB[i]),
            fill=NAVY, edge=AMBER if i in (4, 5) else TEAL,
            lw=2.4 if i in (4, 5) else 1.6, size=12)
        text(ax, x0 + k * w + (w - 0.06) / 2, y - 0.28, 't%d' % (k + 1),
             color=GREY_L, size=12, glowing=False)
    text(ax, 5.4, 6.25, "the stop ']]' starts inside t4 and ends in t5",
         color=TEAL_L, size=16)
    text(ax, 5.4, 5.80, "decoded: %s" % repr(TOK.decode(ids)), color=TEAL, size=14)

    text(ax, 0.75, 4.10, 'at the step that ends on t8,\nthe tail windows are:',
         color=TEAL, size=13, ha='left')
    naive_w = min(len("]]") + 1, 8)
    ext_w = S.stop_match_tail_len(len("]]"), 4, 8)
    yb = 3.85
    xa = x0 + (8 - naive_w) * w
    region(ax, xa, yb, naive_w * w - 0.06, 0.62)
    text(ax, xa + naive_w * w / 2, yb + 0.31,
         'stop_max_len + 1 = %d tokens' % naive_w, color=TEAL_L, size=13)
    cross(ax, xa - 0.42, yb + 0.31, r=0.24)
    text(ax, xa - 0.86, yb + 0.31, 'misses it', color=RED, size=14, ha='right')

    yc = 2.95
    xc = x0 + (8 - ext_w) * w
    region(ax, xc, yc, ext_w * w - 0.06, 0.62)
    text(ax, xc + ext_w * w / 2, yc + 0.31,
         '+ (accepted - 1) = %d tokens' % ext_w, color=TEAL_L, size=13)
    text(ax, xc + ext_w * w + 0.20, yc + 0.31, 'catches it', color=AMBER,
         size=14, ha='left')

    box(ax, 0.75, 1.35, 9.30, 1.20,
        label="with 4 tokens accepted per step the check does not run at t5 at all;\n"
              "the batch lands on t8 and the short window no longer contains ']]'",
        fill=NAVY, edge=RED, lw=2.0, size=14)
    text(ax, 5.4, 0.62, "then decode growing prefixes of the window to turn a "
         "character match back into a token count", color=TEAL, size=14)
    return save(fig, os.path.join(HERE, 'day25_4.png'))


# --------------------------------------------------------------------------
def fig5():
    """Rolling hash, and the verify step that makes it an algorithm."""
    fig, ax = canvas(10.4, 6.4, xlim=(0, 10.4), ylim=(0, 6.4))

    s = "A C G T A C G T G".split()
    w, x0, y = 0.82, 1.60, 5.05
    cells(ax, x0, y, len(s), w=w, h=0.66, labels=s, size=14)
    text(ax, x0 - 0.18, y + 0.33, 'text', color=TEAL, size=14, ha='right')

    region(ax, x0 - 0.05, y - 0.10, 4 * w + 0.10, 0.86)
    text(ax, x0 + 2 * w, y + 1.05, 'window i', color=TEAL_L, size=14)
    region(ax, x0 + w - 0.05, y - 0.72, 4 * w + 0.10, 0.60, fill='#0093a1',
           alpha=0.18)
    text(ax, x0 + 3 * w, y - 1.02, 'window i+1', color=TEAL_L, size=14)

    text(ax, 5.2, 3.10, "h' = (h - s[i]*base^(m-1)) * base + s[i+m]   (mod q)",
         color=AMBER, size=16)
    text(ax, 5.2, 2.62, "one multiply-add per position, whatever the pattern length",
         color=TEAL, size=14)

    # the collision
    ha, hb = _h(COL_A), _h(COL_B)
    box(ax, 0.90, 1.05, 1.55, 0.70, label=repr(COL_A), fill=NAVY, size=14)
    box(ax, 0.90, 0.25, 1.55, 0.70, label=repr(COL_B), fill=NAVY, size=14)
    arrow(ax, (2.55, 1.40), (3.55, 1.10), color=TEAL)
    arrow(ax, (2.55, 0.60), (3.55, 0.90), color=TEAL)
    box(ax, 3.60, 0.65, 1.60, 0.70, label='hash = %d' % ha, fill=GREY,
        edge=RED, lw=2.2, size=14)
    cross(ax, 5.55, 1.00, r=0.26)
    text(ax, 6.05, 1.00, "equal fingerprints, different strings (mod 101)",
         color=RED, size=14, ha='left')
    text(ax, 6.05, 0.48, "so every candidate is compared for real - drop that\n"
         "line and the function is silently wrong",
         color=TEAL, size=13, ha='left')
    assert ha == hb
    return save(fig, os.path.join(HERE, 'day25_5.png'))


# --------------------------------------------------------------------------
def fig6():
    """A KV page id folds in the page before it - and has no verify step."""
    fig, ax = canvas(10.2, 7.0, xlim=(0, 10.2), ylim=(0, 7.0))

    system = list(range(1000, 1008))
    a_ids = system + [1, 2, 3, 4]
    b_ids = system + [1, 2, 9, 9]
    ha = S.page_hashes(a_ids, 4)
    hb = S.page_hashes(b_ids, 4)

    text(ax, 5.1, 6.62, 'a page id is a hash of the page and of every page before it',
         color=TEAL_L, size=16)

    def lab(page):
        return '[%d .. %d]' % (page[0], page[-1]) if page[-1] - page[0] == 3 \
            else str(page)

    w, gap, x0 = 2.35, 0.75, 0.80
    for ids, hs, name, yy in [(a_ids, ha, 'req A', 4.55), (b_ids, hb, 'req B', 2.85)]:
        text(ax, x0 - 0.18, yy + 0.42, name, color=TEAL, size=14, ha='right')
        for p in range(3):
            page = ids[p * 4:(p + 1) * 4]
            same = ha[p] == hb[p]
            x = x0 + p * (w + gap)
            box(ax, x, yy, w, 0.84,
                label=lab(page) + '\n' + hs[p][:8],
                fill=NAVY, edge=TEAL if same else RED,
                lw=1.6 if same else 2.4, size=13)
            if p < 2:
                arrow(ax, (x + w, yy + 0.42), (x + w + gap, yy + 0.42),
                      color=TEAL if same else AMBER, lw=1.6)

    brace(ax, x0, x0 + 2 * w + gap, 5.55, height=0.24,
          label='identical ids: exactly the shared prefix', size=14)
    text(ax, x0 + 2 * (w + gap) + w / 2, 2.35,
         'the first page that differs', color=RED, size=13)
    text(ax, 5.10, 1.95, 'and every id after it differs too, because the id folds in '
         'its whole history', color=TEAL, size=14)

    box(ax, 0.80, 0.30, 8.60, 1.30,
        label="no verify step here, unlike Rabin-Karp: the id is used to fetch\n"
              "somebody else's KV blocks and the tokens are not there to compare -\n"
              "so the collision probability has to be carried by the hash itself",
        fill=NAVY, edge=AMBER, lw=2.0, size=14)
    return save(fig, os.path.join(HERE, 'day25_6.png'))


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print(f())
