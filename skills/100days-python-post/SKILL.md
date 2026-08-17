---
name: 100days-python-post
description: Produce one day of the "100 Days of Python" series (KingRei/100DaysPython + Medium blog) - day folder, Jupyter notebook, README.md, Medium article draft, and diagrams in the series' signature dark-navy/teal glow style. Use when asked to write day N, add a topic to the series, or generate diagrams matching the existing image style.
---

# 100 Days of Python - daily post builder

Produces one day of the "100 Days of Python" series: a day folder in this repo
containing a notebook, a README, a Medium article draft, and diagrams that match the
series' established visual style.

## Step 0 - always read ROADMAP.md first

`ROADMAP.md` at the repo root is the source of truth for what day N is about, which
LeetCode problem pairs with it, and which earlier/later days it should call back to.
**It changes over time - re-read it every session, never rely on a remembered plan.**
The dependency list at the bottom records the deliberate callbacks (e.g. Day 37 LRU ->
Day 66 KV cache -> Day 67 RadixAttention); weave the relevant one into the article so
the series feels connected.

## Deliverables for day N

Folder name follows the existing convention: lowercase `day NN - topic`.

```
day NN - <topic>/
    <Topic>.ipynb      # live code, runnable top to bottom
    <topic>.py         # standalone runnable demo - python <topic>.py prints the whole story
    README.md          # GitHub-facing summary
    demo.html          # interactive animated walkthrough, single file, zh/en toggle
    medium_en.html     # the article, English - open in a browser, select all, paste
    medium_zh.html     # the article, 繁體中文
    imgs/
        dayNN_1.png
        make_figs.py   # the script that generated the images
```

### README.md format (match the existing days exactly)

```markdown
# <Topic Title>

More details in:
https://medium.com/100-days-of-python/day-NN-<slug>

![img](https://github.com/KingRei/100DaysPython/blob/master/day%20NN%20-%20<topic>/imgs/dayNN_1.png?raw=true)

<2-4 short paragraphs, Wikipedia-ish register>

## <Sub-heading per variant/approach>

## Complexity
| Operation | Time | Space |

## References
- [Wiki](...)
```

Leave the Medium URL as a placeholder when the article is unpublished, and say so in
the hand-off message.

### Notebook

Markdown cell explaining each section, then a clean commented implementation, then a
demo cell that prints output. Prefer the standard library. The notebook must carry
stored outputs and the code must actually run. This VM has no `nbformat` and pip is
blocked, so build the `.ipynb` JSON by hand: keep the cells in a `CELLS` list of
`('md'|'code', source)` tuples, `exec` the code cells in order into a shared namespace
while capturing stdout, and embed the captured text as a `stream` output. End the
notebook's test cell with asserts so a silent regression fails loudly.

### Standalone demo script

Every day ships a plain `.py` file that can be run directly (`python graph_traversal.py`)
and prints a self-explanatory walkthrough of the day's topic: the implementation, a
worked example with labelled sections, the paired LeetCode solution, and asserts that
prove it. Same code as the notebook, but importable and copy-pasteable - no Jupyter
required. Give it a `if __name__ == '__main__': main()` entry point, print section
headers so the output reads top to bottom, and run it before delivering.

### Medium articles - BOTH languages, every day

**First line of every article is the code link.** Right under the H1 title, before the
`*100 Days of Python - day NN*` line, put:

```markdown
程式碼：[day NN on GitHub](https://github.com/KingRei/100DaysPython/tree/master/day%20NN%20-%20<topic>)
Code: [day NN on GitHub](https://github.com/KingRei/100DaysPython/tree/master/day%20NN%20-%20<topic>)
```

(the first form for the 中文 article, the second for the English one - the folder name
is URL-encoded, spaces become `%20`).

Two articles, same content and structure, not machine-translated stubs:

- English, first person, teaching tone, 800-1400 words.
- 繁體中文，同樣的架構與程式碼區塊；技術名詞保留英文（BFS、queue、adjacency list…），
  行文用台灣習慣的說法，不要翻譯得太生硬。程式碼註解維持英文，說明文字用中文。

**Only the generated `.html` goes into the repo.** Write each draft as Markdown in the
scratch/outputs directory (e.g. `outputs/medium_src/dayNN_en.md`), convert it, and drop
only the HTML into the day folder - Eric does not want the `.md` drafts committed.

**Do not promise specific future days.** Saying "this comes back on Day 98" is a
commitment the roadmap may not keep. Name the *algorithms and applications* that will
use today's topic ("autograd's backward pass is a reverse topological traversal",
"Kruskal leans on Union-Find") and leave the day numbers out. Referring backwards to
days already published is fine, and so is a one-line teaser for tomorrow.

Structure that works for both:

1. One-sentence hook tying the topic to something practical.
2. Where this sits in the series - one line back to the previous day.
3. Intuition first, with the main diagram.
4. The algorithm step by step on a small worked example.
5. Code in chunks, each followed by prose.
6. Complexity.
7. The paired LeetCode problem, solved, with the interview-relevant caveat.
8. "Why this matters later", using the ROADMAP dependency list.
9. Next-day teaser + repo link.

### Medium-ready HTML - the paste step

Medium's editor ignores Markdown syntax on paste, so never hand Eric the `.md` file to
copy. After both drafts are final, run the converter:

```bash
python3 tools/md2medium.py <scratch>/dayNN_en.md -o "day NN - <topic>/medium_en.html"
# or simply run it next to a draft and move the .html into the day folder
```

Eric opens the HTML in a browser, Ctrl+A, Ctrl+C, and pastes into a new Medium story -
headings, bold, links, lists, quotes and code blocks all survive as rich text. Images
referenced relatively (`imgs/dayNN_1.png`) are **inlined as base64 `data:` URLs**, so
the page renders before anything is pushed and the browser puts real images on the
clipboard; `--raw-urls` switches to raw.githubusercontent links instead (smaller file,
but blank until the PNGs are on `master`). Markdown tables become an aligned monospace
block, because Medium has no table element. The page carries no banner or helper text -
what you see is exactly what gets pasted.

### Interactive demo page - demo.html, every day

Every day also ships `demo.html`: one self-contained page that animates the day's
algorithm and the paired LeetCode problem, so the idea can be *watched* rather than
read. `day 13 - topological sort/demo.html` is the reference implementation - copy its
structure rather than inventing a new one.

Requirements:

- **Single file, no build step, no CDN.** Inline CSS and JS; it must render offline
  straight off disk.
- **Same visual language as the figures**: navy `#002639` fills, teal `#0093a1`
  outlines, `#00aaaa` / cyan `#00ffff` highlights, amber `#ffc000` for the active
  element, red `#ff0000` for failure, Latin Modern Roman / serif, dark background.
- **中英切換按鈕** in the header, `data-i18n` keys plus a `tr()` helper, exactly like
  StarGZR and 黑洞觀測台. Step narration is `{zh, en}` objects, not a single string.
- **Tabs**, one per idea (algorithm variant 1, variant 2, the failure case, the
  LeetCode problem), and optional variant buttons inside a tab.
- **Frame-based animation**: the algorithm runs once up front and *records* frames
  (`{nodes, edges, panels, line, msg}`); the UI just replays them. Never hand-write the
  steps - they must come from actually executing the algorithm, so the page can never
  drift from `<topic>.py`.
- Controls: reset / prev / play-pause / next, a speed slider, a step counter, and
  arrow-key + spacebar shortcuts.
- Right-hand column shows the live data structures (in-degree, queue, colours, stack,
  output) as chips, the Python code with the current line highlighted, and a short
  "the idea" note.
- Show *why the naive version breaks* somewhere: the cycle, the wrong edge direction,
  the off-by-one - that failure case is usually the most valuable tab.

**Verify it headlessly.** The VM has no browser, but it has `node`. Concatenate the
`<script>` bodies into a `.js` file, stub a minimal DOM (`getElementById`,
`createElement(NS)`, `appendChild`, `setAttribute`, `classList`, `querySelectorAll`,
`setInterval`), then drive every tab x variant x frame x language through `render()` and
assert: no exception, no empty narration, every `line` index exists in the code array,
and no `NaN`/`undefined` in any SVG attribute. Also check the final orders the engine
produces match what `<topic>.py` prints.

## Diagram style

All figures come from `tools/diagram_style.py` - import it, do not reinvent the look:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from diagram_style import *

fig, ax = canvas(9, 6, xlim=(0, 9), ylim=(0, 6))
box(ax, 0.2, 4.5, 2.0, 0.9, label="key : 'a'\nvalue: 5")
node(ax, 6, 3, '1'); edge(ax, (6, 3), (7.5, 4), weight=7)
arrow(ax, (2.2, 5.0), (5.2, 5.0))
cross(ax, 3.5, 2.0, label='collision')
region(ax, 2.6, 0.4, 1.9, 5.4, label='Hash Function')
save(fig, 'day12_1.png')
```

House rules:

- **Transparent background** always - the images sit on both Medium (white) and GitHub
  dark mode. `save()` handles it.
- **Palette** (sampled from the day 08-11 PNGs): fill `#002639` navy; outlines, arrows
  and body text `#0093a1` teal; larger labels `#00aaaa`; glow `#00ffff` cyan;
  index/secondary cells `#292c30` grey; failure paths and the circled-X marker
  `#ff0000` red; secondary annotation amber `#ffc000` / yellow `#ffff00`; grouping
  regions pale cyan `#d6f2f4` with a dashed teal border.
- **Serif typography** - Latin Modern Roman, falling back to Liberation Serif. Never
  sans-serif.
- **Glow is for short labels only.** Free-floating labels of roughly 16 characters or
  fewer ('start', 'collision', 'L0', '3 islands') get the signature cyan glow. Anything
  longer - a sentence, an explanation, a visit-order line - is drawn as plain text,
  because the neon smear makes long strings hard to read. `text()` decides this
  automatically via `GLOW_MAX_CHARS`; do not force `glowing=True` on a sentence.
- Labels inside filled blocks are plain white with no glow.
- Thin outlines (1.6-1.8 lw), generous whitespace, no titles, no axes, no matplotlib
  legend box - use `legend()` from the module if a key is needed.
- To highlight a subset of cells, restyle their edges (thick amber) rather than drawing
  dashed rings around them - the rings are fiddly to align and read worse.
- One idea per figure. Two or three simple figures beat one dense one.
- ~800-900 px wide at dpi 140.

Save the generating script as `imgs/make_figs.py` so figures can be regenerated or
restyled in bulk later.

### Verifying a figure

The VM has no browser. Copy each figure into the outputs directory and read it back as
an image before delivering - overlapping labels are the most common failure, and they
are invisible unless you look.

## Editing discipline

Write and edit repo files through bash heredocs or python scripts rather than the file
editing tools; on some mounts of this repo the file tools have silently truncated large
writes. Verify with `wc -l` and `python3 -m py_compile`. Back up before patching an
existing file (`cp "$f" "$f.bak"`). Delete `__pycache__` after running the figure
script.

## Finishing

Hand back with computer:// links to the notebook, the runnable `.py`, README,
`demo.html`, both article `.html` files and the figures, and flag anything that
still needs Eric's input (published Medium URL, ROADMAP tweaks). Once Eric publishes and
sends the real Medium URL, replace the placeholder in that day's `README.md`.
