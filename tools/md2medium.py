#!/usr/bin/env python3
"""Turn a Medium draft written in Markdown into HTML you can copy-paste into Medium.

Medium's editor ignores Markdown syntax on paste, but it *does* understand rich text
from the clipboard.  So the workflow is:

    python3 tools/md2medium.py draft_en.md -o "day 12 - graph traversal/medium_en.html"

which writes the HTML where you asked (without ``-o``: next to the source).  Open that file in a browser,
Ctrl+A, Ctrl+C, then paste into a fresh Medium story.  Headings, bold, italics, links,
lists, block quotes, code blocks and images all survive.

Two things worth knowing:

* Relative image paths (``imgs/day12_1.png``) are inlined as ``data:`` URLs, so the
  page shows the figures even before the repo is pushed and the browser puts real
  images on the clipboard.  Pass ``--raw-urls`` to link to raw.githubusercontent
  instead (smaller file, but nothing renders until the images are on master).
* Medium has no table element.  Markdown tables are emitted as a monospace block with
  the columns padded, which pastes as a code block and stays readable.

No third-party dependencies: this repo's VM cannot install packages.
"""

from __future__ import annotations

import base64
import html
import os
import re
import sys
import urllib.parse

REPO = 'KingRei/100DaysPython'
BRANCH = 'master'
RAW = 'https://raw.githubusercontent.com/{repo}/{branch}/{path}'
RAW_URLS = False   # --raw-urls flips this
IMG_DIRS: list = []   # extra directories to resolve relative image paths against

CSS = """
body   { max-width: 46rem; margin: 3rem auto; padding: 0 1.5rem;
         font-family: Georgia, 'Times New Roman', serif; font-size: 20px;
         line-height: 1.7; color: #191919; background: #fff; }
h1     { font-size: 2.1em; line-height: 1.2; margin: 0 0 .6em; }
h2     { font-size: 1.5em; line-height: 1.3; margin: 1.8em 0 .4em; }
h3     { font-size: 1.2em; margin: 1.5em 0 .3em; }
p      { margin: 0 0 1.2em; }
a      { color: #191919; }
img    { max-width: 100%; display: block; margin: 1.6em auto; }
pre    { background: #f2f2f2; border-radius: 4px; padding: 1em 1.2em;
         overflow-x: auto; font-size: 15px; line-height: 1.5;
         font-family: 'SFMono-Regular', Menlo, Consolas, monospace; }
code   { background: #f2f2f2; padding: .1em .35em; border-radius: 3px;
         font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: .85em; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { border-left: 3px solid #191919; margin: 0 0 1.2em; padding-left: 1.2em;
             font-style: italic; }
hr     { border: none; border-top: 1px solid #ddd; margin: 2.5em 0; }
li     { margin: .3em 0; }
"""

# --------------------------------------------------------------------------- inline

MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.gif': 'image/gif', '.svg': 'image/svg+xml', '.webp': 'image/webp'}


def _img_url(src: str, src_dir: str) -> str:
    """Turn a repo-relative image path into something the page can always show.

    By default the image bytes are inlined as a ``data:`` URL: the file then
    renders whether or not the repo has been pushed, and the browser hands a
    real image to the clipboard, which is what Medium needs on paste.

    ``--raw-urls`` instead points at ``raw.githubusercontent.com`` - smaller
    files, but nothing displays until the images are on ``master``.
    """
    if src.startswith(('http://', 'https://', 'data:')):
        return src

    abs_path = os.path.normpath(os.path.join(src_dir, src))
    if not os.path.exists(abs_path):
        # the draft may live outside the day folder; try the output folder too
        for d in IMG_DIRS:
            cand = os.path.normpath(os.path.join(d, src))
            if os.path.exists(cand):
                abs_path = cand
                break

    if not RAW_URLS and os.path.exists(abs_path):
        mime = MIME.get(os.path.splitext(abs_path)[1].lower(), 'image/png')
        with open(abs_path, 'rb') as fh:
            return 'data:%s;base64,%s' % (
                mime, base64.b64encode(fh.read()).decode('ascii'))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = os.path.relpath(abs_path, repo_root).replace(os.sep, '/')
    return RAW.format(repo=REPO, branch=BRANCH, path=urllib.parse.quote(rel))


def inline(text: str, src_dir: str) -> str:
    """Markdown inline markup -> HTML.  Code spans are protected from the rest."""
    spans: list[str] = []

    def stash(m):
        spans.append('<code>%s</code>' % html.escape(m.group(1)))
        return '\x00%d\x00' % (len(spans) - 1)

    text = re.sub(r'`([^`]+)`', stash, text)
    text = html.escape(text)

    text = re.sub(r'!\[([^\]]*)\]\(([^)\s]+)\)',
                  lambda m: '<img alt="%s" src="%s">' % (m.group(1), _img_url(m.group(2), src_dir)),
                  text)
    text = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'<em>\1</em>', text)
    text = re.sub(r'(?<![\w_])_([^_\n]+)_(?![\w_])', r'<em>\1</em>', text)

    return re.sub(r'\x00(\d+)\x00', lambda m: spans[int(m.group(1))], text)


# ---------------------------------------------------------------------------- table

def table_to_pre(rows: list[str]) -> str:
    """Medium has no tables - render one as an aligned monospace block instead."""
    def cells(line):
        return [_plain(c.strip()) for c in line.strip().strip('|').split('|')]

    header, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    grid = [header] + body
    width = [max(_display_len(r[i]) for r in grid if i < len(r))
             for i in range(len(header))]

    def fmt(row):
        return '  '.join(c + ' ' * (width[i] - _display_len(c))
                         for i, c in enumerate(row)).rstrip()

    out = [fmt(header), '  '.join('-' * w for w in width)] + [fmt(r) for r in body]
    return '<pre><code>%s</code></pre>' % html.escape('\n'.join(out))


def _plain(s: str) -> str:
    """Strip inline Markdown markers - the table becomes preformatted text, where
    ``**bold**`` and backticks would just be noise."""
    s = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', r'\1', s)
    s = re.sub(r'[`*_]', '', s)
    return s


def _display_len(s: str) -> int:
    """CJK glyphs are double width in a monospace font."""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


# --------------------------------------------------------------------------- blocks

def convert(md: str, src_dir: str) -> str:
    lines = md.replace('\r\n', '\n').split('\n')
    out: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # fenced code block
        m = re.match(r'^\s*```+\s*(\w*)', line)
        if m:
            i += 1
            buf = []
            while i < n and not re.match(r'^\s*```+\s*$', lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append('<pre><code>%s</code></pre>' % html.escape('\n'.join(buf)))
            continue

        # heading
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            lvl = len(m.group(1))
            out.append('<h%d>%s</h%d>' % (lvl, inline(m.group(2), src_dir), lvl))
            i += 1
            continue

        # horizontal rule
        if re.match(r'^\s*([-*_])\s*(\1\s*){2,}$', line):
            out.append('<hr>')
            i += 1
            continue

        # table
        if line.lstrip().startswith('|') and i + 1 < n and re.match(
                r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
            rows = []
            while i < n and lines[i].lstrip().startswith('|'):
                rows.append(lines[i])
                i += 1
            out.append(table_to_pre(rows))
            continue

        # blockquote
        if line.lstrip().startswith('>'):
            buf = []
            while i < n and lines[i].lstrip().startswith('>'):
                buf.append(re.sub(r'^\s*>\s?', '', lines[i]))
                i += 1
            out.append('<blockquote><p>%s</p></blockquote>' % inline(' '.join(buf), src_dir))
            continue

        # list
        m = re.match(r'^\s*([-*+]|\d+\.)\s+', line)
        if m:
            ordered = bool(re.match(r'^\s*\d+\.', line))
            items: list[str] = []
            while i < n and lines[i].strip():
                mm = re.match(r'^\s*(?:[-*+]|\d+\.)\s+(.*)$', lines[i])
                if mm:
                    items.append(mm.group(1))
                elif items:                      # continuation line
                    items[-1] += ' ' + lines[i].strip()
                else:
                    break
                i += 1
            tag = 'ol' if ordered else 'ul'
            out.append('<%s>%s</%s>' % (
                tag, ''.join('<li>%s</li>' % inline(x, src_dir) for x in items), tag))
            continue

        # paragraph
        buf = []
        while i < n and lines[i].strip() and not re.match(
                r'^\s*(#{1,6}\s|```|>|\||([-*+]|\d+\.)\s)', lines[i]):
            buf.append(lines[i].strip())
            i += 1
        para = inline(' '.join(buf), src_dir)
        # a paragraph that is nothing but an image should not be wrapped in <p>
        out.append(para if re.fullmatch(r'<img [^>]*>', para) else '<p>%s</p>' % para)

    return '\n'.join(out)


def render(md_path: str) -> str:
    src_dir = os.path.dirname(os.path.abspath(md_path))
    md = open(md_path, encoding='utf-8').read()
    title = next((l[2:].strip() for l in md.split('\n') if l.startswith('# ')),
                 os.path.basename(md_path))
    lang = 'zh-Hant' if md_path.endswith('_zh.md') else 'en'
    return ('<!DOCTYPE html>\n<html lang="%s">\n<head>\n<meta charset="utf-8">\n'
            '<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n%s\n'
            '</body>\n</html>\n'
            % (lang, html.escape(title), CSS, convert(md, src_dir)))


def main(argv: list[str]) -> int:
    global RAW_URLS
    RAW_URLS = '--raw-urls' in argv

    args, out = [], None
    rest = [a for a in argv[1:] if a != '--raw-urls']
    while rest:
        a = rest.pop(0)
        if a in ('-o', '--out'):
            out = rest.pop(0) if rest else None
        else:
            args.append(a)

    if not args or (out and len(args) > 1):
        print(__doc__)
        return 1

    for path in args:
        out_path = out or os.path.splitext(path)[0] + '.html'
        IMG_DIRS[:] = [os.path.dirname(os.path.abspath(out_path))]
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write(render(path))
        print('wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
