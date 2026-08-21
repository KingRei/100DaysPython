# demokit — shared builder for the daily `demo.html` pages

Every `day NN - topic/demo.html` from Day 01 onwards is generated from three pieces:

* `shell_head.html` / `shell_tail.html` — the page shell: palette (teal `#12b3b8`,
  purple `#9d6bff`, orange `#ff9736`), header, tab bar, stage, narration bar,
  控制列 and the zh/en toggle. Placeholders: `{{DAY}} {{TITLE_ZH}} {{TITLE_EN}}
  {{SUB_ZH}} {{FOLDER}} {{MEDIUM}}`.
* `common.js` — the shared engine: style table `STY`, colour table `COL`,
  `Frames`, shape helpers `S.r/S.c/S.e/S.t`, `cellRow`, `heapTreeShapes`,
  the SVG renderer, panels, code pane, legend, i18n (`tr()`), playback.
* `days/dayNN.js` — the per-day engine. Header comments carry the metadata:

  ```js
  // DAY: 09
  // TITLE_ZH: ...
  // TITLE_EN: ...
  // SUB_ZH: ...
  // FOLDER: day%2009%20-%20hash%20table%20-%20chaining
  // MEDIUM: https://medium.com/...
  ```

  and the file ends with `const DAY_META = {title, sub, tabs:[...]}` where each tab is
  `{id, label, stage, view, variants?, idea, legend, code, build}` and `build(variantIndex)`
  returns a list of recorded frames `{shapes, panels, view, line, msg}`.

**Frame-recording rule**: the algorithm runs once and *records* frames; the UI only
replays them. A page therefore cannot drift away from the reference implementation.

## Build & verify

```bash
python3 build.py days/day09.js      # -> out/day09.html  and  out/day09.html.js
node domtest.mjs out/day09.html.js  # replays every tab x variant x frame x language
```

`domtest.mjs` stubs a minimal DOM (the VM has node but no browser and no jsdom) and
`drive.js` asserts that narration is non-empty, that the highlighted code line exists,
and that no SVG attribute contains `NaN` or `undefined`.

Copy `out/dayNN.html` to `day NN - topic/demo.html` when it passes.
