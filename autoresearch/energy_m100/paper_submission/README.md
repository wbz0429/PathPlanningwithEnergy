# Paper Submission — LaTeX

## Files
- `neurips_2026.tex` — full paper source (compiles with `pdflatex`)
- `figures/` — 8 paper figures (PNG, publication quality)
- `paper_neurips.md` — markdown version (YAML frontmatter + 8 figure captions)
- `paper_preview.pdf` — Chrome-rendered preview (visual check)
- `Makefile` — `make` to build, `make clean` to clean

## Build
```bash
make
# or:
pdflatex neurips_2026.tex   # run twice for cross-references
```
The paper compiles with plain `pdflatex` using the bundled `geometry` fallback. For the
official NeurIPS style, drop `neurips_2026.sty` into this directory and uncomment the
`\usepackage{neurips_2026}` line (and remove the fallback geometry/other packages that
the .sty already loads).

## Missing LaTeX packages (if TinyTeX/basictex)
If `algorithmic`/`algpseudocode` or `threeparttable` are missing:
```bash
tlmgr install algorithmicx algorithm2e threeparttable caption natbib
```

## Figures
All 8 figures regenerate from the experiment scripts via `../pubfigs.py` and the figure
scripts. The PNGs here are the current committed versions.

## Submission strategy (2026-08-01)

- NeurIPS 2026 main: abstract 2026-05-04 / full 2026-05-06 — **already passed**.
- ICML 2026 AI-Scientists workshop: 2026-05-07 — passed.
- **Realistic targets (upcoming deadlines)**: NeurIPS 2026 workshops (calls open late
  2026); ICML 2027 / AAAI 2027 / IJCAI 2027 main tracks. The paper targets the
  "AI4Science trustworthiness" / "AI scientists" track, where the CMU-pitfalls gap and
  our trustworthiness protocol + capability-boundary framing are strongest.

## Note
- Authors: Binzhu Wang, Le Zhang (Xi'an Jiaotong University). Update `\author` before
  submission.
- Abstract is 199 words (NeurIPS limit 200).
- All numbers trace to the frozen evaluator `m100_eval.py` and committed JSONs.
