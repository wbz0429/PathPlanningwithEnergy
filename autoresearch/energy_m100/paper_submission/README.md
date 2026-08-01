# NeurIPS 2026 Submission — LaTeX

## Files
- `neurips_2026.tex` — full paper source (compiles with `pdflatex`)
- `figures/` — 6 paper figures (PNG, publication quality)
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
tlmgr install algorithmicx algorithm2e threeparttable caption
```

## Figures
All figures regenerate from the experiment scripts (`../energy_model_zoo.py` etc.) via
`../pubfigs.py` and the figure scripts. The PNGs here are the current committed versions.

## Note
- Author: Binzhu Wang (Xi'an Jiaotong University). Update `\author` before submission.
- Abstract is 199 words (NeurIPS limit 200).
- All numbers trace to the frozen evaluator `m100_eval.py` and committed JSONs.
