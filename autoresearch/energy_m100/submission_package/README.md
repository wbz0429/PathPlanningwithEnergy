# Trustworthy LLM-Agent Autoresearch — Submission Package

Paper: *"When Does Guided Search Help? A Trustworthy LLM-Agent Autoresearch Protocol for Noisy Engineering Domains, with a Pre-Loop Headroom Diagnostic"*

## Contents
- `PAPER_DRAFT.md` (in `..`) — full manuscript
- `figures/` — all paper figures (18 PNG, publication-quality, `pubstyle.py`)
- `scripts/` — every experiment script (version-controlled, deterministic)
- `results/` — every result JSON (from the frozen evaluator `m100_eval.py`)

## Reproduce pipeline (in order)

| Step | Script | Output |
|---|---|---|
| 1. Knowledge base | `energy_model_zoo.py` | 14 literature models, forms + sources |
| 2. Zero-fit transfer | `as_published_benchmark.py` | zero-fit ARE 53–76% (all fail) |
| 3. Refit leaderboard | `benchmark_model_zoo.py` | refit ARE: Tseng 1.90% best; physics ~7% |
| 4. Fair complexity sweep | `complexity_scaling_fair.py` | advantage ≈ 0 at K=8…200 (energy) |
| 5. Headroom diagnostic | `headroom_diagnostic.py` | predicts tie at all 9 K levels |
| 6. Diagnostic mechanism probe | `headroom_mechanism_probe.py` | scope: saturation detector, not oracle |
| 7. Diagnostic deployment value | `headroom_value.py` | skip-in-saturated vs invest-in-escapable rule |
| 8. Energy significance | `significance_test.py` | energy tie z=−0.35 |
| 9. Planning significance | `planner_headroom.py` + `planner_significance.py` (in `../..` = `autoresearch/`) | config-tie + cross-space escape win z=−2.38 |
| 10. Safeguard ablations | `safeguard_ablation.py` | every mechanism load-bearing |

## Frozen evaluator
`m100_eval.py` — DJI M100 209 flights (Rodrigues 2021, CC-BY), flight-level split, per-flight energy ARE. **Never edited** — the loop only changes featurize functions.

## Figures
All `figures/*.png` regenerate via `../pubfigs.py` + the figure scripts. Fonts: STHeiti (full CJK coverage). PNG at 200–300 dpi; PDF vector versions in `../experiments/pub/`.

## Key numbers (audited against results/)
- Zero-fit transfer: energy ARE 53–76%
- Refit best (Tseng polynomial): 1.90%; physics models 6.7–7.0%
- Loop product: 1.93% (matches human expert 1.90%)
- Energy loop vs random: z=−0.35 (tie); fine-grid 9 levels all predict tie
- Planning loop vs random: z=−2.38, +8.1%, 0% of random runs beat it
- Cross-space decomposition: config-only loop 3264 ≈ random 3135 (tie); +evolved code → 2882 (−382 escape gain)
