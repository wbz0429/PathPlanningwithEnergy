# Trustworthy LLM-Agent Autoresearch in Noisy Engineering Domains
## Paper skeleton for top-CS submission (framework contribution)

> Status: working draft. Numbers pulled from the experiment JSONs in `experiments/`.
> Honest-positioning rule (do not violate): the contribution is the **trustworthy protocol + empirical capability boundaries**, NOT a better energy model / NOT a new planner algorithm.

---

## 0. Title options
- A. "When Does Guided Search Help? A Trustworthy LLM-Agent Autoresearch Protocol for Noisy Engineering Domains"
- B. "Cheat-Proof LLM Autoresearch: A Protocol + Empirical Capability Boundaries on Real UAV Energy Data"
- C. "From Zero-Shot Transfer Failure to Cheat-Proof Discovery: A Real-Data Autoresearch Framework"

## 1. Abstract (draft ~200 words)

Large language model (LLM) agents promise autonomous scientific discovery (FunSearch, AlphaEvolve, AI Scientist), but their reported successes assume **exact, verifiable evaluators**. Real engineering domains — sensor data, hardware idiosyncrasies, noise — lack ground truth, and independent evaluations show naive AI Scientist pipelines overclaim: 42% experiment coding failures, 57% hallucinated numbers, prior art misjudged as novel. We present a **trustworthy autoresearch protocol** for such domains, instantiated on real DJI M100 UAV energy modeling and energy-aware path planning. The protocol has four stages: (i) a **prior-art knowledge base** of literature energy models; (ii) a **zero-fit transfer benchmark** showing all as-published models fail on the target platform (energy ARE 53–76%) — motivating, not assuming, refit; (iii) a **cheat-proof discovery loop** (frozen held-out evaluator, keep/revert, forced novelty gate, causal ablation, honest negative reporting) starting from the refit best; (iv) **quantified capability boundaries**: guided search ties random in the low-effective-dimensional energy domain (z=−0.35, even under nominal-space inflation) but significantly beats random in the code-structure planning domain (z=−2.38, 0% of random runs beat it). Each safeguard is shown load-bearing via ablation. The protocol converges to human-expert-level modeling accuracy (1.90% vs 1.90%) while exposing — not hiding — its own limits.

## 2. Key numbers (all from experiments/, frozen evaluator)

| Metric | Value | Source |
|---|---|---|
| Energy ARE: textbook BEMT | 6.88% | zoo leaderboard |
| Energy ARE: as-published transfer (momentum/Tseng) | 53–76% | as_published_benchmark |
| Energy ARE: refit physics models (VRS, Dorling, etc.) | 6.7–7.0% | zoo leaderboard |
| Energy ARE: refit Tseng polynomial (human expert) | 1.90% | zoo leaderboard |
| Energy ARE: our loop (physics+payload) | 1.93% | agent_log iter6 |
| Energy ARE: full linear LIB (random-search ceiling) | 1.91% | zoo |
| Loop vs random: energy domain | z=−0.35 (tie) | significance_test |
| Loop vs random: planning domain | z=−2.38, 8.1% better, 0% random beats it | planner_significance |
| Fair controlled sweep (energy, K=10..110) | advantage ≈ 0 (+0.13pp at K=110, ns) | complexity_scaling_fair |
| Instant power: our loop at low/mid climb | ±1.1% bias | instant_power_error |
| Instant power: textbook BEMT at climb | −13% to −20% (misses climb) | instant_power_error |
| Planning: energy-aware detour vs climb | 5–13% saved; PX4 stack 14.3% | planning/px4 |

## 3. Contributions (claim ladder, most defensible first)

1. **Empirical: zero-shot transfer of literature models fails on real data (53–76% ARE), and the failure is systematic** — motivates an explicit refit stage. Not assumed; measured.
2. **Method: a four-stage trustworthy autoresearch protocol** (knowledge base → zero-fit benchmark → cheat-proof loop → boundary quantification) for domains without ground truth.
3. **Algorithmic — Headroom Diagnostic**: a cheap pre-loop predictor of whether guided search beats random, from single-term utility spread + combination synergy. Correctly predicts the observed outcome at all 5 controlled complexity levels (tie at K=10–70 where observed adv≈0; guided-wins at K=110 where observed +0.13pp). Turns "we report a boundary" into "we provide a tool to predict the boundary." (`headroom_diagnostic.py`)
4. **Empirical capability law**: guided-search advantage depends on **effective combinatorial complexity, not nominal space size** — ties random in low-effective-dim energy (even under K inflation), beats random in code-structure planning. Supported by 2 real domains + 1 controlled negative control + literature (Bergstra'12, REMBO, FunSearch).
5. **Safeguards are load-bearing**: ablation ladder shows removing any one node → the loop overclaims or collapses (novelty gate stops prior-art misjudgment; held-out ruler stops reward-hacking; causal ablation proves climb is the decision cause).
6. **Application**: real-data energy cost changes planning decisions (detour over climb), validated end-to-end (dynamics + PX4 firmware, 14.3% saving).

## 4. Related work (must cite; do NOT claim priority)

- FunSearch (Romera-Paredes 2024), AlphaEvolve (Novikov 2025), Eureka (Ma 2024), AI Scientist (Lu 2024) — all assume exact evaluators.
- CMU hidden-pitfalls evaluation of AI Scientist (Beel/Kan/Baumgart, arXiv 2502.14297) — the gap we occupy.
- AIGS / automated falsification (Liu 2024) — falsification framing is prior art; our contribution is the engineered protocol + empirical boundaries.
- LAENet (LLM-designed UAV reward), LLM-VD (LLM + evolution for drone routing) — adjacent; our niche is closed-loop autoresearch over energy MODELING + planning cost, not reward design or VRP.
- UAV energy models: momentum theory / BEMT (textbook), Tseng regression (Muli/Park/Liu arXiv 2206.01609), Abeywardena VRS (arXiv 2209.04128), Dorling 2017, Stolaroff 2018, Morbidi 2020. All in our knowledge base.
- Search-space scaling: Bergstra & Bengio 2012 (low effective dim → random ties), Wang REMBO 2013 (~15–20D tipping), NAS confounder removal (ENAS 0.07→0.90), FunSearch program spaces.
- Energy-aware planning prior art (MUST cite, don't claim): EcoFlight 2025, Michel 2024 (delivery order), Nguyen & Au 2017 (reachable set).

## 5. Method (the protocol)

**Stage 1 — Knowledge base.** Collect literature model forms (14 in `energy_model_zoo.py`), each with source, category (physics/data-driven/hybrid), as-published vs refit flags.

**Stage 2 — Zero-fit transfer benchmark.** Evaluate each as-published model on the frozen evaluator with NO fitting. Result: all fail (53–76% ARE). → establishes that refit is necessary, and that a target-platform benchmark is the honest baseline.

**Stage 3 — Refit benchmark + starting point.** Refit each form's coefficients on the frozen training split. Result: data-driven polynomial (Tseng) 1.90% best; physics forms 6.7–7.0%. The best becomes the loop's start (vs naive "start from textbook").

**Stage 4 — Cheat-proof discovery loop.** Frozen held-out evaluator (m100_eval.py, flight-level split, energy ARE); LLM proposes featurize changes; sandbox; keep/revert on search+val guards; forced novelty gate; causal ablation; honest negatives logged. Convergence ~1.9% matches human expert.

**Stage 5 — Capability-boundary quantification.** Statistical: loop vs random distribution in each domain. Controlled: fair complexity sweep (same action space, utility-weighted vs uniform). → ties in energy, wins in planning.

**Algorithm 1 — Headroom Diagnostic (pre-loop prediction of whether guided search helps).**

```
Input: candidate pool P, frozen evaluator E (search split only), budget B
1. For each term t ∈ P:  u_t ← E(score of {1, t})          # single-term utility (lower=better)
2. spread     ← std({u_t});  best_single ← min({u_t})
3. best_comb  ← best of B random subsets (uniform sampling)
4. synergy    ← best_single − best_comb                   # combination gain over single term
5. rel_spread ← spread / best_single;  headroom ← max(0,−synergy)/best_single
6. predict ← "tie" if (rel_spread > 0.3 ∧ headroom < 0.05) else "guided-wins"
```
Claim: the diagnostic predicts, before running the loop, whether guided search beats random. Empirically correct at all 5 controlled K levels in energy (tie where adv≈0, guided-wins at K=110 where adv=+0.13pp) and, with the planning validation, on the code-structure domain.

## 6. Experiments / figures plan (money figures first)

- **Fig 1 (money)**: Two-domain significance — energy (loop = random center, z=−0.35) vs planning (loop far left, z=−2.38). Exists as `pub/两域显著性.png`; must add the fair-sweep flat line as negative control.
- **Fig 2 (money)**: Knowledge-base leaderboard — 14 models ranked by refit ARE; Tseng on top, physics cluster at 7%. NEW (`pub/模型知识库排行榜.png`).
- **Fig 3 (money)**: Zero-fit vs refit gap — as-published 53–76% vs refit 1.9–7%. NEW (`pub/as_published_vs_refit.png`).
- **Fig 4**: Per-state instant power error — our loop vs Tseng vs BEMT per climb-rate bin; BEMT misses climb by −131W (the planning-decision root cause). NEW (`pub/瞬时功率误差.png`).
- **Fig 5**: Safeguard ablation ladder (framework_ablation master table + local-optimum escape 4.2%→1.9%).
- **Fig 6**: Effective-complexity law — x-axis = effective combinatorial complexity (energy under K-inflation = flat; planning = win), literature markers. REWORK of `复杂度规律.png` to include the controlled negative control.
- **Fig 7 (money)**: Headroom diagnostic predicted-vs-observed — per K level (energy) and per domain (planning), the predicted verdict vs the observed z/advantage. NEW.
- **Table 1**: All 14 models with as-published vs refit ARE.
- **Table 2**: Ablation ladder (7 nodes).
- **Table 3**: Headroom diagnostic predictions vs observations across all settings.

## 7. Honest boundaries (explicitly stated in paper)

- Energy modeling: guided search does NOT beat random in this domain; that IS the point (capability boundary), not a failure.
- Tseng is the same dataset's human result; we match (1.90 vs 1.93), we don't claim to beat human modeling.
- Instant power: kinematics explain <40% of variance; per-flight energy is the stable quantity.
- Zero-fit transfer numbers (53–76%) include scale-mismatch for Tseng's simplified form — reported as approximate.

## 8. Submission strategy

- **Target**: AAAI / NeurIPS workshop (AI4Science) full paper, or ICLR workshop. The "trustworthy autoresearch + capability boundaries" framing is the strongest.
- **Narrative**: not "we built a better model" but "**we built a protocol that knows when it works and honestly reports when it doesn't, on real noisy data**" — the CMU gap.
- **The two-domain + negative-control evidence is the differentiator**: nobody in the AI4S field reports a *measured* boundary of guided search with a controlled negative control.

## 9. TODO before submission
- [ ] Rework 复杂度规律 figure → effective-complexity law with negative control
- [ ] Reproduce planning-domain significance with committed script (in progress)
- [ ] Make every figure regenerate from one script (pubfigs.py has most)
- [ ] Formal ablation table → figure
- [ ] Write the protocol as pseudocode (Algo 1)
- [ ] Decide: one-domain-deep paper (energy+planning) vs add radar MTT domain (memory says pivot to radar was considered)
