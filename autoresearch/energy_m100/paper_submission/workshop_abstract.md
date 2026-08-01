# When Does Guided Search Help?
## A Trustworthy LLM-Agent Autoresearch Protocol for Noisy Engineering Domains, with a Pre-Loop Headroom Diagnostic

**Submitted to: AI4Science / AI-Scientists workshop track**

## Abstract (workshop version, ~180 words)

LLM agents for autonomous discovery (FunSearch, AlphaEvolve, AI Scientist) assume exact, verifiable evaluators — a condition that fails in most engineering domains, where data is noisy and no ground truth exists. Independent evaluation documents the cost: 42% of AI Scientist experiments failed on coding errors, 57% of manuscripts contained hallucinated numbers, and published prior art was reported as novel. We present a trustworthy autoresearch protocol for such domains, instantiated on real DJI M100 UAV energy modeling and energy-aware path planning, plus a pre-loop headroom diagnostic that predicts whether guided search will beat random. The protocol: a knowledge base of 14 published energy models; a zero-fit transfer benchmark showing all as-published models fail on the target platform (energy ARE 53–76%); a cheat-proof loop (frozen held-out evaluator, keep/revert, forced novelty gate, causal ablation, honest negatives); and quantified capability boundaries. The diagnostic correctly predicts the energy domain is saturated (tie at 9 controlled complexity levels) and attributes the planning win (z=−2.38) to cross-space escape into program space. Every safeguard is load-bearing by ablation.

## Keywords
LLM agents · automated science · trustworthy AI · capability boundaries · UAV energy · program-space search

## Why this fits the workshop

The workshop's central question — "AI scientists: tools, co-authors, or founders?" — is
exactly what our paper addresses with measurement, not assertion:
- **When is an AI scientist a useful tool?** Our headroom diagnostic answers this *before*
  running the loop: skip in saturated spaces (energy domain, tie), invest where a
  higher-complexity program space is escapable (planning domain, +8.1%).
- **When is it untrustworthy?** The CMU pitfalls (overclaim, hallucinated numbers, prior
  art misjudged as novel) are exactly what our protocol's safeguards block — and each
  safeguard is proven load-bearing by ablation.
- **What is its honest capability boundary?** We report the settings where the loop does
  NOT beat random — the first measured capability boundary in this line of work.

## Contact
Binzhu Wang, Le Zhang — Xi'an Jiaotong University
