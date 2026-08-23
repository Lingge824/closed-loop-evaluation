# Project Mainline

## Research question

When evaluator verdicts enter an adaptive self-improvement loop, which errors have the largest downstream causal effect, and can those effects improve allocation of a fixed trusted-audit budget beyond static error probability alone?

## Ordered gates

1. **G1 — Paired causal identification**
   - Reproduce the `dedupe_1` smoke failure with diagnostics.
   - Run a no-corruption persistent-memory baseline.
   - Branch from the exact clean prefix.
   - Reuse the exact intervention-round solution and test outcomes.
   - Change only the delivered proxy verdict.
   - Require at least one downstream task.
2. **G2 — Heterogeneity and power**
   - Repeat across predeclared seeds and intervention positions.
   - Estimate signed harm and magnitude separately.
   - Quantify uncertainty and detectable effect size.
3. **G3 — OOD prediction**
   - Test whether pre-intervention risk features predict held-out causal effects.
4. **G4 — Equal-budget auditing**
   - Compare random, error-probability, and causal-value audit allocation under the same trusted-audit budget.
5. **Paper gate**
   - Robustness, negative regimes, related work, writing, and submission.

## Anti-drift rule

Do not add new benchmarks, agent architectures, corruption families, or theory branches unless the current gate has passed or the addition is required to diagnose its failure.

## Current status

- Toy V0–V3 code and figures: preserved.
- Original coding-agent pilots: preserved in `archive/`.
- Expanded coding-agent V1 harness: repaired; local regression tests pass `8/8`.
- First live seed-17 smoke: strict pairing and downstream solution divergence
  confirmed, but invalidated for memory-distance estimation because provider
  reasoning text entered persistent memory.
- Memory contract and Groq credential selection: repaired; live smoke rerun
  required before the no-corruption baseline.
