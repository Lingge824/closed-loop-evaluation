# G1.1 Confirmatory Result: Resource-Sensitive Causal Propagation

Date: 2026-08-25

## Status

**Valid confirmatory negative result.**

The preregistered G1.1 decision criterion was not met.

## Provenance

- Preregistration commit: `17cf56e`
- Preregistration tag: `g1.1-preregistered`
- Resource-oracle implementation commit: `568d39b`
- Model: `qwen/qwen3.6-27b`
- Temperature: `0`
- Seeds: `7, 17, 29, 41, 53`
- Intervention: verdict-only flip at round 0 on `dedupe_1`

The preregistration was committed before implementation and before any
G1.1 model run.

## Research Question

Can one false evaluator verdict cause persistent downstream resource harm
when later task contracts explicitly state that elements are hashable and
that implementations must have expected near-linear complexity?

## Validity Audit

All five runs passed the required validity checks:

- exact clean/counterfactual intervention pairing: `5/5`;
- only the delivered proxy verdict changed at intervention: `5/5`;
- compact persistent-memory contract: `5/5`;
- clean correctness outcomes: `15/15`;
- clean resource-sensitive downstream outcomes: `10/10`;
- resource metrics present for both downstream probes in every branch.

The deterministic resource oracle correctly distinguishes known linear
and quadratic reference implementations in local regression tests.

## Aggregate Results

| Seed | Oracle harm | Resource harm | Operation excess | Memory-distance growth |
|---:|---:|---:|---:|---:|
| 7  | 0.000 | 0.000 | -0.500 | 0.291 |
| 17 | 0.000 | 0.000 | -0.500 | 0.290 |
| 29 | 0.000 | 0.000 | -0.500 | 0.291 |
| 41 | 0.000 | 0.000 | -0.500 | 0.291 |
| 53 | 0.000 | 0.000 | -0.500 | 0.291 |

Additional observations:

- positive future resource harm: `0/5` seeds;
- mean paired normalized operation excess: `-0.500`;
- downstream solution text changed in `2/2` tasks for every seed;
- only `2` unique complete trajectory fingerprints appeared across the
  five nominal seeds.

## Preregistered Decision

G1.1 required both:

1. positive future oracle harm in at least `3/5` paired seeds; and
2. higher mean normalized resource cost in the flip branch.

Observed:

1. positive future oracle harm occurred in `0/5` seeds; and
2. the mean operation excess was negative (`-0.500`).

Therefore:

**PREREGISTERED G1.1 DECISION: NOT MET.**

No threshold, task prompt, seed, or outcome definition will be changed to
reverse this result.

## Mechanistic Interpretation

The false verdict consistently changed persistent memory and future code
text. However, the explicit downstream contract stated that:

- all elements were hashable; and
- expected near-linear complexity was required.

Under this contract, both branches selected near-linear strategies.

For `first_unique_1`, representative clean and flip solutions used a
manual dictionary and `collections.Counter`, respectively. Both had
linear operation-count growth. The flip implementation used slightly
fewer counted operations.

For `stable_intersection_1`, both branches used the same set-based
algorithm. Their only observed difference was the ordering of two
independent statements.

Thus, textual solution divergence did not imply algorithmic or utility
divergence.

## Scientific Conclusion

The experiment supports:

> A false evaluator verdict can enter persistent state and change future
> generated code, while an explicit authoritative task contract can prevent
> that state divergence from producing correctness or resource harm.

The experiment does not support:

> A single false verdict causes reproducible downstream resource harm even
> when later task requirements explicitly rule out the erroneous lesson.

## Limitations

1. The task sequence contains only three small controlled coding tasks.
2. Temperature zero produced only two unique trajectory fingerprints, so
   the five seeds are reproducibility checks rather than five independent
   stochastic samples.
3. The resource oracle targets hash lookup versus repeated list scanning;
   it does not measure every form of computational cost.
4. The experiment tests one model and one persistent-memory architecture.
5. G1.1 is a boundary-condition experiment, not a direct matched-accuracy
   comparison between two evaluators.

## Consequence for the Main Research Program

G1.1 rules out an overly strong version of the derailment hypothesis and
identifies explicit task contracts as a candidate closed-loop stabilization
mechanism.

The next decisive experiment must move beyond the three-task pilot and
directly test the paper's central claim:

> evaluators with matched static accuracy but different error structure can
> produce different long-run outcomes on a preregistered naturalistic
> longitudinal task distribution.

G1 exploratory and G1.1 confirmatory results must remain preserved and
reported regardless of later outcomes.
