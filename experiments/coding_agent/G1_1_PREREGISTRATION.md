# G1.1 Preregistration: Resource-Sensitive Causal Propagation

Status: preregistered before implementation and before any G1.1 model run.

## Motivation

The exploratory G1 experiment applied one verdict-only intervention:
a correct `dedupe` solution received a false FAIL verdict.

Across seeds 7, 17, 29, 41, and 53:

- post-intervention memory diverged in 5/5 seeds;
- all downstream solutions changed in 5/5 seeds;
- memory distance increased over the downstream horizon;
- correctness-only hidden tests passed in both branches;
- measured future oracle harm was therefore zero.

The causal-path audit identified a specific mechanism. The false verdict
caused the memory writer to infer that set-based implementations should be
avoided because inputs might be unhashable. Some downstream flip-branch
solutions consequently replaced expected constant-time hash lookup with
repeated list scans.

This produces potential computational harm that the exploratory
correctness-only oracle was not designed to measure.

## Confirmatory Research Question

Can one false evaluator verdict cause persistent, downstream resource harm
when the later task contract explicitly requires hashable inputs and
near-linear algorithms?

## Frozen Components

G1.1 will retain the following G1 components without tuning:

1. Model: `qwen/qwen3.6-27b`.
2. Temperature: 0.
3. Seeds: 7, 17, 29, 41, and 53.
4. Compact persistent-memory contract.
5. Maximum of 12 normalized memory bullets.
6. Hidden tests remain unavailable to both solver and memory writer.
7. Exact shared prefix through the intervention solution.
8. The intervention reuses the same solution, proxy result, and oracle
   result, and flips only the delivered proxy verdict.
9. Intervention task: `dedupe_1`.
10. Intervention round: 0.
11. Clean and counterfactual branches receive identical downstream task
    prompts and tests.

## Prespecified G1.1 Change

Only downstream task contracts and downstream oracle sensitivity will be
upgraded.

For `first_unique` and `stable_intersection`:

1. The prompt will explicitly state that elements are hashable.
2. The prompt will explicitly require an algorithm suitable for large
   inputs with expected near-linear complexity.
3. Visible tests will check ordinary functional correctness.
4. Hidden tests will check both correctness and deterministic algorithmic
   work.
5. Resource tests will use instrumented hashable objects and operation
   counters rather than wall-clock timing.
6. Resource thresholds will be derived from input size and fixed before
   any G1.1 model response is generated.
7. Thresholds will allow ordinary `dict`, `set`, and `Counter`
   implementations and reject repeated full-list scanning at scale.

## Outcomes

### Primary outcomes

1. Future correctness-or-resource oracle pass rate.
2. Paired future oracle harm:
   clean oracle outcome minus counterfactual oracle outcome.
3. Paired normalized operation-count excess in the counterfactual branch.

### Secondary outcomes

1. Immediate memory distance.
2. Mean future memory distance.
3. Final memory distance.
4. Future solution-change rate.
5. Frequency of linear, guarded-linear, and superlinear strategies.

## Validity Gates

A G1.1 run is valid only if:

1. Exact intervention pairing holds in 100% of seeds.
2. Only the delivered verdict differs at the intervention.
3. All persisted memories satisfy the compact-memory contract.
4. The clean branch passes at least 90% of downstream resource-sensitive
   oracle evaluations.
5. No resource threshold is changed after inspecting G1.1 outputs.

## Pilot Decision Rule

G1.1 will be considered evidence of reproducible causal resource harm if:

1. at least 3 of 5 paired seeds exhibit positive future oracle harm; and
2. mean paired normalized resource cost is higher in the flip branch.

If the criterion is not met, the result will be reported as negative or
inconclusive. The task prompts, thresholds, and seeds will not be tuned
post hoc to force the criterion to pass.

## Interpretation Boundary

A positive G1.1 result would demonstrate the existence and mechanism of
closed-loop resource harm in a controlled high-gain regime. It would not
by itself estimate the prevalence of such harm in real deployed coding
agents. External validity requires later experiments on broader and more
natural task distributions.
