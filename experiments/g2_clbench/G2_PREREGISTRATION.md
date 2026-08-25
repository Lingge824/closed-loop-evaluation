# G2 Preregistration: Matched-Error Structural Leverage in CL-Bench

Status: preregistered on 2026-08-25 before implementation and before any
G2 model call.

## Purpose

G1 established that a verdict-only evaluator error can change persistent
state and all later solutions in a controlled coding-agent loop. G1.1 found
no downstream correctness or resource harm after later task contracts were
made explicit. G2 therefore tests the narrower claim that remains central to
the project:

> Holding evaluator accuracy, confusion-matrix error type, intervention time,
> and remaining horizon fixed, errors at positions with greater future reuse
> of latent task knowledge can cause greater downstream harm.

This experiment is a causal pilot and a go/no-go test for the structural
leverage hypothesis. It is not intended to estimate the prevalence of such
harm in deployed agents.

## Upstream Benchmark Pin

Repository: `pgasawa/continual-learning-bench`

Upstream commit:

`5f8c50eb1e84b2eda2ef4faff757dfc812a0ea26`

Task: `database_exploration`

Schedule: `default` (`schema_drift`, 20 pre-migration questions followed by
20 post-migration questions)

The following SHA-256 hashes freeze the local benchmark data:

| File | SHA-256 |
|---|---|
| `products.db` | `edf8ee80ff125de0bfd6c37a1d185efa9e3037ce28eb1bd1d32ae0829bd264a6` |
| `products_drifted.db` | `a53d523f70604be0e4328f3722417895250ac35e1127cc504b609576aee70fad` |
| `questions.json` | `f98e515c14b74f4d30c29401fb79cca9ceeb15fe9ee2432b10265a4a649d1771` |
| `questions_post_drift.json` | `c05aa6a33def3878e6b655a1cba6454e55e1a9c16ded8c1fe53b7811986a45de` |

Raw database files, questions, answers, SQL, and complete traces will not be
committed to this repository. Public artifacts will contain aggregate metrics,
configuration, hashes, and redacted or hashed identifiers only.

## Frozen Agent Configuration

Primary system: CL-Bench `icl`.

The primary system is intentionally the minimal linear-context baseline. Its
`observe` method persists task observations in the future context, but it has
no separate memory writer. This reduces ambiguity about whether an observed
effect was created by a custom memory mechanism.

Configuration:

- provider: Groq through LiteLLM;
- model identifier: `groq/qwen/qwen3.6-27b`;
- provider mode: `litellm_chat`;
- temperature: `0`;
- request seeds: `7`, `17`, `29`, `41`, and `53`;
- task seed: `42`;
- maximum exploratory queries per question: `15`;
- response timeout: `60` seconds;
- system prompt: empty, matching the official ICL default;
- context and reserve-token behavior: official ICL defaults;
- official task questions, order-construction function, scoring, and database
  behavior remain unchanged.

The pinned upstream ICL implementation does not expose sampling temperature or
request seed and calls `litellm.completion` without either argument. Before any
model call, a focused overlay patch will add these controls. Unit tests must
verify that every G2 request uses the frozen temperature and the prespecified
per-turn seed. No G2 result is valid without this gate.

`icl_notepad` is not part of the primary experiment. It may be evaluated later
as a higher-feedback-gain replication, and any such experiment will be labeled
exploratory unless separately preregistered.

## Standardized Feedback Channel

The official task uses different feedback payloads for correct and incorrect
answers and reveals the correct answer after an error. That format cannot
support a verdict-only intervention because flipping the verdict would also
change diagnostic content.

All G2 arms will therefore receive the same standardized envelope:

```text
Question <index>: EVALUATOR VERDICT: <PASS|FAIL>
Your submitted answer: <submitted answer>
Exploratory queries used: <count>
```

The payload never reveals the ground-truth answer or SQL. Clean, high-leverage,
and low-leverage arms all use this envelope. The task computes and stores the
true correctness, regret, reward, and `InstanceOutcome` before the delivered
feedback is transformed. Corruption may change only the `PASS` or `FAIL` token.

Timeout and query-budget failures are scored by the official task and receive a
truthful `FAIL`; they are ineligible as intervention instances.

## Structural Leverage Proxy

Each benchmark question has an expert-authored `knowledge_dependencies` list.
For question position `t`, let `K_t` be its set of dependencies and let `b(t)`
be the end of its current schema stage: position 20 before migration and
position 40 after migration.

The prespecified structural leverage proxy is

```text
theta_t = (1 / |K_t|) * sum over k in K_t and u in (t+1 ... b(t))
          of 1{k is in K_u}.
```

Thus `theta_t` is the average number of later same-stage questions that reuse a
dependency of the current question. It uses benchmark metadata and schedule
order only. It does not use model responses, evaluator outputs, downstream
performance, raw answer values, or intervention results.

This quantity is a pre-outcome structural proxy for causal leverage, not the
causal effect itself.

## Matched-Position Candidate Selection

The official schedule provides five deterministic permutations, indexed 0--4.
At each absolute position, high and low schedules have exactly the same
intervention time and remaining horizon. They differ in the structural reuse
of the question occupying that position.

The primary post-drift candidate order is frozen by decreasing leverage gap:

| Rank | Position | High run / score | Low run / score | Gap | Remaining |
|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 0 / 12.000 | 4 / 7.500 | 4.500 | 12 |
| 2 | 25 | 3 / 15.000 | 1 / 11.000 | 4.000 | 15 |
| 3 | 23 | 0 / 13.333 | 4 / 9.750 | 3.583 | 17 |
| 4 | 24 | 3 / 13.000 | 2 / 9.600 | 3.400 | 16 |
| 5 | 27 | 4 / 11.000 | 2 / 7.750 | 3.250 | 13 |
| 6 | 30 | 4 / 8.667 | 1 / 5.500 | 3.167 | 10 |
| 7 | 31 | 4 / 9.000 | 0 / 6.000 | 3.000 | 9 |
| 8 | 29 | 0 / 9.333 | 3 / 6.500 | 2.833 | 11 |
| 9 | 26 | 2 / 11.333 | 1 / 9.000 | 2.333 | 14 |
| 10 | 32 | 0 / 6.333 | 1 / 4.000 | 2.333 | 8 |

For each request seed, the harness will select the first ranked candidate for
which the two clean intervention responses:

1. both complete normally without timeout or budget exhaustion; and
2. have the same true verdict.

If both are true `PASS`, both counterfactual branches receive one false `FAIL`
and therefore have one false negative. If both are true `FAIL`, both receive one
false `PASS` and therefore have one false positive. If the two true verdicts
differ, the candidate is skipped without inspecting any downstream metric.

Selection must be performed automatically from intervention-instance validity
and true verdict only. Clean downstream metrics must remain hidden from the
selection procedure. If no candidate is eligible for a seed, that seed is
reported as ineligible rather than changing the ranking.

The strongest pre-drift candidate, position 5 with high run 3, low run 1, and
gap 2.500, is reserved for a later cross-migration robustness study. It is not
part of the primary G2 decision.

## Counterfactual Construction

For every eligible high or low arm:

1. Run and record the clean trajectory.
2. Reconstruct the exact clean prefix without additional model calls.
3. Reuse the exact query sequence, exploratory actions, submitted answer,
   ground-truth evaluation, and standardized truthful feedback through the
   intervention response.
4. At the intervention, change only `PASS` to `FAIL` or `FAIL` to `PASS` in
   the standardized envelope.
5. Allow the counterfactual trajectory to evolve naturally afterward under
   the same task, model, sampling controls, and per-turn request seeds.

All clean and counterfactual requests at a corresponding post-intervention turn
use the same deterministic seed schedule. Provider response identifiers,
system fingerprints when available, token usage, context truncation, retries,
and rate-limit events are logged.

Each counterfactual evaluator makes exactly one error across 40 questions:
accuracy is therefore 39/40 = 97.5%. Within each matched high/low pair, the
error count and confusion-matrix type are identical.

## Outcomes

For downstream question `u`, official regret is the number of exploratory
queries when the answer is correct and the full budget of 15 when incorrect.

For arm `a` in `{high, low}`, signed downstream causal harm is

```text
H_a = mean over u > t of
      (counterfactual_regret_u - clean_regret_u) / 15.
```

Positive values mean the evaluator error caused harm. The primary matched
difference-in-differences outcome is

```text
D = H_high - H_low.
```

This comparison subtracts each schedule's own clean outcome before comparing
the high- and low-leverage schedules.

Primary outcomes:

1. `H_high`;
2. `H_low`;
3. `D`;
4. the number of request seeds with `D > 0`.

Secondary outcomes:

1. downstream accuracy harm;
2. normalized exploratory-query excess;
3. first downstream action divergence;
4. downstream action-divergence rate;
5. context-token growth and truncation events;
6. effect persistence by downstream distance.

All signed effects will be reported even when they indicate benefit rather than
harm. Absolute divergence is not treated as evidence of harm.

## Validity Gates

A seed is valid only if:

1. upstream commit and all four dataset hashes match this document;
2. temperature and request-seed tests pass;
3. the selected high/low pair has the same position and horizon;
4. its true intervention verdicts match;
5. both branches reuse the exact clean prefix and intervention response;
6. only the delivered verdict token differs at intervention;
7. each counterfactual evaluator has exactly one error and the matched pair has
   the same FP/FN type;
8. official ground-truth scoring is unchanged by feedback corruption;
9. retries and provider failures are logged and do not silently alter prompts;
10. raw benchmark contents are not committed or printed in public summaries.

Invalid seeds are reported with reasons and are not silently replaced by new
seeds.

## Pilot Decision Rule

The structural-leverage pilot is considered positive enough to justify broader
replication if all validity gates hold and:

1. mean `D` across eligible seeds is positive;
2. at least 4 of 5 eligible request seeds have `D > 0`; and
3. mean `H_high` is positive.

If fewer than five seeds are eligible, the result is inconclusive rather than
positive. If the gates hold but the decision rule is not met, the primary ICL
result is reported as negative or inconclusive. Candidate order, feedback
format, leverage definition, metrics, and thresholds will not be tuned after
observing G2 outcomes.

## Interpretation Boundary

A positive result would show that a pre-outcome dependency-reuse score predicts
the relative causal harm of matched evaluator errors in a natural stateful
database task. It would not establish that dependency reuse is the unique or
optimal leverage measure, that the same effect occurs for every agent memory
architecture, or that a one-error 97.5%-accurate evaluator represents deployed
error prevalence.

A negative result would constrain the theory: in the official ICL setting,
dependency reuse may not be sufficient to convert a verdict error into
measurable downstream regret. Such a result will not be hidden or repaired by
post-hoc task or candidate tuning.
