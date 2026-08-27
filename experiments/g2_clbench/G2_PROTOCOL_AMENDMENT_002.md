# G2 Protocol Amendment 002: Use Groq JSON Object Mode for Qwen 3.6

Status: recorded after the second incomplete G2 launch and before any complete
clean trajectory, candidate selection, counterfactual trajectory, or G2 outcome.

Date: 2026-08-25.

## Trigger

After amendment 001 was committed, tagged as `g2-provider-recovery-v1`, and
pushed, the formal run was restarted from question 1. It again stopped at
request seed 7, clean run 0, question 1, turn 7. The initial request and all
five bounded identical retries returned Groq HTTP 400 with error code
`tool_use_failed` and message `Failed to call a function`.

At the time of this amendment:

- no clean trajectory had completed or been cached;
- no structural-leverage candidate had been selected;
- no counterfactual branch had run;
- no downstream causal-harm metric had been computed or inspected;
- both failed-attempt logs remained in the ignored private results directory.

The repeated fixed-seed failure showed that this was not a transient provider
error. It was an unsupported structured-output transport for the frozen model.

## Root Cause

The adapter passed a Pydantic model as `response_format`. LiteLLM classified
`groq/qwen/qwen3.6-27b` as supporting that parameter and translated the schema
into a forced function call. Groq's current documentation lists Qwen 3.6 27B
as supporting JSON Object Mode, but does not list it among models supporting
JSON-Schema Structured Outputs. The forced schema-shaped function call failed
deterministically in the observed context.

Primary provider documentation consulted on 2026-08-25:

- https://console.groq.com/docs/structured-outputs
- https://console.groq.com/docs/vision

## Frozen Operational Change

For the exact model identifier `groq/qwen/qwen3.6-27b`, the structured-output
adapter now:

1. injects the same Pydantic JSON schema into the final user message using the
   adapter's existing prompt-based schema instruction;
2. requests Groq's documented `response_format={"type": "json_object"}`;
3. parses the returned JSON and validates it locally against the same Pydantic
   model;
4. retains the existing bounded retry and validation behavior.

The adapter still forwards temperature, request seed, and timeout unchanged.
It does not change the task prompt before the formatting instruction, relax
the action schema, accept invalid actions, switch models, or perturb a failed
request. A regression test forces LiteLLM's capability probe to report schema
support and verifies that the provider-specific override still selects JSON
Object Mode, includes the schema instruction, validates locally, and preserves
all frozen sampling controls.

## Unchanged Scientific Protocol

This amendment does not change:

- the model or provider;
- any request seed or task seed;
- temperature, timeout, or retry count;
- task questions, order, database, or scoring;
- standardized feedback;
- structural-leverage scores or candidate ranking;
- candidate eligibility rules;
- exact-prefix replay or verdict-only corruption;
- primary or secondary outcomes;
- validity gates or the preregistered decision rule.

Both clean and counterfactual trajectories use the same corrected output
transport from question 1. Neither incomplete launch contributes observations
or cached state to the restarted experiment.

The amendment is versioned as overlay 0007 and must be committed and tagged
before the formal run is restarted again from question 1.
