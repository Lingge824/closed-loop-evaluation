# G2b Provider and Model Feasibility Screen

Status: preregistered before any screen call or G2b task call.

Date: 2026-08-27.

## Purpose

The screen selects experimental infrastructure using only transport criteria.
It never loads the CL-Bench database, real questions, task feedback, candidate
positions, or G2 outcomes. It cannot observe whether a model would support the
research hypothesis.

The screen tests the exact operational properties that blocked G2:

- schema-valid structured output;
- nonempty final content;
- growing multi-turn context;
- fixed sampling controls;
- sequential throughput under a conservative Tier-1 token rate;
- API-reported usage and cached-token accounting.

## Frozen Candidate Order

Candidates are not compared on task accuracy. They are attempted in this fixed
order and the screen stops at the first passing snapshot:

1. primary: `openai/gpt-5.4-mini-2026-03-17`;
2. compatibility fallback: `openai/gpt-4.1-mini-2025-04-14`.

The fallback may run only if the primary reaches a model-level screen failure
(unsupported request, invalid/empty structured output, exact-response failure,
context rejection, or exhausted transient retries). Authentication, billing,
organization, or account-rate-limit failures stop the entire screen instead of
causing a model switch.

The choice is motivated before measurement: GPT-5.4 Mini is the current fixed
mini snapshot, supports structured outputs, has a 400k context window, and is
priced for high-volume workloads. GPT-4.1 Mini is the fixed compatibility
fallback with structured outputs and a larger context window. No G2 behavior is
part of this ordering.

Primary documentation consulted on 2026-08-27:

- https://developers.openai.com/api/docs/models/gpt-5.4-mini
- https://developers.openai.com/api/docs/models/gpt-4.1-mini
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/docs/guides/rate-limits

Recorded prices per one million tokens:

| Snapshot | Input | Cached input | Output |
| --- | ---: | ---: | ---: |
| GPT-5.4 Mini | $0.75 | $0.075 | $4.50 |
| GPT-4.1 Mini | $0.40 | $0.10 | $1.60 |

Pricing is recorded for audit and budgeting; the selection rule does not rank
passing models by observed cost because the first passing snapshot ends the
screen.

## Frozen Probe

Each candidate receives 24 calls across three synthetic, growing conversations:

| Session | Calls | Synthetic text added per call | Approximate final scale |
| --- | ---: | ---: | ---: |
| short | 6 | 2,500 characters | about 6k tokens |
| medium | 8 | 6,000 characters | about 20k tokens |
| long | 10 | 19,000 characters | about 80k tokens |

The text is generated locally from a fixed generic vocabulary. It contains no
benchmark question, SQL task, database fact, evaluator verdict, intervention,
or answer. Every call uses a two-field schema shaped like the task action:

```json
{"action": "ANSWER", "content": "<frozen call nonce>"}
```

The expected content is specified literally in the current synthetic prompt.
The response must match it exactly. Prior synthetic prompts and validated
assistant responses remain in the conversation, so the screen exercises
growing-prefix transport and prompt caching.

Frozen request settings:

- temperature: `0.0`;
- seed: `20260827`;
- timeout: 90 seconds;
- maximum completion tokens: 128;
- GPT-5.4 Mini reasoning effort: `none`;
- minimum start-to-start interval: 12 seconds;
- retry budget: two retries after the initial request;
- retry delays: 20 and 40 seconds;
- retryable events: connection/timeout errors, HTTP 408, 425, 429, and 5xx;
- schema errors, empty content, and wrong exact content are never resampled.
- hard cost guards per candidate: 1.2 million reported input tokens or $2.00
  estimated spend, whichever is reached first.

## Passing Rule

A candidate passes if and only if:

1. all 24 calls return a nonempty action;
2. all 24 actions pass the frozen Pydantic schema;
3. all 24 actions equal the frozen expected value;
4. no call exhausts its retry budget;
5. no nonretryable provider error occurs;
6. total transient retries do not exceed two;
7. API usage metadata supplies a nonnegative input-token count for every call.
8. neither frozen cost guard is crossed.

The runner records response IDs and fingerprints when available, but their
absence is not a failure because providers and SDK paths expose them
inconsistently.

## Separation From G2b

Passing the screen authorizes only preparation of a G2b preregistration. It does
not authorize a task call. The selected model snapshot, provider mode, context
limit, response format, retry policy, request controls, cost guard, and G2b
decision rule must be committed and tagged before the first real G2b question.

Screen outputs are stored separately from G2/G2b trajectories. No screen
response may be replayed into G2b.
