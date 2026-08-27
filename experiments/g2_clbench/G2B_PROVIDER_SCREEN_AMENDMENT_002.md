# G2b Provider-Screen Amendment 002: Advance to Current Gemini 3 Models

Status: preregistered correction after a zero-token availability rejection and
before any successful provider-screen call or G2b task call.

Date: 2026-08-27.

## Operational evidence requiring the correction

Amendment 001 correctly restored the funded Gemini Developer API route, but
its first screen execution completed zero calls for both candidates. A
subsequent metadata-only check established that the API key and project could
resolve both model records. One exact, synthetic reproduction of the first
frozen request then returned HTTP 404 before generation:

> This model models/gemini-2.5-flash is no longer available to new users.

The provider directed new users to `gemini-3.6-flash`. The rejected screen and
diagnostic loaded no CL-Bench question, database, answer, feedback,
intervention, or G2 outcome. The screen reported zero completed calls, zero
tokens, and zero estimated cost for both Amendment 001 candidates. These events
therefore provide only endpoint-availability evidence and cannot select a model
using research behavior.

The earlier commits, tags, failure summaries, and private diagnostic log remain
immutable audit records.

## Frozen current-model route

The candidate order is advanced once, before a successful screen observation:

1. primary: stable `gemini-3.6-flash`;
2. compatibility fallback: stable `gemini-3.5-flash-lite`.

The provider explicitly named the primary replacement. The fallback is the
current stable, cost-efficient Flash-Lite model on the same native API. Both
models expose 1,048,576-token input windows and support structured outputs and
minimal thinking. No task-performance evidence enters the ordering.

Standard paid-tier prices recorded on 2026-08-27, per one million tokens, are:

| Model | Input | Cached input | Output, including thinking |
| --- | ---: | ---: | ---: |
| Gemini 3.6 Flash | $0.75 | $0.075 | $3.75 |
| Gemini 3.5 Flash-Lite | $0.30 | $0.03 | $2.50 |

The Gemini 3.6 Flash prices above are the published rates through 2026-12-31.
Pricing is used only by the existing cost guard and never to compare outcomes.

Official documentation consulted on 2026-08-27:

- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash
- https://ai.google.dev/gemini-api/docs/deprecations
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/thinking
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/api/generate-content

## Gemini 3 request controls

Gemini 3 uses thinking levels rather than the Gemini 2.5 integer thinking
budget. Each request therefore freezes `thinkingLevel=MINIMAL`. The old
`thinkingBudget=0` field is removed. Gemini's current model guide recommends
the default temperature `1.0` for Gemini 3 and warns that lower values can
produce looping or degraded behavior, so the transport screen freezes
temperature `1.0`. The decoding seed remains `20260827`.

The native `generateContent` endpoint remains the screened route. Stateful
Interactions would hide prefix management and complicate the exact branching
required by the later clean/counterfactual design. For client-managed Gemini 3
history, the runner now replays the complete provider-returned model content,
including every opaque thought signature, without alteration. Parsed JSON is
still validated locally with the original `ScreenAction` Pydantic model.

The complete frozen settings are:

- temperature `1.0`;
- seed `20260827`;
- maximum output tokens `128`;
- JSON response MIME type and the unchanged `ScreenAction` JSON Schema;
- thinking level `MINIMAL`;
- exact provider-returned model content and thought-signature replay.

Provider HTTP status and message are retained only in the private screen
record. Public summaries continue to expose the failure category without
publishing account-specific provider text.

## Unchanged probe and decision rule

The locally generated synthetic text, 24-call short/medium/long schedule,
literal nonces, growing conversation histories, schema, timeout, 12-second
pacing, two-retry budget, retry delays, cost guards, first-pass selection rule,
and pass/fail criteria remain unchanged.

A model-level failure may trigger only the frozen fallback. Authentication,
billing, project, or account-rate-limit failure stops the screen. Invalid,
empty, or incorrect model content is never resampled.

## Separation from G2b

Passing this screen authorizes only preparation and freezing of a separate G2b
task preregistration. It does not authorize a real G2b call. The selected model,
provider route, Gemini 3 controls, signature handling, task protocol, metrics,
and decision rule must be committed and tagged before any real CL-Bench
material is loaded.
