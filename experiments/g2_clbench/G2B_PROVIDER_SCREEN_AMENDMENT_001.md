# G2b Provider-Screen Amendment 001: Restore the Funded Gemini Route

Status: preregistered correction before any successful provider-screen call or
G2b task call.

Date: 2026-08-27.

## Reason for the correction

The frozen `g2b-provider-screen-v1` artifact mistakenly named OpenAI candidates
even though the experimental infrastructure available to the investigator was
the already-funded Gemini Developer API. The first execution produced zero
successful calls, zero reported input or output tokens, zero estimated cost,
and no model response. Its OpenAI primary was rejected locally by LiteLLM and
its fallback failed authentication. It therefore supplied no provider/model
feasibility observation and did not authorize G2b.

The original commit and tag remain immutable as an audit record. This amendment
corrects only the provider/API route before any successful screen observation.
It does not reinterpret the failed run as evidence against either model.

## Corrected frozen route

The screen uses the investigator's Gemini Developer API account through the
native `generateContent` REST endpoint at API version `v1beta`. Authentication
is read only from `GEMINI_API_KEY`. It does not use OpenAI, an OpenAI key,
LiteLLM, Vertex AI, OpenRouter, or a compatibility proxy.

The corrected candidate order is:

1. primary: stable `gemini-2.5-flash`;
2. compatibility fallback: stable `gemini-2.5-flash-lite`.

The primary is Google's stable price-performance model for high-volume agentic
workloads. The fallback is the stable low-cost model from the same family. Both
support a 1,048,576-token input window and structured outputs. Their selection
uses no CL-Bench question, answer, feedback, intervention, or G2 outcome.

Recorded standard paid-tier prices per one million tokens are:

| Model | Input | Cached input | Output, including thinking |
| --- | ---: | ---: | ---: |
| Gemini 2.5 Flash | $0.30 | $0.03 | $2.50 |
| Gemini 2.5 Flash-Lite | $0.10 | $0.01 | $0.40 |

Official documentation consulted on 2026-08-27:

- https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/api/generate-content
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/api-key

## Preserved controls and passing rule

The synthetic text, 24-call short/medium/long schedule, exact nonces, growing
conversation histories, Pydantic action schema, timeout, pacing, retry budget,
cost guards, candidate ordering rule, and pass/fail criteria from version 1 are
unchanged.

Every native Gemini request freezes:

- temperature `0.0`;
- decoding seed `20260827`;
- maximum output tokens `128`;
- JSON response MIME type and the original `ScreenAction` JSON Schema;
- Gemini 2.5 thinking budget `0`.

The screen still stops at the first passing candidate. Account, billing, or
project failures stop the screen rather than trigger the fallback. Model-level
transport or schema failure may trigger only the already-preregistered
fallback. No failed or successful screen output may enter a G2b trajectory.

## Separation from G2b

A passing corrected screen authorizes only preparation and freezing of the G2b
task preregistration. It does not authorize a real task call. The G2b harness
must use the same selected Gemini model, native provider route, request controls,
schema path, pacing, and retry policy, and must be committed and tagged before
the first real CL-Bench question is loaded.
