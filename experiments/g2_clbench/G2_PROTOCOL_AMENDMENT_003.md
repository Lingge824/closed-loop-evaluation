# G2 Protocol Amendment 003: Local Validation of Prompt-Shaped JSON

Status: recorded after the third incomplete G2 launch and before any complete
clean trajectory, candidate selection, counterfactual trajectory, or G2 outcome.

Date: 2026-08-27.

## Trigger

Amendment 002 was committed as `dc8e909`, tagged as `g2-json-mode-v1`, and
pushed before the formal run was restarted from question 1. The corrected
Groq JSON Object Mode completed the first two requests, then stopped at request
seed 7, clean run 0, question 1, turn 3. Groq returned HTTP 400 with error code
`json_validate_failed` and message `Failed to validate JSON`.

At the time of this amendment:

- question 1 had not completed;
- no clean trajectory had completed or been cached;
- no structural-leverage candidate had been selected;
- no counterfactual branch had run;
- no downstream causal-harm metric had been computed or inspected;
- all three failed-attempt logs remained in the ignored private results
  directory.

The run therefore produced no G2 observation or outcome. It exposed a second
provider-side constraint: JSON Object Mode still rejects malformed model text
before the benchmark's tolerant parser and strict local validator can inspect
it.

## Root Cause

Groq documents JSON Object Mode as syntactic JSON validation without schema
enforcement and states that it can error when the model does not produce valid
JSON. The frozen Qwen model produced such a server-rejected generation on the
third request. Because the server returned no usable completion, the existing
local recovery scanner and Pydantic validation never ran.

Primary provider documentation consulted:

- https://console.groq.com/docs/structured-outputs
- https://console.groq.com/docs/reasoning

## Frozen Operational Change

For the exact model identifier `groq/qwen/qwen3.6-27b`, the adapter now:

1. retains the same prompt-based Pydantic schema instruction introduced by
   amendment 002;
2. omits provider-side `response_format`, allowing ordinary text completion;
3. sets Groq's `reasoning_format="parsed"`, which keeps model reasoning enabled
   but separates it from final content;
4. extracts the final JSON object with the existing tolerant parser;
5. validates the extracted action locally against the unchanged Pydantic
   schema and rejects any invalid action;
6. retains the existing bounded retry behavior for local parse or validation
   failures.

The recovery scanner is extended so that, if an earlier reasoning object parses
as JSON but lacks required action fields, it continues scanning the original
text for a later schema-shaped action. A regression test covers an intermediate
reasoning object followed by the final valid action.

No free-form response is accepted as an action. Only an object passing the
original Pydantic schema reaches the task. Temperature, request seed, and
timeout remain unchanged and are asserted in tests.

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

Both clean and counterfactual trajectories use the same corrected transport
from question 1. None of the three incomplete launches contributes cached
state or observations to the restarted experiment.

The amendment is versioned as overlay 0008 and must be committed and tagged
before the formal run is restarted again from question 1.
