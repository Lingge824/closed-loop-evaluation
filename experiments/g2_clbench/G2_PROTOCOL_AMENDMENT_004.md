# G2 Protocol Amendment 004: Raw Reasoning Transport

Status: recorded after the fourth incomplete G2 launch and before any complete
question, clean trajectory, candidate selection, counterfactual trajectory, or
G2 outcome.

Date: 2026-08-27.

## Trigger

Amendment 003 was committed as `1d0889f`, tagged as `g2-local-json-v1`, and
pushed before the formal run was restarted from question 1. The local JSON
validator accepted four consecutive schema-valid actions for request seed 7,
clean run 0, question 1. The fifth action request then exhausted the existing
bounded retry loop because Groq repeatedly returned an empty final-content
field. Question 1 did not complete.

At the time of this amendment:

- no question had completed;
- no clean trajectory had completed or been cached;
- no structural-leverage candidate had been selected;
- no counterfactual branch had run;
- no downstream causal-harm metric had been computed or inspected;
- all four incomplete-attempt logs remained in the ignored private results
  directory.

The launch therefore produced no G2 observation or outcome. It showed that
provider-side schema enforcement was no longer the blocker, but that parsed
reasoning transport could still leave `message.content` empty after all
pre-existing retries.

## Root Cause and Existing Retry Boundary

The structured-output adapter already classifies empty content as transient and
uses a frozen maximum of six total attempts: the initial request plus five
retries. The terminal `_EmptyContentError` can only escape after that budget is
exhausted. Increasing the retry count would therefore add repeated sampling
without addressing the response representation that caused local validation to
receive no text.

Groq documents `reasoning_format` as controlling how reasoning is presented:

- `parsed` moves reasoning into a dedicated `message.reasoning` field;
- `raw` returns reasoning in `<think>` tags inside `message.content`;
- `hidden` returns only the final answer.

Primary provider documentation consulted:

- https://console.groq.com/docs/reasoning
- https://console.groq.com/docs/api-reference

## Frozen Operational Change

For the exact model identifier `groq/qwen/qwen3.6-27b`, the adapter changes only
`reasoning_format` from `parsed` to `raw`.

The existing local path then:

1. receives the complete provider text in `message.content`;
2. ignores `<think>` prose unless it contains a candidate object;
3. searches for a later object with the required action fields;
4. accepts an action only if the unchanged Pydantic schema validates it;
5. retries empty, malformed, or schema-invalid responses under the unchanged
   six-attempt bound and with identical request parameters.

No reasoning prose is directly accepted as an action. A reasoning-only response
without a schema-valid action remains invalid and eventually terminates the run
after the frozen retry budget.

## Estimand and Reporting Boundary

The operational estimand is the agent action conditional on obtaining a
schema-valid, nonempty action within the frozen retry budget. This was already
the behavior of the structured-output adapter before Amendment 004; the
amendment changes only whether the local validator can see the complete text
returned by Groq.

Provider failures remain separate from task errors. They are preserved in the
private terminal log and are not scored as wrong database answers. Clean and
counterfactual branches use the same request format, retry policy, validation
rule, and timeout.

## Unchanged Scientific Protocol

This amendment does not change:

- the model or provider;
- task or request seeds;
- temperature, timeout, or retry count;
- task questions, order, database, or scoring;
- standardized feedback;
- structural-leverage scores or candidate ranking;
- candidate eligibility rules;
- exact-prefix replay or verdict-only corruption;
- primary or secondary outcomes;
- validity gates or the preregistered decision rule.

None of the four incomplete launches contributes cached state or observations
to the restarted experiment. The amendment is versioned as overlay 0009 and
must be committed and tagged before the formal run restarts from question 1.

## Hard Stop

If the same frozen model-provider path cannot complete a formal trajectory
after this amendment, G2 will not receive another transport amendment. The
provider-model pair will be reported as operationally infeasible for this
protocol, and any replacement model will require a separately preregistered G2b.
