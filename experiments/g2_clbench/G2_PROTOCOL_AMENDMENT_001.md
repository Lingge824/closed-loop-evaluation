# G2 Protocol Amendment 001: Retry Groq Tool-Use Transport Failures

Status: recorded after the first attempted G2 launch and before any complete
clean trajectory, candidate selection, counterfactual trajectory, or G2 outcome.

Date: 2026-08-25.

## Trigger

The first formal launch stopped at request seed 7, clean run 0, question 1,
turn 7. Groq returned HTTP 400 with error code `tool_use_failed` and message
`Failed to call a function`. The harness did not save a clean trajectory
because private caches are written only after all 40 questions complete.

At the time of this amendment:

- no clean trajectory had completed;
- no ranked candidate had been selected;
- no counterfactual branch had run;
- no downstream causal-harm metric had been computed or inspected;
- the failed-attempt log was retained in the ignored private results directory.

This was therefore a provider-transport failure, not an experimental outcome.

## Frozen Operational Change

The structured-output adapter now classifies either of these Groq
`BadRequestError` signatures as retryable:

- `tool_use_failed`;
- `failed to call a function`.

Recovery uses the existing retry loop, maximum retry count, and exponential
backoff. Every retry receives the identical completion arguments: model,
messages, response schema, temperature, request seed, and timeout. The adapter
does not add a reminder, switch to prompt-based JSON, alter task context, or
change a successful response. If all prespecified retries fail, the run still
terminates and is reported as a provider failure.

A regression test records both attempts and requires their complete request
dictionaries to be equal.

## Stored-Patch Artifact Repair

Commit `1833a36` normalized whitespace inside stored overlay 0005 so that a
generic `git diff --check` would accept the patch as a newly added text file.
In a unified diff, however, a single-space line is the required marker for an
empty context line. Removing that marker made the stored copy syntactically
invalid even though the already-applied CL-Bench source and all 104 tests were
unchanged. This amendment restores the original valid overlay 0005 bytes and
hash. Future staged whitespace checks exclude `*.patch` artifacts while still
checking all executable source and documentation.

This repair changes no installed CL-Bench code and no experimental behavior.

## Unchanged Scientific Protocol

This amendment does not change:

- the model or provider;
- any request seed or task seed;
- temperature or timeout;
- task questions, order, database, or scoring;
- standardized feedback;
- structural-leverage scores or candidate ranking;
- candidate eligibility rules;
- exact-prefix replay or verdict-only corruption;
- primary or secondary outcomes;
- validity gates or the preregistered decision rule.

The amendment is versioned as overlay 0006 and will be committed and tagged
before the formal run is restarted from question 1.
