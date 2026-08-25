# G2 CL-Bench harness

This directory implements the preregistered matched-error structural-leverage
pilot in `G2_PREREGISTRATION.md`.

## What is frozen

- CL-Bench commit `5f8c50eb1e84b2eda2ef4faff757dfc812a0ea26`;
- Database Exploration `default` schema-drift schedule;
- `groq/qwen/qwen3.6-27b`, temperature `0`, request seeds
  `7, 17, 29, 41, 53`;
- task seed `42`, query budget `15`, request timeout `60` seconds;
- standardized verdict-only feedback;
- ranked high/low candidate list and the preregistered decision rule.

The harness lazily runs only the clean permutations needed to identify the
first eligible candidate. Candidate selection reads intervention completion
status and true verdict only. It cannot inspect clean downstream outcomes.

## Causal construction

For each selected high/low arm, the clean action and exact serialized assistant
record are cached privately. The counterfactual reconstructs the ICL state
through the intervention response without a model call, flips only the
`PASS`/`FAIL` token in task feedback, then resumes live generation. The audit
rejects any prefix, task-state, scoring, metadata, error-count, or
confusion-type mismatch.

Raw actions and observations remain under ignored `results/private/`. Public
JSON summaries contain only frozen configuration, aggregate signed outcomes,
digests, and validity results. Never add the private cache with `git add -f`;
the public summaries are deliberately eligible for later review and commit.

## Verify

From this directory:

```bash
./verify_g2.sh
```

`CLBENCH_ROOT` may be set when the pinned checkout is not at
`~/Downloads/continual-learning-bench`.

## Run

With `GROQ_API_KEY` already exported:

```bash
./run_g2.sh all
```

An interrupted run reuses only complete, configuration-matched clean caches.
Use `./run_g2.sh all --fresh` only when intentionally archiving old caches and
restarting every preregistered seed.
