# G2b Preregistration: Matched Evaluator Error on Gemini 3.6 Flash

Status: frozen after a synthetic provider screen passed and before any G2b task
call.

Date: 2026-08-27.

## Why G2b exists

The original G2 protocol could not complete one CL-Bench question with the
frozen Groq/Qwen route. Six preserved logs record tool-call failures, JSON
validation failures, empty content, and rate limiting; no clean trajectory,
counterfactual trajectory, public seed summary, or research outcome exists.
That evidence is published in the operational infeasibility audit and is not a
negative result for the hypothesis.

G2b changes only the experimental instrument needed to obtain valid model
actions. The task, task data, task seed, five request seeds, ranked intervention
candidates, clean/counterfactual construction, evaluator manipulation, outcome
metrics, validity audits, and aggregate decision rule remain those of G2.

## Infrastructure selection completed without task evidence

The frozen synthetic screen in
`results/provider_screen/public/screen_summary.json` selected
`gemini-3.6-flash` before this preregistration. It loaded no G2 question,
database, answer, feedback, intervention, candidate position, or prior outcome.

Selection evidence:

- screen version: `g2b-provider-screen-gemini-amendment-002`;
- selected candidate: primary `gemini-3.6-flash`;
- calls: 24 of 24 completed across short, medium, and long growing sessions;
- every call passed the frozen schema and exact nonce on its first request;
- transient retries: 0;
- reported input tokens: 859,479, of which 473,304 were cached;
- reported output tokens, including thinking: 537;
- estimated screen cost: $0.327143;
- fallback `gemini-3.5-flash-lite` was not called;
- public summary SHA-256:
  `8859cc566193b6145bd3dcccf04736920b32208f0722c3cbcb4b41c7f7daa130`.

No additional model comparison or task-based model selection is permitted.

## Frozen task and sampling configuration

- upstream CL-Bench commit:
  `5f8c50eb1e84b2eda2ef4faff757dfc812a0ea26`;
- Database Exploration schedule: `default`;
- task seed: `42`;
- task run indices: `0` through `4` as referenced by the ranked candidates;
- questions per trajectory: `40`;
- exploratory-query budget per question: `15`;
- task response timeout: `60` seconds, unchanged from G2;
- feedback: standardized `verdict_only`;
- request seeds: `7, 17, 29, 41, 53`;
- model: stable `gemini-3.6-flash`;
- native route: Gemini Developer API `v1beta generateContent`;
- temperature: `1.0`;
- thinking level: `MINIMAL`;
- provider request timeout: `90` seconds;
- maximum model output: `4,096` tokens;
- client context ceiling: `1,048,576` tokens;
- reserved context allowance: `8,192` tokens;
- response format: `application/json` with the task Pydantic JSON Schema;
- minimum process-wide request start interval: `12` seconds.

The provider timeout, temperature, thinking level, native endpoint, schema path,
and pacing match the route that passed the screen. Keeping the task timeout at
60 seconds preserves the original task rule independently of provider transport.

## Frozen transport-failure rule

Only connection/timeout failures and HTTP 408, 425, 429, or 5xx are transient.
Each call has one initial request and at most two identical retries, delayed 20
and 40 seconds. Prompt, history, seed, temperature, schema, and every generation
setting remain identical on retry.

Empty final content, malformed JSON, schema-invalid output, a wrong action,
content rejection, authentication failure, billing failure, and every other 4xx
are not resampled. Such an event makes the run operationally incomplete; it is
not scored as a wrong benchmark answer and is not a negative hypothesis result.

The total preregistered G2b spend guard is $100, calculated from provider usage
with the frozen rates of $0.75/M uncached input tokens, $0.075/M cached input
tokens, and $3.75/M output tokens including thinking. Every billable response is
written to a private response-ID ledger before the guard is checked. Crossing
the guard stops further calls and yields an operationally incomplete study.

## Exact Gemini prefix replay

Gemini 3 can return opaque thought signatures in model content. Every successful
clean turn stores two different records privately:

1. the locally validated action JSON used by CL-Bench; and
2. the exact provider-returned model content, including all thought signatures.

The counterfactual reconstructs the clean prefix without a provider call. It
replays the exact action and exact provider model content for every turn through
the terminal response at the intervention question. The harness rejects the
counterfactual if any query digest, action, provider record, provider-visible
state, task observation, task score, or prefix usage count differs. Provider
calls in the replayed prefix must equal zero.

After the clean terminal action at the intervention question has been replayed,
the task changes only the literal evaluator verdict token `PASS` to `FAIL` or
`FAIL` to `PASS`. Submitted answer, true verdict, query count, task metadata,
score, and all other feedback text must remain identical. Generation resumes
live only after this one-token intervention.

## Frozen ranked candidates

Candidate selection may inspect only normal completion at the intervention and
whether the truthful high/low verdicts match. It may not inspect downstream
clean outcomes. The first eligible row is selected independently for each
request seed.

| Rank | Position | High run | Low run | Structural gap | Remaining questions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28 | 0 | 4 | 4.500 | 12 |
| 2 | 25 | 3 | 1 | 4.000 | 15 |
| 3 | 23 | 0 | 4 | 3.583 | 17 |
| 4 | 24 | 3 | 2 | 3.400 | 16 |
| 5 | 27 | 4 | 2 | 3.250 | 13 |
| 6 | 30 | 4 | 1 | 3.167 | 10 |
| 7 | 31 | 4 | 0 | 3.000 | 9 |
| 8 | 29 | 0 | 3 | 2.833 | 11 |
| 9 | 26 | 2 | 1 | 2.333 | 14 |
| 10 | 32 | 0 | 1 | 2.333 | 8 |

If no row is eligible for a seed, that seed is marked ineligible and the final
study is inconclusive rather than silently substituting a new candidate.

## Outcomes and causal contrast

For each selected candidate, G2b runs the high and low clean trajectories and
their matched one-error counterfactuals. For downstream questions only, each arm
reports:

- signed regret harm, normalized by the query budget;
- downstream accuracy harm;
- normalized exploratory-query excess;
- first action-divergence distance and divergence rate;
- context-token and truncation-count differences;
- distance-indexed persistence of regret and action divergence.

Let `H_high` and `H_low` be the signed normalized regret harm in the high- and
low-leverage arms. The seed-level estimand is

`D = H_high - H_low`.

The high and low counterfactuals must have the same confusion type. Every
counterfactual must contain exactly one evaluator error at the preregistered
position and truthful feedback everywhere else.

## Frozen aggregate rule

The result is positive if and only if all of the following hold:

1. all five request seeds have an eligible candidate;
2. all five pass every causal validity audit;
3. mean `D` across the five seeds is greater than zero;
4. `D > 0` in at least four of five seeds; and
5. mean `H_high` is greater than zero.

Fewer than five eligible seeds is `inconclusive_fewer_than_five_eligible`. Any
causal audit failure is `invalid`. If all audits pass but the positive rule is
not met, the result is `negative_or_inconclusive`. Provider failure or the cost
guard produces no research classification.

## Execution and reporting gate

The harness, Gemini transport overlay, this preregistration, tests, runner, and
public provider-screen selection must be committed and annotated-tagged, and
both commit and tag must be pushed, before `GEMINI_API_KEY` is exported for the
first real task call.

Complete raw actions, provider records, observations, response IDs, and the cost
ledger stay under ignored private results. Public files contain frozen
configuration, hashes, aggregate metrics, resource totals, and validity results.
No private trajectory may be committed.
