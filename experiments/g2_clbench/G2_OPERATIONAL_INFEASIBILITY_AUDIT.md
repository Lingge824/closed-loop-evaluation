# G2 Operational Infeasibility Audit

Status: frozen after the Amendment 004 hard stop and before any G2b provider
screen or replacement-model task run.

Date: 2026-08-27.

## Scope

This audit records an experimental-equipment failure, not a scientific result.
The frozen G2 model-provider pair was:

- model: `qwen/qwen3.6-27b`;
- provider: Groq;
- transport: LiteLLM chat completions;
- task: the preregistered 40-question CL-Bench database-exploration trajectory;
- request seeds: 7, 17, 29, 41, and 53.

The original protocol and Amendments 001--004 remain the authoritative record
of each change. The private terminal logs remain preserved and ignored by Git.
`g2_operational_audit.py` hashes those logs and emits a public, content-free
inventory so the evidence can be checked without publishing prompts, actions,
or observations.

## Observed Operational Sequence

The incomplete launches exposed, in order:

1. provider request/timeout behavior that required a bounded transient-error
   policy;
2. deterministic `tool_use_failed` responses when LiteLLM translated the
   Pydantic schema into a forced function call;
3. `json_validate_failed` under Groq JSON Object Mode;
4. repeated empty final content after local schema validation replaced
   provider-side JSON validation;
5. repeated responses without a schema-valid action under raw reasoning
   transport, followed by the Groq free-tier token-per-minute limit.

Amendment 004 explicitly declared a hard stop if the same model-provider pair
could not complete a formal trajectory. That condition was met.

## Scientific Boundary

At the hard stop:

- question 1 had not completed;
- no clean trajectory had completed or been cached;
- no structural-leverage candidate had been selected;
- no counterfactual branch had run;
- no downstream causal-harm metric had been computed;
- no G2 positive, negative, or inconclusive decision existed.

Consequently, none of the incomplete launches is evidence for or against the
Same Accuracy, Different Futures hypothesis. They establish only that this
model-provider-rate-limit combination was operationally infeasible for the
frozen protocol.

## Disposition

The original G2 is closed without a scientific outcome. There will be no
Amendment 005 to its transport. A replacement experiment must be named G2b,
must use a separately frozen model/provider screen that contains no real G2
question, and must commit its selected snapshot and protocol before any G2b
task call.
