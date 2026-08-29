# Same Accuracy, Different Futures

**Causal attribution of evaluator errors in adaptive agents**

Static accuracy says how often an evaluator is wrong. This project asks what those errors *cause* after verdicts change persistent memory, future actions, task selection, or other adaptive state.

The core method is an **exact-prefix paired intervention**: replay the complete clean history through one evaluation decision, change only the delivered verdict, and then let the clean and counterfactual branches evolve naturally. This separates the effect of the verdict from ordinary model and environment variation.

> **Current scientific status:** the controlled system is mostly contractive and additive; coding-agent experiments establish persistent behavioral divergence but not downstream oracle harm; the original G2 route was operationally infeasible; the preregistered G2b benchmark has no scientific outcome yet.

## Research question

When evaluator verdicts enter an adaptive loop:

1. which individual errors have the largest downstream causal effect;
2. do those effects decay, persist, amplify, or interact;
3. can local leverage predict harm under unseen error processes; and
4. can a fixed trusted-audit budget prevent more regret than accuracy- or uncertainty-only allocation?

For state update `S_(t+1) = F_t(S_t, X_t, D_t)`, the horizon-`h` effect of one corrupted verdict is the paired outcome difference between `do(D_t = corrupt)` and `do(D_t = clean)` under the same pre-intervention history.

## What the evidence supports

| Stage | Status | Main conclusion |
| --- | --- | --- |
| Toy V2.6/V3 | Verified negative regime | Early leverage dominates; later dynamics add modest signal and are mostly contractive/additive. |
| Coding-agent G1 | Exploratory mechanism evidence | One verdict flip changes persistent memory and most later solutions across five seeds, without hidden-oracle accuracy decline. |
| Coding-agent G1.1 | Preregistered confirmatory negative | Explicit near-linear task requirements prevent the erroneous memory lesson from causing resource harm (`0/5` positive-harm seeds). |
| G2: Groq/Qwen | Operationally closed | Zero complete trajectories; provider failures are not a research result. |
| G2b: Gemini 3.6 Flash | Preregistered, outcome pending | Synthetic screen passed `24/24`; formal benchmark execution has not produced a complete trajectory. |

The project does **not** currently claim that evaluator errors generically snowball, that G1 caused downstream correctness harm, or that G2/G2b supports the central hypothesis.

## Start here

- [Working technical note](docs/technical_note/SAME_ACCURACY_DIFFERENT_FUTURES_TECHNICAL_NOTE.md)
- [Verified toy-system findings](experiments/toy_system/RESULTS_INDEX.md)
- [G1.1 preregistered confirmatory result](experiments/coding_agent/audits/g1_1_confirmatory_20260825.md)
- [G2 operational infeasibility audit](experiments/g2_clbench/G2_OPERATIONAL_INFEASIBILITY_AUDIT.md)
- [Frozen G2b preregistration](experiments/g2_clbench/G2B_PREREGISTRATION.md)
- [Task-blind Gemini provider-screen summary](experiments/g2_clbench/results/provider_screen/public/screen_summary.json)
- [Ordered project gates](PROJECT_MAINLINE.md)

## Repository layout

- `docs/proposal/`: current research proposal and archived earlier versions.
- `docs/technical_note/`: concise public technical synthesis.
- `experiments/toy_system/`: controlled simulations, figures, and validation logs.
- `experiments/coding_agent/`: exact-prefix persistent-memory G1/G1.1 harness and audits.
- `experiments/g2_clbench/`: frozen G2/G2b protocols, transport overlays, tests, and public operational records.

Private trajectories, response IDs, provider records, cost ledgers, and terminal logs are intentionally excluded from Git.

## Reproduce the public checks

Toy experiments:

```bash
cd experiments/toy_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MPLBACKEND=Agg python pilot_v3.py
```

Coding-agent local tests:

```bash
cd experiments/coding_agent
python -m unittest discover -s tests -v
```

G2b verification requires the frozen upstream CL-Bench checkout described in the preregistration:

```bash
cd experiments/g2_clbench
./verify_g2b.sh
```

No API key is required for these local verification commands. Live model runs remain subject to their frozen protocol gates.

## Next decisive tests

1. Complete the frozen G2b matched-error experiment in a supported execution region.
2. Freeze a leverage estimator from isolated interventions and test held-out ranking beyond strong early-performance baselines.
3. Compare `error probability × preventable harm` auditing with random, uncertainty-only, and error-probability-only allocation at equal cost.
4. Test a second update authority beyond persistent memory.

## Author

Lingge Meng — Mathematics–Computer Science, University of California San Diego
Working research repository, August 2026.
