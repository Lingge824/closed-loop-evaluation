# Same Accuracy, Different Futures

Research repository for **From Static Accuracy to Closed-Loop Stability**.

The central hypothesis is that evaluators with similar static accuracy can have different downstream causal effects when their verdicts change persistent agent state and future data, task selection, or behavior.

The uploaded toy experiments do **not** support a strong runaway-instability claim. V2.6 finds that early perturbation leverage explains most predictive power and closed-loop dynamics add a modest residual signal; V3 is mostly additive and contractive. The paper mainline is therefore **closed-loop evaluation leverage**, with instability/snowball behavior treated as a conditional secondary regime.

## Current mainline

The current gate is **G1: establish a valid paired causal effect in a persistent-memory coding agent**. The clean and perturbed trajectories must share an exact pre-intervention prefix and the same intervention-round solution and test outcomes. Only the delivered proxy verdict may change at the intervention; later trajectories then evolve naturally.

Do not scale models, tasks, seeds, corruption processes, or audit policies until G1 passes.

## Repository layout

- `docs/proposal/`: current proposal plus preserved proposal versions.
- `experiments/toy_system/`: V0, V1, V2, V2.5, V2.6, and V3 simulations with the uploaded figures.
- `experiments/coding_agent/`: active G1 harness and regression tests.
- `experiments/coding_agent/archive/`: untouched original 6-task and expanded V1 pilots.
- `PROJECT_MAINLINE.md`: ordered gates and stop conditions.

The tag `toy-upload-snapshot` preserves the exact flat-layout toy-system upload before repository reorganization.
