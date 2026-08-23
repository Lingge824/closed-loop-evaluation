# Verified Toy-System Findings

The uploaded scripts were rerun in isolated directories on 2026-08-23 with Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, and `MPLBACKEND=Agg`. Original uploaded figures were not overwritten. Full console outputs are under `validation_logs/`.

## V2.5 falsification

- Held-out early-performance baseline: Spearman rho `0.7768`.
- Impact plus growth: rho `0.8682`.
- Shuffle control: rho `0.3787`.
- LOORO median rho: early performance `0.6877`; impact plus growth `0.6938`.
- Script decision: **falsification concern**. A simple baseline explains too much of the V2 result, so V2 alone is not sufficient evidence of strong closed-loop instability.

## V2.6 early-performance control

- Train interventions: `2400`; held-out interventions: `2400`.
- Residual dynamics pooled rho: `0.3459`.
- Median within-regime residual rho: `0.2975`.
- Adding dynamics beyond the early-performance baseline changes log-space R² by `0.00565`.
- Within-regime permutation p-value: approximately `0.0499`.
- Script decision: **leverage dominates, dynamics add modest signal**.

## V3 propagation and interaction

- At `T=320`, median late/early performance ratio: `0.1434`.
- Performance snowball fraction at `T=320`: `0.0417`.
- Capability snowball fraction at `T=320`: `0.0083`.
- Error-density late-deviation scaling exponent: `0.8718`.
- `H(20%) / [2 H(10%)]`: `1.0012`, approximately additive.
- Median normalized pairwise interaction: `0.0`.
- Fraction with normalized interaction above `0.25`: `0.0889`.
- Burst50/IID late-deviation ratio: `0.8513`; Burst25/IID: `0.6383`.
- Snowball evidence count: `0/4`.
- Script decision: **mostly additive leverage**.

## Consequence for the paper

These are not failed experiments. They establish an important negative regime: in this controlled adaptive system, errors act mostly through local update leverage and then contract rather than producing runaway snowball effects. The coding-agent G1 pilot now tests whether persistent language-model memory produces stronger, heterogeneous causal consequences under the same strict paired intervention design.
