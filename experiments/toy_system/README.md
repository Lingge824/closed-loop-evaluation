# Toy Closed-Loop Experiments

This directory preserves the uploaded simulation sequence and figures.

| Script | Role |
| --- | --- |
| `pilot_v0.py` | Initial matched-accuracy and impulse experiment. |
| `pilot_v1.py` | Stronger dynamics, impulse responses, and pre-specified stability grid. |
| `pilot_v2.py` | Held-out feedback-regime prediction and leave-one-regime-out evaluation. |
| `pilot_v25.py` | Falsification via simpler baselines, ablations, and shuffle control. |
| `pilot_v26.py` | Controls early true-performance deviation and tests incremental explanatory value. |
| `pilot_v3.py` | Single-error propagation, density scaling, pairwise interaction, and temporal persistence. |
| `pilot.py` | Byte-identical copy of `pilot_v0.py` from the original folder; retained for provenance only. |

The existing PNGs in `results/` are preserved research artifacts. Run scripts from this directory so their relative imports resolve correctly. Use the explicit versioned filenames; do not treat `pilot.py` as the latest experiment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MPLBACKEND=Agg python pilot_v3.py
```
