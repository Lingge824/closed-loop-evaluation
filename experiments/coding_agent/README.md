# Coding-Agent G1 Pilot

This is the active G1 harness. It is based on the uploaded expanded V1 pilot rather than the earlier six-task prototype.

## What is fixed

- Twelve tasks across four related families are preserved.
- The perturbed trajectory copies the exact clean prefix.
- The first intervention reuses the exact clean solution and visible/hidden test outcomes.
- Only the delivered proxy verdict changes at the intervention.
- Solver and memory-update seeds are paired by round.
- Test failures retain return code, stdout, stderr, and timeout state.
- Provider `system_fingerprint` is logged when available.
- Memory-writer reasoning tags are removed and persisted memory is restricted
  to at most twelve Markdown bullets.
- Groq endpoints prefer `GROQ_API_KEY` even if an unrelated
  `OPENAI_API_KEY` exists in the shell.
- Double-flip interactions are disabled by default and, when enabled later, use a common outcome horizon.

The untouched source files are under `archive/`.

## Setup with the Groq free tier

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY='...'
export PILOT_MODEL='qwen/qwen3.6-27b'
export OPENAI_BASE_URL='https://api.groq.com/openai/v1'
```

## Mainline execution order

```bash
# Local causal-identification regression tests; no API key needed.
python -m unittest discover -s tests -v

# G1 paired three-task smoke.
python pilot.py --mode smoke --seed 17

# No-corruption twelve-task persistent-memory baseline.
python pilot.py --mode baseline --seed 17

# Single-flip G1 run. Interactions remain off.
python pilot.py --mode full --seed 17
```

Do not use `--include-interactions` until the single-flip G1 gate passes.
