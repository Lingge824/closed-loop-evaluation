# Coding-Agent Evaluation Leverage Pilot

Minimal paired counterfactual experiment with persistent `skills.md`.

- visible tests = proxy evaluator
- hidden tests = oracle
- one proxy verdict is flipped exactly once
- memory update sees only proxy verdict
- later coding prompts always include persistent memory

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY='...'
export PILOT_MODEL='your-model'
# optional OpenAI-compatible endpoint:
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'

python pilot.py
```

Read `results/summary.csv` and the per-round JSON traces.
