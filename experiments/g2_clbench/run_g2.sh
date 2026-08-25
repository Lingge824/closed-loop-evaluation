#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLBENCH_ROOT="${CLBENCH_ROOT:-$HOME/Downloads/continual-learning-bench}"
PYTHON="$CLBENCH_ROOT/.venv/bin/python"
SEED="${1:-all}"
shift || true

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing CL-Bench virtual environment: $PYTHON" >&2
  exit 1
fi

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is not set" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/g2_harness.py" \
  --clbench-root "$CLBENCH_ROOT" \
  run --seed "$SEED" "$@"
