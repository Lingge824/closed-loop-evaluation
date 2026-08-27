#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLBENCH_ROOT="${CLBENCH_ROOT:-$HOME/Downloads/continual-learning-bench}"
PYTHON="$CLBENCH_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing CL-Bench virtual environment: $PYTHON" >&2
  exit 1
fi

cd "$CLBENCH_ROOT"

"$PYTHON" -m py_compile \
  src/systems/icl/system.py \
  src/systems/utils/provider_adapters.py \
  src/systems/utils/structured_output.py \
  src/tasks/database_exploration/task.py \
  "$SCRIPT_DIR/g2_harness.py" \
  "$SCRIPT_DIR/g2b_harness.py" \
  "$SCRIPT_DIR/g2b_provider_screen.py" \
  "$SCRIPT_DIR/tests/test_g2_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_screen.py"

"$PYTHON" -m pytest -q \
  tests/test_database_exploration_verdict_feedback.py \
  tests/test_structured_output.py \
  tests/test_provider_adapters.py \
  tests/test_token_budget_systems.py \
  "$SCRIPT_DIR/tests/test_g2_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_screen.py"

"$PYTHON" -m ruff check \
  src/systems/icl/system.py \
  src/systems/utils/provider_adapters.py \
  src/systems/utils/structured_output.py \
  "$SCRIPT_DIR/g2_harness.py" \
  "$SCRIPT_DIR/g2b_harness.py" \
  "$SCRIPT_DIR/g2b_provider_screen.py" \
  "$SCRIPT_DIR/tests/test_g2_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_harness.py" \
  "$SCRIPT_DIR/tests/test_g2b_screen.py"

"$PYTHON" "$SCRIPT_DIR/g2b_harness.py" \
  --clbench-root "$CLBENCH_ROOT" \
  preflight

echo "G2b HARNESS VERIFICATION PASSED"
