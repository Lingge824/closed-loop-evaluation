#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python g2_operational_audit.py
python g2b_provider_screen.py preflight
python g2b_provider_screen.py run "$@"
