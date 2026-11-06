#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="${1:-"$ROOT_DIR/artifacts/latest"}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m qa_agents.run_test --goal "Toggle Wi-Fi off and on" --env mock --logs_dir "$LOGS_DIR"
echo "Done. See $LOGS_DIR"
