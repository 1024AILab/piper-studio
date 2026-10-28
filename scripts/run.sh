#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' '请先运行：python3 -m venv .venv && .venv/bin/python -m pip install -e .'
  exit 1
fi
exec .venv/bin/python -m piper_studio.app "$@"
