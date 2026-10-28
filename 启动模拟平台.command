#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' '首次使用请按 README 安装 Python 环境。' '终端命令：python3 -m venv .venv' '.venv/bin/python -m pip install -e .'
  read -r -p '按 Enter 退出。'
  exit 1
fi
exec .venv/bin/python -m piper_studio.app --config configs/sim.json --connect-sim --open
