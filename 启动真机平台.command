#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./scripts/run.sh --config configs/mac_connected.local.json --open
