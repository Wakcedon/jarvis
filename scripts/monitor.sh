#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python -m jarvis.monitor --status runtime/status.json
