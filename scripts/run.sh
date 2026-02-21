#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# чтобы появилась команда .venv/bin/jarvis и подтянулись entrypoints
if [[ ! -x .venv/bin/jarvis ]] || [[ pyproject.toml -nt .venv/bin/jarvis ]]; then
	pip install -e . >/dev/null 2>&1 || true
fi

# Если параллельно запущен systemd user-сервис, он может конфликтовать с ручным запуском
# (микрофон/аудио). Останавливаем на время ручного теста.
systemctl --user stop jarvis.service jarvis-tray.service 2>/dev/null || true

# держим кэш HF локально, чтобы всё лежало в проекте/установке
export HF_HOME="$PWD/models/hf"

# so relative paths in config.yaml resolve to this project root
export JARVIS_ROOT="$PWD"
# keep status in XDG state so UI/systemd/dev all point to one place
export JARVIS_STATUS="$HOME/.local/state/jarvis/status.json"

python -m jarvis --config config.yaml
