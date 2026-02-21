#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# Если параллельно запущен systemd user-сервис, он может конфликтовать с ручным запуском
# (микрофон/аудио). Останавливаем на время ручного теста.
systemctl --user stop jarvis.service jarvis-tray.service 2>/dev/null || true

# держим кэш HF локально, чтобы всё лежало в проекте/установке
export HF_HOME="$PWD/models/hf"

python -m jarvis --config config.yaml
