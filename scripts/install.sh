#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$HOME/.local/opt/jarvis"

echo "[1/6] Установка системных зависимостей и venv…"
cd "$SRC_DIR"
bash scripts/setup_ubuntu.sh

echo "[2/6] Копирование проекта в $DEST_DIR (без пробелов в пути)…"
mkdir -p "$DEST_DIR"
rsync -a --delete \
  --exclude ".venv" \
  --exclude "models" \
  --exclude "data" \
  --exclude "**/__pycache__" \
  "$SRC_DIR/" "$DEST_DIR/"

echo "[3/6] Установка Python-зависимостей в $DEST_DIR/.venv…"
cd "$DEST_DIR"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "[4/6] Скачивание моделей (Vosk + Piper + Whisper)…"
bash scripts/download_models.sh

echo "[5/6] Установка Ollama (если нет), моделей LLM и systemd user сервисов…"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama не найдена — устанавливаю вручную с прогрессом…"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  ver="$(python3.11 -c 'import json, urllib.request; url="https://api.github.com/repos/ollama/ollama/releases/latest"; req=urllib.request.Request(url, headers={"User-Agent":"jarvis-installer"}); data=json.load(urllib.request.urlopen(req, timeout=20)); print((data.get("tag_name") or "").strip())')"
  if [[ -z "$ver" ]]; then
    echo "Не смог определить версию Ollama для скачивания." >&2
    exit 1
  fi

  url="https://github.com/ollama/ollama/releases/download/${ver}/ollama-linux-amd64.tar.zst"
  echo "Скачиваю (~1.7GB): $url"
  echo "(это может занять время; прогресс/скорость будет виден ниже)"
  curl --fail --location --progress-bar "$url" -o "$tmpdir/ollama.tar.zst"

  echo "Распаковываю… (sudo)"
  sudo tar --use-compress-program=unzstd -xvf "$tmpdir/ollama.tar.zst" -C /usr/local

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama не появилась в PATH после установки. Проверьте /usr/local/bin/ollama" >&2
    exit 1
  fi
fi

mkdir -p "$HOME/.config/systemd/user"
cp -f systemd/user/ollama.service "$HOME/.config/systemd/user/ollama.service"
systemctl --user daemon-reload
systemctl --user enable --now ollama.service

done
echo "Скачиваю LLM модели (fast + quality)…"
FAST_MODEL="${FAST_MODEL:-qwen2.5:0.5b-instruct}"
QUALITY_MODEL="${QUALITY_MODEL:-qwen2.5:1.5b-instruct}"
FAST_MODEL="$FAST_MODEL" QUALITY_MODEL="$QUALITY_MODEL" bash scripts/pull_llm_models.sh

python - <<PY
from __future__ import annotations

from pathlib import Path

import yaml

path = Path("config.yaml")
raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
llm = dict(llm)

llm.setdefault("enabled", True)
llm.setdefault("backend", "ollama")
llm["dual_model_enabled"] = True
llm["fast_model"] = "${FAST_MODEL}"
llm["quality_model"] = "${QUALITY_MODEL}"
# для совместимости: некоторые места читают llm.model
llm["model"] = llm.get("quality_model")

raw["llm"] = llm
path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("LLM configured:", llm["fast_model"], "/", llm["quality_model"])
PY

echo "Установка ярлыка и systemd user сервисов (jarvis + tray)…"
mkdir -p "$HOME/.local/share/applications"
cp -f desktop/jarvis-tray.desktop "$HOME/.local/share/applications/jarvis-tray.desktop"

cp -f systemd/user/jarvis.service "$HOME/.config/systemd/user/jarvis.service"
cp -f systemd/user/jarvis-tray.service "$HOME/.config/systemd/user/jarvis-tray.service"
systemctl --user daemon-reload
systemctl --user enable --now jarvis-tray.service

# сам ассистент можно включить сразу, но если микрофон будет занят — лучше запускать из трея
systemctl --user enable --now jarvis.service

echo "[6/6] Готово. Иконка Jarvis должна появиться в трее."
echo "Если не появилась — перезайдите в сессию или выполните: systemctl --user restart jarvis-tray.service"

chmod +x scripts/*.sh || true
