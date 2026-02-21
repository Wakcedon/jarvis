#!/usr/bin/env bash
set -euo pipefail

mkdir -p models data

echo "[1/3] Скачивание Vosk RU (small)..."
VOSK_ZIP="models/vosk-model-small-ru-0.22.zip"
VOSK_DIR="models/vosk-model-small-ru-0.22"
if [[ ! -d "$VOSK_DIR" ]]; then
  curl -L -o "$VOSK_ZIP" "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
  unzip -q "$VOSK_ZIP" -d models
else
  echo "Vosk модель уже есть: $VOSK_DIR"
fi

echo "[2/3] Скачивание Piper RU голоса (${PIPER_VOICE:-dmitri}/${PIPER_QUALITY:-medium})..."
mkdir -p models/piper

PIPER_VOICE="${PIPER_VOICE:-dmitri}"
PIPER_QUALITY="${PIPER_QUALITY:-medium}"

PIPER_ONNX="models/piper/ru_RU-${PIPER_VOICE}-${PIPER_QUALITY}.onnx"
PIPER_JSON="models/piper/ru_RU-${PIPER_VOICE}-${PIPER_QUALITY}.onnx.json"
if [[ ! -f "$PIPER_ONNX" ]]; then
  curl -L -o "$PIPER_ONNX" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/${PIPER_VOICE}/${PIPER_QUALITY}/ru_RU-${PIPER_VOICE}-${PIPER_QUALITY}.onnx?download=true"
else
  echo "Piper onnx уже есть: $PIPER_ONNX"
fi

if [[ ! -f "$PIPER_JSON" ]]; then
  curl -L -o "$PIPER_JSON" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/${PIPER_VOICE}/${PIPER_QUALITY}/ru_RU-${PIPER_VOICE}-${PIPER_QUALITY}.onnx.json?download=true"
else
  echo "Piper json уже есть: $PIPER_JSON"
fi

echo "[3/3] Скачивание Whisper-модели (faster-whisper) локально..."
WHISPER_SIZE="${WHISPER_SIZE:-base}"
WHISPER_OUT="models/whisper/faster-whisper-${WHISPER_SIZE}"
if [[ -f ".venv/bin/python" ]]; then
  . .venv/bin/activate
  python scripts/download_whisper_model.py --size "$WHISPER_SIZE" --out "$WHISPER_OUT"
else
  python3.11 scripts/download_whisper_model.py --size "$WHISPER_SIZE" --out "$WHISPER_OUT"
fi

echo "OK"

