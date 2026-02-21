#!/usr/bin/env bash
set -euo pipefail

FAST_MODEL="${FAST_MODEL:-qwen2.5:0.5b-instruct}"
QUALITY_MODEL="${QUALITY_MODEL:-qwen2.5:1.5b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama не найден в PATH"
  exit 1
fi

echo "Pull fast:    $FAST_MODEL"
ollama pull "$FAST_MODEL"

echo "Pull quality: $QUALITY_MODEL"
ollama pull "$QUALITY_MODEL"

echo "OK"
