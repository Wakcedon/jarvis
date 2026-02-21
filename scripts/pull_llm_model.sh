#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen2.5:1.5b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama не найден в PATH"
  exit 1
fi

echo "Pull: $MODEL"
ollama pull "$MODEL"

echo "OK"
