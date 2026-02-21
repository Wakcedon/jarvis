#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Нужен python3.11. Установите его и повторите." >&2
  exit 1
fi

echo "[1/3] Установка системных зависимостей (sudo)..."
set +e
sudo apt-get update
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  # Частая проблема: битый PPA linvinus/rhvoice (404). Отключаем его и пробуем ещё раз.
  if [[ -f /etc/apt/sources.list.d/linvinus-ubuntu-rhvoice-jammy.list ]]; then
    echo "Обнаружен проблемный репозиторий linvinus/rhvoice — отключаю его и повторяю apt-get update..."
    sudo mv /etc/apt/sources.list.d/linvinus-ubuntu-rhvoice-jammy.list \
      /etc/apt/sources.list.d/linvinus-ubuntu-rhvoice-jammy.list.disabled
  fi
  sudo apt-get update
fi
sudo apt-get install -y \
  python3.11-venv \
  python3.11-dev \
  build-essential \
  curl \
  unzip \
  zstd \
  rsync \
  pkg-config \
  gobject-introspection \
  libgirepository1.0-dev \
  libcairo2-dev \
  libportaudio2 \
  portaudio19-dev \
  libsndfile1 \
  ffmpeg \
  pulseaudio-utils \
  playerctl \
  xdotool \
  wl-clipboard \
  xclip \
  gnome-screenshot \
  python3-gi \
  gir1.2-gtk-3.0 \
  gir1.2-ayatanaappindicator3-0.1 \
  gnome-shell-extension-appindicator

echo "[2/3] Создание виртуального окружения (.venv)..."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[3/3] Установка Python-зависимостей..."
pip install -r requirements.txt

echo "OK"
