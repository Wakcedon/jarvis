# Jarvis (локальный голосовой ассистент, Ubuntu, RU)

Прототип: **активация по слову → запись фразы → распознавание → команда/LLM → озвучка**.

**Стек (выбранный):**

- Wake word: Vosk (keyword spotting/ограниченная грамматика)
- STT: faster-whisper
- LLM: Ollama (локальный HTTP API)
- TTS: Piper (`piper-tts`)

## Установка одной командой (рекомендуется)

```bash
cd "/home/string/Рабочий стол/Projects/jarvis"
./scripts/install.sh
```

Примечание: если у вас подключён битый PPA `linvinus/rhvoice` (404), скрипт установки автоматически отключит его,
чтобы `apt-get update` не падал.

## Быстрый старт (без установки в ~/.local/opt)

1. Установить системные зависимости и Python-пакеты:

```bash
cd "$(dirname "$0")"
bash scripts/setup_ubuntu.sh
```

1. Скачать модели (Vosk RU + Piper RU голос + Whisper):

```bash
bash scripts/download_models.sh
```

1. Поднять Ollama (один раз) и скачать модель:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

1. Запуск ассистента:

```bash
bash scripts/run.sh
```

## Настройка

Все настройки в [config.yaml](config.yaml):

- `wake_word.phrases` — слова активации
- `stt.model_size/device/compute_type` — модель Whisper и режим
- `tts.model_path/config_path` — выбранный голос Piper
- `llm.model` — модель Ollama
- `commands.allowed_apps` — список приложений, которые можно запускать командой

## Примеры команд

- «открой браузер»
- «погода в Москве»
- «запиши заметку купить молоко»
- «который час»

Если команда не распознана как “системная”, текст отправляется в LLM (Ollama).

## Как добавить новую команду

Откройте [jarvis/commands.py](jarvis/commands.py) и добавьте правило в `CommandRouter.handle()`.
Возвращайте `CommandResponse(text=...)` для ответа ассистента.

## Как сменить модель/голос

- Whisper: поменяйте `stt.model_size` и (опционально) `stt.compute_type`.
- Piper: скачайте другой голос и поменяйте `tts.model_path`/`tts.config_path`.

## Запуск в фоне (systemd)

Шаблон юнита: [systemd/jarvis.service](systemd/jarvis.service).
В нём нужно заменить пути на ваши (из-за пробелов в пути лучше указывать полный путь и экранировать/проверить).
