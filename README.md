# Jarvis (локальный голосовой ассистент, Ubuntu, RU)

Коротко: **сказал “Джарвис” → сказал фразу → Jarvis ответил голосом**.

Стек:
- Wake word: Vosk
- STT: faster-whisper
- LLM: Ollama (локально)
- TTS: Piper

Главная идея: всё управляется командами `jarvis …` и через трей — без ручного редактирования файлов.

## Установка одной командой (рекомендуется)

```bash
cd "/home/string/Рабочий стол/Projects/jarvis"
./scripts/install.sh
```

После установки проект будет в `~/.local/opt/jarvis`, а настройки/данные — в XDG:
- config: `~/.config/jarvis/config.yaml`
- data: `~/.local/share/jarvis/`
- state/status: `~/.local/state/jarvis/status.json`

Примечание: если у вас подключён битый PPA `linvinus/rhvoice` (404), скрипт установки автоматически отключит его,
чтобы `apt-get update` не падал.

## Быстрый старт (без установки)

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

Вариант через CLI:

```bash
# создать XDG-конфиг (если ещё нет)
"$PWD/.venv/bin/python" -m jarvis init

# запустить, используя config.yaml из проекта
"$PWD/.venv/bin/python" -m jarvis run --config config.yaml
```

## Настройка

По умолчанию Jarvis ищет конфиг так:
1) `JARVIS_CONFIG`, если задан
2) `~/.config/jarvis/config.yaml`
3) `./config.yaml` (portable-режим)

Настраивать можно без ручного редактирования:

```bash
# если команда `jarvis` не найдена, используй:
# ~/.local/opt/jarvis/.venv/bin/jarvis …

# включить/выключить LLM
jarvis config toggle llm.enabled

# включить/выключить wake word
jarvis config toggle wake_word.enabled

# включить/выключить стриминг ответа Ollama прямо в TTS (быстрее первый звук)
jarvis config set llm.stream_to_tts true

# запретить опасные действия
jarvis config set capabilities.can_power false
jarvis config set capabilities.can_shell false
```

Если хочешь посмотреть текущее значение:

```bash
jarvis config get llm.enabled
```

Открыть конфиг в редакторе:

```bash
jarvis config open
```

Если реагирует медленно — включи быстрый пресет (одной командой):

```bash
jarvis preset speed
jarvis service restart jarvis.service
```

Если микрофон далеко / шумно и Jarvis то не слышит, то срабатывает на шум:

```bash
jarvis preset room
jarvis service restart jarvis.service
```

Если всё равно приходится говорить вплотную — проверь, что выбран правильный микрофон:

```bash
# посмотреть устройства
"$PWD/.venv/bin/python" scripts/list_audio_devices.py

# выставить input_device (пример)
jarvis config set audio.input_device 1
jarvis service restart jarvis.service
```

Ручные “крутилки” wake word:
- меньше ложных срабатываний: `jarvis config set wake_word.min_confidence 0.75`
- лучше слышит издалека (но может чаще ошибаться): `jarvis config set wake_word.noise_gate_multiplier 3.0`

Ключевые настройки в конфиге:
- `wake_word.phrases` — слова активации
- `wake_word.min_confidence` — сделай выше, если срабатывает на лишние слова (например 0.75)
- `wake_word.cooldown_s` — защита от повторных срабатываний подряд
- `stt.model_size/device/compute_type` — модель Whisper и режим
- `tts.model_path/config_path` — выбранный голос Piper
- `llm.enabled` / `wake_word.enabled`
- `llm.stream_to_tts` — меньше задержка до первой озвучки
- `audio.barge_in_enabled` — можно перебить Jarvis голосом (остановит озвучку)
- `storage.backend` — `sqlite` (по умолчанию) или `files`
- `capabilities.*` — политика прав (питание/shell/буфер/сеть и т.д.)

Если хочешь, чтобы Jarvis **вообще не реагировал на разговоры** без обращения к нему:
- оставь `wake_word.enabled: true` (рекомендуется)
- или, если хочешь open-mic, поставь `wake_word.enabled: false` — тогда Jarvis ответит только на фразы, начинающиеся с «джарвис …» / «ассистент …»

Если плохо распознаёт короткие команды (типа «погода») и отвечает невпопад:
- увеличь точность Whisper без смены модели: `jarvis config set stt.beam_size 3`
- чуть увеличь паузу после активации, чтобы TTS не попадал в запись: `jarvis config set audio.activation_delay_s 0.15`
- если wake word иногда срабатывает лишний раз: `jarvis config set wake_word.min_confidence 0.75`

## Примеры команд

- «открой браузер»
- «погода в Москве»
- «запиши заметку купить молоко»
- «который час»

Если команда не распознана как “системная”, текст отправляется в LLM (Ollama).

## Как добавить новую команду

Команды разнесены по навыкам (skills) в папке [jarvis/skills/](jarvis/skills/).
Добавь новый файл-скилл и подключи его в [jarvis/app.py](jarvis/app.py).

## Как сменить модель/голос

- Whisper: поменяйте `stt.model_size` и (опционально) `stt.compute_type`.
- Piper: скачайте другой голос и поменяйте `tts.model_path`/`tts.config_path`.

## Запуск в фоне (systemd)

Шаблон юнита: [systemd/jarvis.service](systemd/jarvis.service).
В нём нужно заменить пути на ваши (из-за пробелов в пути лучше указывать полный путь и экранировать/проверить).

Управление сервисами:

```bash
jarvis service status jarvis.service
jarvis service restart jarvis.service
jarvis service status jarvis-tray.service
```
