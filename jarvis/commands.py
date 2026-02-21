from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import signal
from typing import Any

import requests

from jarvis.desktop_apps import find_app, launch_app


@dataclass(frozen=True)
class CommandResponse:
    text: str


@dataclass(frozen=True)
class TimerSetResponse(CommandResponse):
    seconds: float


@dataclass(frozen=True)
class TimerCancelResponse(CommandResponse):
    pass


@dataclass(frozen=True)
class TimerStatusRequest(CommandResponse):
    pass


@dataclass(frozen=True)
class PendingSystemActionResponse(CommandResponse):
    action: str  # poweroff|reboot|suspend|lock


@dataclass(frozen=True)
class VerbositySetResponse(CommandResponse):
    mode: str  # short|normal|long


@dataclass(frozen=True)
class SpeakStatusRequest(CommandResponse):
    pass


@dataclass(frozen=True)
class RepeatLastHeardRequest(CommandResponse):
    pass


class CommandRouter:
    def __init__(
        self,
        *,
        notes_path: Path,
        memory_path: Path,
        weather_cache_path: Path,
        reminders_path: Path,
        todo_path: Path,
        screenshots_dir: Path,
        default_browser_url: str,
        default_city: str,
        allowed_apps: tuple[str, ...],
        allow_shell: bool,
        allow_desktop_launch: bool,
    ) -> None:
        self._notes_path = notes_path
        self._memory_path = memory_path
        self._weather_cache_path = weather_cache_path
        self._reminders_path = reminders_path
        self._todo_path = todo_path
        self._screenshots_dir = screenshots_dir
        self._default_browser_url = default_browser_url
        self._default_city = default_city
        self._allowed_apps = set(allowed_apps)
        self._allow_shell = allow_shell
        self._allow_desktop_launch = allow_desktop_launch

    def handle(self, text: str) -> CommandResponse | None:
        t = text.strip().lower()
        # Нормализация: Whisper часто добавляет точки/вопросительные знаки.
        t = re.sub(r"[\s\u00A0]+", " ", t)
        t = re.sub(r"[\.!\?…,:;]+$", "", t).strip()
        if not t:
            return None

        # --- Быстрые режимы ответа ---
        if t in {"говори короче", "покороче", "короче"}:
            return VerbositySetResponse(text="Ок. Буду короче.", mode="short")
        if t in {"говори нормально", "обычно", "нормально"}:
            return VerbositySetResponse(text="Ок.", mode="normal")
        if t in {"говори подробнее", "подетальнее", "развернуто"}:
            return VerbositySetResponse(text="Ок. Буду подробнее.", mode="long")

        if t in {"скажи статус", "статус", "как дела у системы"}:
            return SpeakStatusRequest(text="")
        if t in {"повтори", "повтори пожалуйста", "что я сказал", "повтори что услышал"}:
            return RepeatLastHeardRequest(text="")

        # --- Память (локально) ---
        m = re.search(r"\bзапомни\b.*\b(меня\s+зовут|имя)\s+(.+)$", t)
        if m:
            name = m.group(2).strip().strip(".!")
            if name:
                self._mem_set("name", name)
                return CommandResponse(text=f"Хорошо. Запомнил: тебя зовут {name}.")

        if re.search(r"\bкак\s+меня\s+зовут\b", t):
            name = self._mem_get("name")
            return CommandResponse(
                text=f"Тебя зовут {name}." if name else "Я пока не знаю твоё имя. Скажи: «запомни, меня зовут …»."
            )

        m = re.search(r"\bзапомни\b.*\b(мой\s+город|город)\s+(.+)$", t)
        if m:
            city = m.group(2).strip().strip(".!")
            if city:
                self._mem_set("city", city)
                return CommandResponse(text=f"Запомнил. Твой город — {city}.")

        if re.search(r"\bкакой\s+мой\s+город\b", t):
            city = self._mem_get("city")
            return CommandResponse(
                text=f"Твой город — {city}." if city else "Я пока не знаю твой город. Скажи: «запомни мой город …»."
            )

        if re.search(r"\bочисти\s+память\b", t):
            self._mem_clear()
            return CommandResponse(text="Память очищена.")

        # --- Todo ---
        if t in {"покажи задачи", "список задач", "todo", "туду", "дела"}:
            items = _todo_load(self._todo_path)
            if not items:
                return CommandResponse(text="Список задач пуст.")
            undone = [x for x in items if not x.get("done")]
            done = [x for x in items if x.get("done")]
            parts: list[str] = []
            if undone:
                parts.append("Активные: " + "; ".join(f"{x['id']}) {x['text']}" for x in undone[:6]))
            if done:
                parts.append("Готово: " + "; ".join(f"{x['id']}) {x['text']}" for x in done[:4]))
            return CommandResponse(text=". ".join(parts) + ".")

        m = re.search(r"\bдобавь\s+задач[ау]\b\s*(.+)$", t)
        if m:
            item_text = m.group(1).strip()
            if not item_text:
                return CommandResponse(text="Скажи текст задачи после «добавь задачу». ")
            new_id = _todo_add(self._todo_path, item_text)
            return CommandResponse(text=f"Добавил задачу {new_id}.")

        m = re.search(r"\b(сделано|выполнено|закрой)\s+задач[ау]\b\s*(\d+)\b", t)
        if m:
            task_id = int(m.group(2))
            ok = _todo_mark_done(self._todo_path, task_id, done=True)
            return CommandResponse(text="Отметил как выполненную." if ok else "Не нашёл такую задачу.")

        m = re.search(r"\bудали\s+задач[ау]\b\s*(\d+)\b", t)
        if m:
            task_id = int(m.group(1))
            ok = _todo_delete(self._todo_path, task_id)
            return CommandResponse(text="Удалил." if ok else "Не нашёл такую задачу.")

        # --- Напоминания ---
        if t in {"покажи напоминания", "список напоминаний", "напоминания"}:
            items = _reminders_load(self._reminders_path)
            pending = [x for x in items if not x.get("fired")]
            if not pending:
                return CommandResponse(text="Активных напоминаний нет.")
            now = time.time()
            pending.sort(key=lambda x: _as_float(x.get("due_ts"), 0.0))
            parts: list[str] = []
            for x in pending[:6]:
                due = _as_float(x.get("due_ts"), 0.0)
                rem = max(0.0, due - now)
                parts.append(f"{x['id']}) через {_fmt_duration(rem)}: {x['text']}")
            return CommandResponse(text=". ".join(parts) + ".")

        m = re.search(
            r"\bнапомни\b\s+через\s+(\d+)\s*(секунд[уы]?|минут[уы]?|час(?:а|ов)?)\b\s*(.+)?$",
            t,
        )
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            msg = (m.group(3) or "").strip()
            if not msg:
                msg = "напоминание"
            mult = 1.0
            if unit.startswith("мин"):
                mult = 60.0
            elif unit.startswith("час"):
                mult = 3600.0
            seconds = float(n) * mult
            if seconds <= 0:
                return CommandResponse(text="Скажи длительность больше нуля.")
            due_ts = time.time() + seconds
            rid = _reminders_add(self._reminders_path, due_ts=due_ts, text=msg)
            return CommandResponse(text=f"Ок. Напомню через {_fmt_duration(seconds)} (номер {rid}).")

        m = re.search(r"\bудали\s+напоминани[ея]\b\s*(\d+)\b", t)
        if m:
            rid = int(m.group(1))
            ok = _reminders_delete(self._reminders_path, rid)
            return CommandResponse(text="Удалил." if ok else "Не нашёл такое напоминание.")

        # --- Таймер ---
        if re.search(r"\bотмени\s+таймер\b", t) or re.search(r"\bстоп\s+таймер\b", t):
            return TimerCancelResponse(text="Ок. Таймер отменён.")

        if re.search(r"\bсколько\s+осталось\b", t) or re.search(r"\bтаймер\s+статус\b", t):
            return TimerStatusRequest(text="")

        m = re.search(r"\bтаймер\s+на\s+(\d+)\s*(секунд[уы]?|минут[уы]?|час(?:а|ов)?)\b", t)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            mult = 1.0
            if unit.startswith("мин"):
                mult = 60.0
            elif unit.startswith("час"):
                mult = 3600.0
            seconds = float(n) * mult
            if seconds <= 0:
                return CommandResponse(text="Скажи длительность больше нуля.")
            return TimerSetResponse(text=f"Запускаю таймер на {_fmt_duration(seconds)}.", seconds=seconds)

        if re.search(r"\b(который\s+час|сколько\s+времени)\b", t):
            now = dt.datetime.now().strftime("%H:%M")
            return CommandResponse(text=f"Сейчас {now}.")

        if re.search(r"\b(какой\s+сегодня\s+день|какая\s+сегодня\s+дата)\b", t):
            now = dt.datetime.now()
            # 0=Mon
            dow = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"][now.weekday()]
            month = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"][now.month-1]
            return CommandResponse(text=f"Сегодня {now.day} {month} {now.year} года, {dow}.")

        if re.search(r"\bкак\s+дела\b", t):
            return CommandResponse(text="Нормально. Чем помочь?")

        if re.search(r"\bоткрой\s+браузер\b", t):
            subprocess.Popen(["xdg-open", self._default_browser_url])
            return CommandResponse(text="Открываю браузер.")

        if t in {"открой загрузки", "открой папку загрузки"}:
            subprocess.Popen(["xdg-open", str(Path.home() / "Загрузки")])
            return CommandResponse(text="Открываю загрузки.")
        if t in {"открой рабочий стол", "открой десктоп"}:
            subprocess.Popen(["xdg-open", str(Path.home() / "Рабочий стол")])
            return CommandResponse(text="Открываю рабочий стол.")
        if t in {"открой документы", "открой папку документы"}:
            subprocess.Popen(["xdg-open", str(Path.home() / "Документы")])
            return CommandResponse(text="Открываю документы.")

        m = re.search(r"\bоткрой\s+([\w\-]+)\b", t)
        if m:
            app = m.group(1)
            if app in self._allowed_apps:
                subprocess.Popen([app])
                return CommandResponse(text=f"Открываю {app}.")
            # fallback: поиск по .desktop (почти любое приложение)
            if self._allow_desktop_launch:
                found = find_app(app)
                if found and launch_app(found):
                    return CommandResponse(text=f"Открываю {found.name}.")
            return CommandResponse(text=f"Я не могу открыть «{app}». Попробуй назвать как в меню приложений.")

        m = re.search(r"\bоткрой\s+(.+)$", t)
        if m and self._allow_desktop_launch:
            q = m.group(1).strip()
            if q:
                found = find_app(q)
                if found and launch_app(found):
                    return CommandResponse(text=f"Открываю {found.name}.")
                return CommandResponse(text=f"Не нашёл приложение «{q}». Попробуй другое название.")

        m = re.search(r"\bзапиши\s+заметку\b\s*(.*)$", t)
        if m:
            note = m.group(1).strip()
            if not note:
                return CommandResponse(text="Скажи текст заметки после фразы «запиши заметку».")
            self._notes_path.parent.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().isoformat(timespec="seconds")
            self._notes_path.write_text(self._notes_path.read_text(encoding="utf-8") + f"\n- [{ts}] {note}\n" if self._notes_path.exists() else f"- [{ts}] {note}\n", encoding="utf-8")
            return CommandResponse(text="Записал.")

        if t in {"покажи заметки", "открой заметки"}:
            self._notes_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._notes_path.exists():
                self._notes_path.write_text("", encoding="utf-8")
            subprocess.Popen(["xdg-open", str(self._notes_path)])
            return CommandResponse(text="Открываю заметки.")

        m = re.search(r"\bнайди\s+в\s+заметках\b\s*(.+)$", t)
        if m:
            q = m.group(1).strip()
            if not q:
                return CommandResponse(text="Скажи, что искать.")
            if not self._notes_path.exists():
                return CommandResponse(text="Заметок пока нет.")
            text_notes = self._notes_path.read_text(encoding="utf-8")
            hits = [line.strip() for line in text_notes.splitlines() if q.lower() in line.lower()]
            if not hits:
                return CommandResponse(text="Не нашёл.")
            return CommandResponse(text="Нашёл: " + "; ".join(hits[:4]) + ".")

        m = re.search(r"\bпогода\s+в\s+(.+)$", t)
        if m:
            city = m.group(1).strip()
            return CommandResponse(text=_weather(city, cache_path=self._weather_cache_path))

        if re.search(r"\bкакая\s+погода\b", t) or t == "погода":
            city = self._mem_get("city") or self._default_city
            if city:
                return CommandResponse(text=_weather(city, cache_path=self._weather_cache_path))
            return CommandResponse(text="Скажи город, например: «погода в Москве»." )

        # --- Скриншоты ---
        if t in {"скриншот", "сделай скриншот", "скрин"}:
            p = _take_screenshot(self._screenshots_dir, mode="full")
            if p:
                return CommandResponse(text="Скриншот сделал.")
            return CommandResponse(text="Не смог сделать скриншот (нет gnome-screenshot).")

        if t in {"скриншот окна", "скрин окна"}:
            p = _take_screenshot(self._screenshots_dir, mode="window")
            if p:
                return CommandResponse(text="Скриншот окна сделал.")
            return CommandResponse(text="Не смог сделать скриншот окна.")

        # --- Громкость / микрофон ---
        if t in {"громче", "погромче", "сделай громче"}:
            ok = _volume_step(+5)
            return CommandResponse(text="Сделал громче." if ok else "Не смог изменить громкость.")

        if t in {"тише", "потише", "сделай тише"}:
            ok = _volume_step(-5)
            return CommandResponse(text="Сделал тише." if ok else "Не смог изменить громкость.")

        m = re.search(r"\bгромкост[ьи]\s+(\d{1,3})\b", t)
        if m:
            val = max(0, min(150, int(m.group(1))))
            ok = _volume_set_percent(val)
            return CommandResponse(text=f"Ставлю громкость {val}%." if ok else "Не смог выставить громкость.")

        if t in {"выключи звук", "мут", "без звука", "выруби звук"}:
            ok = _volume_mute(True)
            return CommandResponse(text="Ок. Без звука." if ok else "Не смог выключить звук.")

        if t in {"включи звук", "убери мут", "со звуком"}:
            ok = _volume_mute(False)
            return CommandResponse(text="Ок." if ok else "Не смог включить звук.")

        if t in {"выключи микрофон", "выруби микрофон", "замуть микрофон"}:
            ok = _mic_mute(True)
            return CommandResponse(text="Ок. Микрофон выключен." if ok else "Не смог выключить микрофон.")

        if t in {"включи микрофон", "размуть микрофон"}:
            ok = _mic_mute(False)
            return CommandResponse(text="Ок. Микрофон включен." if ok else "Не смог включить микрофон.")

        # --- Wi‑Fi / Bluetooth ---
        if t in {"выключи вайфай", "выключи wi fi", "вайфай офф", "wifi off"}:
            ok = _set_radio("wifi", False)
            return CommandResponse(text="Wi‑Fi выключен." if ok else "Не смог выключить Wi‑Fi.")
        if t in {"включи вайфай", "включи wi fi", "вайфай он", "wifi on"}:
            ok = _set_radio("wifi", True)
            return CommandResponse(text="Wi‑Fi включен." if ok else "Не смог включить Wi‑Fi.")
        if t in {"выключи блютуз", "выключи bluetooth", "блютуз офф"}:
            ok = _set_radio("bluetooth", False)
            return CommandResponse(text="Bluetooth выключен." if ok else "Не смог выключить Bluetooth.")
        if t in {"включи блютуз", "включи bluetooth", "блютуз он"}:
            ok = _set_radio("bluetooth", True)
            return CommandResponse(text="Bluetooth включен." if ok else "Не смог включить Bluetooth.")

        # --- Окна / фокус (если есть xdotool) ---
        if t in {"закрой окно", "закрой текущее окно"}:
            ok = _xdotool_key("alt+f4")
            return CommandResponse(text="Закрываю окно." if ok else "Не могу: нет xdotool.")
        if t in {"следующее окно", "переключи окно", "альт таб"}:
            ok = _xdotool_key("alt+tab")
            return CommandResponse(text="Переключаю." if ok else "Не могу: нет xdotool.")

        # --- Медиаплеер (MPRIS через playerctl) ---
        if t in {"пауза", "поставь на паузу"}:
            ok = _playerctl("pause")
            return CommandResponse(text="Пауза." if ok else "Не могу управлять плеером (нет playerctl).")
        if t in {"играй", "воспроизведение", "продолжай"}:
            ok = _playerctl("play")
            return CommandResponse(text="Ок." if ok else "Не могу управлять плеером (нет playerctl).")
        if t in {"плей пауза", "переключи паузу", "play pause"}:
            ok = _playerctl("play-pause")
            return CommandResponse(text="Ок." if ok else "Не могу управлять плеером (нет playerctl).")
        if t in {"следующий трек", "следующая песня", "дальше"}:
            ok = _playerctl("next")
            return CommandResponse(text="Дальше." if ok else "Не могу управлять плеером (нет playerctl).")
        if t in {"предыдущий трек", "предыдущая песня", "назад"}:
            ok = _playerctl("previous")
            return CommandResponse(text="Назад." if ok else "Не могу управлять плеером (нет playerctl).")

        # --- Буфер обмена ---
        m = re.search(r"\bскопируй\b\s+(.+)$", t)
        if m:
            payload = m.group(1).strip()
            if not payload:
                return CommandResponse(text="Скажи, что скопировать.")
            ok = _clipboard_set(payload)
            return CommandResponse(text="Скопировал в буфер." if ok else "Не смог записать в буфер (нет wl-copy/xclip).")

        if t in {"что в буфере", "прочитай буфер", "буфер"}:
            s = _clipboard_get()
            return CommandResponse(text=s if s else "Буфер пуст или недоступен.")

        # --- Калькулятор / конвертер ---
        m = re.search(r"\bпосчитай\b\s+(.+)$", t)
        if m:
            expr = m.group(1).strip()
            v = _safe_calc(expr)
            return CommandResponse(text=f"Получается {v}." if v is not None else "Не смог посчитать.")

        m = re.search(r"\bпереведи\b\s+(\d+(?:[\.,]\d+)?)\s*([a-zа-яё°]+)\s+в\s+([a-zа-яё°]+)\b", t)
        if m:
            val = float(m.group(1).replace(",", "."))
            src = m.group(2)
            dst = m.group(3)
            out = _convert_units(val, src, dst)
            if out is None:
                return CommandResponse(text="Не знаю такую конвертацию.")
            return CommandResponse(text=f"Это примерно {out}.")

        # --- Питание (требует подтверждения) ---
        if t in {"выключи компьютер", "выключи пк", "выключай компьютер", "выключи систему"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, выключай». ", action="poweroff")
        if t in {"перезагрузи компьютер", "перезагрузи пк", "перезагрузка"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, перезагружай». ", action="reboot")
        if t in {"сон", "спящий режим", "усыпи компьютер", "приостанови"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, в сон». ", action="suspend")
        if t in {"заблокируй", "заблокируй экран", "лок"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, блокируй». ", action="lock")

        if t.startswith("выполни ") and self._allow_shell:
            cmd = t.removeprefix("выполни ").strip()
            if cmd:
                subprocess.Popen(cmd, shell=True)
                return CommandResponse(text="Выполняю.")

        return None

    def _mem_get(self, key: str) -> str | None:
        try:
            if not self._memory_path.exists():
                return None
            data = json.loads(self._memory_path.read_text(encoding="utf-8"))
            v = (data or {}).get(key)
            s = str(v).strip() if v is not None else ""
            return s or None
        except Exception:
            return None

    def _mem_set(self, key: str, value: str) -> None:
        try:
            self._memory_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, str] = {}
            if self._memory_path.exists():
                try:
                    data = json.loads(self._memory_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            data[str(key)] = str(value)
            self._memory_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _mem_clear(self) -> None:
        try:
            if self._memory_path.exists():
                self._memory_path.unlink()
        except Exception:
            pass


def _fmt_duration(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60:
        return f"{int(round(seconds))} секунд"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(round(seconds % 60))
        return f"{m} минут {s} секунд" if s else f"{m} минут"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h} часов {m} минут" if m else f"{h} часов"


def _read_json(path: Path, default: object) -> object:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _todo_load(path: Path) -> list[dict[str, object]]:
    raw = _read_json(path, default={})
    items = []
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        for x in raw["items"]:
            if isinstance(x, dict) and "id" in x and "text" in x:
                items.append(x)
    return items


def _todo_next_id(items: list[dict[str, object]]) -> int:
    mx = 0
    for x in items:
        mx = max(mx, _as_int(x.get("id"), 0))
    return mx + 1


def _todo_add(path: Path, text: str) -> int:
    items = _todo_load(path)
    new_id = _todo_next_id(items)
    items.append({"id": new_id, "text": text, "done": False, "ts": time.time()})
    _write_json(path, {"items": items})
    return new_id


def _todo_mark_done(path: Path, task_id: int, *, done: bool) -> bool:
    items = _todo_load(path)
    ok = False
    for x in items:
        if _as_int(x.get("id"), -1) == int(task_id):
            x["done"] = bool(done)
            ok = True
    if ok:
        _write_json(path, {"items": items})
    return ok


def _todo_delete(path: Path, task_id: int) -> bool:
    items = _todo_load(path)
    new_items = [x for x in items if _as_int(x.get("id"), -1) != int(task_id)]
    if len(new_items) == len(items):
        return False
    _write_json(path, {"items": new_items})
    return True


def _reminders_load(path: Path) -> list[dict[str, object]]:
    raw = _read_json(path, default={})
    items: list[dict[str, object]] = []
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        for x in raw["items"]:
            if isinstance(x, dict) and "id" in x and "text" in x and "due_ts" in x:
                items.append(x)
    return items


def _reminders_next_id(items: list[dict[str, object]]) -> int:
    mx = 0
    for x in items:
        mx = max(mx, _as_int(x.get("id"), 0))
    return mx + 1


def _reminders_add(path: Path, *, due_ts: float, text: str) -> int:
    items = _reminders_load(path)
    rid = _reminders_next_id(items)
    items.append(
        {
            "id": rid,
            "text": text,
            "due_ts": float(due_ts),
            "created_ts": time.time(),
            "fired": False,
        }
    )
    _write_json(path, {"items": items})
    return rid


def _reminders_delete(path: Path, rid: int) -> bool:
    items = _reminders_load(path)
    new_items = [x for x in items if _as_int(x.get("id"), -1) != int(rid)]
    if len(new_items) == len(items):
        return False
    _write_json(path, {"items": new_items})
    return True


def _as_int(value: object, default: int) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return int(default)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return int(default)
            return int(float(s.replace(",", ".")))
        return int(default)
    except Exception:
        return int(default)


def _as_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return float(default)
            return float(s.replace(",", "."))
        return float(default)
    except Exception:
        return float(default)


def _resolve_exe(name: str) -> str | None:
    from shutil import which

    return which(name)


def _run(cmd: list[str], *, timeout_s: float = 4.0) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s)
        return True
    except Exception:
        return False


def _volume_step(delta_percent: int) -> bool:
    wpctl = _resolve_exe("wpctl")
    if wpctl:
        sign = "+" if delta_percent > 0 else "-"
        return _run([wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{abs(delta_percent)}%{sign}"])
    pactl = _resolve_exe("pactl")
    if pactl:
        op = "+" if delta_percent > 0 else "-"
        return _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{abs(delta_percent)}%{op}"])
    return False


def _volume_set_percent(percent: int) -> bool:
    wpctl = _resolve_exe("wpctl")
    if wpctl:
        return _run([wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent}%"])
    pactl = _resolve_exe("pactl")
    if pactl:
        return _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
    return False


def _volume_mute(mute: bool) -> bool:
    wpctl = _resolve_exe("wpctl")
    if wpctl:
        return _run([wpctl, "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if mute else "0"])
    pactl = _resolve_exe("pactl")
    if pactl:
        return _run([pactl, "set-sink-mute", "@DEFAULT_SINK@", "1" if mute else "0"])
    return False


def _mic_mute(mute: bool) -> bool:
    wpctl = _resolve_exe("wpctl")
    if wpctl:
        return _run([wpctl, "set-mute", "@DEFAULT_AUDIO_SOURCE@", "1" if mute else "0"])
    pactl = _resolve_exe("pactl")
    if pactl:
        return _run([pactl, "set-source-mute", "@DEFAULT_SOURCE@", "1" if mute else "0"])
    return False


def _set_radio(kind: str, on: bool) -> bool:
    nmcli = _resolve_exe("nmcli")
    if nmcli:
        return _run([nmcli, "radio", kind, "on" if on else "off"], timeout_s=6.0)

    if kind == "wifi":
        rfkill = _resolve_exe("rfkill")
        if rfkill:
            return _run([rfkill, "unblock" if on else "block", "wifi"], timeout_s=6.0)
    if kind == "bluetooth":
        bluetoothctl = _resolve_exe("bluetoothctl")
        if bluetoothctl:
            return _run([bluetoothctl, "power", "on" if on else "off"], timeout_s=6.0)
    return False


def _take_screenshot(dir_path: Path, *, mode: str) -> Path | None:
    dir_path.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = dir_path / f"screenshot_{ts}.png"
    gnome = _resolve_exe("gnome-screenshot")
    if gnome:
        if mode == "window":
            ok = _run([gnome, "-w", "-f", str(out)], timeout_s=15.0)
        else:
            ok = _run([gnome, "-f", str(out)], timeout_s=15.0)
        return out if ok else None
    return None


def _xdotool_key(keys: str) -> bool:
    xdotool = _resolve_exe("xdotool")
    if not xdotool:
        return False
    return _run([xdotool, "key", keys], timeout_s=2.0)


def _playerctl(cmd: str) -> bool:
    playerctl = _resolve_exe("playerctl")
    if not playerctl:
        return False
    return _run([playerctl, cmd], timeout_s=2.0)


def _clipboard_set(text: str) -> bool:
    wl = _resolve_exe("wl-copy")
    if wl:
        try:
            subprocess.run([wl], input=text.encode("utf-8"), check=True, timeout=2.0)
            return True
        except Exception:
            return False
    xclip = _resolve_exe("xclip")
    if xclip:
        try:
            subprocess.run([xclip, "-selection", "clipboard"], input=text.encode("utf-8"), check=True, timeout=2.0)
            return True
        except Exception:
            return False
    return False


def _clipboard_get() -> str | None:
    wl = _resolve_exe("wl-paste")
    if wl:
        try:
            out = subprocess.check_output([wl, "-n"], timeout=2.0)
            s = out.decode("utf-8", errors="ignore").strip()
            return s or None
        except Exception:
            return None
    xclip = _resolve_exe("xclip")
    if xclip:
        try:
            out = subprocess.check_output([xclip, "-selection", "clipboard", "-o"], timeout=2.0)
            s = out.decode("utf-8", errors="ignore").strip()
            return s or None
        except Exception:
            return None
    return None


def _safe_calc(expr: str) -> str | None:
    expr = expr.strip()
    if not expr:
        return None
    # простая нормализация: запятые в десятичные точки
    expr = expr.replace(",", ".")
    # иногда диктуют "x" вместо "*"
    expr = re.sub(r"\bх\b", "*", expr)
    expr = expr.replace("×", "*")
    expr = expr.replace("÷", "/")

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _eval(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Mod):
                return a % b
            if isinstance(n.op, ast.Pow):
                return a**b
        raise ValueError("unsupported")

    try:
        v = _eval(node)
        if math.isfinite(v):
            # красивое форматирование: 2.0 -> 2
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.6g}"
        return None
    except Exception:
        return None


def _convert_units(value: float, src: str, dst: str) -> str | None:
    def norm(u: str) -> str:
        u = u.strip().lower().replace(" ", "")
        u = u.replace("°", "")
        return u

    s = norm(src)
    d = norm(dst)

    table: dict[tuple[str, str], float] = {
        ("км", "м"): 1000.0,
        ("м", "км"): 1.0 / 1000.0,
        ("м", "см"): 100.0,
        ("см", "м"): 1.0 / 100.0,
        ("кг", "г"): 1000.0,
        ("г", "кг"): 1.0 / 1000.0,
        ("л", "мл"): 1000.0,
        ("мл", "л"): 1.0 / 1000.0,
    }

    if (s, d) in table:
        v = value * table[(s, d)]
        return f"{v:.6g} {dst}"

    # температуры
    if s in {"c", "с", "ц"} and d in {"f", "ф"}:
        v = value * 9.0 / 5.0 + 32.0
        return f"{v:.6g} {dst}"
    if s in {"f", "ф"} and d in {"c", "с", "ц"}:
        v = (value - 32.0) * 5.0 / 9.0
        return f"{v:.6g} {dst}"

    return None


def _weather(city: str, *, cache_path: Path) -> str:
    def _timeout_handler(_signum, _frame):
        raise TimeoutError()

    old = None
    try:
        # Жёсткий общий таймаут на всю команду (включая DNS), чтобы не висло бесконечно.
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(6)

        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru"},
            timeout=(3, 3),
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"Не нашёл город «{city}»."
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        name = results[0].get("name", city)

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=(3, 3),
        ).json()
        cur = (w.get("current") or {})
        temp = cur.get("temperature_2m")
        wind = cur.get("wind_speed_10m")
        if temp is None:
            return f"Не удалось получить погоду для {name}."
        t_int = int(round(float(temp)))
        parts = [f"Сейчас в {name} примерно {t_int}°C"]
        if wind is not None:
            w_int = int(round(float(wind)))
            parts.append(f"ветер {w_int} м/с")
        answer = ", ".join(parts) + "."

        # cache
        try:
            cache = {"ts": time.time(), "city": name, "answer": answer}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

        return answer
    except TimeoutError:
        cached = _load_cached_weather(cache_path)
        return cached or "Погода сейчас недоступна (таймаут сети)."
    except Exception:
        cached = _load_cached_weather(cache_path)
        return cached or "Не удалось получить погоду (нет интернета или API недоступно)."
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
        if old is not None:
            try:
                signal.signal(signal.SIGALRM, old)
            except Exception:
                pass
def _load_cached_weather(cache_path: Path, max_age_s: float = 3 * 3600) -> str | None:
    try:
        if not cache_path.exists():
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        ts = float(data.get("ts", 0.0) or 0.0)
        if (time.time() - ts) > max_age_s:
            return None
        ans = str(data.get("answer", "") or "").strip()
        if not ans:
            return None
        city = str(data.get("city", "") or "").strip()
        return f"(по кэшу) {ans}" if city else f"(по кэшу) {ans}"
    except Exception:
        return None
