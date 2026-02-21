from __future__ import annotations

import re
from pathlib import Path
import datetime as dt

from jarvis.commands import CommandResponse
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command

# reuse weather implementation from existing code to avoid behavior drift
from jarvis.commands import _weather  # type: ignore


class OpenNotesWeatherSkill:
    name = "open-notes-weather"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

        if re.search(r"\bоткрой\s+браузер\b", t):
            ok = ctx.executor.open_url(ctx.default_browser_url)
            return CommandResponse(text="Открываю браузер." if ok else "Не смог открыть браузер.")

        if t in {"открой загрузки", "открой папку загрузки"}:
            ok = ctx.executor.open_path(Path.home() / "Загрузки")
            return CommandResponse(text="Открываю загрузки." if ok else "Не смог открыть загрузки.")

        if t in {"открой рабочий стол", "открой десктоп"}:
            ok = ctx.executor.open_path(Path.home() / "Рабочий стол")
            return CommandResponse(text="Открываю рабочий стол." if ok else "Не смог открыть рабочий стол.")

        if t in {"открой документы", "открой папку документы"}:
            ok = ctx.executor.open_path(Path.home() / "Документы")
            return CommandResponse(text="Открываю документы." if ok else "Не смог открыть документы.")

        m = re.search(r"\bоткрой\s+([\w\-]+)\b", t)
        if m:
            app = m.group(1)
            if app in ctx.allowed_apps:
                ok = ctx.executor.launch_app(app)
                return CommandResponse(text=f"Открываю {app}." if ok else f"Не смог открыть {app}.")
            if ctx.allow_desktop_launch:
                ok = ctx.executor.launch_desktop(app)
                return CommandResponse(text="Открываю." if ok else f"Я не могу открыть «{app}». Попробуй название как в меню приложений.")
            return CommandResponse(text=f"Я не могу открыть «{app}».")

        m = re.search(r"\bоткрой\s+(.+)$", t)
        if m and ctx.allow_desktop_launch:
            q = m.group(1).strip()
            if q:
                ok = ctx.executor.launch_desktop(q)
                return CommandResponse(text="Открываю." if ok else f"Не нашёл приложение «{q}» или запуск запрещён.")

        m = re.search(r"\bзапиши\s+заметку\b\s*(.*)$", t)
        if m:
            note = m.group(1).strip()
            if not note:
                return CommandResponse(text="Скажи текст заметки после фразы «запиши заметку». ")
            ctx.notes_path.parent.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().isoformat(timespec="seconds")
            if ctx.notes_path.exists():
                prev = ctx.notes_path.read_text(encoding="utf-8")
                ctx.notes_path.write_text(prev + f"\n- [{ts}] {note}\n", encoding="utf-8")
            else:
                ctx.notes_path.write_text(f"- [{ts}] {note}\n", encoding="utf-8")
            return CommandResponse(text="Записал.")

        if t in {"покажи заметки", "открой заметки"}:
            ctx.notes_path.parent.mkdir(parents=True, exist_ok=True)
            if not ctx.notes_path.exists():
                ctx.notes_path.write_text("", encoding="utf-8")
            ok = ctx.executor.open_path(ctx.notes_path)
            return CommandResponse(text="Открываю заметки." if ok else "Не смог открыть заметки.")

        m = re.search(r"\bнайди\s+в\s+заметках\b\s*(.+)$", t)
        if m:
            q = m.group(1).strip()
            if not q:
                return CommandResponse(text="Скажи, что искать.")
            if not ctx.notes_path.exists():
                return CommandResponse(text="Заметок пока нет.")
            text_notes = ctx.notes_path.read_text(encoding="utf-8")
            hits = [line.strip() for line in text_notes.splitlines() if q.lower() in line.lower()]
            if not hits:
                return CommandResponse(text="Не нашёл.")
            return CommandResponse(text="Нашёл: " + "; ".join(hits[:4]) + ".")

        m = re.search(r"\bпогода\s+в\s+(.+)$", t)
        if m:
            city = m.group(1).strip()
            return CommandResponse(text=_weather(city, cache_path=ctx.weather_cache_path))

        if re.search(r"\bкакая\s+погода\b", t) or t == "погода":
            # try memory city first
            city = None
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                try:
                    city = ctx.storage.mem_get("city")
                except Exception:
                    city = None
            if not city:
                city = ctx.default_city
            if city:
                return CommandResponse(text=_weather(str(city), cache_path=ctx.weather_cache_path))
            return CommandResponse(text="Скажи город, например: «погода в Москве». ")

        return None
