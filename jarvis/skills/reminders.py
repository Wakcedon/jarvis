from __future__ import annotations

import json
import re
import time
from pathlib import Path

from jarvis.commands import CommandResponse
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command


def _read_items(path: Path) -> list[dict]:
    try:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8")) or {}
        items = raw.get("items") if isinstance(raw, dict) else None
        return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
    except Exception:
        return []


def _write_items(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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


class RemindersSkill:
    name = "reminders"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

        if t in {"покажи напоминания", "список напоминаний", "напоминания"}:
            now = time.time()
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                items = [x for x in ctx.storage.reminders_list() if not x.fired]
                if not items:
                    return CommandResponse(text="Активных напоминаний нет.")
                parts: list[str] = []
                for x in items[:6]:
                    rem = max(0.0, float(x.due_ts) - now)
                    parts.append(f"{x.id}) через {_fmt_duration(rem)}: {x.text}")
                return CommandResponse(text=". ".join(parts) + ".")

            items = [x for x in _read_items(ctx.reminders_path) if not x.get("fired")]
            if not items:
                return CommandResponse(text="Активных напоминаний нет.")
            items.sort(key=lambda x: float(x.get("due_ts") or 0.0))
            parts: list[str] = []
            for x in items[:6]:
                due = float(x.get("due_ts") or 0.0)
                rem = max(0.0, due - now)
                parts.append(f"{x.get('id')}) через {_fmt_duration(rem)}: {x.get('text')}")
            return CommandResponse(text=". ".join(parts) + ".")

        m = re.search(
            r"\bнапомни\b\s+через\s+(\d+)\s*(секунд[уы]?|минут[уы]?|час(?:а|ов)?)\b\s*(.+)?$",
            t,
        )
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            msg = (m.group(3) or "").strip() or "напоминание"
            mult = 1.0
            if unit.startswith("мин"):
                mult = 60.0
            elif unit.startswith("час"):
                mult = 3600.0
            seconds = float(n) * mult
            if seconds <= 0:
                return CommandResponse(text="Скажи длительность больше нуля.")
            due_ts = time.time() + seconds

            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                rid = ctx.storage.reminders_add(due_ts=due_ts, text=msg)
            else:
                items = _read_items(ctx.reminders_path)
                rid = max([int(x.get("id") or 0) for x in items] + [0]) + 1
                items.append({"id": rid, "text": msg, "due_ts": float(due_ts), "created_ts": time.time(), "fired": False})
                _write_items(ctx.reminders_path, items)

            return CommandResponse(text=f"Ок. Напомню через {_fmt_duration(seconds)} (номер {rid}).")

        m = re.search(r"\bудали\s+напоминани[ея]\b\s*(\d+)\b", t)
        if m:
            rid = int(m.group(1))
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                ok = ctx.storage.reminders_delete(rid)
            else:
                items = _read_items(ctx.reminders_path)
                new_items = [x for x in items if int(x.get("id") or -1) != rid]
                ok = len(new_items) != len(items)
                if ok:
                    _write_items(ctx.reminders_path, new_items)
            return CommandResponse(text="Удалил." if ok else "Не нашёл такое напоминание.")

        return None
