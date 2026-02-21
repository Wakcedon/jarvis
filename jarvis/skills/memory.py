from __future__ import annotations

import json
import re
from pathlib import Path

from jarvis.commands import CommandResponse
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class MemorySkill:
    name = "memory"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

        m = re.search(r"\bзапомни\b.*\b(меня\s+зовут|имя)\s+(.+)$", t)
        if m:
            name = m.group(2).strip().strip(".!")
            if name:
                self._mem_set(ctx, "name", name)
                return CommandResponse(text=f"Хорошо. Запомнил: тебя зовут {name}.")

        if re.search(r"\bкак\s+меня\s+зовут\b", t):
            name = self._mem_get(ctx, "name")
            return CommandResponse(
                text=f"Тебя зовут {name}." if name else "Я пока не знаю твоё имя. Скажи: «запомни, меня зовут …»."
            )

        m = re.search(r"\bзапомни\b.*\b(мой\s+город|город)\s+(.+)$", t)
        if m:
            city = m.group(2).strip().strip(".!")
            if city:
                self._mem_set(ctx, "city", city)
                return CommandResponse(text=f"Запомнил. Твой город — {city}.")

        if re.search(r"\bкакой\s+мой\s+город\b", t):
            city = self._mem_get(ctx, "city")
            return CommandResponse(
                text=f"Твой город — {city}." if city else "Я пока не знаю твой город. Скажи: «запомни мой город …»."
            )

        if re.search(r"\bочисти\s+память\b", t):
            self._mem_clear(ctx)
            return CommandResponse(text="Память очищена.")

        return None

    def _mem_get(self, ctx: SkillContext, key: str) -> str | None:
        if ctx.storage_backend == "sqlite" and ctx.storage is not None:
            try:
                v = ctx.storage.mem_get(key)
                s = str(v).strip() if v is not None else ""
                return s or None
            except Exception:
                return None

        data = _read_json(ctx.memory_path)
        v = data.get(key)
        s = str(v).strip() if v is not None else ""
        return s or None

    def _mem_set(self, ctx: SkillContext, key: str, value: str) -> None:
        if ctx.storage_backend == "sqlite" and ctx.storage is not None:
            try:
                ctx.storage.mem_set(str(key), str(value))
                return
            except Exception:
                pass

        data = _read_json(ctx.memory_path)
        data[str(key)] = str(value)
        _write_json(ctx.memory_path, data)

    def _mem_clear(self, ctx: SkillContext) -> None:
        if ctx.storage_backend == "sqlite" and ctx.storage is not None:
            try:
                ctx.storage.mem_clear()
                return
            except Exception:
                pass
        try:
            if ctx.memory_path.exists():
                ctx.memory_path.unlink()
        except Exception:
            pass
