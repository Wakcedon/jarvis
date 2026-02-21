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


def _fmt(items: list[dict]) -> str:
    if not items:
        return "Список задач пуст."
    undone = [x for x in items if not x.get("done")]
    done = [x for x in items if x.get("done")]
    parts: list[str] = []
    if undone:
        parts.append("Активные: " + "; ".join(f"{x.get('id')}) {x.get('text')}" for x in undone[:6]))
    if done:
        parts.append("Готово: " + "; ".join(f"{x.get('id')}) {x.get('text')}" for x in done[:4]))
    return ". ".join(parts) + "."


class TodoSkill:
    name = "todo"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

        if t in {"покажи задачи", "список задач", "todo", "туду", "дела"}:
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                items = [
                    {"id": it.id, "text": it.text, "done": it.done, "ts": it.created_ts}
                    for it in ctx.storage.todo_list()
                ]
                return CommandResponse(text=_fmt(items))
            return CommandResponse(text=_fmt(_read_items(ctx.todo_path)))

        m = re.search(r"\bдобавь\s+задач[ау]\b\s*(.+)$", t)
        if m:
            item_text = m.group(1).strip()
            if not item_text:
                return CommandResponse(text="Скажи текст задачи после «добавь задачу». ")
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                new_id = ctx.storage.todo_add(item_text)
                return CommandResponse(text=f"Добавил задачу {new_id}.")
            items = _read_items(ctx.todo_path)
            new_id = max([int(x.get("id") or 0) for x in items] + [0]) + 1
            items.append({"id": new_id, "text": item_text, "done": False, "ts": time.time()})
            _write_items(ctx.todo_path, items)
            return CommandResponse(text=f"Добавил задачу {new_id}.")

        m = re.search(r"\b(сделано|выполнено|закрой)\s+задач[ау]\b\s*(\d+)\b", t)
        if m:
            task_id = int(m.group(2))
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                ok = ctx.storage.todo_mark_done(task_id, done=True)
                return CommandResponse(text="Отметил как выполненную." if ok else "Не нашёл такую задачу.")
            items = _read_items(ctx.todo_path)
            ok = False
            for x in items:
                if int(x.get("id") or -1) == task_id:
                    x["done"] = True
                    ok = True
            if ok:
                _write_items(ctx.todo_path, items)
            return CommandResponse(text="Отметил как выполненную." if ok else "Не нашёл такую задачу.")

        m = re.search(r"\bудали\s+задач[ау]\b\s*(\d+)\b", t)
        if m:
            task_id = int(m.group(1))
            if ctx.storage_backend == "sqlite" and ctx.storage is not None:
                ok = ctx.storage.todo_delete(task_id)
                return CommandResponse(text="Удалил." if ok else "Не нашёл такую задачу.")
            items = _read_items(ctx.todo_path)
            new_items = [x for x in items if int(x.get("id") or -1) != task_id]
            if len(new_items) == len(items):
                return CommandResponse(text="Не нашёл такую задачу.")
            _write_items(ctx.todo_path, new_items)
            return CommandResponse(text="Удалил.")

        return None
