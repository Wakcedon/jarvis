from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jarvis.commands import CommandResponse
from jarvis.executor import SystemExecutor
from jarvis.storage import SQLiteStorage


@dataclass(frozen=True)
class SkillContext:
    storage: SQLiteStorage | None
    storage_backend: str
    executor: SystemExecutor

    notes_path: Path
    memory_path: Path
    weather_cache_path: Path
    reminders_path: Path
    todo_path: Path
    screenshots_dir: Path

    default_browser_url: str
    default_city: str
    allowed_apps: set[str]
    allow_desktop_launch: bool
    allow_shell: bool


class Skill(Protocol):
    name: str

    def handle(self, text: str, *, ctx: SkillContext) -> CommandResponse | None:
        ...
