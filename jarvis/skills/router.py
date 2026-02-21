from __future__ import annotations

from dataclasses import dataclass

from jarvis.skills.base import Skill, SkillContext
from jarvis.commands import CommandResponse


@dataclass
class SkillRouter:
    skills: list[Skill]

    def handle(self, text: str, *, ctx: SkillContext) -> CommandResponse | None:
        for s in self.skills:
            try:
                out = s.handle(text, ctx=ctx)
            except Exception:
                continue
            if out is not None:
                return out
        return None
