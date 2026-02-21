from __future__ import annotations

from dataclasses import dataclass

from jarvis.commands import CommandRouter
from jarvis.commands import CommandResponse
from jarvis.skills.base import SkillContext


@dataclass
class BuiltinCommandsSkill:
    router: CommandRouter
    name: str = "builtin-commands"

    def handle(self, text: str, *, ctx: SkillContext) -> CommandResponse | None:
        return self.router.handle(text)
