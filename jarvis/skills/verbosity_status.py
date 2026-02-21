from __future__ import annotations

import re

from jarvis.commands import (
    RepeatLastHeardRequest,
    SpeakStatusRequest,
    VerbositySetResponse,
)
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command


class VerbosityStatusSkill:
    name = "verbosity-status"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

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

        # keep compatibility with extra punctuation
        if re.fullmatch(r"статус[.!?…]*", t):
            return SpeakStatusRequest(text="")

        return None
