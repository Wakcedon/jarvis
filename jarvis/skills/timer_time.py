from __future__ import annotations

import datetime as dt
import re

from jarvis.commands import CommandResponse, TimerCancelResponse, TimerSetResponse, TimerStatusRequest
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command


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


class TimerTimeSkill:
    name = "timer-time"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

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
            dow = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"][now.weekday()]
            month = [
                "января",
                "февраля",
                "марта",
                "апреля",
                "мая",
                "июня",
                "июля",
                "августа",
                "сентября",
                "октября",
                "ноября",
                "декабря",
            ][now.month - 1]
            return CommandResponse(text=f"Сегодня {now.day} {month} {now.year} года, {dow}.")

        if re.search(r"\bкак\s+дела\b", t):
            return CommandResponse(text="Нормально. Чем помочь?")

        return None
