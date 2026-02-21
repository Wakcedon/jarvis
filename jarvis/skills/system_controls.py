from __future__ import annotations

import re

from jarvis.commands import CommandResponse, PendingSystemActionResponse
from jarvis.skills.base import SkillContext
from jarvis.skills.text import normalize_command

# reuse these helpers (behavior is already tested in this project)
from jarvis.commands import _convert_units, _safe_calc  # type: ignore


class SystemControlsSkill:
    name = "system-controls"

    def handle(self, text: str, *, ctx: SkillContext):
        t = normalize_command(text)
        if not t:
            return None

        # --- Screenshots ---
        if t in {"скриншот", "сделай скриншот", "скрин"}:
            p = ctx.executor.take_screenshot(ctx.screenshots_dir, mode="full")
            return CommandResponse(text="Скриншот сделал." if p else "Не смог сделать скриншот.")

        if t in {"скриншот окна", "скрин окна"}:
            p = ctx.executor.take_screenshot(ctx.screenshots_dir, mode="window")
            return CommandResponse(text="Скриншот окна сделал." if p else "Не смог сделать скриншот окна.")

        # --- Volume ---
        if t in {"громче", "погромче", "сделай громче"}:
            ok = ctx.executor.volume_step(+5)
            return CommandResponse(text="Сделал громче." if ok else "Не смог изменить громкость.")

        if t in {"тише", "потише", "сделай тише"}:
            ok = ctx.executor.volume_step(-5)
            return CommandResponse(text="Сделал тише." if ok else "Не смог изменить громкость.")

        m = re.search(r"\bгромкост[ьи]\s+(\d{1,3})\b", t)
        if m:
            val = max(0, min(150, int(m.group(1))))
            ok = ctx.executor.volume_set_percent(val)
            return CommandResponse(text=f"Ставлю громкость {val}%." if ok else "Не смог выставить громкость.")

        if t in {"выключи звук", "мут", "без звука", "выруби звук"}:
            ok = ctx.executor.volume_mute(True)
            return CommandResponse(text="Ок. Без звука." if ok else "Не смог выключить звук.")

        if t in {"включи звук", "убери мут", "со звуком"}:
            ok = ctx.executor.volume_mute(False)
            return CommandResponse(text="Ок." if ok else "Не смог включить звук.")

        # --- Mic mute ---
        if t in {"выключи микрофон", "выруби микрофон", "замуть микрофон"}:
            ok = ctx.executor.mic_mute(True)
            return CommandResponse(text="Ок. Микрофон выключен." if ok else "Не смог выключить микрофон.")

        if t in {"включи микрофон", "размуть микрофон"}:
            ok = ctx.executor.mic_mute(False)
            return CommandResponse(text="Ок. Микрофон включен." if ok else "Не смог включить микрофон.")

        # --- Wi‑Fi / Bluetooth ---
        if t in {"выключи вайфай", "выключи wi fi", "вайфай офф", "wifi off"}:
            ok = ctx.executor.set_radio("wifi", False)
            return CommandResponse(text="Wi‑Fi выключен." if ok else "Не смог выключить Wi‑Fi.")
        if t in {"включи вайфай", "включи wi fi", "вайфай он", "wifi on"}:
            ok = ctx.executor.set_radio("wifi", True)
            return CommandResponse(text="Wi‑Fi включен." if ok else "Не смог включить Wi‑Fi.")
        if t in {"выключи блютуз", "выключи bluetooth", "блютуз офф"}:
            ok = ctx.executor.set_radio("bluetooth", False)
            return CommandResponse(text="Bluetooth выключен." if ok else "Не смог выключить Bluetooth.")
        if t in {"включи блютуз", "включи bluetooth", "блютуз он"}:
            ok = ctx.executor.set_radio("bluetooth", True)
            return CommandResponse(text="Bluetooth включен." if ok else "Не смог включить Bluetooth.")

        # --- Window keys (xdotool) ---
        if t in {"закрой окно", "закрой текущее окно"}:
            ok = ctx.executor.xdotool_key("alt+f4")
            return CommandResponse(text="Закрываю окно." if ok else "Не могу: запрещено политикой или нет xdotool.")
        if t in {"следующее окно", "переключи окно", "альт таб"}:
            ok = ctx.executor.xdotool_key("alt+tab")
            return CommandResponse(text="Переключаю." if ok else "Не могу: запрещено политикой или нет xdotool.")

        # --- Media (playerctl) ---
        if t in {"пауза", "поставь на паузу"}:
            ok = ctx.executor.playerctl("pause")
            return CommandResponse(text="Ок." if ok else "Не могу: запрещено политикой или нет playerctl.")
        if t in {"продолжай", "играй", "воспроизведение"}:
            ok = ctx.executor.playerctl("play")
            return CommandResponse(text="Ок." if ok else "Не могу: запрещено политикой или нет playerctl.")
        if t in {"следующий трек", "следующая песня"}:
            ok = ctx.executor.playerctl("next")
            return CommandResponse(text="Ок." if ok else "Не могу: запрещено политикой или нет playerctl.")
        if t in {"предыдущий трек", "предыдущая песня"}:
            ok = ctx.executor.playerctl("previous")
            return CommandResponse(text="Ок." if ok else "Не могу: запрещено политикой или нет playerctl.")

        # --- Clipboard ---
        m = re.search(r"\bскопируй\b\s*(.+)$", t)
        if m:
            payload = m.group(1).strip()
            if not payload:
                return CommandResponse(text="Скажи, что скопировать.")
            ok = ctx.executor.clipboard_set(payload)
            return CommandResponse(text="Скопировал в буфер." if ok else "Не смог записать в буфер.")

        if t in {"что в буфере", "прочитай буфер", "буфер"}:
            s = ctx.executor.clipboard_get()
            return CommandResponse(text=s if s else "Буфер пуст или недоступен.")

        # --- Calc / convert ---
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

        # --- Power (requires confirmation in app) ---
        if t in {"выключи компьютер", "выключи пк", "выключай компьютер", "выключи систему"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, выключай». ", action="poweroff")
        if t in {"перезагрузи компьютер", "перезагрузи пк", "перезагрузка"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, перезагружай». ", action="reboot")
        if t in {"сон", "спящий режим", "усыпи компьютер", "приостанови"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, в сон». ", action="suspend")
        if t in {"заблокируй", "заблокируй экран", "лок"}:
            return PendingSystemActionResponse(text="Подтверди: скажи «да, блокируй». ", action="lock")

        # --- Shell (policy guarded in executor) ---
        if t.startswith("выполни ") and ctx.allow_shell:
            cmd = t.removeprefix("выполни ").strip()
            if cmd:
                ok = ctx.executor.run_shell(cmd)
                return CommandResponse(text="Выполняю." if ok else "Запрещено политикой.")

        return None
