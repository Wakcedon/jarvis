from __future__ import annotations

import re


_HOTWORDS = (
    "погода",
    "заметка",
    "заметку",
    "задача",
    "задачи",
    "напоминание",
    "напоминания",
    "таймер",
    "громкость",
    "микрофон",
    "браузер",
    "включи",
    "выключи",
    "пауза",
    "продолжай",
)


def _levenshtein_1(a: str, b: str) -> bool:
    """True if edit distance <= 1 (fast path for short words)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # substitution
    if la == lb:
        diff = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diff += 1
                if diff > 1:
                    return False
        return True
    # insertion/deletion
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    used = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if used:
            return False
        used = True
        j += 1
    return True


def _fix_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if len(tok) < 5:
            out.append(tok)
            continue
        if tok in _HOTWORDS:
            out.append(tok)
            continue

        best = None
        for w in _HOTWORDS:
            if _levenshtein_1(tok, w):
                best = w
                break
        out.append(best or tok)
    return out


def normalize_command(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s\u00A0]+", " ", t)
    t = re.sub(r"[\.!\?…,:;]+$", "", t).strip()
    if not t:
        return t
    # lightweight fuzzy fix for common command words
    tokens = t.split()
    tokens = _fix_tokens(tokens)
    return " ".join(tokens).strip()
