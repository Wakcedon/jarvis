from __future__ import annotations

import re
from num2words import num2words


_MONTHS_GEN = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


_ORDINAL_DAY_NEUTER: dict[int, str] = {
    1: "первое",
    2: "второе",
    3: "третье",
    4: "четвертое",
    5: "пятое",
    6: "шестое",
    7: "седьмое",
    8: "восьмое",
    9: "девятое",
    10: "десятое",
    11: "одиннадцатое",
    12: "двенадцатое",
    13: "тринадцатое",
    14: "четырнадцатое",
    15: "пятнадцатое",
    16: "шестнадцатое",
    17: "семнадцатое",
    18: "восемнадцатое",
    19: "девятнадцатое",
    20: "двадцатое",
    21: "двадцать первое",
    22: "двадцать второе",
    23: "двадцать третье",
    24: "двадцать четвертое",
    25: "двадцать пятое",
    26: "двадцать шестое",
    27: "двадцать седьмое",
    28: "двадцать восьмое",
    29: "двадцать девятое",
    30: "тридцатое",
    31: "тридцать первое",
}


def _int_words(n: int) -> str:
    if n < 0:
        return "минус " + num2words(abs(n), lang="ru")
    return num2words(n, lang="ru")


def normalize_for_tts(text: str) -> str:
    s = text
    s = s.replace("°C", " градусов")
    s = s.replace("°c", " градусов")
    s = s.replace("м/с", " метров в секунду")
    s = s.replace("м/сек", " метров в секунду")

    # Дата dd.mm.yyyy -> "двадцатое февраля две тысячи двадцать шесть года"
    def repl_date(m: re.Match[str]) -> str:
        dd = int(m.group(1))
        mm = int(m.group(2))
        yy = int(m.group(3))
        month = _MONTHS_GEN.get(mm)
        if not month:
            return m.group(0)
        day = _ORDINAL_DAY_NEUTER.get(dd, str(dd))
        year = _int_words(yy)
        return f"{day} {month} {year} года"

    s = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", repl_date, s)

    # День месяца: "20 февраля" -> "двадцатое февраля"
    months = "|".join(re.escape(m) for m in _MONTHS_GEN.values())

    def repl_day_month(m: re.Match[str]) -> str:
        dd = int(m.group(1))
        mon = m.group(2)
        day = _ORDINAL_DAY_NEUTER.get(dd)
        if not day:
            return m.group(0)
        return f"{day} {mon}"

    s = re.sub(rf"\b(\d{{1,2}})\s+({months})\b", repl_day_month, s, flags=re.IGNORECASE)

    # Температура/ветер: переводим целые числа в слова в контексте единиц
    def repl_temp(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return f"{_int_words(n)} градусов"

    s = re.sub(r"(?<!\d)(-?\d+)\s+градусов\b", repl_temp, s)

    def repl_wind(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return f"{_int_words(n)} метров в секунду"

    s = re.sub(r"(?<!\d)(-?\d+)\s+метров\s+в\s+секунду\b", repl_wind, s)

    # "2026 года" -> "две тысячи двадцать шестого года" (приближенно)
    def repl_year(m: re.Match[str]) -> str:
        yy = int(m.group(1))
        ord_year = num2words(yy, lang="ru", to="ordinal")  # "... шестой"
        # грубое склонение в род.п.
        ord_year = re.sub(r"(ый|ий|ой)$", "ого", ord_year)
        ord_year = re.sub(r"(ая)$", "ой", ord_year)
        ord_year = re.sub(r"(ое)$", "ого", ord_year)
        return f"{ord_year} года"

    s = re.sub(r"\b(\d{4})\s+года\b", repl_year, s)

    # ВАЖНО: у некоторых piper-голосов '. ' вызывает артефакты/шум.
    # Заменяем конечную пунктуацию на запятые (пауза без "выстрела" шума).
    s = re.sub(r"[\.!\?…]+", ",", s)
    s = re.sub(r"[,\s]+,", ",", s)
    s = re.sub(r",{2,}", ",", s)

    # Десятичные числа: 7.0 -> 7; -13.9 -> "минус тринадцать целых девять десятых"
    def repl_decimal(m: re.Match[str]) -> str:
        sign = "-" if m.group(1) else ""
        a = int(m.group(2))
        frac = m.group(3)
        if set(frac) == {"0"}:
            return f"{sign}{a}"

        # ограничим до 2 знаков для речи
        frac = frac[:2]
        b = int(frac)
        denom_word = "десятых" if len(frac) == 1 else "сотых"
        # "минус" отдельно, чтобы num2words не давал странных форм
        head = "минус " if sign else ""
        return f"{head}{num2words(a, lang='ru')} целых {num2words(b, lang='ru')} {denom_word}"

    s = re.sub(r"(?<!\d)(-)?(\d+)\.(\d+)", repl_decimal, s)

    # Единицы: "7 метров" (убрать точку в конце единицы, если появилось)
    s = re.sub(r"\s+градусов\b", " градусов", s)

    # финальная нормализация пробелов
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(,\s*)+$", "", s).strip()

    # "..., Чем" -> "..., чем" (чуть естественнее для TTS)
    s = re.sub(r",\s+([А-ЯЁ])", lambda m: ", " + m.group(1).lower(), s)
    return s
