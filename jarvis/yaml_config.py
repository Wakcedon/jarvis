from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _ensure_mapping(root: Any) -> dict[str, Any]:
    if root is None:
        return {}
    if isinstance(root, dict):
        return root
    return {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    y = _yaml()
    with path.open("r", encoding="utf-8") as f:
        data = y.load(f)
    return _ensure_mapping(data)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = _yaml()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        y.dump(data, f)
    tmp.replace(path)


def get_dotted_bool(path: Path, dotted_key: str, default: bool) -> bool:
    try:
        section, key = dotted_key.split(".", 1)
    except ValueError:
        return default

    data = read_yaml(path)
    sec = data.get(section)
    if not isinstance(sec, dict):
        return default
    val = sec.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "1", "on")
    return bool(val) if isinstance(val, (int, float)) else default


def set_dotted_bool(path: Path, dotted_key: str, value: bool) -> None:
    try:
        section, key = dotted_key.split(".", 1)
    except ValueError:
        raise ValueError("dotted_key must be like 'section.key'")

    data = read_yaml(path)
    sec = data.get(section)
    if not isinstance(sec, dict):
        sec = {}
        data[section] = sec
    sec[key] = bool(value)
    write_yaml(path, data)


def get_dotted(path: Path, dotted_key: str, default: Any = None) -> Any:
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        return default
    data: Any = read_yaml(path)
    cur: Any = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _parse_scalar(s: str) -> Any:
    low = s.strip().lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "yes", "on", "1"):
        return True
    if low in ("false", "no", "off", "0"):
        return False
    try:
        if "." in low or "," in low:
            return float(low.replace(",", "."))
        return int(low)
    except Exception:
        return s


def set_dotted(path: Path, dotted_key: str, value: Any) -> None:
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        raise ValueError("dotted_key must be non-empty")
    data = read_yaml(path)
    cur: Any = data
    for p in parts[:-1]:
        nxt = cur.get(p) if isinstance(cur, dict) else None
        if not isinstance(nxt, dict):
            nxt = {}
            if isinstance(cur, dict):
                cur[p] = nxt
        cur = nxt
    last = parts[-1]
    if isinstance(value, str):
        cur[last] = _parse_scalar(value)
    else:
        cur[last] = value
    write_yaml(path, data)
