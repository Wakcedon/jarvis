from __future__ import annotations

import configparser
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DESKTOP_DIRS = (
    Path("/usr/share/applications"),
    Path.home() / ".local/share/applications",
    Path("/var/lib/snapd/desktop/applications"),
)


@dataclass(frozen=True)
class DesktopApp:
    desktop_id: str
    name: str
    generic_name: str
    keywords: tuple[str, ...]


def _iter_desktop_files() -> Iterable[Path]:
    for d in DESKTOP_DIRS:
        if d.exists():
            yield from d.glob("*.desktop")


def _parse_desktop(path: Path) -> DesktopApp | None:
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read(path, encoding="utf-8")
    except Exception:
        return None

    if "Desktop Entry" not in cp:
        return None

    e = cp["Desktop Entry"]
    if e.get("NoDisplay", "false").strip().lower() == "true":
        return None
    if e.get("Hidden", "false").strip().lower() == "true":
        return None
    if e.get("Type", "Application").strip() != "Application":
        return None

    name = (e.get("Name") or "").strip()
    if not name:
        return None
    generic = (e.get("GenericName") or "").strip()
    keywords = tuple(k.strip() for k in (e.get("Keywords") or "").split(";") if k.strip())
    desktop_id = path.stem
    return DesktopApp(desktop_id=desktop_id, name=name, generic_name=generic, keywords=keywords)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def find_app(query: str) -> DesktopApp | None:
    q = _normalize(query)
    if not q:
        return None

    best: tuple[float, DesktopApp] | None = None
    for f in _iter_desktop_files():
        app = _parse_desktop(f)
        if app is None:
            continue

        hay = " ".join(
            [
                _normalize(app.desktop_id),
                _normalize(app.name),
                _normalize(app.generic_name),
                " ".join(_normalize(k) for k in app.keywords),
            ]
        )

        score = 0.0
        if q == _normalize(app.desktop_id):
            score = 10.0
        elif q == _normalize(app.name):
            score = 9.0
        elif q in hay:
            score = 7.0
        else:
            # токены
            qt = set(q.split())
            ht = set(hay.split())
            if qt and ht:
                score = 3.0 * (len(qt & ht) / len(qt))

        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, app)

    return best[1] if best else None


def launch_app(app: DesktopApp) -> bool:
    # предпочтительно gtk-launch
    try:
        r = subprocess.run(["gtk-launch", app.desktop_id], check=False)
        return r.returncode == 0
    except Exception:
        return False
