from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_path, user_data_path, user_state_path


@dataclass(frozen=True)
class JarvisPaths:
    root: Path
    config_dir: Path
    data_dir: Path
    state_dir: Path

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.sqlite3"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"


def get_root() -> Path:
    env = os.environ.get("JARVIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def get_paths(*, app_name: str = "jarvis") -> JarvisPaths:
    root = get_root()
    cfg = user_config_path(app_name, ensure_exists=True)
    data = user_data_path(app_name, ensure_exists=True)
    state = user_state_path(app_name, ensure_exists=True)
    return JarvisPaths(root=root, config_dir=Path(cfg), data_dir=Path(data), state_dir=Path(state))


def find_config_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    env = os.environ.get("JARVIS_CONFIG")
    if env:
        return Path(env).expanduser().resolve()

    p = get_paths().config_path
    if p.exists():
        return p

    # portable fallback
    candidate = get_root() / "config.yaml"
    return candidate


def ensure_default_config(*, template_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    if template_path.exists():
        dest_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
