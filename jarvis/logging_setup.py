from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from jarvis.paths import get_paths


def setup_logging(*, name: str = "jarvis") -> logging.Logger:
    level_name = (os.environ.get("JARVIS_LOG_LEVEL") or "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    try:
        log_path = get_paths().state_dir / "jarvis.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            fh = RotatingFileHandler(
                str(log_path),
                maxBytes=1_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)
    except Exception:
        pass

    return logging.getLogger(name)
