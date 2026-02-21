from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jarvis.paths import get_paths


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(x: float) -> int:
    try:
        return int(max(0.0, min(1.0, float(x) / 0.2)) * 100)
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Показывает живой статус Jarvis в терминале")
    p.add_argument(
        "--status",
        default="auto",
        help='Путь к status.json (или "auto" для XDG ~/.local/state/jarvis/status.json)',
    )
    p.add_argument("--interval", type=float, default=0.2, help="Интервал обновления (сек)")
    args = p.parse_args(argv)

    status_arg = str(args.status).strip() if args.status is not None else ""
    if not status_arg or status_arg.lower() in ("auto", "xdg"):
        path = get_paths().status_path
    else:
        path = Path(status_arg).expanduser().resolve()
    last_ts = -1.0

    try:
        while True:
            data = _load(path)
            ts = float(data.get("ts", 0.0) or 0.0)
            if ts != last_ts:
                last_ts = ts
                state = str(data.get("state", "?"))
                msg = str(data.get("message", ""))
                mic = _pct(float(data.get("mic_level", 0.0) or 0.0))
                thr = float(data.get("recording_threshold", 0.0) or 0.0)
                heard = str(data.get("last_heard", "") or "")
                err = str(data.get("last_error", "") or "")

                lr = float(data.get("latency_record_ms", 0.0) or 0.0)
                ls = float(data.get("latency_stt_ms", 0.0) or 0.0)
                ll = float(data.get("latency_llm_ms", 0.0) or 0.0)
                lt = float(data.get("latency_tts_ms", 0.0) or 0.0)
                ltot = float(data.get("latency_total_ms", 0.0) or 0.0)

                if len(heard) > 60:
                    heard = heard[:57] + "…"
                if len(err) > 60:
                    err = err[:57] + "…"

                line = f"{state:<18} mic={mic:>3}% thr={thr:.3f} lat(ms) r={lr:.0f} s={ls:.0f} l={ll:.0f} t={lt:.0f} Σ={ltot:.0f}"
                if msg:
                    line += f" | {msg}"
                if heard:
                    line += f" | last: {heard}"
                if err:
                    line += f" | ERR: {err}"

                sys.stdout.write("\r" + line[:200].ljust(200))
                try:
                    sys.stdout.flush()
                except BrokenPipeError:
                    return 0

            time.sleep(float(args.interval))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 0
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
