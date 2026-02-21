from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from jarvis.audio import raw_mic_stream


@dataclass
class BargeInConfig:
    enabled: bool = True
    rms_threshold: float = 0.020
    min_hold_ms: int = 160
    cooldown_s: float = 1.0


class BargeInMonitor:
    def __init__(
        self,
        *,
        sample_rate: int,
        input_device: int | str | None,
        blocksize: int,
        cfg: BargeInConfig,
        is_speaking: Callable[[], bool],
        on_barge_in: Callable[[], None],
    ) -> None:
        self._sample_rate = int(sample_rate)
        self._input_device = input_device
        self._blocksize = int(blocksize)
        self._cfg = cfg
        self._is_speaking = is_speaking
        self._on_barge_in = on_barge_in
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jarvis-barge-in", daemon=True)

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        hold_needed = max(1, int((self._cfg.min_hold_ms / 1000.0) * (self._sample_rate / max(1, self._blocksize))))
        hold = 0
        last_fire = 0.0

        for chunk in raw_mic_stream(
            sample_rate=self._sample_rate,
            input_device=self._input_device,
            blocksize=self._blocksize,
        ):
            if self._stop.is_set():
                return

            if not self._is_speaking():
                hold = 0
                continue

            samples = np.frombuffer(chunk, dtype=np.int16)
            if samples.size == 0:
                continue
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
            if rms >= float(self._cfg.rms_threshold):
                hold += 1
            else:
                hold = 0

            if hold >= hold_needed:
                now = time.monotonic()
                if (now - last_fire) >= float(self._cfg.cooldown_s):
                    last_fire = now
                    try:
                        self._on_barge_in()
                    except Exception:
                        pass
                hold = 0
