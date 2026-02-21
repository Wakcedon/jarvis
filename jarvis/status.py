from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StatusSnapshot:
    state: str = "starting"  # starting|idle|listening_wakeword|recording|transcribing|thinking|speaking|error
    message: str = ""
    mic_level: float = 0.0
    last_update_ts: float = 0.0
    last_heard: str = ""
    last_heard_ts: float = 0.0
    last_answer: str = ""
    last_answer_ts: float = 0.0
    last_error: str = ""
    recording_threshold: float = 0.0
    recording_gain: float = 1.0
    latency_record_ms: float = 0.0
    latency_stt_ms: float = 0.0
    latency_llm_ms: float = 0.0
    latency_tts_ms: float = 0.0
    latency_total_ms: float = 0.0


class StatusWriter:
    def __init__(self, *, path: Path, min_interval_s: float = 0.5) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._min_interval_s = min_interval_s
        self._snapshot = StatusSnapshot(last_update_ts=0.0)

    def update(
        self,
        *,
        state: str | None = None,
        message: str | None = None,
        mic_level: float | None = None,
        last_heard: str | None = None,
        last_answer: str | None = None,
        last_error: str | None = None,
        recording_threshold: float | None = None,
        recording_gain: float | None = None,
        latency_record_ms: float | None = None,
        latency_stt_ms: float | None = None,
        latency_llm_ms: float | None = None,
        latency_tts_ms: float | None = None,
        latency_total_ms: float | None = None,
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and (now - self._snapshot.last_update_ts) < self._min_interval_s:
            if mic_level is not None:
                self._snapshot.mic_level = float(mic_level)
            return

        if state is not None:
            self._snapshot.state = str(state)
        if message is not None:
            self._snapshot.message = str(message)
        if mic_level is not None:
            self._snapshot.mic_level = float(mic_level)

        now_ts = now
        if last_heard is not None:
            self._snapshot.last_heard = str(last_heard)
            self._snapshot.last_heard_ts = now_ts
        if last_answer is not None:
            self._snapshot.last_answer = str(last_answer)
            self._snapshot.last_answer_ts = now_ts
        if last_error is not None:
            self._snapshot.last_error = str(last_error)
        if recording_threshold is not None:
            self._snapshot.recording_threshold = float(recording_threshold)
        if recording_gain is not None:
            self._snapshot.recording_gain = float(recording_gain)

        if latency_record_ms is not None:
            self._snapshot.latency_record_ms = float(latency_record_ms)
        if latency_stt_ms is not None:
            self._snapshot.latency_stt_ms = float(latency_stt_ms)
        if latency_llm_ms is not None:
            self._snapshot.latency_llm_ms = float(latency_llm_ms)
        if latency_tts_ms is not None:
            self._snapshot.latency_tts_ms = float(latency_tts_ms)
        if latency_total_ms is not None:
            self._snapshot.latency_total_ms = float(latency_total_ms)

        self._snapshot.last_update_ts = now
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "state": self._snapshot.state,
                    "message": self._snapshot.message,
                    "mic_level": self._snapshot.mic_level,
                    "recording_threshold": self._snapshot.recording_threshold,
                    "recording_gain": self._snapshot.recording_gain,
                    "last_heard": self._snapshot.last_heard,
                    "last_heard_ts": self._snapshot.last_heard_ts,
                    "last_answer": self._snapshot.last_answer,
                    "last_answer_ts": self._snapshot.last_answer_ts,
                    "last_error": self._snapshot.last_error,
                    "latency_record_ms": self._snapshot.latency_record_ms,
                    "latency_stt_ms": self._snapshot.latency_stt_ms,
                    "latency_llm_ms": self._snapshot.latency_llm_ms,
                    "latency_tts_ms": self._snapshot.latency_tts_ms,
                    "latency_total_ms": self._snapshot.latency_total_ms,
                    "ts": self._snapshot.last_update_ts,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self._path)
