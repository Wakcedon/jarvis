from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vosk import KaldiRecognizer, Model

from jarvis.audio import raw_mic_stream


@dataclass(frozen=True)
class WakeWordDetected:
    phrase: str


class VoskWakeWordDetector:
    def __init__(
        self,
        *,
        model_path: str,
        sample_rate: int,
        phrases: tuple[str, ...],
        input_device: int | str | None,
        blocksize: int = 1600,
        min_rms: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._phrases = tuple(p.strip().lower() for p in phrases if p.strip())
        self._input_device = input_device
        self._blocksize = int(blocksize)
        self._min_rms = float(min_rms)

        self._model = Model(model_path)
        grammar = json.dumps(list(self._phrases), ensure_ascii=False)
        self._rec = KaldiRecognizer(self._model, sample_rate, grammar)
        self._rec.SetWords(False)

    def listen(self, *, on_mic_level: Callable[[float], None] | None = None) -> WakeWordDetected:
        for chunk in raw_mic_stream(
            sample_rate=self._sample_rate,
            input_device=self._input_device,
            blocksize=self._blocksize,
        ):
            rms: float | None = None
            if on_mic_level is not None:
                samples = np.frombuffer(chunk, dtype=np.int16)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
                    on_mic_level(rms)

            if self._min_rms and rms is not None and rms < self._min_rms:
                continue
            if self._rec.AcceptWaveform(chunk):
                text = json.loads(self._rec.Result()).get("text", "").strip().lower()
                for phrase in self._phrases:
                    if phrase and phrase in text.split():
                        return WakeWordDetected(phrase=phrase)
            else:
                partial = json.loads(self._rec.PartialResult()).get("partial", "").strip().lower()
                for phrase in self._phrases:
                    if phrase and phrase in partial.split():
                        return WakeWordDetected(phrase=phrase)


