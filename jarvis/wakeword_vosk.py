from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
import time

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
        min_confidence: float = 0.65,
        use_partial: bool = False,
        partial_hits: int = 3,
        cooldown_s: float = 1.2,
        confirm_window_s: float = 1.0,
        noise_gate_multiplier: float = 3.2,
        noise_ema_alpha: float = 0.02,
    ) -> None:
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._phrases = tuple(p.strip().lower() for p in phrases if p.strip())
        self._input_device = input_device
        self._blocksize = int(blocksize)
        self._min_rms = float(min_rms)
        self._min_confidence = float(min_confidence)
        self._use_partial = bool(use_partial)
        self._partial_hits = max(1, int(partial_hits))
        self._cooldown_s = float(max(0.0, cooldown_s))
        self._confirm_window_s = float(max(0.1, confirm_window_s))
        self._noise_gate_multiplier = float(max(1.0, noise_gate_multiplier))
        self._noise_ema_alpha = float(min(0.2, max(0.001, noise_ema_alpha)))
        self._last_trigger_ts = 0.0

        self._noise_ema = 0.0

        self._phrase_tokens = [tuple(p.split()) for p in self._phrases]

        self._model = Model(model_path)
        grammar = json.dumps(list(self._phrases), ensure_ascii=False)
        self._rec = KaldiRecognizer(self._model, sample_rate, grammar)
        self._rec.SetWords(True)

    def listen(self, *, on_mic_level: Callable[[float], None] | None = None) -> WakeWordDetected:
        partial_streak = 0
        last_partial_phrase: str | None = None
        pending_phrase: str | None = None
        pending_until = 0.0
        for chunk in raw_mic_stream(
            sample_rate=self._sample_rate,
            input_device=self._input_device,
            blocksize=self._blocksize,
        ):
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0) if samples.size else 0.0
            if on_mic_level is not None:
                on_mic_level(rms)

            if self._min_rms and rms < self._min_rms:
                continue

            # update noise floor EMA when signal is quiet-ish
            if self._noise_ema <= 0.0:
                self._noise_ema = rms
            if rms < (self._noise_ema * 1.4):
                self._noise_ema = (1.0 - self._noise_ema_alpha) * self._noise_ema + self._noise_ema_alpha * rms

            adaptive_gate = max(self._min_rms, self._noise_ema * self._noise_gate_multiplier)
            if rms < adaptive_gate:
                # too quiet relative to room noise
                continue

            # cooldown to reduce re-triggers on the same utterance
            now = time.monotonic()
            if self._cooldown_s and (now - self._last_trigger_ts) < self._cooldown_s:
                continue

            if self._rec.AcceptWaveform(chunk):
                obj = json.loads(self._rec.Result())
                text = str(obj.get("text", "") or "").strip().lower()
                tokens = tuple(text.split())

                conf = None
                try:
                    words = obj.get("result") or []
                    if isinstance(words, list) and words:
                        cs = [float(w.get("conf", 0.0) or 0.0) for w in words if isinstance(w, dict)]
                        if cs:
                            conf = sum(cs) / len(cs)
                except Exception:
                    conf = None

                for phrase, pt in zip(self._phrases, self._phrase_tokens):
                    if not phrase:
                        continue
                    if _contains_token_sequence(tokens, pt):
                        if conf is None or conf >= self._min_confidence:
                            # if we had a pending partial, require matching it
                            if pending_phrase is not None and phrase != pending_phrase:
                                continue
                            self._last_trigger_ts = time.monotonic()
                            return WakeWordDetected(phrase=phrase)
                partial_streak = 0
                last_partial_phrase = None
                pending_phrase = None
            else:
                if not self._use_partial:
                    continue
                # pending candidate expired
                now = time.monotonic()
                if pending_phrase is not None and now > pending_until:
                    pending_phrase = None
                    partial_streak = 0
                    last_partial_phrase = None
                partial = str(json.loads(self._rec.PartialResult()).get("partial", "") or "").strip().lower()
                tokens = tuple(partial.split())
                matched: str | None = None
                for phrase, pt in zip(self._phrases, self._phrase_tokens):
                    if phrase and _contains_token_sequence(tokens, pt):
                        matched = phrase
                        break
                if matched is None:
                    partial_streak = 0
                    last_partial_phrase = None
                    continue
                if last_partial_phrase != matched:
                    partial_streak = 0
                    last_partial_phrase = matched
                partial_streak += 1
                if partial_streak >= self._partial_hits:
                    # candidate: require confirmation by final result shortly
                    pending_phrase = matched
                    pending_until = time.monotonic() + self._confirm_window_s
                    partial_streak = 0

        raise RuntimeError("wake word stream ended unexpectedly")


def _contains_token_sequence(tokens: tuple[str, ...], phrase_tokens: tuple[str, ...]) -> bool:
    if not phrase_tokens:
        return False
    if not tokens:
        return False
    if len(phrase_tokens) > len(tokens):
        return False
    if tokens == phrase_tokens:
        return True
    # contiguous subsequence match
    n = len(phrase_tokens)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_tokens:
            return True
    return False


