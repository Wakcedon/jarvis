from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass
from collections import deque
from typing import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import shutil
import subprocess
from tempfile import NamedTemporaryFile


def beep(sample_rate: int, output_device: int | str | None = None, duration_s: float = 0.12, freq_hz: float = 880.0) -> None:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = 0.2 * np.sin(2.0 * math.pi * freq_hz * t)
    sd.play(wave.astype(np.float32), samplerate=sample_rate, device=output_device)
    sd.wait()


def play_wav(path: Path, output_device: int | str | None = None) -> None:
    # Читаем WAV и делаем safety-обработку: лимитер + короткий fade,
    # чтобы убрать щелчки/артефакты и не допустить слишком громкого шума.
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(data, np.ndarray) and data.size:
        x = data
        if x.ndim > 1:
            x = x.reshape(-1)

        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.95:
            x = x / max(peak, 1e-6) * 0.85

        fade_s = 0.01
        n = int(sr * fade_s)
        if n > 2 and x.size > (n * 2):
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)
            x[:n] *= w
            x[-n:] *= w[::-1]

        data = x.astype(np.float32, copy=False)

    # Предпочитаем системные плееры (PulseAudio/PipeWire/ALSA)
    if output_device is None and (shutil.which("paplay") or shutil.which("aplay")):
        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, data, sr, subtype="PCM_16")
            if shutil.which("paplay"):
                subprocess.run(["paplay", tmp.name], check=False)
                return
            subprocess.run(["aplay", "-q", tmp.name], check=False)
            return

    # Fallback: sounddevice
    sd.play(data, sr, device=output_device)
    sd.wait()


@dataclass(frozen=True)
class RecordedAudio:
    pcm16: np.ndarray  # shape: (n,)
    sample_rate: int


def record_utterance(
    *,
    sample_rate: int,
    input_device: int | str | None = None,
    max_s: float = 12.0,
    start_threshold: float = 0.015,
    end_silence_s: float = 1.0,
    start_immediately: bool = False,
    pre_roll_s: float = 0.25,
    auto_threshold: bool = False,
    noise_calibrate_s: float = 0.4,
    threshold_multiplier: float = 3.5,
    chunk_ms: int = 30,
    on_rms: Callable[[float], None] | None = None,
    auto_gain: bool = True,
    target_rms: float = 0.03,
    max_gain: float = 25.0,
) -> RecordedAudio:
    """Записывает фразу: ждёт начала речи по энергии, затем пишет до паузы."""

    chunk = int(sample_rate * (chunk_ms / 1000.0))
    audio_chunks: list[np.ndarray] = []

    started = False
    silence_s = 0.0
    start_time = time.monotonic()

    # Кольцевой буфер для "начала фразы" (чтобы не отрезать первые слоги)
    pre_roll_chunks: deque[np.ndarray] = deque(
        maxlen=max(1, int((max(0.0, pre_roll_s) * sample_rate) / max(1, chunk)))
    )

    effective_threshold = float(start_threshold)
    calibrate_until = start_time + float(max(0.0, noise_calibrate_s)) if auto_threshold else start_time
    noise_rms: list[float] = []
    calibrated = False

    record_utterance.last_started = False  # type: ignore[attr-defined]
    record_utterance.last_threshold = effective_threshold  # type: ignore[attr-defined]

    started_ts: float | None = None
    peak_rms = 0.0
    min_after_start_s = 0.25

    with sd.RawInputStream(
        samplerate=sample_rate,
        device=input_device,
        dtype="int16",
        channels=1,
        blocksize=chunk,
    ) as stream:
        while True:
            if time.monotonic() - start_time > max_s:
                break

            data, _overflowed = stream.read(chunk)
            data_bytes = bytes(data)
            samples = np.frombuffer(data_bytes, dtype=np.int16)
            if samples.size == 0:
                continue

            pre_roll_chunks.append(samples.copy())

            # rms в шкале 0..1
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
            # отдаём "живой" уровень микрофона через атрибут на функции (простая интеграция без колбэка)
            record_utterance.last_rms = rms  # type: ignore[attr-defined]
            if on_rms is not None:
                try:
                    on_rms(rms)
                except Exception:
                    pass

            # После wake word лучше начинать запись сразу: пороги часто промахиваются.
            if start_immediately and not started:
                started = True
                started_ts = time.monotonic()
                record_utterance.last_started = True  # type: ignore[attr-defined]
                audio_chunks.extend(list(pre_roll_chunks))
                pre_roll_chunks.clear()
                continue

            # Автокалибровка порога старта по шуму помещения (до начала речи)
            if auto_threshold and (not started) and time.monotonic() < calibrate_until:
                # Если вдруг пользователь начал говорить во время калибровки — не игнорируем.
                if rms >= float(start_threshold):
                    calibrated = True
                else:
                    noise_rms.append(rms)
                    continue

            if auto_threshold and (not started) and (not calibrated):
                baseline = float(np.mean(noise_rms)) if noise_rms else 0.0
                effective_threshold = max(float(start_threshold), baseline * float(threshold_multiplier))
                calibrated = True
                record_utterance.last_threshold = effective_threshold  # type: ignore[attr-defined]

            if not started:
                if rms >= effective_threshold:
                    started = True
                    started_ts = time.monotonic()
                    record_utterance.last_started = True  # type: ignore[attr-defined]
                    record_utterance.last_threshold = effective_threshold  # type: ignore[attr-defined]
                    audio_chunks.extend(list(pre_roll_chunks))
                    pre_roll_chunks.clear()
                continue

            audio_chunks.append(samples.copy())

            peak_rms = max(peak_rms, rms)

            # Тишина после старта: адаптируем порог под реальный уровень речи,
            # иначе на тихих микрофонах запись обрывается мгновенно.
            silence_threshold = max(0.002, peak_rms * 0.25, effective_threshold * 0.25)

            if started_ts is not None and (time.monotonic() - started_ts) < min_after_start_s:
                continue

            if rms < silence_threshold:
                silence_s += chunk / sample_rate
                if silence_s >= end_silence_s:
                    break
            else:
                silence_s = 0.0

    if not audio_chunks:
        return RecordedAudio(pcm16=np.zeros((0,), dtype=np.int16), sample_rate=sample_rate)

    pcm16 = np.concatenate(audio_chunks, axis=0).astype(np.int16, copy=False)

    # Авто-усиление: полезно для тихих микрофонов/low gain.
    record_utterance.last_gain = 1.0  # type: ignore[attr-defined]
    if auto_gain and pcm16.size:
        x = pcm16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms > 0:
            gain = min(float(max_gain), float(target_rms) / (rms + 1e-9))
            if gain > 1.2:
                y = np.clip(x * gain, -1.0, 1.0)
                pcm16 = (y * 32767.0).astype(np.int16)
                record_utterance.last_gain = float(gain)  # type: ignore[attr-defined]

    return RecordedAudio(pcm16=pcm16, sample_rate=sample_rate)


# динамическое поле для статуса (см. record_utterance)
record_utterance.last_rms = 0.0  # type: ignore[attr-defined]


def raw_mic_stream(
    *,
    sample_rate: int,
    input_device: int | str | None,
    blocksize: int,
):
    """Генератор байтов int16 из микрофона (для Vosk)."""

    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # не печатаем в stdout, чтобы не мешать логам; можно расширить позже
            pass
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=blocksize,
        device=input_device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            yield q.get()
