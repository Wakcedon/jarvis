from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices() -> list[AudioDevice]:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        devices.append(
            AudioDevice(
                index=int(idx),
                name=str(dev.get("name", "?")),
                hostapi=str(sd.query_hostapis(int(dev.get("hostapi", 0))).get("name", "?")),
                max_input_channels=int(dev.get("max_input_channels", 0) or 0),
                max_output_channels=int(dev.get("max_output_channels", 0) or 0),
                default_samplerate=float(dev.get("default_samplerate", 0.0) or 0.0),
            )
        )
    return devices


def mic_test(*, seconds: float = 5.0, sample_rate: int = 16000, device: int | str | None = None, blocksize: int = 1600):
    """Reads raw mic for a few seconds and returns (avg_rms, peak_rms, samples_count)."""

    seconds = float(max(0.5, seconds))
    sample_rate = int(sample_rate)
    blocksize = int(blocksize)

    end_at = time.monotonic() + seconds
    rms_values: list[float] = []

    def _run(sr: int, bs: int) -> None:
        nonlocal rms_values
        with sd.RawInputStream(
            samplerate=int(sr),
            device=device,
            dtype="int16",
            channels=1,
            blocksize=int(bs),
        ) as stream:
            while time.monotonic() < end_at:
                data, _overflowed = stream.read(int(bs))
                samples = np.frombuffer(bytes(data), dtype=np.int16)
                if samples.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
                rms_values.append(rms)

    try:
        _run(sample_rate, blocksize)
    except Exception:
        # Fallback to device default samplerate (common: 48000 for USB mics)
        try:
            dev_info = sd.query_devices(kind="input") if device is None else sd.query_devices(device)
            fallback_sr = int(float(dev_info.get("default_samplerate", 0.0) or 0.0) or sample_rate)
        except Exception:
            fallback_sr = sample_rate
        # Keep same time duration per block.
        block_dur_s = int(blocksize) / float(max(1, int(sample_rate)))
        fallback_bs = max(1, int(round(int(fallback_sr) * block_dur_s)))
        _run(int(fallback_sr), int(fallback_bs))

    if not rms_values:
        return 0.0, 0.0, 0

    avg = float(np.mean(rms_values))
    peak = float(np.max(rms_values))
    return avg, peak, len(rms_values)
