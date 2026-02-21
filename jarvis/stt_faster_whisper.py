from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class STTResult:
    text: str
    avg_logprob: float | None = None
    max_no_speech_prob: float | None = None


class FasterWhisperSTT:
    def __init__(
        self,
        *,
        model_size: str,
        model_path: Path | None,
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        language: str,
        beam_size: int,
        vad_filter: bool,
        initial_prompt: str,
        min_avg_logprob: float,
        max_no_speech_prob: float,
    ) -> None:
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._initial_prompt = str(initial_prompt)
        self._min_avg_logprob = float(min_avg_logprob)
        self._max_no_speech_prob = float(max_no_speech_prob)

        model_source: str | Path = model_size
        if model_path is not None:
            if not model_path.exists():
                raise RuntimeError(
                    f"Whisper-модель не найдена по пути: {model_path}. "
                    "Запустите scripts/download_models.sh (он скачает Whisper локально), "
                    "или очистите stt.model_path чтобы разрешить авто-скачивание."
                )
            model_source = model_path

        self._model = WhisperModel(
            str(model_source),
            device=device,
            compute_type=compute_type,
            cpu_threads=int(cpu_threads),
            num_workers=int(num_workers),
        )

    def transcribe_pcm16(self, pcm16: np.ndarray, sample_rate: int) -> STTResult:
        if pcm16.size == 0:
            return STTResult(text="")

        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, pcm16, sample_rate, subtype="PCM_16")

            segments, _info = self._model.transcribe(
                tmp.name,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                initial_prompt=self._initial_prompt,
                condition_on_previous_text=False,
            )
            parts: list[str] = []
            logps: list[float] = []
            nospeech: list[float] = []

            for seg in segments:
                parts.append(str(getattr(seg, "text", "")))
                lp = getattr(seg, "avg_logprob", None)
                if lp is not None:
                    try:
                        logps.append(float(lp))
                    except Exception:
                        pass
                nsp = getattr(seg, "no_speech_prob", None)
                if nsp is not None:
                    try:
                        nospeech.append(float(nsp))
                    except Exception:
                        pass

            text = "".join(parts).strip()
            avg_lp = (sum(logps) / len(logps)) if logps else None
            max_nsp = max(nospeech) if nospeech else None

            # Фильтр от явной "тишины"/галлюцинаций: лучше переспросить, чем врать.
            if max_nsp is not None and max_nsp >= self._max_no_speech_prob and len(text) <= 2:
                return STTResult(text="", avg_logprob=avg_lp, max_no_speech_prob=max_nsp)
            if avg_lp is not None and avg_lp < self._min_avg_logprob and len(text) <= 20:
                return STTResult(text="", avg_logprob=avg_lp, max_no_speech_prob=max_nsp)

            return STTResult(text=text, avg_logprob=avg_lp, max_no_speech_prob=max_nsp)
