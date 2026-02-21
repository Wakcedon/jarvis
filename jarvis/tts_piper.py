from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
import shutil
import re
import time

from jarvis.audio import play_wav
from jarvis.text_normalize_ru import normalize_for_tts


@dataclass(frozen=True)
class PiperVoice:
    model_path: Path
    config_path: Path


class PiperTTS:
    def __init__(
        self,
        *,
        voice: PiperVoice,
        output_device: int | str | None,
        length_scale: float = 0.88,
        sentence_silence: float = 0.15,
        volume: float = 1.0,
    ) -> None:
        self._voice = voice
        self._output_device = output_device
        self._length_scale = float(length_scale)
        self._sentence_silence = float(sentence_silence)
        self._volume = float(volume)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._current_proc: subprocess.Popen[bytes] | None = None
        self.last_error: str = ""

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._current_proc is not None and self._current_proc.poll() is None:
                self._current_proc.terminate()
        except Exception:
            pass

    def speak(
        self,
        text: str,
        *,
        length_scale: float | None = None,
        sentence_silence: float | None = None,
        volume: float | None = None,
    ) -> None:
        text = normalize_for_tts(text.strip())
        if not text:
            return

        ls = self._length_scale if length_scale is None else float(length_scale)
        ss = self._sentence_silence if sentence_silence is None else float(sentence_silence)
        vol = self._volume if volume is None else float(volume)

        with self._lock:
            self._stop_event.clear()
            for chunk in _split_text(text):
                self._speak_inner(chunk, length_scale=ls, sentence_silence=ss, volume=vol)

    def _speak_inner(self, text: str, *, length_scale: float, sentence_silence: float, volume: float) -> None:

        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            # piper CLI устанавливается вместе с piper-tts
            try:
                piper_exe = _resolve_piper_executable()
                if self._stop_event.is_set():
                    return
                proc = subprocess.Popen(
                    [
                        str(piper_exe),
                        "--model",
                        str(self._voice.model_path),
                        "--config",
                        str(self._voice.config_path),
                        "--output_file",
                        tmp.name,
                        "--length_scale",
                        str(length_scale),
                        "--sentence_silence",
                        str(sentence_silence),
                        "--volume",
                        str(volume),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_proc = proc
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(text.encode("utf-8"))
                        proc.stdin.close()
                except Exception:
                    pass

                while proc.poll() is None:
                    if self._stop_event.is_set():
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        return
                    time.sleep(0.03)

                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode or 1, "piper")

                if self._stop_event.is_set():
                    return
                play_wav(Path(tmp.name), output_device=self._output_device, stop_event=self._stop_event)
                self.last_error = ""
            except FileNotFoundError:
                self.last_error = "не найден бинарь piper"
                print("TTS: не найден бинарь 'piper' (piper-tts не установлен?)", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                self.last_error = f"ошибка piper ({e.returncode})"
                print(f"TTS: ошибка piper ({e.returncode}). Озвучка отключена.", file=sys.stderr)
            except Exception as e:
                self.last_error = str(e)
                print(f"TTS: ошибка: {e}. Озвучка отключена.", file=sys.stderr)
            finally:
                self._current_proc = None


def _split_text(text: str, max_chars: int = 220) -> list[str]:
    # Простое разбиение, чтобы piper не "сыпался" на больших текстах.
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    buf = ""
    for token in re.split(r"([\.!\?…]+)\s+", text):
        if not token:
            continue
        candidate = (buf + " " + token).strip() if buf else token.strip()
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            parts.append(buf)
            buf = token.strip()
        else:
            # fallback: жёстко режем
            for i in range(0, len(token), max_chars):
                parts.append(token[i : i + max_chars].strip())
            buf = ""
    if buf:
        parts.append(buf)
    return [p for p in parts if p]


def _resolve_piper_executable() -> Path:
    # 1) рядом с текущим python (в venv это .venv/bin/piper)
    candidate = Path(sys.executable).resolve().parent / "piper"
    if candidate.exists():
        return candidate

    # 2) PATH
    w = shutil.which("piper")
    if w:
        return Path(w)

    # 3) fallback
    return Path("piper")
