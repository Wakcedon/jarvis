from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from jarvis.tts_piper import PiperTTS


@dataclass(frozen=True)
class SpeakRequest:
    text: str
    length_scale: float
    sentence_silence: float
    volume: float
    done: threading.Event


class TTSWorker:
    def __init__(self, *, tts: PiperTTS) -> None:
        self._tts = tts
        self._q: queue.Queue[SpeakRequest] = queue.Queue()
        self._stop_all = threading.Event()
        self._speaking = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jarvis-tts", daemon=True)
        self._thread.start()

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self) -> None:
        self._stop_all.set()
        try:
            self._tts.stop()
        except Exception:
            pass
        # drain queue
        try:
            while True:
                req = self._q.get_nowait()
                req.done.set()
        except Exception:
            pass

    def cancel_current(self) -> None:
        """Stops current speech and clears queued speech, but keeps the worker alive."""
        try:
            self._tts.stop()
        except Exception:
            pass
        try:
            while True:
                req = self._q.get_nowait()
                req.done.set()
        except Exception:
            pass

    def speak_blocking(self, text: str, *, length_scale: float, sentence_silence: float, volume: float) -> None:
        if not text.strip():
            return
        ev = threading.Event()
        self._q.put(
            SpeakRequest(
                text=text,
                length_scale=float(length_scale),
                sentence_silence=float(sentence_silence),
                volume=float(volume),
                done=ev,
            )
        )
        ev.wait()

    def speak_async(self, text: str, *, length_scale: float, sentence_silence: float, volume: float) -> None:
        if not text.strip():
            return
        self._q.put(
            SpeakRequest(
                text=text,
                length_scale=float(length_scale),
                sentence_silence=float(sentence_silence),
                volume=float(volume),
                done=threading.Event(),
            )
        )

    def _run(self) -> None:
        while True:
            req = self._q.get()
            if self._stop_all.is_set():
                req.done.set()
                continue
            try:
                self._speaking.set()
                self._tts.speak(
                    req.text,
                    length_scale=req.length_scale,
                    sentence_silence=req.sentence_silence,
                    volume=req.volume,
                )
            except Exception:
                pass
            finally:
                self._speaking.clear()
                req.done.set()
