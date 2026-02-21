from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import requests


class OllamaUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str  # system|user|assistant
    content: str


class OllamaClient:
    def __init__(self, *, base_url: str, model: str, timeout_s: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=min(3, self._timeout_s))
            return r.status_code == 200
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=min(5, self._timeout_s))
            r.raise_for_status()
            data = r.json()
            models = data.get("models") or []
            for m in models:
                name = (m.get("name") or "").strip()
                if name == self._model or name.split(":")[0] == self._model.split(":")[0]:
                    return True
            return False
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        num_predict: int | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_thread: int | None = None,
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = str(keep_alive)
        options: dict[str, Any] = {}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if temperature is not None:
            options["temperature"] = float(temperature)
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)
        if num_thread is not None:
            options["num_thread"] = int(num_thread)
        if options:
            payload["options"] = options
        try:
            r = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=(3, self._timeout_s),
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise OllamaUnavailable(str(e)) from e
        msg = data.get("message") or {}
        return str(msg.get("content", "")).strip()

    def chat_fast(
        self,
        messages: list[ChatMessage],
        *,
        num_predict: int,
        temperature: float,
        num_ctx: int,
        num_thread: int,
        keep_alive: str | None = None,
        max_sentences: int = 1,
        max_chars: int = 260,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "num_predict": int(num_predict),
                "temperature": float(temperature),
                "num_ctx": int(num_ctx),
                "num_thread": int(num_thread),
            },
        }
        if keep_alive is not None:
            payload["keep_alive"] = str(keep_alive)

        acc: list[str] = []

        try:
            with requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(3, self._timeout_s),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = (obj.get("message") or {}).get("content") or ""
                    if msg:
                        acc.append(str(msg))
                        text = "".join(acc)
                        if len(text) >= int(max_chars) or _count_sentences(text) >= int(max_sentences):
                            break
                    if obj.get("done") is True:
                        break
        except requests.RequestException as e:
            raise OllamaUnavailable(str(e)) from e

        text = "".join(acc).strip()
        return _truncate_sentences(text, int(max_sentences)).strip()


def _count_sentences(text: str) -> int:
    import re

    return len([m for m in re.finditer(r"[\.\!\?…]", text)])


def _truncate_sentences(text: str, max_sentences: int) -> str:
    import re

    if max_sentences <= 0:
        return text
    ends = [m.end() for m in re.finditer(r"[\.\!\?…]", text)]
    if len(ends) >= max_sentences:
        return text[: ends[max_sentences - 1]].strip()
    return text
