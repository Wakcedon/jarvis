from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class LlamaCppUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LlamaCppClient:
    """Клиент для llama.cpp server (OpenAI-compatible endpoint).

    Ожидается `llama-server` с `/v1/chat/completions`.
    """

    def __init__(self, *, base_url: str, timeout_s: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = int(timeout_s)

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/health", timeout=min(2, self._timeout_s))
            return r.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)

        try:
            r = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=self._timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise LlamaCppUnavailable(str(e)) from e

        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0].get("message") or {})
        return str(msg.get("content", "")).strip()
