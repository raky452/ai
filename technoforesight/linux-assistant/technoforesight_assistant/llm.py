"""Minimal client for a local Ollama server (no third-party dependencies)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator

SYSTEM_PROMPT = (
    "You are {name}, an offline assistant running privately on the user's Linux laptop. "
    "Answer clearly and briefly. If asked about files, only use the text you are given."
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Some models (e.g. Qwen3) emit hidden reasoning in <think> tags; users shouldn't see it."""
    return _THINK_RE.sub("", text)


class ThinkFilter:
    """Streaming version of strip_thinking: hides everything between <think> and </think>."""

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out = []
        while self._buffer:
            if self._thinking:
                end = self._buffer.find("</think>")
                if end == -1:
                    # keep a tail in case the closing tag is split across chunks
                    self._buffer = self._buffer[-len("</think>"):]
                    break
                self._buffer = self._buffer[end + len("</think>"):].lstrip()
                self._thinking = False
            else:
                start = self._buffer.find("<think>")
                if start == -1:
                    # don't emit a partial "<thi" that might become a tag
                    safe = len(self._buffer)
                    for i in range(1, len("<think>")):
                        if self._buffer.endswith("<think>"[:i]):
                            safe = len(self._buffer) - i
                            break
                    out.append(self._buffer[:safe])
                    self._buffer = self._buffer[safe:]
                    break
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len("<think>"):]
                self._thinking = True
        return "".join(out)

    def flush(self) -> str:
        rest, self._buffer = ("" if self._thinking else self._buffer), ""
        return rest


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, assistant_name: str = "TechnoForesight Assistant"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = SYSTEM_PROMPT.format(name=assistant_name)

    def _request(self, path: str, payload: dict | None = None, timeout: float = 600):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - local server only
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise OllamaError(f"Ollama error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise OllamaError(
                "Can't reach the local AI engine. Is Ollama running? (start it with: ollama serve)"
            ) from e

    def installed_models(self) -> list[str]:
        with self._request("/api/tags", timeout=5) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]

    def check(self) -> str | None:
        """Returns a human-readable problem, or None if everything is ready."""
        try:
            models = self.installed_models()
        except OllamaError as e:
            return str(e)
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in models:
            return f"Model '{self.model}' isn't downloaded yet. Run: ollama pull {self.model}"
        return None

    def chat_stream(self, user_text: str, history: list[dict] | None = None) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += history or []
        messages.append({"role": "user", "content": user_text})
        payload = {"model": self.model, "messages": messages, "stream": True}
        flt = ThinkFilter()
        with self._request("/api/chat", payload) as resp:
            for line in resp:
                if not line.strip():
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise OllamaError(event["error"])
                piece = flt.feed(event.get("message", {}).get("content", ""))
                if piece:
                    yield piece
                if event.get("done"):
                    break
        tail = flt.flush()
        if tail:
            yield tail

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        return "".join(self.chat_stream(user_text, history)).strip()
