"""Minimal client for a local Ollama server (no third-party dependencies)."""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

# (English name, native name, script) for the response-language picker
LANGUAGES = [
    ("English", "English", "Latin"),
    ("Telugu", "తెలుగు", "Telugu"),
    ("Hindi", "हिन्दी", "Devanagari"),
    ("Tamil", "தமிழ்", "Tamil"),
    ("Kannada", "ಕನ್ನಡ", "Kannada"),
    ("Malayalam", "മലയാളം", "Malayalam"),
    ("Marathi", "मराठी", "Devanagari"),
    ("Bengali", "বাংলা", "Bengali"),
    ("Gujarati", "ગુજરાતી", "Gujarati"),
    ("Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi"),
    ("Odia", "ଓଡ଼ିଆ", "Odia"),
    ("Urdu", "اردو", "Urdu (Perso-Arabic)"),
]
LANGUAGE_BY_NAME = {name: (name, native, script) for name, native, script in LANGUAGES}


def language_instruction(language: str) -> str:
    if language not in LANGUAGE_BY_NAME or language == "English":
        return "Reply in English unless the user clearly asks for another language."
    name, native, script = LANGUAGE_BY_NAME[language]
    return (f"Always reply in {name} ({native}), written in the {script} script, even if the question is in "
            f"English or another language. Keep app names, file names, numbers and technical terms as they are. "
            f"Use simple, natural, everyday {name}. Write only {name} words in the {script} script plus "
            f"English for technical terms; never use words from any other language (for example "
            f"Vietnamese, Chinese or Russian). Write plain text without Markdown symbols like ** or #.")


SYSTEM_PROMPT = (
    "You are {name}, an offline assistant running privately on the user's {os} computer. "
    "Answer clearly and briefly. If asked about files, only use the text you are given."
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CLOSE = "</think>"
# Yielded by the stream when text already shown turns out to be the model's hidden reasoning
# (thinking-only models emit reasoning, then "</think>", then the answer). The UI clears its answer.
RESET = "\x00RESET\x00"


def strip_thinking(text: str) -> str:
    """Some models (e.g. Qwen3) emit hidden reasoning in <think> tags; users shouldn't see it."""
    text = _THINK_RE.sub("", text)
    if _CLOSE in text:  # reasoning with no opening tag
        text = text.split(_CLOSE)[-1].lstrip()
    return text


def join_stream(pieces) -> str:
    """Collects streamed pieces, honouring RESET markers."""
    out: list[str] = []
    for piece in pieces:
        if piece == RESET:
            out.clear()
        else:
            out.append(piece)
    return "".join(out)


class MarkdownCleaner:
    """Streaming clean-up of Markdown the answer card can't render: removes ** and __ bold markers,
    leading # headings and turns "* " / "- " bullets into "• ". Holds back at most a few characters
    so markers split across chunks are still caught."""

    def __init__(self) -> None:
        self._buffer = ""
        self._line_start = True

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        buf = self._buffer
        i = 0
        while i < len(buf):
            rest = buf[i:]
            if self._line_start:
                stripped = rest.lstrip(" \t")
                lead = rest[:len(rest) - len(stripped)]
                if not stripped and not final:
                    break  # wait to see what the line starts with
                if stripped.startswith("#"):
                    hashes = len(stripped) - len(stripped.lstrip("#"))
                    after = stripped[hashes:]
                    if not after and not final:
                        break
                    if after.startswith(" ") or not after:
                        i += len(lead) + hashes + (1 if after.startswith(" ") else 0)
                        self._line_start = False
                        continue
                if stripped[:2] in ("* ", "- "):
                    out.append(lead + "• ")
                    i += len(lead) + 2
                    self._line_start = False
                    continue
                if stripped in ("*", "-") and not final:
                    break
                self._line_start = False
                continue
            ch = buf[i]
            if ch in "*_":
                if i + 1 >= len(buf) and not final:
                    break  # might be the first half of ** / __
                if buf[i + 1:i + 2] == ch:
                    i += 2
                    continue
            out.append(ch)
            if ch == "\n":
                self._line_start = True
            i += 1
        self._buffer = buf[i:]
        if final:
            out.append(self._buffer)
            self._buffer = ""
        return "".join(out)


class ThinkFilter:
    """Streaming version of strip_thinking: hides everything between <think> and </think>."""

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    def feed(self, chunk: str) -> list[str]:
        """Returns the visible pieces for this chunk (may include RESET)."""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._thinking:
                end = self._buffer.find(_CLOSE)
                if end == -1:
                    # keep a tail in case the closing tag is split across chunks
                    self._buffer = self._buffer[-len(_CLOSE):]
                    break
                self._buffer = self._buffer[end + len(_CLOSE):].lstrip()
                self._thinking = False
            else:
                start = self._buffer.find("<think>")
                close = self._buffer.find(_CLOSE)
                if close != -1 and (start == -1 or close < start):
                    # Everything so far was reasoning without an opening tag: throw it away
                    out = [RESET]
                    self._buffer = self._buffer[close + len(_CLOSE):].lstrip()
                    continue
                if start == -1:
                    # don't emit a partial "<thi" / "</thi" that might become a tag
                    safe = len(self._buffer)
                    for tag in ("<think>", _CLOSE):
                        for i in range(len(tag) - 1, 0, -1):
                            if self._buffer.endswith(tag[:i]):
                                safe = min(safe, len(self._buffer) - i)
                                break
                    if safe:
                        out.append(self._buffer[:safe])
                    self._buffer = self._buffer[safe:]
                    break
                if start:
                    out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len("<think>"):]
                self._thinking = True
        return out

    def flush(self) -> str:
        rest, self._buffer = ("" if self._thinking else self._buffer), ""
        return rest


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, assistant_name: str = "TechnoForesight Assistant",
                 max_tokens: int = 700, language: str = "English"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.language = language
        self.base_prompt = SYSTEM_PROMPT.format(name=assistant_name,
                                                os="Windows" if sys.platform == "win32" else "Linux")

    @property
    def system_prompt(self) -> str:
        return self.base_prompt + " " + language_instruction(self.language)

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
                "Can't reach the local AI engine. Install Ollama from ollama.com/download and make sure it's running."
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

    def pull_stream(self) -> Iterator[tuple[str, float]]:
        """Downloads the model through Ollama. Yields (status text, fraction 0..1 or -1)."""
        with self._request("/api/pull", {"model": self.model, "stream": True}, timeout=3600) as resp:
            for line in resp:
                if not line.strip():
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise OllamaError(event["error"])
                total, done = event.get("total") or 0, event.get("completed") or 0
                yield event.get("status", ""), (done / total if total else -1.0)

    def _chat_payload(self, messages: list[dict], think: bool | None) -> dict:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            # Short, focused answers; a hard cap stops runaway generation on slow PCs
            # Qwen's recommended sampling for its instruct models; presence/repeat penalties stop the
            # looping repetition small quantised models fall into, especially in Indian languages.
            "options": {"num_predict": self.max_tokens, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "repeat_penalty": 1.1, "presence_penalty": 1.2, "repeat_last_n": 256},
            "keep_alive": "30m",  # stay loaded so the next question starts fast
        }
        if think is not None:
            payload["think"] = think
        return payload

    def chat_stream(self, user_text: str, history: list[dict] | None = None,
                    cancel: threading.Event | None = None) -> Iterator[str]:
        """Streams the answer. Thinking/reasoning mode is switched off: on a laptop CPU it can
        add minutes of silent work before the first word. Setting `cancel` stops generation
        (closing the connection makes Ollama abort the request)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += history or []
        messages.append({"role": "user", "content": user_text})
        try:
            resp = self._request("/api/chat", self._chat_payload(messages, think=False))
        except OllamaError as e:
            if "think" not in str(e).lower():
                raise
            # Model or older Ollama doesn't support the think switch: ask without it
            resp = self._request("/api/chat", self._chat_payload(messages, think=None))
        flt = ThinkFilter()
        md = MarkdownCleaner()
        with resp:
            for line in resp:
                if cancel is not None and cancel.is_set():
                    return
                if not line.strip():
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise OllamaError(event["error"])
                for piece in flt.feed(event.get("message", {}).get("content", "")):
                    if piece == RESET:
                        md = MarkdownCleaner()
                        yield piece
                    else:
                        clean = md.feed(piece)
                        if clean:
                            yield clean
                if event.get("done"):
                    break
        tail = md.feed(flt.flush() or "") + md.flush()
        if tail:
            yield tail

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        return join_stream(self.chat_stream(user_text, history)).strip()

    def speed_test(self) -> float:
        """Loads the model and generates a few words. Returns tokens per second (0 if unknown).
        Raises OllamaError if the model can't actually run (e.g. not enough memory)."""
        payload = {"model": self.model, "prompt": "Reply with one short sentence: hello.", "stream": False,
                   "think": False, "options": {"num_predict": 24}, "keep_alive": "30m"}
        try:
            resp = self._request("/api/generate", payload, timeout=300)
        except OllamaError as e:
            if "think" not in str(e).lower():
                raise
            payload.pop("think")
            resp = self._request("/api/generate", payload, timeout=300)
        with resp:
            data = json.load(resp)
        if "error" in data:
            raise OllamaError(data["error"])
        count, dur = data.get("eval_count") or 0, data.get("eval_duration") or 0
        return count / (dur / 1e9) if count and dur else 0.0
