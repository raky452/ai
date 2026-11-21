"""Carries out router actions. Shared by the GTK window and the command-line mode."""

from __future__ import annotations

import subprocess
import urllib.parse
from collections.abc import Iterator

from . import apps as apps_mod
from . import files
from .config import Config
from .llm import OllamaClient
from .router import Action, AskAssistant, FindFiles, OpenApp, OpenUrl, Summarize, WebSearch


def _xdg_open(target: str) -> None:
    subprocess.Popen(
        ["xdg-open", target],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )


def run(action: Action, cfg: Config, llm: OllamaClient) -> tuple[str, Iterator[str] | None, list]:
    """
    Returns (status message, optional stream of AI text, list of file results).
    The UI shows the message immediately, then appends streamed text as it arrives.
    """
    if isinstance(action, OpenApp):
        apps_mod.launch(action.app)
        return f"Opening {action.app.name}…", None, []

    if isinstance(action, OpenUrl):
        _xdg_open(action.url)
        return f"Opening {action.url}", None, []

    if isinstance(action, WebSearch):
        _xdg_open("https://duckduckgo.com/?q=" + urllib.parse.quote_plus(action.query))
        return f"Searching the web for “{action.query}”", None, []

    if isinstance(action, FindFiles):
        found = files.find_files(action.query, cfg.search_roots)
        msg = f"Found {len(found)} file(s) matching “{action.query}”" if found else (
            f"No files matching “{action.query}”"
        )
        return msg, None, found

    if isinstance(action, Summarize):
        text = files.read_document(action.path, cfg.max_document_chars)
        prompt = (
            f"Summarise this document ({action.path.name}) in 5 short bullet points, "
            f"then list any dates, amounts or action items.\n\n---\n{text}\n---"
        )
        return f"Summarising {action.path.name}…", llm.chat_stream(prompt), []

    if isinstance(action, AskAssistant):
        return "", llm.chat_stream(action.text), []

    raise TypeError(f"Unknown action: {action!r}")
