"""Carries out router actions. Shared by the window and the command-line mode."""

from __future__ import annotations

import ctypes
import os
import sys
import urllib.parse
import webbrowser
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import apps as apps_mod
from . import files
from .config import Config, save_connector
from .llm import OllamaClient
from .router import (
    Action, AskAssistant, Connector, FindFiles, OpenApp, OpenSettings, OpenUrl, Summarize, System, WebSearch,
)


@dataclass
class Result:
    message: str = ""
    stream: Iterator[str] | None = None
    files: list[Path] = field(default_factory=list)
    hide_window: bool = False
    quit_app: bool = False


def _open(target: str) -> None:
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
    else:
        webbrowser.open(target)


def run(action: Action, cfg: Config, llm: OllamaClient) -> Result:
    if isinstance(action, OpenApp):
        apps_mod.launch(action.app)
        return Result(f"Opening {action.app.name}…", hide_window=True)

    if isinstance(action, OpenSettings):
        _open(action.uri)
        return Result(f"Opening Settings → {action.page}", hide_window=True)

    if isinstance(action, OpenUrl):
        _open(action.url)
        return Result(f"Opening {action.url}", hide_window=True)

    if isinstance(action, WebSearch):
        webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(action.query))
        return Result(f"Searching the web for “{action.query}”", hide_window=True)

    if isinstance(action, Connector):
        if action.manage == "list":
            rows = "\n".join(f"  {k:<16} {v}" for k, v in sorted(cfg.connectors.items()))
            help_ = ("\n\nAdd your own:  add connector store https://your-site.com\n"
                     "Use {q} for searches:  add connector amz https://example.com/search?q={q}\n"
                     "Remove:  remove connector store")
            return Result(f"{len(cfg.connectors)} connectors", stream=iter([rows + help_]))
        if action.manage == "add":
            cfg.connectors[action.name] = action.url
            save_connector(action.name, action.url)
            return Result(f"✔ Connector '{action.name}' added. Type '{action.name}' to open {action.url}")
        if action.manage == "remove":
            existed = cfg.connectors.pop(action.name, None) is not None
            save_connector(action.name, None)
            return Result(f"✔ Connector '{action.name}' removed." if existed else f"No connector named '{action.name}'.")
        _open(action.url)
        return Result(f"🔗 {action.name} → {action.url}", hide_window=True)

    if isinstance(action, System):
        if action.command == "lock":
            ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
            return Result("Locking…", hide_window=True)
        return Result("Goodbye!", quit_app=True)

    if isinstance(action, FindFiles):
        found = files.find_files(action.query, cfg.search_roots)
        msg = (f"Found {len(found)} file(s) matching “{action.query}”. Press ↓ and Enter to open."
               if found else f"No files matching “{action.query}” in Desktop, Documents or Downloads.")
        return Result(msg, files=found)

    if isinstance(action, Summarize):
        text = files.read_document(action.path, cfg.max_document_chars)
        prompt = (
            f"Summarise this document ({action.path.name}) in 5 short bullet points, "
            f"then list any dates, amounts or action items.\n\n---\n{text}\n---"
        )
        return Result(f"Summarising {action.path.name}…", stream=llm.chat_stream(prompt))

    if isinstance(action, AskAssistant):
        return Result("", stream=llm.chat_stream(action.text))

    raise TypeError(f"Unknown action: {action!r}")
