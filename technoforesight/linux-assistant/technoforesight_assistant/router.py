"""Turns what the user typed into an action. Rules first (instant), AI for everything else."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .apps import DesktopApp, find_app


@dataclass(frozen=True)
class OpenApp:
    app: DesktopApp


@dataclass(frozen=True)
class FindFiles:
    query: str


@dataclass(frozen=True)
class Summarize:
    path: Path


@dataclass(frozen=True)
class OpenUrl:
    url: str


@dataclass(frozen=True)
class WebSearch:
    query: str


@dataclass(frozen=True)
class AskAssistant:
    text: str


Action = OpenApp | FindFiles | Summarize | OpenUrl | WebSearch | AskAssistant

_OPEN = re.compile(r"^(open|launch|start|run)\s+(.+)$", re.IGNORECASE)
_FIND = re.compile(r"^(find|search|locate)\s+(files?\s+|for\s+)?(.+)$", re.IGNORECASE)
_SUMMARIZE = re.compile(r"^(summari[sz]e|tl;?dr|explain)\s+(.+)$", re.IGNORECASE)
_URL = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)
_WEB = re.compile(r"^(google|web)\s+(.+)$", re.IGNORECASE)


def _looks_like_path(text: str) -> bool:
    return text.startswith(("/", "~", "./")) or bool(re.search(r"\.\w{1,5}$", text))


def route(text: str, apps: list[DesktopApp]) -> Action:
    text = text.strip()

    if m := _URL.match(text):
        url = m.group(1)
        return OpenUrl(url if url.lower().startswith("http") else "https://" + url)

    if m := _WEB.match(text):
        return WebSearch(m.group(2).strip())

    if m := _SUMMARIZE.match(text):
        target = m.group(2).strip().strip("'\"")
        if _looks_like_path(target):
            return Summarize(Path(target).expanduser())

    if m := _OPEN.match(text):
        target = m.group(2).strip()
        if app := find_app(target, apps):
            return OpenApp(app)
        if _looks_like_path(target):
            return FindFiles(Path(target).name)

    if m := _FIND.match(text):
        return FindFiles(m.group(3).strip())

    # A bare app name ("firefox") opens it straight away.
    if app := find_app(text, apps, exact_only=True):
        return OpenApp(app)

    return AskAssistant(text)
