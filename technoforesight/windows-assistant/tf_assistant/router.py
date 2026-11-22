"""Turns typed text into an action. Instant rules first; the AI handles everything else."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .apps import App, find_app

# "settings wifi" -> ms-settings URI. Only well-known, stable pages.
SETTINGS_PAGES = {
    "wifi": "ms-settings:network-wifi",
    "wi-fi": "ms-settings:network-wifi",
    "bluetooth": "ms-settings:bluetooth",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "battery": "ms-settings:batterysaver",
    "update": "ms-settings:windowsupdate",
    "updates": "ms-settings:windowsupdate",
    "apps": "ms-settings:appsfeatures",
    "wallpaper": "ms-settings:personalization-background",
    "background": "ms-settings:personalization-background",
    "dark mode": "ms-settings:colors",
    "colors": "ms-settings:colors",
    "storage": "ms-settings:storagesense",
    "privacy": "ms-settings:privacy",
    "notifications": "ms-settings:notifications",
    "vpn": "ms-settings:network-vpn",
    "printers": "ms-settings:printers",
    "mouse": "ms-settings:mousetouchpad",
    "keyboard": "ms-settings:keyboard",
    "about": "ms-settings:about",
    "": "ms-settings:",
}


@dataclass(frozen=True)
class OpenApp:
    app: App


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
class OpenSettings:
    page: str
    uri: str


@dataclass(frozen=True)
class System:
    command: str  # "lock" | "quit"


@dataclass(frozen=True)
class AskAssistant:
    text: str


Action = OpenApp | FindFiles | Summarize | OpenUrl | WebSearch | OpenSettings | System | AskAssistant

_OPEN = re.compile(r"^(open|launch|start|run)\s+(.+)$", re.IGNORECASE)
_FIND = re.compile(r"^(find|search|locate)\s+(files?\s+|for\s+)?(.+)$", re.IGNORECASE)
_SUMMARIZE = re.compile(r"^(summari[sz]e|tl;?dr|explain)\s+(.+)$", re.IGNORECASE)
_URL = re.compile(r"^(https?://\S+|www\.\S+|[\w-]+\.(com|in|org|net|io|ai|dev)(/\S*)?)$", re.IGNORECASE)
_WEB = re.compile(r"^(google|web|bing)\s+(.+)$", re.IGNORECASE)
_SETTINGS = re.compile(r"^(open\s+)?(settings?)\s*(.*)$", re.IGNORECASE)
_LOCK = re.compile(r"^(lock|lock (pc|computer|screen|laptop))$", re.IGNORECASE)
_QUIT = re.compile(r"^(quit|exit) (assistant|technoforesight)$", re.IGNORECASE)


def _looks_like_path(text: str) -> bool:
    return bool(re.match(r"^([a-zA-Z]:[\\/]|~|\.[\\/]|\\\\)", text)) or bool(re.search(r"\.\w{1,5}$", text))


def route(text: str, apps: list[App]) -> Action:
    text = text.strip()

    if _LOCK.match(text):
        return System("lock")
    if _QUIT.match(text):
        return System("quit")

    if m := _SETTINGS.match(text):
        page = m.group(3).strip().lower()
        if page in SETTINGS_PAGES:
            return OpenSettings(page or "home", SETTINGS_PAGES[page])

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
        target = m.group(2).strip().strip("'\"")
        if _looks_like_path(target) and re.match(r"^([a-zA-Z]:|~)", target):
            return OpenUrl(str(Path(target).expanduser()))
        if app := find_app(target, apps):
            return OpenApp(app)

    if m := _FIND.match(text):
        return FindFiles(m.group(3).strip())

    if app := find_app(text, apps, exact_only=True):
        return OpenApp(app)

    return AskAssistant(text)
