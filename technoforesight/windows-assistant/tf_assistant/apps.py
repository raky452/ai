"""Discover and launch Windows apps: classic desktop apps *and* Microsoft Store apps.

Primary source is PowerShell's Get-StartApps (everything in the Start menu, incl. Calculator,
Settings, WhatsApp...). If that fails we fall back to scanning Start-menu shortcut files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class App:
    name: str
    # Either an AppUserModelID (Store/Start apps) or a path to a .lnk/.url/.exe
    target: str
    is_appid: bool = True

    def matches(self, query: str) -> bool:
        return query.lower() in self.name.lower()


_SKIP_WORDS = ("uninstall", "readme", "help", "documentation", "release notes", "website")


def _useful(name: str) -> bool:
    lower = name.lower()
    return bool(name) and not any(w in lower for w in _SKIP_WORDS)


def _from_get_startapps() -> list[App]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"],
        capture_output=True, text=True, encoding="utf-8", timeout=20, creationflags=CREATE_NO_WINDOW,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return []
    data = json.loads(out.stdout)
    if isinstance(data, dict):
        data = [data]
    return [App(d["Name"], d["AppID"]) for d in data if _useful(d.get("Name", "")) and d.get("AppID")]


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for env in ("APPDATA", "PROGRAMDATA"):
        base = os.environ.get(env)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def scan_shortcuts(dirs: list[Path] | None = None) -> list[App]:
    apps = []
    for d in dirs if dirs is not None else _start_menu_dirs():
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in (".lnk", ".url") and _useful(f.stem):
                apps.append(App(f.stem, str(f), is_appid=False))
    return apps


def dedupe_sort(apps: list[App]) -> list[App]:
    seen: dict[str, App] = {}
    for a in apps:
        seen.setdefault(a.name.lower(), a)
    return sorted(seen.values(), key=lambda a: a.name.lower())


def load_apps() -> list[App]:
    if sys.platform != "win32":
        return []
    try:
        apps = _from_get_startapps()
    except (OSError, subprocess.SubprocessError, ValueError):
        apps = []
    return dedupe_sort(apps or scan_shortcuts())


def find_app(query: str, apps: list[App], exact_only: bool = False) -> App | None:
    q = query.strip().lower()
    if not q:
        return None
    for app in apps:
        if app.name.lower() == q:
            return app
    if exact_only:
        return None
    for app in apps:
        if app.name.lower().startswith(q):
            return app
    # word-start match: "word" -> "Microsoft Word"
    for app in apps:
        if any(w.startswith(q) for w in app.name.lower().split()):
            return app
    for app in apps:
        if app.matches(q):
            return app
    return None


def launch(app: App) -> None:
    if app.is_appid:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app.target}"], creationflags=CREATE_NO_WINDOW)
    else:
        os.startfile(app.target)  # type: ignore[attr-defined]  # noqa: S606
