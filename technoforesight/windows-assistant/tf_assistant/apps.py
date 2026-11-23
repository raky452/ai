"""Discover and launch installed apps.

Windows: PowerShell's Get-StartApps (everything in the Start menu, incl. Store apps), falling back to
Start-menu shortcut files. Linux: freedesktop .desktop files (system, user, Flatpak and Snap).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
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
    exec_line: str = ""                 # Linux: Exec= line from the .desktop file
    keywords: tuple[str, ...] = ()
    terminal: bool = False

    def matches(self, query: str) -> bool:
        q = query.lower()
        return q in self.name.lower() or any(q in k.lower() for k in self.keywords)


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


# ---------- Linux (.desktop files) ----------
_FIELD_CODES = {"%f", "%F", "%u", "%U", "%d", "%D", "%n", "%N", "%i", "%c", "%k", "%v", "%m"}


def application_dirs() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    return ([data_home / "applications"] + [Path(d) / "applications" for d in data_dirs if d]
            + [Path("/var/lib/flatpak/exports/share/applications"),
               data_home / "flatpak" / "exports" / "share" / "applications",
               Path("/var/lib/snapd/desktop/applications")])


def parse_desktop_file(path: Path) -> App | None:
    """Reads the [Desktop Entry] section; None for hidden or non-application entries."""
    entry: dict[str, str] = {}
    in_entry = False
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                in_entry = line == "[Desktop Entry]"
            elif in_entry and "=" in line:
                key, _, value = line.partition("=")
                entry.setdefault(key.strip(), value.strip())
    except OSError:
        return None
    if entry.get("Type") != "Application" or "Name" not in entry or "Exec" not in entry:
        return None
    if entry.get("NoDisplay", "").lower() == "true" or entry.get("Hidden", "").lower() == "true":
        return None
    if not _useful(entry["Name"]):
        return None
    return App(entry["Name"], str(path), is_appid=False, exec_line=entry["Exec"],
               keywords=tuple(k for k in entry.get("Keywords", "").split(";") if k),
               terminal=entry.get("Terminal", "").lower() == "true")


def scan_desktop_files(dirs: list[Path] | None = None) -> list[App]:
    """All visible apps; earlier directories win (user overrides beat system files)."""
    seen: set[str] = set()
    apps: list[App] = []
    for d in dirs if dirs is not None else application_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.desktop")):
            desktop_id = f.relative_to(d).as_posix().replace("/", "-")
            if desktop_id in seen:
                continue
            seen.add(desktop_id)
            app = parse_desktop_file(f)
            if app:
                apps.append(app)
    return apps


def build_command(exec_line: str) -> list[str]:
    return [p for p in shlex.split(exec_line)
            if p not in _FIELD_CODES and not (len(p) == 2 and p.startswith("%"))]


def load_apps() -> list[App]:
    if sys.platform != "win32":
        return dedupe_sort(scan_desktop_files())
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
    if sys.platform != "win32":
        if shutil.which("gio") and not app.terminal:
            cmd = ["gio", "launch", app.target]
        else:
            cmd = build_command(app.exec_line)
            if app.terminal:
                cmd = ["x-terminal-emulator", "-e", *cmd]
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,  # noqa: S603
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return
    if app.is_appid:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app.target}"], creationflags=CREATE_NO_WINDOW)
    else:
        os.startfile(app.target)  # type: ignore[attr-defined]  # noqa: S606
