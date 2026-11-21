"""Discover and launch installed desktop applications (freedesktop .desktop files)."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

# %f %F %u %U %i %c %k etc. are placeholders the launcher must strip.
_FIELD_CODES = {"%f", "%F", "%u", "%U", "%d", "%D", "%n", "%N", "%i", "%c", "%k", "%v", "%m"}


@dataclass(frozen=True)
class DesktopApp:
    name: str
    exec_line: str
    path: str
    keywords: tuple[str, ...] = ()
    terminal: bool = False

    def matches(self, query: str) -> bool:
        q = query.lower()
        return q in self.name.lower() or any(q in k.lower() for k in self.keywords)


def application_dirs() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    dirs = [data_home / "applications"] + [Path(d) / "applications" for d in data_dirs if d]
    # Flatpak and Snap apps
    dirs += [
        Path("/var/lib/flatpak/exports/share/applications"),
        data_home / "flatpak" / "exports" / "share" / "applications",
        Path("/var/lib/snapd/desktop/applications"),
    ]
    return dirs


def parse_desktop_file(path: Path) -> DesktopApp | None:
    """Read the [Desktop Entry] section. Returns None for hidden or non-app entries."""
    entry: dict[str, str] = {}
    in_entry = False
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                in_entry = line == "[Desktop Entry]"
                continue
            if in_entry and "=" in line:
                key, _, value = line.partition("=")
                entry.setdefault(key.strip(), value.strip())
    except OSError:
        return None

    if entry.get("Type") != "Application":
        return None
    if entry.get("NoDisplay", "").lower() == "true" or entry.get("Hidden", "").lower() == "true":
        return None
    if "Name" not in entry or "Exec" not in entry:
        return None
    keywords = tuple(k for k in entry.get("Keywords", "").split(";") if k)
    return DesktopApp(
        name=entry["Name"],
        exec_line=entry["Exec"],
        path=str(path),
        keywords=keywords,
        terminal=entry.get("Terminal", "").lower() == "true",
    )


def load_apps(dirs: list[Path] | None = None) -> list[DesktopApp]:
    """All visible apps. Earlier directories win (user overrides beat system files)."""
    seen_ids: set[str] = set()
    apps: list[DesktopApp] = []
    for d in dirs if dirs is not None else application_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.desktop")):
            desktop_id = f.relative_to(d).as_posix().replace("/", "-")
            if desktop_id in seen_ids:
                continue
            seen_ids.add(desktop_id)
            app = parse_desktop_file(f)
            if app:
                apps.append(app)
    apps.sort(key=lambda a: a.name.lower())
    return apps


def find_app(query: str, apps: list[DesktopApp], exact_only: bool = False) -> DesktopApp | None:
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
    for app in apps:
        if app.matches(q):
            return app
    return None


def build_command(exec_line: str) -> list[str]:
    parts = shlex.split(exec_line)
    return [p for p in parts if p not in _FIELD_CODES and not (len(p) == 2 and p.startswith("%"))]


def launch(app: DesktopApp) -> None:
    cmd = build_command(app.exec_line)
    if app.terminal:
        cmd = ["x-terminal-emulator", "-e", *cmd]
    subprocess.Popen(  # noqa: S603 - command comes from the system's own .desktop files
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
