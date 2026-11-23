"""Linux helpers: single instance (Unix socket), opening files/URLs, locking the screen, settings pages.

Linux desktops don't let apps grab a global hotkey the way Windows does, so the installer binds
Super+Space (GNOME) to `technoforesight-assistant`. Launching it again simply asks the running copy
to show its window through the socket below.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
from collections.abc import Callable

# gnome-control-center panel names for "settings <page>"
SETTINGS_PANELS = {
    "wifi": "wifi", "wi-fi": "wifi", "bluetooth": "bluetooth", "display": "display", "sound": "sound",
    "battery": "power", "update": "", "updates": "", "apps": "applications", "wallpaper": "background",
    "background": "background", "dark mode": "background", "colors": "background", "storage": "",
    "privacy": "privacy", "notifications": "notifications", "vpn": "network", "printers": "printers",
    "mouse": "mouse", "keyboard": "keyboard", "about": "about", "": "",
}


def _quiet_popen(cmd: list[str]) -> None:
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,  # noqa: S603
                     stderr=subprocess.DEVNULL, start_new_session=True)


def open_target(target: str) -> None:
    """Open a file, folder or URL with the desktop's default app."""
    if target.startswith("linux-settings:"):
        open_settings(target.split(":", 1)[1])
        return
    _quiet_popen(["xdg-open", target])


def open_settings(page: str) -> None:
    panel = SETTINGS_PANELS.get(page, "")
    for tool in ("gnome-control-center", "cinnamon-settings", "systemsettings", "xfce4-settings-manager"):
        if shutil.which(tool):
            _quiet_popen([tool, panel] if panel and tool in ("gnome-control-center", "cinnamon-settings") else [tool])
            return
    raise OSError("No settings app found (looked for GNOME, Cinnamon, KDE and Xfce settings).")


def lock_screen() -> None:
    for cmd in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"],
                ["gnome-screensaver-command", "-l"], ["dm-tool", "lock"]):
        if shutil.which(cmd[0]):
            if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,  # noqa: S603
                              timeout=10).returncode == 0:
                return
    raise OSError("Couldn't lock the screen on this desktop.")


def start_hotkey_listener(spec: str, on_press: Callable[[], None]) -> None:
    """Global hotkeys are set by the desktop (see install.sh), not grabbed by the app."""
    return None


class SingleInstance:
    """First instance binds an abstract Unix socket; later launches send 'show' to it."""

    def __init__(self, name: str = "technoforesight-assistant") -> None:
        self.address = f"\0{name}-{os.getuid()}"
        self._sock: socket.socket | None = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.bind(self.address)
            self._sock.listen(4)
            self.is_primary = True
        except OSError:
            self._sock.close()
            self._sock = None
            self.is_primary = False

    def signal_primary(self) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(self.address)
                s.sendall(b"show")
        except OSError:
            pass

    def listen(self, on_signal: Callable[[], None]) -> None:
        if self._sock is None:
            return
        sock = self._sock

        def loop():
            while True:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    return
                with conn:
                    if conn.recv(16).startswith(b"show"):
                        on_signal()

        threading.Thread(target=loop, daemon=True, name="instance-signal").start()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


SHORTCUT_LABEL = "Ctrl+Alt+Space"
_GNOME_KEYS = "org.gnome.settings-daemon.plugins.media-keys"
_GNOME_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/technoforesight/"


def _gsettings(*args: str) -> str | None:
    try:
        out = subprocess.run(["gsettings", *args], capture_output=True, text=True, timeout=5)  # noqa: S603,S607
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def ensure_gnome_shortcut(command: str = "technoforesight-assistant", binding: str = "<Control><Alt>space") -> bool:
    """Adds a GNOME/Ubuntu custom shortcut that opens the assistant. Returns True if it is set.
    Leaves existing shortcuts alone; does nothing on desktops without GNOME's settings daemon."""
    if os.environ.get("TF_NO_SHORTCUT") or not shutil.which("gsettings"):  # tests set TF_NO_SHORTCUT
        return False
    current = _gsettings("get", _GNOME_KEYS, "custom-keybindings")
    if current is None:
        return False
    if _GNOME_PATH in current:
        return True
    paths = [p.strip().strip("'\"") for p in current.removeprefix("@as").strip(" []").split(",") if p.strip()]
    schema = f"{_GNOME_KEYS}.custom-keybinding:{_GNOME_PATH}"
    for key, value in (("name", "TechnoForesight Assistant"), ("command", command), ("binding", binding)):
        if _gsettings("set", schema, key, value) is None:
            return False
    new = "[" + ", ".join(f"'{p}'" for p in [*paths, _GNOME_PATH]) + "]"
    return _gsettings("set", _GNOME_KEYS, "custom-keybindings", new) is not None
