"""Small Win32 helpers via ctypes: global hotkey, single instance, start at login."""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes

MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004, "win": 0x0008}
MOD_NOREPEAT = 0x4000
NAMED_KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "backquote": 0xC0, "`": 0xC0}
NAMED_KEYS.update({f"f{i}": 0x6F + i for i in range(1, 13)})

WM_HOTKEY = 0x0312
ERROR_ALREADY_EXISTS = 183
INFINITE = 0xFFFFFFFF


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+k' -> (modifier flags, virtual-key code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("Empty hotkey")
    mods = 0
    for p in parts[:-1]:
        if p not in MODIFIERS:
            raise ValueError(f"Unknown modifier '{p}' in hotkey '{spec}'")
        mods |= MODIFIERS[p]
    key = parts[-1]
    if key in NAMED_KEYS:
        vk = NAMED_KEYS[key]
    elif len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    else:
        raise ValueError(f"Unknown key '{key}' in hotkey '{spec}'")
    return mods | MOD_NOREPEAT, vk


def start_hotkey_listener(spec: str, on_press: Callable[[], None]) -> threading.Thread | None:
    """Registers a system-wide hotkey. Returns None (and does nothing) off Windows or if the key is taken."""
    if sys.platform != "win32":
        return None
    mods, vk = parse_hotkey(spec)
    ready = threading.Event()
    ok = {"value": False}

    def loop():
        user32 = ctypes.windll.user32
        ok["value"] = bool(user32.RegisterHotKey(None, 1, mods, vk))
        ready.set()
        if not ok["value"]:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                on_press()

    t = threading.Thread(target=loop, daemon=True, name="hotkey")
    t.start()
    ready.wait(2)
    return t if ok["value"] else None


class SingleInstance:
    """First instance owns a named mutex; later launches just signal it to show its window."""

    MUTEX = "Local\\TechnoForesightAssistant.Mutex"
    EVENT = "Local\\TechnoForesightAssistant.Show"

    def __init__(self) -> None:
        self.is_primary = True
        self._event = None
        if sys.platform != "win32":
            return
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateEventW.restype = wintypes.HANDLE
        self._mutex = k32.CreateMutexW(None, False, self.MUTEX)
        self.is_primary = k32.GetLastError() != ERROR_ALREADY_EXISTS
        self._event = k32.CreateEventW(None, False, False, self.EVENT)

    def signal_primary(self) -> None:
        if self._event:
            ctypes.windll.kernel32.SetEvent(self._event)

    def listen(self, on_signal: Callable[[], None]) -> None:
        if not self._event:
            return

        def loop():
            k32 = ctypes.windll.kernel32
            while k32.WaitForSingleObject(self._event, INFINITE) == 0:
                on_signal()

        threading.Thread(target=loop, daemon=True, name="instance-signal").start()
