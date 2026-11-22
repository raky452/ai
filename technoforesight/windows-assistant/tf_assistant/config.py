"""User settings, read from %APPDATA%\\TechnoForesight\\config.toml (all keys optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

APP_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "TechnoForesight"
CONFIG_PATH = APP_DIR / "config.toml"


@dataclass
class Config:
    # Local Ollama server that runs the model. Nothing is sent to the internet.
    ollama_url: str = "http://127.0.0.1:11434"
    # Apache-2.0 model, OK for commercial branding. 8 GB RAM: qwen3:4b; 16 GB+: qwen3:8b
    model: str = "qwen3:4b"
    max_document_chars: int = 12_000
    # Where "find files ..." looks
    search_roots: tuple[str, ...] = ("~/Desktop", "~/Documents", "~/Downloads", "~/OneDrive")
    # Global shortcut that shows the command bar from anywhere
    hotkey: str = "alt+space"
    assistant_name: str = "TechnoForesight Assistant"


def load_config(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if not path.exists() or tomllib is None:
        return cfg
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key in known:
            setattr(cfg, key, tuple(value) if key == "search_roots" else value)
    return cfg


DEFAULT_MODEL = Config.model
FAST_MODEL = "qwen3:1.7b"   # ~1.4 GB, roughly 2x faster on older laptops
SLOW_TOKENS_PER_SEC = 4.0   # below this, offer the fast model


def user_set(key: str, path: Path = CONFIG_PATH) -> bool:
    """True if the user's config file sets `key` explicitly."""
    if not path.exists():
        return False
    return any(line.split("=", 1)[0].strip() == key for line in path.read_text(encoding="utf-8").splitlines()
               if "=" in line and not line.lstrip().startswith("#"))


def save_setting(key: str, value: str, path: Path = CONFIG_PATH) -> None:
    """Sets `key = "value"` in config.toml, keeping every other line as it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    new_line = f'{key} = "{value}"'
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#") and line.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
