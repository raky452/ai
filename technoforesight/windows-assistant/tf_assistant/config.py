"""User settings, read from %APPDATA%\\TechnoForesight\\config.toml (all keys optional)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:  # Python 3.10 (Ubuntu 22.04): python3-tomli
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

if sys.platform == "win32":
    APP_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "TechnoForesight"
else:  # ~/.config/technoforesight (XDG)
    APP_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "technoforesight"
CONFIG_PATH = APP_DIR / "config.toml"

INDIC_MODEL = "gemma4:e4b"   # better Indian-language writing
INDIC_MODEL_SIZE = "6.6 GB"


DEFAULT_CONNECTORS = {
    "yt": "https://www.youtube.com/results?search_query={q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "gmail": "https://mail.google.com",
    "drive": "https://drive.google.com",
    "technoforesight": "https://technoforesight.in",
}


@dataclass
class Config:
    # Local Ollama server that runs the model. Nothing is sent to the internet.
    ollama_url: str = "http://127.0.0.1:11434"
    # Apache-2.0, OK for commercial branding. Must be a NON-thinking model: Ollama's plain
    # "qwen3:4b" tag is the thinking-only variant, which spends minutes reasoning on a laptop CPU.
    model: str = "qwen3:4b-instruct"
    max_document_chars: int = 12_000
    # Where "find files ..." looks
    search_roots: tuple[str, ...] = ("~/Desktop", "~/Documents", "~/Downloads", "~/OneDrive")
    # Global shortcut that shows the command bar from anywhere
    hotkey: str = "alt+space"
    # Language the AI answers in (see llm.LANGUAGES): English, Telugu, Hindi, Tamil, ...
    response_language: str = "English"
    # Model used when the response language isn't English. Gemma 4 (Apache-2.0) writes far cleaner
    # Telugu/Hindi/Tamil than small Qwen models. Until it's downloaded, `model` answers and the app
    # offers the download. Empty = use `model` for every language.
    indic_model: str = INDIC_MODEL
    assistant_name: str = "TechnoForesight Assistant"
    # Connectors: one word opens a website or tool. "{q}" is replaced by the rest of what you type.
    connectors: dict = field(default_factory=lambda: dict(DEFAULT_CONNECTORS))


def load_config(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if not path.exists() or tomllib is None:
        return cfg
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key == "connectors" and isinstance(value, dict):
            cfg.connectors.update({str(k).lower(): str(v) for k, v in value.items()})
        elif key in known:
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
    first_table = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[")), len(lines))
    for i, line in enumerate(lines[:first_table]):
        if "=" in line and not line.lstrip().startswith("#") and line.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        lines.insert(first_table, new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toml_str(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_connector(name: str, url: str | None, path: Path = CONFIG_PATH) -> None:
    """Adds/updates (url) or removes (url=None) a connector in the [connectors] table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    name = name.lower()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "[connectors]"), None)
    if start is None:
        if url is None:
            return
        lines += ["", "[connectors]"]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    entry = f"{name} = {_toml_str(url)}" if url is not None else None
    for i in range(start + 1, end):
        if "=" in lines[i] and lines[i].split("=", 1)[0].strip().strip('"').lower() == name:
            if entry is None:
                del lines[i]
            else:
                lines[i] = entry
            break
    else:
        if entry is not None:
            lines.insert(end, entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
