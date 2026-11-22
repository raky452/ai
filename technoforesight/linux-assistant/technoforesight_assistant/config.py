"""User settings, read from ~/.config/technoforesight/config.toml (all keys optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Ubuntu 22.04 ships Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "technoforesight" / "config.toml"
)


@dataclass
class Config:
    # Local Ollama server that runs the model. Nothing is sent to the internet.
    ollama_url: str = "http://127.0.0.1:11434"
    # Any model pulled into Ollama. Apache-2.0 models are safest for commercial branding.
    # Small laptops (8 GB RAM): "qwen3:4b-instruct".  16 GB+: "qwen3:8b".
    model: str = "qwen3:4b-instruct"  # non-thinking variant; plain qwen3:4b is thinking-only
    # How much of a document to feed the model when summarising.
    max_document_chars: int = 12_000
    # Where "find files ..." looks.
    search_roots: tuple[str, ...] = ("~",)
    assistant_name: str = "TechnoForesight Assistant"


def load_config(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if not path.exists() or tomllib is None:
        return cfg
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key not in known:
            continue
        if key == "search_roots":
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg
