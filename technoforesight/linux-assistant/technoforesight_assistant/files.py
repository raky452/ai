"""Local file search and text extraction, so the assistant can work with the user's documents offline."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

TEXT_SUFFIXES = {
    ".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".html", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".sh", ".java", ".kt", ".c", ".cpp", ".rs", ".go",
}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "snap", ".cache"}


def find_files(query: str, roots: tuple[str, ...] = ("~",), limit: int = 30, max_depth: int = 6) -> list[Path]:
    """Case-insensitive filename search. Skips hidden and bulky folders to stay fast."""
    q = query.strip().lower()
    if not q:
        return []
    results: list[Path] = []
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        base_depth = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base):
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in SKIP_DIRS and depth < max_depth
            ]
            for name in filenames:
                if q in name.lower():
                    results.append(Path(dirpath) / name)
                    if len(results) >= limit:
                        return results
    return results


def read_document(path: Path, max_chars: int) -> str:
    """Plain text for text-like files and PDFs (via poppler's `pdftotext`, if installed)."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not shutil.which("pdftotext"):
            raise RuntimeError("Install 'poppler-utils' to read PDFs (sudo apt install poppler-utils).")
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        text = out.stdout
    elif suffix in TEXT_SUFFIXES or suffix == "":
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise RuntimeError(f"Can't read {suffix} files yet. Supported: PDF and text files.")
    return text[:max_chars]


def open_path(path: Path) -> None:
    subprocess.Popen(
        ["xdg-open", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
