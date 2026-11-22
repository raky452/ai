"""Local file search and text extraction (TXT, PDF, DOCX) so the AI can work with documents offline."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {
    ".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".html", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".bat", ".ps1", ".java", ".c", ".cpp", ".cs",
}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "AppData", "$Recycle.Bin"}


def find_files(query: str, roots: tuple[str, ...], limit: int = 30, max_depth: int = 6) -> list[Path]:
    q = query.strip().lower()
    if not q:
        return []
    results: list[Path] = []
    seen_roots: set[Path] = set()
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir() or base in seen_roots:
            continue
        seen_roots.add(base)
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


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - bundled in the Windows build
        raise RuntimeError("PDF support isn't installed (pip install pypdf).") from e
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_document(path: Path, max_chars: int) -> str:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    elif suffix in TEXT_SUFFIXES or suffix == "":
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise RuntimeError(f"Can't read {suffix} files yet. Supported: PDF, Word (.docx) and text files.")
    text = text.strip()
    if not text:
        raise RuntimeError("That file has no readable text (it may be a scanned image).")
    return text[:max_chars]


def open_path(path: Path) -> None:
    os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
