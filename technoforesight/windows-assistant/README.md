# TechnoForesight Assistant for Windows

An **offline AI command bar** for Windows 10/11. Press **Alt+Space** from anywhere and type:

| You type | It does |
|---|---|
| `open chrome` · `word` · `calculator` · `whatsapp` | Opens any Start-menu app, including Microsoft Store apps |
| `find files invoice` | Searches Desktop, Documents, Downloads and OneDrive. Press ↓ and Enter to open a result |
| `summarise C:\Users\you\Downloads\report.pdf` | Summarises PDF, Word (.docx) or text files with the local AI |
| `settings wifi` · `settings bluetooth` · `settings display` | Jumps straight to that Windows Settings page |
| `lock pc` | Locks the computer |
| `technoforesight.in` · `google ipl score` | Opens a website / searches the web |
| anything else | Answered by an AI model running **on your PC**. Nothing is uploaded |

## Install

1. Download **TechnoForesightAssistant-Setup.exe** from [technoforesight.in/assistant](https://technoforesight.in/assistant) or the [GitHub Releases](https://github.com/raky452/ai/releases).
2. Run it. If Windows shows *"Windows protected your PC"*, click **More info → Run anyway**. (The app isn't code-signed yet.) No admin rights are needed.
3. On first launch, click **Install Ollama**. That's the free engine that runs the AI. After installing it, click **Download AI model** (about 2.5 GB, one time).
4. Press **Alt+Space** and start typing.

**PC requirements:** Windows 10/11, 64-bit. 8 GB RAM works with the default `qwen3:4b-instruct` model (slower laptops: the app offers `qwen3:1.7b`); 16 GB+ can use `qwen3:8b`.

## Settings

Optional file: `%APPDATA%\TechnoForesight\config.toml`

```toml
hotkey = "ctrl+space"        # default alt+space
model = "qwen3:8b"           # any model from ollama.com/library
search_roots = ["~/Desktop", "~/Documents", "D:/Work"]
```

## Build from source

The `Build Windows Assistant` GitHub Action builds everything on every push. Pushing a `win-v*` tag publishes a Release.

Locally, on Windows with Python 3.12 and Inno Setup 6:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

That produces `dist\TechnoForesightAssistant-Setup-<version>.exe` and a portable zip.

## Code map

```
tf_assistant/
  ui.py       tkinter command bar, first-run setup (Ollama + model download with progress)
  winsys.py   global hotkey, single instance, via Win32 ctypes
  apps.py     Start-menu + Store apps (Get-StartApps), launching
  router.py   text → action rules (apps, files, settings, lock, web, AI)
  actions.py  carries out actions
  files.py    file search; TXT/PDF/DOCX text extraction
  llm.py      local Ollama client (streaming chat, model pull)
installer/    Inno Setup script (per-user install, optional start-at-login)
```
