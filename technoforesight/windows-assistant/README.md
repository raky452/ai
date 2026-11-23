# TechnoForesight Assistant for Windows

**Latest release: v0.3.3** · [Download](https://technoforesight.in/assistant) · [Releases](https://github.com/raky452/ai/releases) · Ad: [Telugu](https://youtu.be/6oqI3PJZu7c) / [English](https://youtu.be/US3ozZIHie0)

An **offline AI command bar** for Windows 10/11. Press **Alt+Space** from anywhere and type:

| You type | It does |
|---|---|
| `open chrome` · `word` · `calculator` · `whatsapp` | Opens any Start-menu app, including Microsoft Store apps |
| `find files invoice` | Searches Desktop, Documents, Downloads and OneDrive. Press ↓ and Enter to open a result |
| `summarise C:\Users\you\Downloads\report.pdf` | Summarises PDF, Word (.docx) or text files with the local AI |
| `settings wifi` · `settings bluetooth` · `settings display` | Jumps straight to that Windows Settings page |
| `lock pc` | Locks the computer |
| `yt telugu songs` · `maps hyderabad` · `gmail` | **Connectors**: one word opens a website or tool (`connectors` lists them) |
| `add connector store https://your-site.com` | Adds your own connector (use `{q}` for search links); `remove connector store` |
| 🌐 menu | AI answers in English, Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia or Urdu |
| `use model gemma4:e4b for indian languages` · `use model qwen3:8b` | Switches the AI model (downloads it with a progress bar). The first form applies only to non-English answers |
| `technoforesight.in` · `google ipl score` | Opens a website / searches the web |
| anything else | Answered by an AI model running **on your PC**. Nothing is uploaded |

## Install

1. Download **TechnoForesightAssistant-Setup.exe** from [technoforesight.in/assistant](https://technoforesight.in/assistant) or the [GitHub Releases](https://github.com/raky452/ai/releases).
2. Run it. If Windows shows *"Windows protected your PC"*, click **More info → Run anyway**. (The app isn't code-signed yet.) No admin rights are needed.
3. On first launch, click **Install Ollama**. That's the free engine that runs the AI. After installing it, click **Download AI model** (about 2.5 GB, one time).
4. Press **Alt+Space** and start typing.

**PC requirements:** Windows 10/11, 64-bit. 8 GB RAM works with the default `qwen3:4b-instruct` model (slower laptops: the app offers `qwen3:1.7b`); 16 GB+ can use `qwen3:8b`.

## The window

A borderless, rounded panel with a neon gradient edge:
- **Status pill:** amber while warming up or downloading, green for `LOCAL · OFFLINE` when ready.
- **🌐 language picker** in the header.
- **Quick-action chips:** Ask AI, Open app, Find files, Summarise, Settings, Connectors, Lock PC.
- **Thinking indicator:** animated, showing "Thinking… Ns"; press **Esc** to stop a slow answer.
- **AI RESPONSE card:** streams with a typing cursor, has a **Copy** button, and grows with the answer.
- **Footer:** model and tokens/sec readout, and keyboard hints.
- **Drag** the header to move the window.

On first launch the app checks for Ollama, offers to download the model, then runs a short **speed test**. On slow PCs it offers the faster `qwen3:1.7b`.

When you pick an Indian language (🌐 chip), the app offers a one-time download of **Gemma 4** (`gemma4:e4b`, ~6.6 GB), which writes much cleaner Telugu, Hindi, Tamil and others. Until it is downloaded, the main model keeps answering.

## Settings

Optional file: `%APPDATA%\TechnoForesight\config.toml`

```toml
hotkey = "ctrl+space"        # default alt+space
model = "qwen3:8b"           # any model from ollama.com/library
search_roots = ["~/Desktop", "~/Documents", "D:/Work"]
response_language = "Telugu"
indic_model = "gemma4:e4b"   # default; used only for non-English answers ("" = use model)

[connectors]
store = "https://technoforesight.in"
amz = "https://www.example.com/search?q={q}"
```

## Build from source

The `Build Windows Assistant` GitHub Action runs on every push to this folder. It runs the tests (including opening the real window), builds the exe and installer, smoke-tests the frozen exe, and renders UI preview screenshots (`tools/ui_preview.py`, also printed into the log).

**To publish a release:** bump `__version__` in `tf_assistant/__init__.py`, push, then run the workflow manually with **release = true**: Actions → Build Windows Assistant → Run workflow, or

```bash
gh api -X POST repos/raky452/ai/actions/workflows/windows-assistant.yml/dispatches -f ref=main -F 'inputs[release]=true'
```

It publishes `win-v<version>` with versioned files and stable-named `TechnoForesightAssistant-Setup.exe` / `-Portable.zip`. The website links to the stable names via `releases/latest/download/…`.

Locally, on Windows with Python 3.12 and Inno Setup 6:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

That produces `dist\TechnoForesightAssistant-Setup-<version>.exe` and a portable zip.

## Code map

```
tf_assistant/
  ui.py       neon command bar (tkinter): first-run setup, speed test, language menu, chips, streaming answer card
  winsys.py   global hotkey, single instance, via Win32 ctypes
  apps.py     Start-menu + Store apps (Get-StartApps), launching
  router.py   text → action rules (apps, files, settings, lock, connectors, web, AI)
  actions.py  carries out actions
  files.py    file search; TXT/PDF/DOCX text extraction
  llm.py      Ollama client: streaming chat (thinking off, anti-repetition sampling, cancel),
              model pull, speed test, 12-language system prompt, leaked-reasoning filter
  config.py   config.toml loading/saving, default connectors, default/fast models
installer/    Inno Setup script (per-user, no admin; closes a running copy before upgrading)
tools/        ui_preview.py: renders UI screenshots on CI
tests/        router, apps, files, hotkey, languages, connectors, Ollama client, UI smoke test
```

## Version history

| Version | Changes |
|---|---|
| 0.3.2 | `use model …` command; optional separate model for Indian-language answers (`indic_model`) |
| 0.3.3 | Gemma 4 (`gemma4:e4b`) is the default model for Indian languages: offered as a one-time download when you pick one, with the main model answering until then. Indian-language answers stay in one script; Markdown symbols (`**`, `#`) are cleaned out of answers |
| 0.3.1 | Qwen-recommended sampling and repetition penalties (no more looping Telugu answers) |
| 0.3.0 | Connectors: one word opens your sites; `add/remove connector`; Connectors chip |
| 0.2.0 | 12 response languages; new neon UI; answer card that fits its content |
| 0.1.3 | Default model `qwen3:4b-instruct` (plain `qwen3:4b` is thinking-only); leaked-reasoning filter |
| 0.1.2 | Installer closes the running app before upgrading |
| 0.1.1 | Thinking off, live timer, Esc to stop, speed test with faster-model offer, high-DPI fix |
| 0.1.0 | First release |

## Troubleshooting

- **Blank answer for minutes / reasoning text shown:** you're on a thinking model. Use `qwen3:4b-instruct` (default since 0.1.3) or `qwen3:1.7b`.
- **"DeleteFile failed; code 5" while upgrading:** an old copy is running. Close it in Task Manager (fixed in 0.1.2+).
- **Alt+Space doesn't work:** another app uses it. Set `hotkey = "ctrl+space"` in `config.toml`.
