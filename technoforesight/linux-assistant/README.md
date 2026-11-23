# TechnoForesight Assistant for Linux

The same offline AI command bar as the Windows app, built from the **same code**
([`../windows-assistant/tf_assistant`](../windows-assistant/tf_assistant)). This folder only holds the Linux packaging.

**Download:** [technoforesight.in/assistant-linux](https://technoforesight.in/assistant-linux) · [Releases](https://github.com/raky452/ai/releases) (tags `linux-v*`)

| You type | It does |
|---|---|
| `open firefox`, `files`, `writer` | Launches the app (.desktop files, Flatpak, Snap) |
| `find files invoice` | Searches Desktop, Documents and Downloads; ↓ + Enter opens a result |
| `summarise ~/Downloads/report.pdf` | Summarises a PDF, Word (.docx) or text file with the local AI |
| `settings wifi`, `settings display` | Opens that page of GNOME, Cinnamon, KDE or Xfce settings |
| `lock pc` | Locks the screen (`loginctl`, `xdg-screensaver`, …) |
| `yt telugu songs`, `add connector store https://…` | Connectors |
| 🌐 menu | Answers in English, Telugu, Hindi, Tamil and 8 more languages (Gemma 4 offered for Indian languages) |
| anything else | Answered by the AI model on the computer |

## Install

**Ubuntu / Debian / Mint / Pop!_OS:** `sudo apt install ./TechnoForesightAssistant-Linux.deb`

**Fedora / Arch / openSUSE / others:** install Python's Tk (`python3-tkinter`, `tk`, `python3-tk`), extract
`TechnoForesightAssistant-Linux.tar.gz` and run `./install.sh` (installs to `~/.local`, no root).

On first launch the app:
1. adds the **Ctrl+Alt+Space** shortcut on GNOME/Ubuntu (other desktops: add a custom shortcut for `technoforesight-assistant`);
2. offers Ollama's official installer command (copied to the clipboard) if the AI engine is missing;
3. downloads the model (`qwen3:4b-instruct`), and `gemma4:e4b` when you pick an Indian language.

Command line: `technoforesight-assistant doctor`, `technoforesight-assistant ask "..."`, `technoforesight-assistant do "summarise file.pdf"`.

Settings: `~/.config/technoforesight/config.toml` (same keys as Windows: `model`, `indic_model`, `response_language`, `search_roots`, `[connectors]`).

## How the Linux build differs

| Concern | Linux implementation (`tf_assistant/linuxsys.py`, `apps.py`) |
|---|---|
| Global shortcut | GNOME custom keybinding via `gsettings` (added once, never overwrites others) |
| Single instance | Abstract Unix socket; a second launch tells the running window to show |
| Apps | `.desktop` files from XDG dirs, Flatpak and Snap; launched with `gio launch` |
| Open files/URLs | `xdg-open` |
| Window | Window-manager frame kept (borderless windows can't get keyboard focus reliably on X11/XWayland) |
| PDF text | `python3-pypdf` if installed, otherwise `pdftotext` |
| Fonts | Noto Sans (fontconfig covers every Indian script) |

## Build and test

```bash
./build.sh            # dist/technoforesight-assistant_<version>_all.deb, TechnoForesightAssistant-Linux.tar.gz
```

The `Build Linux Assistant` workflow runs the unit tests under Xvfb, builds both packages, installs the .deb on a
fresh Ubuntu, installs Ollama with real models and runs [`tools/e2e_linux.sh`](../windows-assistant/tools/e2e_linux.sh),
which checks CLI and window answers (including clean Telugu via `check_indic.py`). Run it manually with
**release = true** to publish `linux-v<version>`. The release step only runs after the end-to-end test passes, and it
never marks the release as *latest*.

| Version | Changes |
|---|---|
| 0.1.0 | GTK 4 prototype |
| 0.3.4 | Rebuilt on the shared Windows code: neon UI, 12 languages, Gemma 4 for Indian languages, connectors, settings pages, lock screen, first-run setup, .deb + tarball, end-to-end tested |
