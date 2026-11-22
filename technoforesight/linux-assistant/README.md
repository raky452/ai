# TechnoForesight Assistant for Linux

An **offline AI command bar** for Linux laptops. Press a shortcut, type, and it will:

| You type | It does |
|---|---|
| `open firefox` / `writer` | Launches the app |
| `find files invoice` | Searches your home folder; click a result to open it |
| `summarise ~/Downloads/report.pdf` | Reads the PDF/text file and summarises it with the local AI |
| `www.technoforesight.in` / `google weather hyderabad` | Opens the site / searches the web |
| anything else | Answered by the AI model running **on the laptop**, with nothing sent to the cloud |

## How it works

```
GTK 4 window (ui.py) ──► router.py ──► actions.py ──┬─► apps.py   (.desktop files → launch)
                                                     ├─► files.py  (search, PDF/text reading)
                                                     └─► llm.py    (local Ollama server → model)
```

- **AI engine:** [Ollama](https://ollama.com) runs the model locally. The default is `qwen3:4b-instruct` (Apache 2.0, OK for commercial branding).
- **No pip dependencies at runtime.** GTK comes from the OS, and the Ollama client uses only the Python standard library.

## Install (Ubuntu / Debian / Mint)

```bash
cd technoforesight/linux-assistant
./install.sh                         # set TF_MODEL=qwen3:8b for 16 GB+ laptops
technoforesight-assistant doctor     # should say "Ready ✔"
```

Then bind a shortcut: **Settings → Keyboard → Custom Shortcuts**. Use the command `technoforesight-assistant` and a key such as **Super+Space**.

**Fedora:** `sudo dnf install python3-gobject gtk4 poppler-utils pipx`, then run the same `pipx install` and `ollama pull` steps as `install.sh`.

## Command-line mode (also handy for testing)

```bash
technoforesight-assistant ask "write a leave letter for tomorrow"
technoforesight-assistant do "summarise ~/notes.txt"
```

## Configuration

Optional: `~/.config/technoforesight/config.toml`

```toml
model = "qwen3:8b"                 # any model you've pulled with `ollama pull`
ollama_url = "http://127.0.0.1:11434"
search_roots = ["~/Documents", "~/Downloads"]
max_document_chars = 12000
assistant_name = "TechnoForesight Assistant"
```

| Laptop RAM | Suggested model |
|---|---|
| 8 GB | `qwen3:4b-instruct` or `qwen3:1.7b` |
| 16 GB | `qwen3:8b` |
| GPU / 32 GB | `qwen3:14b` or larger |

## Development

```bash
pip install -e '.[dev]' && pytest     # 15 tests: router, app discovery, files, streaming, Ollama client
```

## Roadmap

- [ ] Chat about a whole folder of documents (local search index)
- [ ] Voice input (whisper.cpp)
- [ ] AI-chosen actions (function calling) so free-form requests work, including Hindi and Telugu
- [ ] `.deb` package and Flatpak
- [ ] Bundle into a branded TechnoForesight Linux image (Ubuntu remix via Cubic). Don't use the Ubuntu name or logo, and publish the source of modified GPL parts
