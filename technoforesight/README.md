# TechnoForesight: Private, offline AI assistants

One brand, three platforms. Each runs open-source AI models **on the user's own device**, so nothing they type is sent to the cloud.

| Folder | Platform | Stack | Status |
|---|---|---|---|
| [`windows-assistant/`](windows-assistant/) | Windows 10/11 | Python, tkinter, Ollama (`qwen3:4b-instruct`), Inno Setup | **v0.3.1 released**, 21 tests, built by GitHub Actions. [Download](https://technoforesight.in/assistant) |
| [`linux-assistant/`](linux-assistant/) | Ubuntu, Debian, Mint, Fedora | Python, GTK 4, Ollama (`qwen3:4b-instruct`) | v0.1, 15 tests |
| [`android-launcher/`](android-launcher/) | Android phones | Kotlin, Jetpack Compose, MediaPipe LLM Inference (Gemma) | Starter skeleton, not yet compiled |
| [`ad/`](ad/) | Launch ad (video) | Python (PIL + numpy) motion graphics, ffmpeg | Telugu and English ads live on YouTube and Instagram |

## Features by platform

| Feature | Windows | Linux | Android |
|---|:-:|:-:|:-:|
| Global shortcut command bar | ✅ Alt+Space | ✅ (system shortcut) | n/a (home screen) |
| Open apps / find files | ✅ | ✅ | ✅ apps |
| Summarise PDF / Word / text | ✅ | ✅ PDF and text | — |
| Answers in 12 languages (Telugu, Hindi, Tamil, …) | ✅ | — | — |
| Connectors (one word opens a site) | ✅ | — | — |
| Windows Settings shortcuts, lock PC | ✅ | — | — |
| First-run setup (install engine, download model, speed test) | ✅ | CLI `doctor` | model download |
| Installer / package | ✅ Setup.exe + portable zip | `install.sh` (pipx) | — |

## Shared design

1. **Command bar first:** instant rules handle apps, files, settings and connectors, with no AI needed.
2. **AI for everything else:** a small Apache-2.0 model answers questions and summarises documents offline.
3. **Swappable engine:** each app talks to the model through a small client, so models can change without UI rewrites.
4. **Non-thinking models only:** Ollama's plain `qwen3:4b` tag is a *thinking* model that spends minutes reasoning on laptop CPUs, so the default is `qwen3:4b-instruct`. Slow PCs are offered `qwen3:1.7b`.
5. **No looping answers:** generation uses Qwen's recommended sampling plus presence/repeat penalties, which keep Indian-language answers from repeating.

## Licensing checklist before selling

- Use Apache-2.0/MIT models (Qwen3, Gemma 4, Ministral 3) and ship their licence notices.
- Don't use "Android", "Windows", "Ubuntu" or other third-party trademarks in product names.
- If you ship a modified Linux distro, publish the source of the GPL parts you change.
- Ads: say that the voice and music are AI-generated, and make only claims the product supports. See [`ad/README.md`](ad/README.md).

*Not legal advice. Get a review before commercial launch.*
