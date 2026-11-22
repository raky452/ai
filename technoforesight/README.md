# TechnoForesight: Private, offline AI assistants

One brand, two platforms. Both run open-source AI models **on the user's own device**, so nothing they type is sent to the cloud.

| Folder | Platform | Stack | Status |
|---|---|---|---|
| [`android-launcher/`](android-launcher/) | Android phones | Kotlin, Jetpack Compose, MediaPipe LLM Inference (Gemma) | Starter skeleton, not yet compiled |
| [`windows-assistant/`](windows-assistant/) | Windows 10/11 laptops | Python, tkinter, Ollama (Qwen3), Inno Setup installer | v0.1, built by GitHub Actions |
| [`linux-assistant/`](linux-assistant/) | Linux laptops (Ubuntu, Debian, Mint, Fedora) | Python, GTK 4, Ollama (Qwen3) | Working v0.1, 15 tests passing |

## Shared idea

1. **Command bar first:** open apps, find things and set alarms using instant rules that work without AI.
2. **AI for everything else:** a small open model (Apache 2.0) answers questions and summarises documents offline.
3. **Swappable engine:** each app talks to the model through a small interface, so models can change without UI rewrites.

## Licensing checklist before selling

- Use Apache-2.0/MIT models (Qwen3/3.5, Gemma 4, Ministral 3) and ship their licence notices.
- Don't use "Android", "Ubuntu" or other third-party trademarks in product names.
- If you ship a modified Linux distro, publish the source of the GPL parts you change.

*Not legal advice. Get a review before commercial launch.*
