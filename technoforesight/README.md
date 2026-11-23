# TechnoForesight: Private, offline AI assistants

One brand, three platforms. Each runs open-source AI models **on the user's own device**, so nothing they type is sent to the cloud.

| Folder | Platform | Stack | Status |
|---|---|---|---|
| [`windows-assistant/`](windows-assistant/) | Windows 10/11 **and the shared app code** | Python, tkinter, Ollama, Inno Setup | **v0.3.4**, 30 tests, real-window test on Windows CI. [Download](https://technoforesight.in/assistant) |
| [`linux-assistant/`](linux-assistant/) | Ubuntu, Debian, Mint, Fedora, Arch | Same `tf_assistant` code, packaged as .deb + tarball | **v0.3.4**, end-to-end tested on CI with real models. [Download](https://technoforesight.in/assistant-linux) |
| [`android-assistant/`](android-assistant/) | Android 8+ phones | Kotlin, Jetpack Compose, LiteRT-LM (Gemma 4 E2B on-device) | **v0.3.4**, unit + emulator end-to-end tests. [Download](https://technoforesight.in/assistant-android) |
| [`android-launcher/`](android-launcher/) | Android home screen | Kotlin, Compose, MediaPipe | Early launcher skeleton (superseded by `android-assistant`) |
| [`ad/`](ad/) | Launch ad (video) | Python (PIL + numpy) motion graphics, ffmpeg | Telugu and English ads live on YouTube and Instagram |

## Features by platform

| Feature | Windows | Linux | Android |
|---|:-:|:-:|:-:|
| Quick access | ✅ Alt+Space | ✅ Ctrl+Alt+Space (GNOME, set automatically) | ✅ app, share text, select text → *Ask TechnoForesight* |
| Open apps | ✅ Start menu + Store apps | ✅ .desktop, Flatpak, Snap | ✅ |
| Find files | ✅ | ✅ | — (Android storage rules) |
| Summarise PDF / Word / text | ✅ | ✅ | ✅ (file picker) |
| Answers in 12 languages (Telugu, Hindi, Tamil, …) | ✅ | ✅ | ✅ |
| Better model for Indian languages | ✅ Gemma 4 E4B (offered automatically) | ✅ Gemma 4 E4B | ✅ Gemma 4 E2B on-device |
| Connectors (one word opens a site) | ✅ | ✅ | ✅ |
| Settings pages, lock screen | ✅ | ✅ GNOME/Cinnamon/KDE/Xfce | ✅ settings pages |
| Use the laptop's AI from the phone | — | — | ✅ `use pc <ip>` |
| Package | Setup.exe + portable zip | .deb + tarball (`install.sh`) | APK |

## How each platform is tested

- **Windows:** unit tests, a real Tk window test and a smoke test of the frozen exe on GitHub's Windows runner, then a manual end-to-end check on a real laptop before a pre-release is promoted to *latest*.
- **Linux:** CI installs the .deb on a fresh Ubuntu, installs Ollama with `qwen3:4b-instruct` and `gemma4:e4b`, and runs [`e2e_linux.sh`](windows-assistant/tools/e2e_linux.sh): CLI answers (English, Telugu email, PDF summary), connectors, then the real window under Xvfb driven with xdotool (Telugu answer, Ctrl+A, PDF summary, single instance). Answers are checked by [`check_indic.py`](windows-assistant/tools/check_indic.py).
- **Android:** JVM unit tests, then an emulator run of [`EndToEndTest`](android-assistant/app/src/androidTest/java/com/technoforesight/assistant/EndToEndTest.kt): Telugu and English answers through "use my PC" (Ollama on the CI host), connectors, and Gemma 4 E2B on-device in Telugu.

## Shared design

1. **Command bar first:** instant rules handle apps, files, settings and connectors, with no AI needed.
2. **AI for everything else:** a small Apache-2.0 model answers questions and summarises documents offline.
3. **One codebase for desktops:** Windows and Linux run the same `tf_assistant` package; only `winsys.py` / `linuxsys.py` differ.
4. **Swappable engine:** each app talks to the model through a small client, so models can change without UI rewrites.
5. **Clean Indian-language output:** answers are filtered while streaming. Markdown symbols are removed, and so are stray tokens from unrelated scripts (small models occasionally emit a Chinese character or a Vietnamese word inside Telugu).
6. **Non-thinking models only:** Ollama's plain `qwen3:4b` tag is a *thinking* model that spends minutes reasoning on laptop CPUs, so the default is `qwen3:4b-instruct`. Slow PCs are offered `qwen3:1.7b`.
7. **No looping answers:** generation uses Qwen's recommended sampling plus presence/repeat penalties, which keep Indian-language answers from repeating.

## Licensing checklist before selling

- Use Apache-2.0/MIT models (Qwen3, Gemma 4, Ministral 3) and ship their licence notices.
- Don't use "Android", "Windows", "Ubuntu" or other third-party trademarks in product names.
- If you ship a modified Linux distro, publish the source of the GPL parts you change.
- Ads: say that the voice and music are AI-generated, and make only claims the product supports. See [`ad/README.md`](ad/README.md).

*Not legal advice. Get a review before commercial launch.*
