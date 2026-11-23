# ai

Rakesh's work on AI and ML models.

| Folder | What it is |
|---|---|
| [`reinforcement/`](reinforcement/) | `QLearningModel.py`: trains a Q-learning model with the OpenAI Gym package |
| [`technoforesight/`](technoforesight/) | **TechnoForesight Assistant**: private, offline AI assistants for Windows, Linux and Android, plus the launch-ad renderer. See its [README](technoforesight/README.md) |

## TechnoForesight Assistant at a glance

- **Windows** ([`technoforesight/windows-assistant`](technoforesight/windows-assistant)): Alt+Space AI command bar, answers in 12 languages including Telugu and Hindi (Gemma 4 for Indian languages), connectors and an installer. **Download:** [technoforesight.in/assistant](https://technoforesight.in/assistant)
- **Linux** ([`technoforesight/linux-assistant`](technoforesight/linux-assistant)): the same app from the same code, as a .deb and a portable tarball. **Download:** [technoforesight.in/assistant-linux](https://technoforesight.in/assistant-linux)
- **Android** ([`technoforesight/android-assistant`](technoforesight/android-assistant)): Gemma 4 running on the phone, plus "use my PC". **Download:** [technoforesight.in/assistant-android](https://technoforesight.in/assistant-android)
- **Launch ad** ([`technoforesight/ad`](technoforesight/ad)): motion-graphics renderer for the Telugu and English ads ([Telugu](https://youtu.be/6oqI3PJZu7c), [English](https://youtu.be/US3ozZIHie0)).

## Automation (GitHub Actions)

| Workflow | What it does |
|---|---|
| `windows-assistant.yml` | Tests, builds the exe and installer on Windows, and renders UI preview screenshots on every push to `technoforesight/windows-assistant/**`. Run it manually with **release = true** to publish `win-v<version>` (add **prerelease = true** to test before the website serves it) |
| `linux-assistant.yml` | Unit tests under Xvfb, builds the .deb and tarball, installs the .deb, runs the end-to-end test with real models, and (with **release = true**) publishes `linux-v<version>` |
| `android-assistant.yml` | Unit tests, builds the APK, runs the emulator end-to-end test (PC mode and on-device Gemma 4), and (with **release = true**, signing secrets required) publishes `android-v<version>` |
| `fetch-fonts.yml` | Puts the open-licence Noto fonts for Indian scripts on the `ad-assets` branch |
| `fetch-media.yml` | Downloads generated voice-overs and music into the `ad-assets` branch |

Linux and Android releases are published with `--latest=false`, so `releases/latest` always means the Windows release that the website's Windows buttons use.

## Branches

- `main`: all source code
- `ad-assets`: fonts, voice-overs, music and rendered ad videos (large binary files kept out of `main`)
