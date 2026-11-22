# ai

Rakesh's work on AI and ML models.

| Folder | What it is |
|---|---|
| [`reinforcement/`](reinforcement/) | `QLearningModel.py`: trains a Q-learning model with the OpenAI Gym package |
| [`technoforesight/`](technoforesight/) | **TechnoForesight Assistant**: private, offline AI assistants for Windows, Linux and Android, plus the launch-ad renderer. See its [README](technoforesight/README.md) |

## TechnoForesight Assistant at a glance

- **Windows** ([`technoforesight/windows-assistant`](technoforesight/windows-assistant)): Alt+Space AI command bar, answers in 12 languages including Telugu and Hindi, with connectors and an installer. **Latest release:** [win-v0.3.2](https://github.com/raky452/ai/releases/latest) · **Download page:** [technoforesight.in/assistant](https://technoforesight.in/assistant)
- **Linux** ([`technoforesight/linux-assistant`](technoforesight/linux-assistant)): GTK 4 command bar on Ollama.
- **Android** ([`technoforesight/android-launcher`](technoforesight/android-launcher)): home-screen launcher skeleton with an on-device model.
- **Launch ad** ([`technoforesight/ad`](technoforesight/ad)): motion-graphics renderer for the Telugu and English ads ([Telugu](https://youtu.be/6oqI3PJZu7c), [English](https://youtu.be/US3ozZIHie0)).

## Automation (GitHub Actions)

| Workflow | What it does |
|---|---|
| `windows-assistant.yml` | Tests, builds the exe and installer on Windows, and renders UI preview screenshots on every push to `technoforesight/windows-assistant/**`. Run it manually with **release = true** to publish a `win-v<version>` release |
| `fetch-fonts.yml` | Puts the open-licence Noto fonts for Indian scripts on the `ad-assets` branch |
| `fetch-media.yml` | Downloads generated voice-overs and music into the `ad-assets` branch |

## Branches

- `main`: all source code
- `ad-assets`: fonts, voice-overs, music and rendered ad videos (large binary files kept out of `main`)
