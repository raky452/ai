# TechnoForesight Launcher

A privacy-first Android home-screen launcher with a built-in **offline AI assistant**.
The AI model runs entirely on the phone, so nothing the user types leaves the device.

> Status: **starter skeleton (v0.1)**. It covers the core structure and the first features, ready to open in Android Studio and extend.

## What works in this version

| Feature | How |
|---|---|
| Set as the phone's default home screen | `HOME` intent filter in `AndroidManifest.xml` |
| Clock, app grid and live app search | `ui/HomeScreen.kt`, `apps/AppRepository.kt` |
| Command bar: `open whatsapp`, `set alarm 6:30 am`, `call 98xxxxxx`, `search cricket score` | `ai/CommandRouter.kt`: instant rule-based routing that works without the model |
| Offline AI answers for anything else | `ai/LlmEngine.kt` (MediaPipe LLM Inference) |
| Model downloaded on first launch rather than bundled in the APK | `ai/ModelManager.kt` |

## Project layout

```
app/src/main/java/com/technoforesight/launcher/
├── MainActivity.kt          # Entry point, theme, back-button handling
├── LauncherViewModel.kt     # UI state, ties apps + commands + AI together
├── apps/AppRepository.kt    # Lists and launches installed apps
├── ai/
│   ├── CommandRouter.kt     # Text → phone action (open app, alarm, dial, search)
│   ├── LlmEngine.kt         # On-device model interface + MediaPipe implementation
│   └── ModelManager.kt      # First-launch model download
└── ui/HomeScreen.kt         # Jetpack Compose home screen
```

## Getting started

1. Install **Android Studio** (latest stable). Open this folder and let Gradle sync.
2. **Get a model.** Download a Gemma `.task` model that's compatible with MediaPipe LLM Inference
   (for example from Google's Gemma pages on Kaggle or Hugging Face; accept the licence there).
   Host the file somewhere you control, then set `MODEL_URL` in `app/build.gradle.kts`.
   - For quick local testing, skip hosting and push the file straight to the phone:
     `adb push model.task /data/local/tmp/` and copy it to `filesDir/assistant-model.task` (or temporarily point `ModelManager.modelFile` at it).
3. Run on a real phone (an emulator is too slow for the model). 6 GB+ RAM is recommended.
4. Press the Home button and choose **TechnoForesight** → **Always**.

## Roadmap (next steps)

- [ ] **Function calling**: have the model output JSON actions so free-form requests work ("wake me before sunrise", Hindi/Telugu commands)
- [ ] **Voice input** with whisper.cpp (offline speech-to-text)
- [ ] **Model choice**: swap `MediaPipeLlmEngine` for LiteRT-LM (Gemma 4) or llama.cpp (Qwen3.5 GGUF). The `LlmEngine` interface keeps the UI unchanged
- [ ] **Low-RAM fallback**: smaller model, or optional cloud mode
- [ ] Home-screen customisation: folders, dock, hide apps, icon packs
- [ ] Pro tier / white-label build flavours (`productFlavors`) for business customers
- [ ] Privacy policy + Play Store listing

## Licensing notes

- Use models whose licence allows commercial use under your brand (e.g. Apache 2.0: Gemma 4, Qwen3.5, Ministral 3).
  Keep the model's licence/NOTICE file inside the app (e.g. an "Open-source licences" screen).
- Don't use "Android" or Google trademarks in the product name.
- Google Play restricts use of the Accessibility API, so avoid it unless the feature truly needs it.

*This isn't legal advice. Have the licences reviewed before a commercial launch.*
