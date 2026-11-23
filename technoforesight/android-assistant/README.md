# TechnoForesight Assistant for Android

Offline AI on the phone: Google's **Gemma 4 E2B** (Apache-2.0) runs on-device through
[LiteRT-LM](https://ai.google.dev/edge/litert-lm). It has the same commands and languages as the desktop apps.

**Download:** [technoforesight.in/assistant-android](https://technoforesight.in/assistant-android) · [Releases](https://github.com/raky452/ai/releases) (tags `android-v*`)

| You type / do | It does |
|---|---|
| anything | Answer from Gemma 4 on the phone, in the 🌐 language (12 languages, same instructions as desktop) |
| `open whatsapp`, `camera` | Opens an installed app |
| `settings wifi`, `settings battery` | Opens that Settings page |
| `summarise` (or the chip) | Pick a PDF, Word or text file, get 5 key points plus dates, amounts and actions |
| select text in any app → **Ask TechnoForesight**, or share text | Explains the text |
| `yt telugu songs`, `add connector store https://…`, `connectors` | Connectors |
| `language telugu` | Changes the answer language |
| `use pc 192.168.1.20` / `use phone` | Use Ollama on your laptop over Wi-Fi (bigger model) / back to on-device |

## How it works

```
MainActivity (Compose UI, share / PROCESS_TEXT / ASSIST intents)
  └─ AssistantViewModel ── route() (ai/CommandRouter.kt, mirrors desktop router.py)
        ├─ OnDeviceEngine  (LiteRT-LM, GPU → CPU fallback)    ← ModelStore: resumable 2.6 GB download
        ├─ OllamaEngine    ("use my PC", streaming /api/chat)
        └─ ScriptFilter → MarkdownCleaner (clean Indian-language output while streaming)
```

- Model: `gemma-4-E2B-it.litertlm` from [litert-community](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), downloaded once into app-private storage.
- PDF text: pdfbox-android. Word: reads `word/document.xml`.
- Network: only the model download (HTTPS) and the optional PC mode (plain HTTP on the home network).

## Build and test

```bash
./gradlew testDebugUnitTest assembleRelease
```

The `Build Android Assistant` workflow:
1. runs the unit tests (router, cleaners, languages) and builds the APK;
2. starts an x86_64 emulator and runs [`ci/emulator-e2e.sh`](ci/emulator-e2e.sh):
   - **PC mode:** Ollama with `gemma4:e4b` on the CI host (`10.0.2.2`). Asks a Telugu and an English question through the real UI.
   - **Connectors:** adds and lists a connector.
   - **On-device:** pushes the Gemma 4 E2B file into the app and asks the Telugu question on the emulator's CPU.
3. with **release = true**, publishes `android-v<version>` (never *latest*, so the website's Windows links don't move).

### Signing

Release APKs are signed from two repository secrets: `ANDROID_KEYSTORE_B64` (base64 of the `.jks`, alias
`technoforesight`) and `ANDROID_KEYSTORE_PASSWORD`. Without them the release build is debug-signed and the workflow
refuses to publish it. **Keep the keystore safe:** every future update must be signed with the same key, or users
have to uninstall before updating.

| Version | Changes |
|---|---|
| 0.3.4 | First release: on-device Gemma 4 E2B, use-my-PC mode, 12 languages, summarise, connectors, settings, share/select-text integration |
