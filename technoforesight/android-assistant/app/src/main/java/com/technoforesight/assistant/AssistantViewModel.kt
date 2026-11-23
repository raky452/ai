package com.technoforesight.assistant

import android.app.Application
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.technoforesight.assistant.ai.AppEntry
import com.technoforesight.assistant.ai.ChatEngine
import com.technoforesight.assistant.ai.Command
import com.technoforesight.assistant.ai.Language
import com.technoforesight.assistant.ai.MarkdownCleaner
import com.technoforesight.assistant.ai.ModelStore
import com.technoforesight.assistant.ai.OllamaEngine
import com.technoforesight.assistant.ai.OnDeviceEngine
import com.technoforesight.assistant.ai.ScriptFilter
import com.technoforesight.assistant.ai.languageByName
import com.technoforesight.assistant.ai.route
import com.technoforesight.assistant.ai.systemPrompt
import com.technoforesight.assistant.apps.AppRepository
import com.technoforesight.assistant.docs.DocumentReader
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch

enum class Tone { INFO, OK, WARN, ERROR }
enum class EngineState(val label: String) {
    STARTING("STARTING"), NEEDS_MODEL("NO MODEL"), DOWNLOADING("DOWNLOADING"), LOADING("LOADING"),
    READY("LOCAL AI"), PC_READY("YOUR PC"), ERROR("ENGINE ERROR")
}

class AssistantViewModel(app: Application) : AndroidViewModel(app) {
    private val ctx get() = getApplication<Application>()
    private val prefs = Prefs(app)
    private val repo = AppRepository(app)
    val modelStore = ModelStore(app, BuildConfig.MODEL_URL, BuildConfig.MODEL_FILE)

    var input by mutableStateOf("")
    var suggestions by mutableStateOf(emptyList<AppEntry>())
    var status by mutableStateOf("Starting…")
    var tone by mutableStateOf(Tone.INFO)
    var engineState by mutableStateOf(EngineState.STARTING)
    var engineLabel by mutableStateOf("")
    var answer by mutableStateOf("")
    var answerTitle by mutableStateOf("AI RESPONSE")
    var streaming by mutableStateOf(false)
    var busy by mutableStateOf(false)
    var progress by mutableStateOf<Float?>(null)
    var language by mutableStateOf(languageByName(prefs.language))
    var showDownload by mutableStateOf(false)

    /** One-off requests to the Activity (open the document picker). */
    private val _events = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val events = _events.asSharedFlow()

    private var apps = emptyList<AppEntry>()
    private var engine: ChatEngine? = null
    private var job: Job? = null
    private var downloadJob: Job? = null

    init {
        viewModelScope.launch { apps = repo.load() }
        startEngine()
    }

    private fun say(text: String, t: Tone = Tone.INFO) { status = text; tone = t }

    // ---------------- engine ----------------
    fun startEngine() {
        engine?.close()
        engine = null
        viewModelScope.launch {
            if (prefs.engine == "pc") {
                val e = OllamaEngine(prefs.pcUrl, prefs.pcModel)
                engineState = EngineState.LOADING
                engineLabel = e.label
                say("Connecting to your PC at ${prefs.pcUrl}…")
                val problem = e.prepare()
                if (problem == null) {
                    engine = e
                    engineState = EngineState.PC_READY
                    say("✔ Connected to your PC. Answers stay on your own devices.", Tone.OK)
                } else {
                    engineState = EngineState.ERROR
                    say("⚠ $problem  (Type 'use phone' to go back to on-device AI.)", Tone.ERROR)
                }
                return@launch
            }
            engineLabel = "${BuildConfig.MODEL_NAME} · on phone"
            if (!modelStore.isDownloaded) {
                engineState = EngineState.NEEDS_MODEL
                showDownload = true
                val resume = if (modelStore.partialBytes > 0) " (resumes where it stopped)" else ""
                say("One-time setup: download the free AI model (${BuildConfig.MODEL_NAME}, about 2.6 GB, Wi-Fi " +
                    "recommended)$resume. After that it works offline.", Tone.WARN)
                return@launch
            }
            loadOnDevice()
        }
    }

    private suspend fun loadOnDevice() {
        val e = OnDeviceEngine(ctx, modelStore.file, BuildConfig.MODEL_NAME)
        engineState = EngineState.LOADING
        say("Loading the AI model into memory (about 10 seconds)…")
        val problem = e.prepare()
        if (problem == null) {
            engine = e
            engineLabel = e.label
            engineState = EngineState.READY
            say("✔ AI ready. Ask me anything. It all stays on this phone.", Tone.OK)
        } else {
            engineState = EngineState.ERROR
            say("⚠ $problem  Tip: close other apps, or type 'use pc <laptop IP>' to use your PC's AI.", Tone.ERROR)
        }
    }

    fun downloadModel() {
        if (downloadJob?.isActive == true) return
        showDownload = false
        engineState = EngineState.DOWNLOADING
        downloadJob = viewModelScope.launch {
            try {
                modelStore.download { done, total ->
                    val frac = if (total > 0) done.toFloat() / total else -1f
                    progress = frac
                    val pct = if (total > 0) " ${(frac * 100).toInt()}%" else ""
                    say("Downloading ${BuildConfig.MODEL_NAME}: ${done / 1_000_000} of ${total / 1_000_000} MB$pct. " +
                        "Keep the app open.")
                }
                progress = null
                loadOnDevice()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                progress = null
                engineState = EngineState.NEEDS_MODEL
                showDownload = true
                say("⚠ ${e.message ?: "Download failed"}. Tap the button to resume.", Tone.ERROR)
            }
        }
    }

    // ---------------- typing ----------------
    fun onInput(text: String) {
        input = text
        val q = text.trim()
        suggestions = if (q.length >= 2 && !busy) {
            apps.filter { q.lowercase() in it.label.lowercase() }.take(5)
        } else emptyList()
    }

    fun openSuggestion(app: AppEntry) {
        repo.launch(app.packageName)
        input = ""
        suggestions = emptyList()
    }

    fun stop() {
        job?.cancel()
        streaming = false
        busy = false
        say("Stopped.", Tone.INFO)
    }

    fun chooseLanguage(lang: Language) {
        language = lang
        prefs.language = lang.name
        say("✔ I'll answer in ${lang.native} from now on.", Tone.OK)
    }

    fun submit(text: String = input) {
        val line = text.trim()
        if (line.isEmpty() || busy) return
        suggestions = emptyList()
        when (val cmd = route(line, apps, prefs.connectors)) {
            is Command.OpenApp -> { repo.launch(cmd.packageName); say("Opening ${cmd.label}…"); input = "" }
            is Command.OpenSettings -> {
                openIntent(Intent(cmd.action), fallback = Intent("android.settings.SETTINGS"))
                say("Opening Settings → ${cmd.page}"); input = ""
            }
            is Command.OpenUrl -> { view(cmd.url); say("Opening ${cmd.url}"); input = "" }
            is Command.WebSearch -> {
                view("https://www.google.com/search?q=" + Uri.encode(cmd.query))
                say("Searching the web for “${cmd.query}”"); input = ""
            }
            is Command.OpenConnector -> { view(cmd.url); say("🔗 ${cmd.name} → ${cmd.url}"); input = "" }
            Command.ListConnectors -> {
                val rows = prefs.connectors.entries.joinToString("\n") { "${it.key}  →  ${it.value}" }
                showAnswer("CONNECTORS", rows + "\n\nAdd your own:  add connector store https://your-site.com\n" +
                    "Use {q} for searches:  add connector amz https://example.com/search?q={q}\n" +
                    "Remove:  remove connector store")
                input = ""
            }
            is Command.AddConnector -> {
                prefs.connectors = prefs.connectors + (cmd.name to cmd.url)
                say("✔ Connector '${cmd.name}' added. Type '${cmd.name}' to open ${cmd.url}", Tone.OK); input = ""
            }
            is Command.RemoveConnector -> {
                val had = cmd.name in prefs.connectors
                prefs.connectors = prefs.connectors - cmd.name
                say(if (had) "✔ Connector '${cmd.name}' removed." else "No connector named '${cmd.name}'.",
                    if (had) Tone.OK else Tone.WARN)
                input = ""
            }
            Command.Summarise -> { input = ""; _events.tryEmit("pick_document") }
            is Command.SetLanguage -> { chooseLanguage(cmd.language); input = "" }
            is Command.UsePc -> {
                prefs.engine = "pc"; prefs.pcUrl = cmd.url; input = ""
                startEngine()
            }
            Command.UsePhone -> { prefs.engine = "phone"; input = ""; startEngine() }
            is Command.Ask -> ask(cmd.text, title = "AI RESPONSE")
        }
    }

    fun summarise(uri: Uri) {
        val name = DocumentReader.displayName(ctx, uri)
        viewModelScope.launch {
            val text = try {
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { DocumentReader.read(ctx, uri) }
            } catch (e: Exception) {
                say("⚠ Couldn't read $name: ${e.message}", Tone.ERROR); return@launch
            }
            ask("Summarise this document ($name) in 5 short bullet points, then list any dates, amounts or " +
                "action items.\n\n---\n$text\n---", title = "SUMMARY · $name", echo = "summarise $name")
        }
    }

    /** Text shared from another app or selected with "Ask TechnoForesight". */
    fun askAbout(text: String) {
        if (text.isBlank()) return
        ask("Explain this briefly:\n\n${text.take(6000)}", title = "AI RESPONSE", echo = text.take(80))
    }

    private fun ask(prompt: String, title: String, echo: String? = null) {
        val e = engine
        if (e == null) {
            say(if (engineState == EngineState.NEEDS_MODEL) "⬇ Download the AI model first (button below)."
                else "The AI engine isn't ready yet.", Tone.WARN)
            return
        }
        input = echo ?: input
        busy = true
        answer = ""
        answerTitle = title
        say("Thinking…")
        val system = systemPrompt("TechnoForesight Assistant", language.name)
        job = viewModelScope.launch {
            val cleaner = MarkdownCleaner()
            val script = if (language.name != "English") ScriptFilter() else null
            try {
                e.stream(system, prompt).collect { piece ->
                    if (!streaming) { streaming = true; say("") }
                    answer += cleaner.feed(script?.feed(piece) ?: piece)
                }
                answer += cleaner.feed(script?.flush() ?: "") + cleaner.flush()
                say(if (answer.isBlank()) "⚠ The model returned an empty answer. Try again." else "", Tone.INFO)
            } catch (c: CancellationException) {
                throw c
            } catch (ex: Exception) {
                say("⚠ ${ex.message ?: "Something went wrong"}", Tone.ERROR)
            } finally {
                streaming = false
                busy = false
            }
        }
    }

    private fun showAnswer(title: String, text: String) {
        answerTitle = title
        answer = text
    }

    private fun view(url: String) = openIntent(Intent(Intent.ACTION_VIEW, Uri.parse(url)))

    private fun openIntent(intent: Intent, fallback: Intent? = null) {
        try {
            ctx.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (e: ActivityNotFoundException) {
            if (fallback != null) ctx.startActivity(fallback.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            else say("⚠ No app can open that.", Tone.ERROR)
        }
    }

    override fun onCleared() {
        engine?.close()
    }
}
