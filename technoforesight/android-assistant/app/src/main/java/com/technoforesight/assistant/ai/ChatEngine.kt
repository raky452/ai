package com.technoforesight.assistant.ai

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL

/** Something that can answer: the on-device model, or Ollama on the user's own PC. */
interface ChatEngine {
    /** Short name shown in the footer, e.g. "Gemma 4 E2B · on phone". */
    val label: String

    /** Loads/checks the engine. Returns null when ready, otherwise a human-readable problem. */
    suspend fun prepare(): String?

    /** Streams the answer in pieces. Cancelling the collector stops generation. */
    fun stream(system: String, user: String): Flow<String>

    fun close() {}
}

/** Hides <think>…</think> reasoning if a thinking model is used on the PC. */
class ThinkFilter {
    private var buffer = ""
    private var thinking = false

    fun feed(chunk: String): String {
        buffer += chunk
        val out = StringBuilder()
        while (buffer.isNotEmpty()) {
            if (thinking) {
                val end = buffer.indexOf("</think>")
                if (end < 0) { buffer = buffer.takeLast(8); break }
                buffer = buffer.substring(end + 8).trimStart()
                thinking = false
            } else {
                val start = buffer.indexOf("<think>")
                if (start < 0) {
                    // keep a tail that could be the start of a split "<think>" tag
                    val keep = (minOf(6, buffer.length) downTo 1).firstOrNull { "<think>".startsWith(buffer.takeLast(it)) } ?: 0
                    val cut = buffer.length - keep
                    out.append(buffer, 0, cut)
                    buffer = buffer.substring(cut)
                    break
                }
                out.append(buffer, 0, start)
                buffer = buffer.substring(start + 7)
                thinking = true
            }
        }
        return out.toString()
    }

    fun flush(): String = if (thinking) "" else buffer.also { buffer = "" }
}

/** Ollama running on the user's laptop, reached over Wi-Fi (e.g. http://192.168.1.20:11434). */
class OllamaEngine(private val baseUrl: String, private val model: String) : ChatEngine {
    override val label get() = "$model · on your PC"

    private fun open(path: String, body: JSONObject? = null, timeoutMs: Int = 10_000): HttpURLConnection {
        val conn = URL(baseUrl.trimEnd('/') + path).openConnection() as HttpURLConnection
        conn.connectTimeout = 5_000
        conn.readTimeout = timeoutMs
        if (body != null) {
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.use { it.write(body.toString().toByteArray()) }
        }
        return conn
    }

    override suspend fun prepare(): String? = withContext(Dispatchers.IO) {
        try {
            val conn = open("/api/tags")
            val text = conn.inputStream.bufferedReader().use(BufferedReader::readText)
            val models = JSONObject(text).optJSONArray("models") ?: JSONArray()
            val names = (0 until models.length()).map { models.getJSONObject(it).optString("name") }
            val wanted = if (":" in model) model else "$model:latest"
            if (wanted !in names) "The PC doesn't have $model yet. On the PC run: ollama pull $model" else null
        } catch (e: Exception) {
            "Can't reach your PC at $baseUrl. Make sure Ollama is running there with OLLAMA_HOST=0.0.0.0 " +
                "and the phone is on the same Wi-Fi."
        }
    }

    override fun stream(system: String, user: String): Flow<String> = flow {
        val payload = JSONObject()
            .put("model", model)
            .put("stream", true)
            .put("think", false)
            .put("keep_alive", "30m")
            .put("messages", JSONArray()
                .put(JSONObject().put("role", "system").put("content", system))
                .put(JSONObject().put("role", "user").put("content", user)))
            .put("options", JSONObject()
                .put("num_predict", 700).put("temperature", 0.7).put("top_p", 0.8).put("top_k", 20)
                .put("repeat_penalty", 1.1).put("presence_penalty", 1.2).put("repeat_last_n", 256))
        val conn = open("/api/chat", payload, timeoutMs = 600_000)
        val think = ThinkFilter()
        try {
            if (conn.responseCode >= 400) {
                val err = conn.errorStream?.bufferedReader()?.readText().orEmpty()
                throw IllegalStateException(JSONObject(err.ifBlank { "{}" }).optString("error", "PC error ${conn.responseCode}"))
            }
            conn.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    currentCoroutineContext().ensureActive()
                    if (line.isBlank()) continue
                    val event = JSONObject(line)
                    if (event.has("error")) throw IllegalStateException(event.getString("error"))
                    val piece = think.feed(event.optJSONObject("message")?.optString("content").orEmpty())
                    if (piece.isNotEmpty()) emit(piece)
                    if (event.optBoolean("done")) break
                }
            }
            think.flush().takeIf { it.isNotEmpty() }?.let { emit(it) }
        } finally {
            conn.disconnect()
        }
    }.flowOn(Dispatchers.IO)
}
