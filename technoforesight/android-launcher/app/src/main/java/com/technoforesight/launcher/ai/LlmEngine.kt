package com.technoforesight.launcher.ai

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Runtime-agnostic interface for the on-device model.
 * Swap the implementation (MediaPipe, LiteRT-LM, llama.cpp) without touching the UI.
 */
interface LlmEngine {
    val isReady: Boolean
    suspend fun load(modelFile: File)
    suspend fun generate(prompt: String): String
    fun close()
}

/** Runs Gemma-family `.task` models fully offline via Google's MediaPipe LLM Inference API. */
class MediaPipeLlmEngine(private val context: Context) : LlmEngine {

    private var llm: LlmInference? = null
    override val isReady: Boolean get() = llm != null

    override suspend fun load(modelFile: File) = withContext(Dispatchers.IO) {
        if (llm != null) return@withContext
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)
            .setMaxTokens(1024)
            .build()
        llm = LlmInference.createFromOptions(context, options)
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        val engine = llm ?: return@withContext "The assistant model isn't loaded yet."
        engine.generateResponse(buildPrompt(prompt)).trim()
    }

    override fun close() {
        llm?.close()
        llm = null
    }

    private fun buildPrompt(userText: String): String = """
        <start_of_turn>user
        You are the TechnoForesight phone assistant. Answer briefly and helpfully.
        $userText<end_of_turn>
        <start_of_turn>model
    """.trimIndent()
}
