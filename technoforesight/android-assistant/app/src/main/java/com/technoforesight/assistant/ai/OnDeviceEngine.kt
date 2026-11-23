package com.technoforesight.assistant.ai

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** Gemma 4 on the phone itself through Google's LiteRT-LM runtime. Nothing leaves the device. */
class OnDeviceEngine(
    private val context: Context,
    private val modelFile: File,
    private val modelName: String,
) : ChatEngine {
    private var engine: Engine? = null
    private var backendName = ""
    private val lock = Mutex()

    override val label get() = "$modelName · on phone" + if (backendName.isNotEmpty()) " ($backendName)" else ""

    override suspend fun prepare(): String? = withContext(Dispatchers.IO) {
        lock.withLock {
            if (engine != null) return@withContext null
            if (!modelFile.isFile) return@withContext "The AI model isn't downloaded yet."
            // GPU is several times faster where supported; fall back to the CPU everywhere else.
            val attempts = listOf("GPU" to { Backend.GPU() }, "CPU" to { Backend.CPU() })
            var lastError: Throwable? = null
            for ((name, backend) in attempts) {
                try {
                    val e = Engine(EngineConfig(modelPath = modelFile.absolutePath, backend = backend(),
                                                cacheDir = context.cacheDir.path))
                    e.initialize()
                    // Some phones load the GPU backend but can't run it (no OpenCL driver): the failure only
                    // shows up when generating. Generate one short reply now so we fall back to the CPU here.
                    try {
                        e.createConversation(ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)))
                            .use { it.sendMessage("Hi") }
                    } catch (t: Throwable) {
                        e.close()
                        throw t
                    }
                    engine = e
                    backendName = name
                    return@withContext null
                } catch (t: Throwable) {
                    lastError = t
                }
            }
            "The AI model couldn't start on this phone: ${lastError?.message ?: "unknown error"}"
        }
    }

    override fun stream(system: String, user: String): Flow<String> = flow {
        val e = engine ?: throw IllegalStateException("The AI model isn't loaded yet.")
        val config = ConversationConfig(
            systemInstruction = Contents.of(system),
            samplerConfig = SamplerConfig(topK = 40, topP = 0.9, temperature = 0.7),
        )
        e.createConversation(config).use { conversation ->
            conversation.sendMessageAsync(user).collect { message -> emit(message.toString()) }
        }
    }.flowOn(Dispatchers.Default)

    override fun close() {
        engine?.close()
        engine = null
    }
}
