package com.technoforesight.launcher.ai

import android.content.Context
import com.technoforesight.launcher.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads the model once on first launch (keeps the Play Store download small)
 * and stores it in the app's private storage.
 */
class ModelManager(context: Context) {

    val modelFile: File = File(context.filesDir, BuildConfig.MODEL_FILE_NAME)

    val isDownloaded: Boolean get() = modelFile.exists() && modelFile.length() > 0

    /** @param onProgress fraction 0f..1f, or -1f if the server doesn't report size. */
    suspend fun download(onProgress: (Float) -> Unit) = withContext(Dispatchers.IO) {
        val tmp = File(modelFile.parentFile, modelFile.name + ".part")
        val conn = URL(BuildConfig.MODEL_URL).openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 30_000
        try {
            conn.connect()
            check(conn.responseCode == HttpURLConnection.HTTP_OK) {
                "Model download failed: HTTP ${conn.responseCode}"
            }
            val total = conn.contentLengthLong
            var done = 0L
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buffer = ByteArray(1 shl 16)
                    while (true) {
                        ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        done += read
                        onProgress(if (total > 0) done.toFloat() / total else -1f)
                    }
                }
            }
            check(tmp.renameTo(modelFile)) { "Could not save model file" }
        } finally {
            conn.disconnect()
            if (tmp.exists()) tmp.delete()
        }
    }
}
