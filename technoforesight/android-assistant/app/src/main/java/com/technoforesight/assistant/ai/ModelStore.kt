package com.technoforesight.assistant.ai

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL

/** Downloads the on-device model once (resumable), into app-private storage. */
class ModelStore(context: Context, private val url: String, fileName: String) {
    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val file = File(dir, fileName)
    private val part = File(dir, "$fileName.part")

    val isDownloaded get() = file.isFile && file.length() > 0
    val partialBytes get() = if (part.isFile) part.length() else 0L

    fun freeBytes(): Long = StatFs(dir.path).availableBytes

    /** Downloads with HTTP Range resume. onProgress(doneBytes, totalBytes) is called often. */
    suspend fun download(onProgress: (Long, Long) -> Unit) = withContext(Dispatchers.IO) {
        if (isDownloaded) return@withContext
        var redirects = 0
        var target = url
        var conn: HttpURLConnection
        while (true) {  // follow cross-host redirects (Hugging Face -> CDN), keeping the Range header
            conn = URL(target).openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 15_000
            conn.readTimeout = 60_000
            if (partialBytes > 0) conn.setRequestProperty("Range", "bytes=$partialBytes-")
            val code = conn.responseCode
            if (code in 301..308 && redirects++ < 8) {
                target = URL(URL(target), conn.getHeaderField("Location")).toString()
                conn.disconnect()
                continue
            }
            break
        }
        val code = conn.responseCode
        if (code !in listOf(200, 206)) throw IllegalStateException("Download failed (HTTP $code)")
        val resumed = code == 206
        val start = if (resumed) partialBytes else 0L
        val total = start + conn.contentLengthLong.coerceAtLeast(0)
        if (total > 0 && total - start > freeBytes() - 200_000_000) {
            throw IllegalStateException("Not enough free space: the model needs ${total / 1_000_000} MB.")
        }
        RandomAccessFile(part, "rw").use { out ->
            out.setLength(start)
            out.seek(start)
            conn.inputStream.use { input ->
                val buf = ByteArray(1 shl 16)
                var done = start
                var lastReport = 0L
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    done += n
                    if (done - lastReport > 2_000_000) { onProgress(done, total); lastReport = done }
                }
                onProgress(done, total)
            }
        }
        conn.disconnect()
        if (total > 0 && part.length() != total) throw IllegalStateException("Download incomplete, tap to resume.")
        if (!part.renameTo(file)) throw IllegalStateException("Couldn't save the model file.")
    }

    fun delete() {
        file.delete()
        part.delete()
    }
}
