package com.technoforesight.assistant.docs

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.util.zip.ZipInputStream

/** Reads the text of a PDF, Word (.docx) or text file the user picked. */
object DocumentReader {

    fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0) ?: "document"
        }
        return uri.lastPathSegment ?: "document"
    }

    fun read(context: Context, uri: Uri, maxChars: Int = 12_000): String {
        val name = displayName(context, uri).lowercase()
        val mime = context.contentResolver.getType(uri).orEmpty()
        val resolver = context.contentResolver
        val text = when {
            mime == "application/pdf" || name.endsWith(".pdf") -> {
                PDFBoxResourceLoader.init(context.applicationContext)
                resolver.openInputStream(uri)!!.use { input ->
                    PDDocument.load(input).use { doc -> PDFTextStripper().getText(doc) }
                }
            }
            name.endsWith(".docx") || mime.contains("wordprocessingml") ->
                resolver.openInputStream(uri)!!.use { docxText(ZipInputStream(it)) }
            else -> resolver.openInputStream(uri)!!.use { it.readBytes().toString(Charsets.UTF_8) }
        }.trim()
        if (text.isEmpty()) throw IllegalStateException("That file has no readable text (it may be a scanned image).")
        return text.take(maxChars)
    }

    fun docxText(zip: ZipInputStream): String {
        while (true) {
            val entry = zip.nextEntry ?: break
            if (entry.name == "word/document.xml") {
                val xml = zip.readBytes().toString(Charsets.UTF_8)
                return xml.replace("</w:p>", "\n").replace(Regex("<[^>]+>"), "")
                    .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
            }
        }
        throw IllegalStateException("This Word file has no text.")
    }
}
