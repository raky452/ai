package com.technoforesight.assistant.ai

/**
 * Streaming filter for Indian-language answers (port of the desktop version). Small multilingual models
 * occasionally leak a token from an unrelated language (Gemma 4: "贵" or "dịch vụ" inside Telugu).
 * Drops stray CJK/kana/Hangul/Thai characters and whole words written in Vietnamese-only Latin letters
 * or Cyrillic. Holds back the current word until it ends.
 */
class ScriptFilter {
    private val word = StringBuilder()
    private var skipSpace = false

    private fun inRanges(c: Char, ranges: Array<IntRange>) = ranges.any { c.code in it }

    private fun cleanWord(w: String): String {
        if (w.any { inRanges(it, DROP_WORDS) }) return ""
        return w.filterNot { inRanges(it, DROP_CHARS) }
    }

    fun feed(chunk: String): String {
        val out = StringBuilder()
        for (ch in chunk) {
            if (ch.isWhitespace()) {
                val original = word.toString()
                val clean = cleanWord(original)
                if (original.isNotEmpty() && clean.isEmpty()) skipSpace = true else if (clean.isNotEmpty()) skipSpace = false
                out.append(clean)
                word.clear()
                if (ch == ' ' && skipSpace) continue
                skipSpace = false
                out.append(ch)
            } else {
                word.append(ch)
            }
        }
        return out.toString()
    }

    fun flush(): String = cleanWord(word.toString()).also { word.clear() }

    private companion object {
        val DROP_CHARS = arrayOf(0x2E80..0x9FFF, 0xF900..0xFAFF, 0xAC00..0xD7AF, 0x1100..0x11FF, 0x0E00..0x0E7F, 0xFF00..0xFFEF)
        val DROP_WORDS = arrayOf(0x1E00..0x1EFF, 0x0400..0x04FF)
    }
}
