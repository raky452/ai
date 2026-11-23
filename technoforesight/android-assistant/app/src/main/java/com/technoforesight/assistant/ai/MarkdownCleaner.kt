package com.technoforesight.assistant.ai

/**
 * Streaming clean-up of Markdown the answer card doesn't render (port of the desktop version):
 * removes ** and __ bold markers and leading # headings, and turns "* " / "- " bullets into "• ".
 * Holds back a few characters so markers split across chunks are still caught.
 */
class MarkdownCleaner {
    private var buffer = StringBuilder()
    private var lineStart = true

    fun feed(chunk: String): String {
        buffer.append(chunk)
        return drain(final = false)
    }

    fun flush(): String = drain(final = true)

    private fun drain(final: Boolean): String {
        val buf = buffer.toString()
        val out = StringBuilder()
        var i = 0
        loop@ while (i < buf.length) {
            if (lineStart) {
                val rest = buf.substring(i)
                val stripped = rest.trimStart(' ', '\t')
                val lead = rest.length - stripped.length
                if (stripped.isEmpty() && !final) break@loop
                if (stripped.startsWith("#")) {
                    val hashes = stripped.length - stripped.trimStart('#').length
                    val after = stripped.substring(hashes)
                    if (after.isEmpty() && !final) break@loop
                    if (after.isEmpty() || after.startsWith(" ")) {
                        i += lead + hashes + (if (after.startsWith(" ")) 1 else 0)
                        lineStart = false
                        continue@loop
                    }
                }
                if (stripped.startsWith("* ") || stripped.startsWith("- ")) {
                    out.append(rest, 0, lead).append("• ")
                    i += lead + 2
                    lineStart = false
                    continue@loop
                }
                if ((stripped == "*" || stripped == "-") && !final) break@loop
                lineStart = false
                continue@loop
            }
            val ch = buf[i]
            if (ch == '*' || ch == '_') {
                if (i + 1 >= buf.length && !final) break@loop
                if (i + 1 < buf.length && buf[i + 1] == ch) {
                    i += 2
                    continue@loop
                }
            }
            out.append(ch)
            if (ch == '\n') lineStart = true
            i++
        }
        buffer = StringBuilder(buf.substring(i))
        if (final) {
            out.append(buffer)
            buffer = StringBuilder()
        }
        return out.toString()
    }
}
