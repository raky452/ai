package com.technoforesight.assistant

import com.technoforesight.assistant.ai.AppEntry
import com.technoforesight.assistant.ai.Command
import com.technoforesight.assistant.ai.DEFAULT_CONNECTORS
import com.technoforesight.assistant.ai.LANGUAGES
import com.technoforesight.assistant.ai.MarkdownCleaner
import com.technoforesight.assistant.ai.ScriptFilter
import com.technoforesight.assistant.ai.ThinkFilter
import com.technoforesight.assistant.ai.languageInstruction
import com.technoforesight.assistant.ai.route
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CoreTest {
    private val apps = listOf(
        AppEntry("WhatsApp", "com.whatsapp"), AppEntry("Google Maps", "com.google.android.apps.maps"),
        AppEntry("Camera", "com.android.camera"),
    )

    private fun r(text: String) = route(text, apps, DEFAULT_CONNECTORS)

    @Test fun opensApps() {
        assertEquals(Command.OpenApp("WhatsApp", "com.whatsapp"), r("open whatsapp"))
        assertEquals(Command.OpenApp("Google Maps", "com.google.android.apps.maps"), r("launch maps app".removeSuffix(" app")))
        assertEquals(Command.OpenApp("Camera", "com.android.camera"), r("camera"))
    }

    @Test fun settingsUrlsAndSearch() {
        assertEquals(Command.OpenSettings("wifi", "android.settings.WIFI_SETTINGS"), r("settings wifi"))
        assertEquals(Command.OpenSettings("home", "android.settings.SETTINGS"), r("settings"))
        assertEquals(Command.OpenUrl("https://technoforesight.in"), r("technoforesight.in"))
        assertEquals(Command.WebSearch("weather hyderabad"), r("google weather hyderabad"))
    }

    @Test fun connectors() {
        assertEquals(Command.ListConnectors, r("connectors"))
        assertEquals(Command.AddConnector("store", "https://technoforesight.in"), r("add connector store technoforesight.in"))
        assertEquals(Command.RemoveConnector("store"), r("remove connector store"))
        assertEquals(Command.OpenConnector("yt", "https://www.youtube.com/results?search_query=telugu+songs"), r("yt telugu songs"))
        assertEquals(Command.OpenConnector("gmail", "https://mail.google.com"), r("gmail"))
    }

    @Test fun languageEngineAndAsk() {
        assertEquals(Command.SetLanguage(LANGUAGES[1]), r("language telugu"))
        assertEquals(Command.SetLanguage(LANGUAGES[2]), r("answer in हिन्दी"))
        assertEquals(Command.UsePc("http://192.168.1.20:11434"), r("use pc 192.168.1.20"))
        assertEquals(Command.UsePc("http://10.0.2.2:11434"), r("use pc http://10.0.2.2:11434"))
        assertEquals(Command.UsePhone, r("use phone"))
        assertEquals(Command.Summarise, r("summarise"))
        assertEquals(Command.Ask("explain what UPI is in 3 simple lines"), r("explain what UPI is in 3 simple lines"))
    }

    @Test fun markdownCleanerSurvivesAnyChunking() {
        val src = "**Subject:** [మీ పేరు]\n## Heading\n* one\n- two\nkeep 2*3 and file_name.txt\n"
        val want = "Subject: [మీ పేరు]\nHeading\n• one\n• two\nkeep 2*3 and file_name.txt\n"
        for (size in listOf(1, 2, 3, 5, 64)) {
            val m = MarkdownCleaner()
            val out = src.chunked(size).joinToString("") { m.feed(it) } + m.flush()
            assertEquals("chunk size $size", want, out)
        }
    }

    @Test fun thinkFilterHidesReasoning() {
        val src = "<think>secret plan</think>Hello <b>world</b>"
        for (size in listOf(1, 3, 7, 100)) {
            val f = ThinkFilter()
            val out = src.chunked(size).joinToString("") { f.feed(it) } + f.flush()
            assertEquals("Hello <b>world</b>", out)
        }
    }

    @Test fun indianLanguageInstructionKeepsOneScript() {
        val t = languageInstruction("Telugu")
        assertTrue(t.contains("Telugu script") && t.contains("Vietnamese") && t.contains("Markdown"))
        assertTrue(languageInstruction("English").startsWith("Reply in English"))
        assertEquals(12, LANGUAGES.size)
    }

    @Test fun scriptFilterDropsLeakedForeignTokens() {
        val src = "మా ప్రాంతంలో贵 సంస్థ యొక్క. ప్రజలు మీ dịch vụని అమితంగా\nUPI అంటే Unified Payments Interface 12,500"
        val want = "మా ప్రాంతంలో సంస్థ యొక్క. ప్రజలు మీ అమితంగా\nUPI అంటే Unified Payments Interface 12,500"
        for (size in listOf(1, 2, 5, 200)) {
            val f = ScriptFilter()
            assertEquals("chunk size $size", want, src.chunked(size).joinToString("") { f.feed(it) } + f.flush())
        }
    }
}
