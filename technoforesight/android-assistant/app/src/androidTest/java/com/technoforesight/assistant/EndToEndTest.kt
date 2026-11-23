package com.technoforesight.assistant

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextReplacement
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Drives the real app UI. Run by CI on an emulator with instrumentation arguments:
 *   pcUrl   = Ollama on the host (http://10.0.2.2:11434), pcModel = model pulled there
 *   onDevice = "true" when the Gemma 4 .litertlm file was pushed into the app's storage
 * Screenshots land in the app's external files dir (pulled by CI).
 */
@RunWith(AndroidJUnit4::class)
class EndToEndTest {
    @get:Rule val compose = createEmptyComposeRule()

    private val ctx: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()

    private fun prefs(engine: String, language: String) {
        val p = Prefs(ctx)
        p.engine = engine
        p.language = language
        args.getString("pcUrl")?.let { p.pcUrl = it }
        args.getString("pcModel")?.let { p.pcModel = it }
    }

    private fun text(tag: String): String = try {
        compose.onNodeWithTag(tag, useUnmergedTree = true).fetchSemanticsNode()
            .config.getOrElse(SemanticsProperties.Text) { emptyList() }.joinToString("") { it.text }
    } catch (e: AssertionError) { "" }

    private fun waitFor(what: String, timeoutMs: Long, cond: () -> Boolean) {
        try {
            compose.waitUntil(timeoutMs) { cond() }
        } catch (e: Throwable) {
            shot("timeout-$what")
            throw AssertionError("Timed out waiting for $what. Status: '${text("status")}', engine: '${text("engine")}'")
        }
    }

    private fun shot(name: String) {
        val bmp = InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot() ?: return
        val dir = File(ctx.getExternalFilesDir(null), "e2e").apply { mkdirs() }
        File(dir, "$name.png").outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        File(dir, "$name.txt").writeText("status: ${text("status")}\nengine: ${text("engine")}\n\n${text("answer")}")
    }

    private fun ask(question: String, timeoutMs: Long): String {
        compose.onNodeWithTag("input").performTextReplacement(question)
        compose.onNodeWithTag("send").performClick()
        // finished = some answer, no streaming cursor, not "Thinking…"
        waitFor("answer to '$question'", timeoutMs) {
            val a = text("answer")
            a.length > 2 && !a.endsWith("▍") && !text("status").startsWith("Thinking")
        }
        compose.waitForIdle()
        return text("answer")
    }

    /** Most words in Telugu script, nothing from unrelated scripts, no Markdown. */
    private fun assertCleanTelugu(answer: String) {
        val words = answer.split(Regex("\\s+")).filter { w -> w.any { it.isLetter() } }
        val telugu = words.count { w -> w.any { it in 'ఀ'..'౿' } }
        val foreign = answer.filter { it in 'Ḁ'..'ỿ' || it in '一'..'鿿' || it in 'Ѐ'..'ӿ' }
        val share = telugu.toFloat() / maxOf(1, words.size)
        assertTrue("Telugu share $share, foreign '$foreign':\n$answer", share > 0.5f && foreign.isEmpty() && "**" !in answer)
    }

    @Test fun pcEngine_teluguAndEnglishAnswers() {
        assumeTrue("pcUrl not given", args.getString("pcUrl") != null)
        prefs("pc", "Telugu")
        ActivityScenario.launch(MainActivity::class.java).use {
            waitFor("PC engine", 60_000) { text("engine").contains("YOUR PC") }
            shot("01-ready-pc")
            val te = ask("explain what UPI is in 3 simple lines", 420_000)
            shot("02-telugu-upi-pc")
            assertCleanTelugu(te)
        }
        prefs("pc", "English")
        ActivityScenario.launch(MainActivity::class.java).use {
            waitFor("PC engine", 60_000) { text("engine").contains("YOUR PC") }
            val en = ask("What is the capital of India? Answer in one word.", 300_000)
            shot("03-english-pc")
            assertTrue(en, en.contains("Delhi", ignoreCase = true))
        }
    }

    @Test fun connectorsWithoutAi() {
        prefs("pc", "English")
        ActivityScenario.launch(MainActivity::class.java).use {
            compose.onNodeWithTag("input").performTextReplacement("add connector store technoforesight.in")
            compose.onNodeWithTag("send").performClick()
            waitFor("connector saved", 10_000) { text("status").contains("store") }
            compose.onNodeWithTag("input").performTextReplacement("connectors")
            compose.onNodeWithTag("send").performClick()
            waitFor("connector list", 10_000) { text("answer").contains("store") }
            shot("04-connectors")
        }
    }

    @Test fun onDevice_teluguAnswer() {
        assumeTrue("on-device model not pushed", args.getString("onDevice") == "true")
        prefs("phone", "Telugu")
        ActivityScenario.launch(MainActivity::class.java).use {
            waitFor("on-device model", 300_000) { text("engine").contains("LOCAL") || text("engine").contains("ERROR") }
            shot("05-ready-on-device")
            assertTrue("engine failed: ${text("status")}", text("engine").contains("LOCAL"))
            val te = ask("explain what UPI is in 3 simple lines", 900_000)
            shot("06-telugu-upi-on-device")
            assertCleanTelugu(te)
        }
    }
}
