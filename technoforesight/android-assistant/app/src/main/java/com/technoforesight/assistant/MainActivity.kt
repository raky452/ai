package com.technoforesight.assistant

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.lifecycle.lifecycleScope
import com.technoforesight.assistant.ui.AssistantScreen
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val vm: AssistantViewModel by viewModels()

    private val pickDocument = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.summarise(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { AssistantScreen(vm) }
        lifecycleScope.launch {
            vm.events.collect {
                if (it == "pick_document") pickDocument.launch(arrayOf(
                    "application/pdf", "text/plain",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            }
        }
        if (savedInstanceState == null) handle(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handle(intent)
    }

    /** Text shared to the app, or selected text with "Ask TechnoForesight" in any app's menu. */
    private fun handle(intent: Intent?) {
        val text = when (intent?.action) {
            Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            else -> null
        }
        if (!text.isNullOrBlank()) vm.askAbout(text)
    }
}
