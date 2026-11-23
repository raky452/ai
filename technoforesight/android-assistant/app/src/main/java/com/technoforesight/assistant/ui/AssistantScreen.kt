package com.technoforesight.assistant.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.technoforesight.assistant.AssistantViewModel
import com.technoforesight.assistant.BuildConfig
import com.technoforesight.assistant.EngineState
import com.technoforesight.assistant.Tone
import com.technoforesight.assistant.ai.LANGUAGES

private val BG = Color(0xFF0A0F1E)
private val SURFACE = Color(0xFF111A2E)
private val LINE = Color(0xFF1E2C48)
private val FG = Color(0xFFE8EEFC)
private val MUTED = Color(0xFF7F8DB0)
private val CYAN = Color(0xFF22D3EE)
private val VIOLET = Color(0xFF8B5CF6)
private val PINK = Color(0xFFEC4899)
private val GREEN = Color(0xFF34D399)
private val AMBER = Color(0xFFFBBF24)
private val ERR = Color(0xFFFCA5A5)
private val NEON = Brush.horizontalGradient(listOf(CYAN, VIOLET, PINK))

private val CHIPS = listOf(
    "✦ Ask AI" to "", "▶ Open app" to "open ", "▤ Summarise" to "summarise", "⚙ Settings" to "settings ",
    "🔗 Connectors" to "connectors", "💻 Use my PC" to "use pc ",
)

@Composable
fun AssistantScreen(vm: AssistantViewModel) {
    Box(
        Modifier.fillMaxSize()
            .background(Brush.verticalGradient(listOf(Color(0xFF060914), BG, Color(0xFF0D1026))))
            .safeDrawingPadding().imePadding()
    ) {
        Column(Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 10.dp)) {
            Box(Modifier.fillMaxWidth().height(3.dp).clip(RoundedCornerShape(2.dp)).background(NEON))
            Spacer(Modifier.height(12.dp))
            Header(vm)
            Spacer(Modifier.height(12.dp))
            InputBar(vm)
            Spacer(Modifier.height(10.dp))
            Chips(vm)
            Spacer(Modifier.height(8.dp))
            StatusLine(vm)
            SetupCard(vm)
            Suggestions(vm)
            AnswerCard(vm, Modifier.weight(1f, fill = true))
            Footer(vm)
        }
    }
}

@Composable
private fun Header(vm: AssistantViewModel) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Canvas(Modifier.size(26.dp)) {
            val s = size.minDimension
            for (i in 12 downTo 1) {
                val r = s / 2 * i / 12f
                val p = Path().apply {
                    moveTo(s / 2, s / 2 - r); lineTo(s / 2 + r, s / 2); lineTo(s / 2, s / 2 + r); lineTo(s / 2 - r, s / 2); close()
                }
                drawPath(p, Brush.linearGradient(listOf(CYAN, VIOLET, PINK), Offset.Zero, Offset(s, s)), alpha = 0.35f + 0.65f * (1 - i / 12f))
            }
        }
        Spacer(Modifier.width(10.dp))
        Text("TechnoForesight", color = FG, fontWeight = FontWeight.Bold, fontSize = 18.sp)
        Spacer(Modifier.width(8.dp))
        Text("AI", color = BG, fontWeight = FontWeight.Bold, fontSize = 11.sp,
            modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(NEON).padding(horizontal = 7.dp, vertical = 2.dp))
        Spacer(Modifier.weight(1f))
        EnginePill(vm)
        Spacer(Modifier.width(8.dp))
        LanguageChip(vm)
    }
}

@Composable
private fun EnginePill(vm: AssistantViewModel) {
    val color = when (vm.engineState) {
        EngineState.READY, EngineState.PC_READY -> GREEN
        EngineState.ERROR -> ERR
        else -> AMBER
    }
    val pulse by rememberInfiniteTransition(label = "pulse")
        .animateFloat(0.35f, 1f, infiniteRepeatable(tween(900), RepeatMode.Reverse), label = "dot")
    Row(
        Modifier.testTag("engine").clip(RoundedCornerShape(20.dp)).background(SURFACE)
            .border(1.dp, LINE, RoundedCornerShape(20.dp)).padding(horizontal = 9.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(7.dp).alpha(pulse).clip(CircleShape).background(color))
        Spacer(Modifier.width(6.dp))
        Text(vm.engineState.label, color = color, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun LanguageChip(vm: AssistantViewModel) {
    var open by remember { mutableStateOf(false) }
    Box {
        Text("🌐 ${vm.language.native}", color = FG, fontSize = 13.sp,
            modifier = Modifier.testTag("language").clip(RoundedCornerShape(20.dp)).background(SURFACE)
                .border(1.dp, LINE, RoundedCornerShape(20.dp)).clickable { open = true }
                .padding(horizontal = 10.dp, vertical = 5.dp))
        DropdownMenu(expanded = open, onDismissRequest = { open = false }, modifier = Modifier.background(SURFACE)) {
            LANGUAGES.forEach { lang ->
                DropdownMenuItem(
                    text = { Text("${lang.native}   ·   ${lang.name}", color = if (lang == vm.language) CYAN else FG) },
                    onClick = { vm.chooseLanguage(lang); open = false },
                )
            }
        }
    }
}

@Composable
private fun InputBar(vm: AssistantViewModel) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(SURFACE)
            .border(BorderStroke(1.5.dp, if (vm.busy) NEON else Brush.horizontalGradient(listOf(LINE, LINE))), RoundedCornerShape(16.dp))
            .padding(start = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("❯", color = CYAN, fontSize = 22.sp, fontWeight = FontWeight.Bold)
        TextField(
            value = vm.input,
            onValueChange = vm::onInput,
            modifier = Modifier.weight(1f).testTag("input"),
            placeholder = { Text("Ask anything, or: open whatsapp · settings wifi · summarise", color = MUTED, fontSize = 15.sp) },
            singleLine = true,
            textStyle = androidx.compose.ui.text.TextStyle(color = FG, fontSize = 18.sp),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { vm.submit() }),
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent, unfocusedIndicatorColor = Color.Transparent, cursorColor = CYAN,
            ),
        )
        val label = if (vm.busy) "■" else "➤"
        Text(label, color = BG, fontWeight = FontWeight.Bold, fontSize = 16.sp,
            modifier = Modifier.testTag("send").padding(6.dp).clip(RoundedCornerShape(12.dp)).background(NEON)
                .clickable { if (vm.busy) vm.stop() else vm.submit() }.padding(horizontal = 14.dp, vertical = 10.dp))
    }
}

@Composable
private fun Chips(vm: AssistantViewModel) {
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        CHIPS.forEach { (label, fill) ->
            Text(label, color = FG, fontSize = 13.sp,
                modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(SURFACE)
                    .border(1.dp, LINE, RoundedCornerShape(20.dp))
                    .clickable { if (fill == "summarise" || fill == "connectors") vm.submit(fill) else vm.onInput(fill) }
                    .padding(horizontal = 12.dp, vertical = 7.dp))
        }
    }
}

@Composable
private fun StatusLine(vm: AssistantViewModel) {
    val color = when (vm.tone) { Tone.OK -> GREEN; Tone.WARN -> AMBER; Tone.ERROR -> ERR; Tone.INFO -> MUTED }
    AnimatedVisibility(vm.status.isNotEmpty()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (vm.status.startsWith("Thinking")) ThinkingDots()
            Text(vm.status, color = color, fontSize = 13.sp, modifier = Modifier.testTag("status").padding(vertical = 4.dp))
        }
    }
}

@Composable
private fun ThinkingDots() {
    val t by rememberInfiniteTransition(label = "dots")
        .animateFloat(0f, 3f, infiniteRepeatable(tween(1200)), label = "t")
    Row(Modifier.padding(end = 8.dp)) {
        listOf(CYAN, VIOLET, PINK).forEachIndexed { i, c ->
            val a = if (t.toInt() == i) 1f else 0.3f
            Box(Modifier.padding(end = 4.dp).size(7.dp).alpha(a).clip(CircleShape).background(c))
        }
    }
}

@Composable
private fun SetupCard(vm: AssistantViewModel) {
    val p = vm.progress
    if (!vm.showDownload && p == null) return
    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        if (vm.showDownload) {
            Text("⬇  Download AI model (${BuildConfig.MODEL_NAME}, ~2.6 GB)", color = BG, fontWeight = FontWeight.Bold,
                modifier = Modifier.testTag("download").fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(CYAN)
                    .clickable { vm.downloadModel() }.padding(14.dp))
        }
        if (p != null) {
            Spacer(Modifier.height(8.dp))
            if (p >= 0f) LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
                color = CYAN, trackColor = SURFACE)
            else LinearProgressIndicator(Modifier.fillMaxWidth().height(6.dp).clip(CircleShape), color = CYAN, trackColor = SURFACE)
        }
    }
}

@Composable
private fun Suggestions(vm: AssistantViewModel) {
    if (vm.suggestions.isEmpty()) return
    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(12.dp)).background(SURFACE)) {
        vm.suggestions.forEach { app ->
            Text("›   ${app.label}", color = FG, fontSize = 15.sp,
                modifier = Modifier.fillMaxWidth().clickable { vm.openSuggestion(app) }.padding(horizontal = 14.dp, vertical = 11.dp))
        }
    }
}

@Composable
private fun AnswerCard(vm: AssistantViewModel, modifier: Modifier) {
    if (vm.answer.isEmpty() && !vm.streaming) { Spacer(modifier); return }
    val clipboard = LocalClipboardManager.current
    val blink by rememberInfiniteTransition(label = "cursor")
        .animateFloat(0f, 1f, infiniteRepeatable(tween(600), RepeatMode.Reverse), label = "b")
    Column(
        modifier.fillMaxWidth().padding(top = 8.dp).clip(RoundedCornerShape(16.dp)).background(SURFACE)
            .border(1.dp, LINE, RoundedCornerShape(16.dp)).padding(14.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("◆ ${vm.answerTitle}", color = CYAN, fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            if (vm.language.name != "English") Text("  ·  ${vm.language.native}", color = MUTED, fontSize = 11.sp)
            Spacer(Modifier.weight(1f))
            Text("⧉ Copy", color = MUTED, fontSize = 12.sp, modifier = Modifier.clickable {
                clipboard.setText(AnnotatedString(vm.answer.trim()))
            })
        }
        Spacer(Modifier.height(8.dp))
        SelectionContainer(Modifier.verticalScroll(rememberScrollState())) {
            Text(vm.answer + if (vm.streaming) "▍" else "", color = FG, fontSize = 16.sp, lineHeight = 25.sp,
                modifier = Modifier.testTag("answer").alpha(if (vm.streaming && vm.answer.isEmpty()) blink else 1f))
        }
    }
}

@Composable
private fun Footer(vm: AssistantViewModel) {
    Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("⚡ ${vm.engineLabel}", color = MUTED, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
        Spacer(Modifier.weight(1f))
        Text("technoforesight.in", color = VIOLET, fontSize = 11.sp)
    }
}
