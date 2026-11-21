package com.technoforesight.launcher.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.technoforesight.launcher.LauncherViewModel
import com.technoforesight.launcher.ModelStatus
import com.technoforesight.launcher.apps.AppInfo
import kotlinx.coroutines.delay
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@Composable
fun HomeScreen(viewModel: LauncherViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .padding(horizontal = 16.dp),
    ) {
        Spacer(Modifier.height(24.dp))
        Clock()
        Spacer(Modifier.height(20.dp))

        CommandBar(
            text = state.query,
            onTextChange = viewModel::onQueryChange,
            onSubmit = viewModel::submit,
        )

        if (state.isThinking || state.assistantReply != null) {
            Spacer(Modifier.height(12.dp))
            AssistantCard(reply = state.assistantReply, thinking = state.isThinking)
        }

        Spacer(Modifier.height(12.dp))
        ModelStatusRow(status = state.modelStatus, onDownload = viewModel::downloadModel)
        Spacer(Modifier.height(12.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(4),
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            items(state.filteredApps, key = { it.packageName }) { app ->
                AppIcon(app = app, onClick = { viewModel.launchApp(app) })
            }
        }
    }
}

@Composable
private fun Clock() {
    var now by remember { mutableStateOf(LocalDateTime.now()) }
    LaunchedEffect(Unit) {
        while (true) {
            now = LocalDateTime.now()
            delay(1_000)
        }
    }
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            now.format(DateTimeFormatter.ofPattern("hh:mm")),
            fontSize = 64.sp,
            fontWeight = FontWeight.Light,
            color = Color.White,
        )
        Text(
            now.format(DateTimeFormatter.ofPattern("EEEE, d MMMM")),
            color = Color.White.copy(alpha = 0.8f),
        )
    }
}

@Composable
private fun CommandBar(text: String, onTextChange: (String) -> Unit, onSubmit: () -> Unit) {
    OutlinedTextField(
        value = text,
        onValueChange = onTextChange,
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        shape = RoundedCornerShape(28.dp),
        leadingIcon = { Icon(Icons.Filled.AutoAwesome, contentDescription = null) },
        placeholder = { Text("Search apps, or ask: \"set alarm 6 am\"") },
        trailingIcon = {
            IconButton(onClick = onSubmit, enabled = text.isNotBlank()) {
                Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Go")
            }
        },
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Go),
        keyboardActions = KeyboardActions(onGo = { onSubmit() }),
        colors = OutlinedTextFieldDefaults.colors(
            focusedContainerColor = MaterialTheme.colorScheme.surface,
            unfocusedContainerColor = MaterialTheme.colorScheme.surface,
            focusedTextColor = Color.White,
            unfocusedTextColor = Color.White,
        ),
    )
}

@Composable
private fun AssistantCard(reply: String?, thinking: Boolean) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(Modifier.padding(16.dp)) {
            if (thinking) {
                CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
            } else if (reply != null) {
                Text(reply, color = Color.White)
            }
        }
    }
}

@Composable
private fun ModelStatusRow(status: ModelStatus, onDownload: () -> Unit) {
    when (status) {
        ModelStatus.NotDownloaded -> Button(onClick = onDownload, modifier = Modifier.fillMaxWidth()) {
            Text("Download offline assistant (~2 GB, Wi-Fi recommended)")
        }
        is ModelStatus.Downloading -> Column {
            Text("Downloading assistant…", color = Color.White)
            if (status.progress >= 0f) {
                LinearProgressIndicator(progress = { status.progress }, modifier = Modifier.fillMaxWidth())
            } else {
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }
        ModelStatus.Loading -> Text("Starting assistant…", color = Color.White)
        ModelStatus.Ready -> Unit // nothing to show; assistant is available in the command bar
        is ModelStatus.Error -> Column {
            Text(status.message, color = Color(0xFFFCA5A5))
            Button(onClick = onDownload) { Text("Retry") }
        }
    }
}

@Composable
private fun AppIcon(app: AppInfo, onClick: () -> Unit) {
    Column(
        modifier = Modifier.clickable(onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Image(bitmap = app.icon, contentDescription = app.label, modifier = Modifier.size(52.dp))
        Text(
            app.label,
            color = Color.White,
            fontSize = 12.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            textAlign = TextAlign.Center,
        )
    }
}
