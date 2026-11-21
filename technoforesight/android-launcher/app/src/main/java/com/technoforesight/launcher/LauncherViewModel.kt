package com.technoforesight.launcher

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.technoforesight.launcher.ai.CommandRouter
import com.technoforesight.launcher.ai.MediaPipeLlmEngine
import com.technoforesight.launcher.ai.ModelManager
import com.technoforesight.launcher.ai.PhoneAction
import com.technoforesight.launcher.apps.AppInfo
import com.technoforesight.launcher.apps.AppRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

sealed interface ModelStatus {
    data object NotDownloaded : ModelStatus
    data class Downloading(val progress: Float) : ModelStatus
    data object Loading : ModelStatus
    data object Ready : ModelStatus
    data class Error(val message: String) : ModelStatus
}

data class LauncherUiState(
    val apps: List<AppInfo> = emptyList(),
    val query: String = "",
    val assistantReply: String? = null,
    val isThinking: Boolean = false,
    val modelStatus: ModelStatus = ModelStatus.NotDownloaded,
) {
    val filteredApps: List<AppInfo>
        get() = if (query.isBlank()) apps
        else apps.filter { it.label.contains(query.trim(), ignoreCase = true) }
}

class LauncherViewModel(app: Application) : AndroidViewModel(app) {

    private val appRepository = AppRepository(app)
    private val modelManager = ModelManager(app)
    private val llm = MediaPipeLlmEngine(app)
    private val router = CommandRouter()

    private val _state = MutableStateFlow(LauncherUiState())
    val state: StateFlow<LauncherUiState> = _state.asStateFlow()

    init {
        refreshApps()
        if (modelManager.isDownloaded) loadModel()
    }

    fun refreshApps() = viewModelScope.launch {
        val apps = appRepository.loadApps()
        _state.update { it.copy(apps = apps) }
    }

    fun onQueryChange(text: String) = _state.update { it.copy(query = text, assistantReply = null) }

    fun launchApp(app: AppInfo) {
        appRepository.launch(app.packageName)
        _state.update { it.copy(query = "", assistantReply = null) }
    }

    /** Called when the user presses Enter / the send button in the command bar. */
    fun submit() {
        val input = _state.value.query
        if (input.isBlank()) return
        when (val action = router.route(input, _state.value.apps)) {
            is PhoneAction.AskAssistant -> askAssistant(action.text)
            else -> {
                val message = router.execute(getApplication(), action, appRepository)
                _state.update { it.copy(query = "", assistantReply = message) }
            }
        }
    }

    private fun askAssistant(text: String) {
        if (!llm.isReady) {
            _state.update {
                it.copy(assistantReply = "Download the offline assistant (button below) to ask questions.")
            }
            return
        }
        viewModelScope.launch {
            _state.update { it.copy(isThinking = true, assistantReply = null) }
            val reply = runCatching { llm.generate(text) }
                .getOrElse { "Sorry, something went wrong: ${it.message}" }
            _state.update { it.copy(isThinking = false, assistantReply = reply) }
        }
    }

    fun downloadModel() = viewModelScope.launch {
        _state.update { it.copy(modelStatus = ModelStatus.Downloading(0f)) }
        runCatching {
            modelManager.download { p -> _state.update { it.copy(modelStatus = ModelStatus.Downloading(p)) } }
        }.onSuccess { loadModel() }
            .onFailure { e -> _state.update { it.copy(modelStatus = ModelStatus.Error(e.message ?: "Download failed")) } }
    }

    private fun loadModel() = viewModelScope.launch {
        _state.update { it.copy(modelStatus = ModelStatus.Loading) }
        runCatching { llm.load(modelManager.modelFile) }
            .onSuccess { _state.update { it.copy(modelStatus = ModelStatus.Ready) } }
            .onFailure { e -> _state.update { it.copy(modelStatus = ModelStatus.Error(e.message ?: "Could not load model")) } }
    }

    override fun onCleared() {
        llm.close()
        super.onCleared()
    }
}
