package com.technoforesight.launcher.ai

import android.app.SearchManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import com.technoforesight.launcher.apps.AppInfo
import com.technoforesight.launcher.apps.AppRepository

/** Something the launcher can do on the phone. */
sealed interface PhoneAction {
    data class OpenApp(val app: AppInfo) : PhoneAction
    data class SetAlarm(val hour: Int, val minute: Int) : PhoneAction
    data class Dial(val number: String) : PhoneAction
    data class WebSearch(val query: String) : PhoneAction
    /** Not a phone command: hand the text to the on-device model. */
    data class AskAssistant(val text: String) : PhoneAction
}

/**
 * Turns what the user typed into a [PhoneAction].
 *
 * Version 1 uses fast, predictable rules (works instantly, even before the model is downloaded).
 * Next step: let the model emit these actions itself as JSON ("function calling") so it
 * understands free-form requests like "wake me before sunrise" or Hindi/Telugu phrasing.
 */
class CommandRouter {

    private val openRegex = Regex("""^(open|launch|start)\s+(.+)$""", RegexOption.IGNORE_CASE)
    private val alarmRegex = Regex(
        """^(set\s+)?(an\s+)?alarm\s+(for\s+|at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$""",
        RegexOption.IGNORE_CASE,
    )
    private val callRegex = Regex("""^(call|dial)\s+([+\d][\d\s-]{2,})$""", RegexOption.IGNORE_CASE)
    private val searchRegex = Regex("""^(search|google)\s+(.+)$""", RegexOption.IGNORE_CASE)

    fun route(input: String, apps: List<AppInfo>): PhoneAction {
        val text = input.trim()

        openRegex.find(text)?.let { m ->
            findApp(m.groupValues[2], apps)?.let { return PhoneAction.OpenApp(it) }
        }

        alarmRegex.find(text)?.let { m ->
            var hour = m.groupValues[4].toInt()
            val minute = m.groupValues[5].ifEmpty { "0" }.toInt()
            when (m.groupValues[6].lowercase()) {
                "pm" -> if (hour < 12) hour += 12
                "am" -> if (hour == 12) hour = 0
            }
            if (hour in 0..23 && minute in 0..59) return PhoneAction.SetAlarm(hour, minute)
        }

        callRegex.find(text)?.let { m ->
            return PhoneAction.Dial(m.groupValues[2].filter { it.isDigit() || it == '+' })
        }

        searchRegex.find(text)?.let { m -> return PhoneAction.WebSearch(m.groupValues[2]) }

        // A bare app name ("whatsapp") opens the app directly.
        findApp(text, apps, exactOnly = true)?.let { return PhoneAction.OpenApp(it) }

        return PhoneAction.AskAssistant(text)
    }

    private fun findApp(name: String, apps: List<AppInfo>, exactOnly: Boolean = false): AppInfo? {
        val q = name.trim().lowercase()
        if (q.isEmpty()) return null
        return apps.firstOrNull { it.label.lowercase() == q }
            ?: if (exactOnly) null else apps.firstOrNull { it.label.lowercase().startsWith(q) }
                ?: apps.firstOrNull { it.label.lowercase().contains(q) }
    }

    /** Performs actions that don't need the model. Returns a short status message for the UI. */
    fun execute(context: Context, action: PhoneAction, appRepository: AppRepository): String? {
        fun start(intent: Intent) = context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        return try {
            when (action) {
                is PhoneAction.OpenApp -> {
                    appRepository.launch(action.app.packageName); null
                }
                is PhoneAction.SetAlarm -> {
                    start(
                        Intent(AlarmClock.ACTION_SET_ALARM)
                            .putExtra(AlarmClock.EXTRA_HOUR, action.hour)
                            .putExtra(AlarmClock.EXTRA_MINUTES, action.minute)
                            .putExtra(AlarmClock.EXTRA_MESSAGE, "TechnoForesight alarm"),
                    )
                    "Alarm set for %02d:%02d".format(action.hour, action.minute)
                }
                is PhoneAction.Dial -> {
                    // ACTION_DIAL opens the dialler pre-filled; the user taps call. No CALL_PHONE permission needed.
                    start(Intent(Intent.ACTION_DIAL, Uri.parse("tel:${action.number}"))); null
                }
                is PhoneAction.WebSearch -> {
                    start(Intent(Intent.ACTION_WEB_SEARCH).putExtra(SearchManager.QUERY, action.query)); null
                }
                is PhoneAction.AskAssistant -> null // handled by the ViewModel via the LLM
            }
        } catch (e: Exception) {
            "Couldn't do that on this phone: ${e.message}"
        }
    }
}
