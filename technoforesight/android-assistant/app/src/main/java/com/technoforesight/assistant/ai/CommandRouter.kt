package com.technoforesight.assistant.ai

import java.net.URLEncoder

/** What a line typed into the bar means. Mirrors the desktop router (tf_assistant/router.py). */
sealed interface Command {
    data class OpenApp(val label: String, val packageName: String) : Command
    data class OpenSettings(val page: String, val action: String) : Command
    data class OpenUrl(val url: String) : Command
    data class WebSearch(val query: String) : Command
    data object ListConnectors : Command
    data class AddConnector(val name: String, val url: String) : Command
    data class RemoveConnector(val name: String) : Command
    data class OpenConnector(val name: String, val url: String) : Command
    data object Summarise : Command
    data class SetLanguage(val language: Language) : Command
    data class UsePc(val url: String) : Command
    data object UsePhone : Command
    data class Ask(val text: String) : Command
}

data class AppEntry(val label: String, val packageName: String)

val DEFAULT_CONNECTORS = linkedMapOf(
    "yt" to "https://www.youtube.com/results?search_query={q}",
    "maps" to "https://www.google.com/maps/search/{q}",
    "gmail" to "https://mail.google.com",
    "drive" to "https://drive.google.com",
    "technoforesight" to "https://technoforesight.in",
)

/** "settings wifi" -> Android Settings intent action. Only stable, public actions. */
val SETTINGS_PAGES = mapOf(
    "" to "android.settings.SETTINGS",
    "wifi" to "android.settings.WIFI_SETTINGS",
    "wi-fi" to "android.settings.WIFI_SETTINGS",
    "bluetooth" to "android.settings.BLUETOOTH_SETTINGS",
    "display" to "android.settings.DISPLAY_SETTINGS",
    "sound" to "android.settings.SOUND_SETTINGS",
    "battery" to "android.intent.action.POWER_USAGE_SUMMARY",
    "apps" to "android.settings.APPLICATION_SETTINGS",
    "storage" to "android.settings.INTERNAL_STORAGE_SETTINGS",
    "privacy" to "android.settings.PRIVACY_SETTINGS",
    "security" to "android.settings.SECURITY_SETTINGS",
    "location" to "android.settings.LOCATION_SOURCE_SETTINGS",
    "vpn" to "android.settings.VPN_SETTINGS",
    "airplane" to "android.settings.AIRPLANE_MODE_SETTINGS",
    "date" to "android.settings.DATE_SETTINGS",
    "language" to "android.settings.LOCALE_SETTINGS",
    "accessibility" to "android.settings.ACCESSIBILITY_SETTINGS",
    "nfc" to "android.settings.NFC_SETTINGS",
    "about" to "android.settings.DEVICE_INFO_SETTINGS",
)

private val OPEN = Regex("^(open|launch|start|run)\\s+(.+)$", RegexOption.IGNORE_CASE)
private val URL = Regex("^(https?://\\S+|www\\.\\S+|[\\w-]+\\.(com|in|org|net|io|ai|dev)(/\\S*)?)$", RegexOption.IGNORE_CASE)
private val WEB = Regex("^(google|web|bing|search)\\s+(.+)$", RegexOption.IGNORE_CASE)
private val SETTINGS = Regex("^(open\\s+)?(settings?)\\s*(.*)$", RegexOption.IGNORE_CASE)
private val SUMMARISE = Regex("^(summari[sz]e|tl;?dr)(\\s+.*)?$", RegexOption.IGNORE_CASE)
private val LANGUAGE = Regex("^(language|answer in|reply in|speak)\\s+(\\S+)$", RegexOption.IGNORE_CASE)
private val USE_PC = Regex("^use (pc|laptop|computer)\\s+(\\S+)$", RegexOption.IGNORE_CASE)
private val USE_PHONE = Regex("^use (phone|on-device|offline|this phone)$", RegexOption.IGNORE_CASE)
private val CONN_LIST = Regex("^(connectors|list connectors|show connectors)$", RegexOption.IGNORE_CASE)
private val CONN_ADD = Regex("^(add|connect)\\s+connector\\s+([\\w-]+)\\s+(\\S+)$", RegexOption.IGNORE_CASE)
private val CONN_DEL = Regex("^(remove|delete)\\s+connector\\s+([\\w-]+)$", RegexOption.IGNORE_CASE)

fun connectorUrl(template: String, rest: String): String =
    if ("{q}" in template) template.replace("{q}", URLEncoder.encode(rest, "UTF-8")) else template

fun findApp(query: String, apps: List<AppEntry>, exactOnly: Boolean = false): AppEntry? {
    val q = query.trim().lowercase()
    if (q.isEmpty()) return null
    apps.firstOrNull { it.label.lowercase() == q }?.let { return it }
    if (exactOnly) return null
    apps.firstOrNull { it.label.lowercase().startsWith(q) }?.let { return it }
    apps.firstOrNull { app -> app.label.lowercase().split(" ").any { it.startsWith(q) } }?.let { return it }
    return apps.firstOrNull { q in it.label.lowercase() }
}

fun route(input: String, apps: List<AppEntry>, connectors: Map<String, String>): Command {
    val text = input.trim()

    if (CONN_LIST.matches(text)) return Command.ListConnectors
    CONN_ADD.matchEntire(text)?.let { m ->
        var url = m.groupValues[3]
        if (!url.startsWith("http://", true) && !url.startsWith("https://", true)) url = "https://$url"
        return Command.AddConnector(m.groupValues[2].lowercase(), url)
    }
    CONN_DEL.matchEntire(text)?.let { return Command.RemoveConnector(it.groupValues[2].lowercase()) }
    val first = text.substringBefore(" ").lowercase()
    connectors[first]?.let { template ->
        val rest = if (" " in text) text.substringAfter(" ").trim() else ""
        return Command.OpenConnector(first, connectorUrl(template, rest))
    }

    USE_PHONE.matchEntire(text)?.let { return Command.UsePhone }
    USE_PC.matchEntire(text)?.let { m ->
        var url = m.groupValues[2]
        if (!url.startsWith("http")) url = "http://$url"
        if (!Regex(":\\d+").containsMatchIn(url.substringAfter("://"))) url = url.trimEnd('/') + ":11434"
        return Command.UsePc(url)
    }
    LANGUAGE.matchEntire(text)?.let { m ->
        val lang = LANGUAGES.firstOrNull { it.name.equals(m.groupValues[2], true) || it.native == m.groupValues[2] }
        if (lang != null) return Command.SetLanguage(lang)
    }

    SETTINGS.matchEntire(text)?.let { m ->
        val page = m.groupValues[3].trim().lowercase()
        SETTINGS_PAGES[page]?.let { return Command.OpenSettings(page.ifEmpty { "home" }, it) }
    }
    URL.matchEntire(text)?.let { m ->
        val url = m.groupValues[1]
        return Command.OpenUrl(if (url.startsWith("http", true)) url else "https://$url")
    }
    WEB.matchEntire(text)?.let { return Command.WebSearch(it.groupValues[2].trim()) }
    if (SUMMARISE.matches(text)) return Command.Summarise
    OPEN.matchEntire(text)?.let { m ->
        findApp(m.groupValues[2].trim().trim('"', '\''), apps)?.let { return Command.OpenApp(it.label, it.packageName) }
    }
    findApp(text, apps, exactOnly = true)?.let { return Command.OpenApp(it.label, it.packageName) }
    return Command.Ask(text)
}
