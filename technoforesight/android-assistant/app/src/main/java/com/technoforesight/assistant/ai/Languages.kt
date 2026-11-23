package com.technoforesight.assistant.ai

/** Same 12 languages and instructions as the Windows and Linux apps. */
data class Language(val name: String, val native: String, val script: String)

val LANGUAGES = listOf(
    Language("English", "English", "Latin"),
    Language("Telugu", "తెలుగు", "Telugu"),
    Language("Hindi", "हिन्दी", "Devanagari"),
    Language("Tamil", "தமிழ்", "Tamil"),
    Language("Kannada", "ಕನ್ನಡ", "Kannada"),
    Language("Malayalam", "മലയാളം", "Malayalam"),
    Language("Marathi", "मराठी", "Devanagari"),
    Language("Bengali", "বাংলা", "Bengali"),
    Language("Gujarati", "ગુજરાતી", "Gujarati"),
    Language("Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi"),
    Language("Odia", "ଓଡ଼ିଆ", "Odia"),
    Language("Urdu", "اردو", "Urdu (Perso-Arabic)"),
)

fun languageByName(name: String): Language =
    LANGUAGES.firstOrNull { it.name.equals(name, ignoreCase = true) || it.native == name } ?: LANGUAGES[0]

fun languageInstruction(name: String): String {
    val lang = LANGUAGES.firstOrNull { it.name == name }
    if (lang == null || lang.name == "English") {
        return "Reply in English unless the user clearly asks for another language."
    }
    return "Always reply in ${lang.name} (${lang.native}), written in the ${lang.script} script, even if the " +
        "question is in English or another language. Keep app names, file names, numbers and technical terms " +
        "as they are. Use simple, natural, everyday ${lang.name}. Write only ${lang.name} words in the " +
        "${lang.script} script plus English for technical terms; never use words from any other language " +
        "(for example Vietnamese, Chinese or Russian). Write plain text without Markdown symbols like ** or #."
}

fun systemPrompt(assistantName: String, language: String): String =
    "You are $assistantName, an offline assistant running privately on the user's Android phone. " +
        "Answer clearly and briefly. If asked about a document, only use the text you are given. " +
        languageInstruction(language)
