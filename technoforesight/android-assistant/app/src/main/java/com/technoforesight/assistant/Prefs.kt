package com.technoforesight.assistant

import android.content.Context
import com.technoforesight.assistant.ai.DEFAULT_CONNECTORS
import org.json.JSONObject

/** Small settings store: language, engine choice, PC address, connectors. */
class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var language: String
        get() = sp.getString("language", "English")!!
        set(v) = sp.edit().putString("language", v).apply()

    /** "phone" (on-device Gemma) or "pc" (Ollama on the user's laptop over Wi-Fi). */
    var engine: String
        get() = sp.getString("engine", "phone")!!
        set(v) = sp.edit().putString("engine", v).apply()

    var pcUrl: String
        get() = sp.getString("pc_url", "http://192.168.1.2:11434")!!
        set(v) = sp.edit().putString("pc_url", v).apply()

    var pcModel: String
        get() = sp.getString("pc_model", "gemma4:e4b")!!
        set(v) = sp.edit().putString("pc_model", v).apply()

    var connectors: Map<String, String>
        get() {
            val raw = sp.getString("connectors", null) ?: return LinkedHashMap(DEFAULT_CONNECTORS)
            val obj = JSONObject(raw)
            return obj.keys().asSequence().associateWithTo(LinkedHashMap()) { obj.getString(it) }
        }
        set(v) = sp.edit().putString("connectors", JSONObject(v as Map<*, *>).toString()).apply()
}
