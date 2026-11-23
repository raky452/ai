package com.technoforesight.assistant.apps

import android.content.Context
import android.content.Intent
import com.technoforesight.assistant.ai.AppEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Every app with a launcher icon (visible through the <queries> entry in the manifest). */
class AppRepository(private val context: Context) {

    @Suppress("DEPRECATION")
    suspend fun load(): List<AppEntry> = withContext(Dispatchers.IO) {
        val pm = context.packageManager
        pm.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
            .asSequence()
            .filter { it.activityInfo.packageName != context.packageName }
            .map { AppEntry(it.loadLabel(pm).toString(), it.activityInfo.packageName) }
            .distinctBy { it.packageName }
            .sortedBy { it.label.lowercase() }
            .toList()
    }

    fun launch(packageName: String): Boolean {
        val intent = context.packageManager.getLaunchIntentForPackage(packageName) ?: return false
        context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        return true
    }
}
