package com.technoforesight.launcher

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.graphics.Color
import com.technoforesight.launcher.ui.HomeScreen

class MainActivity : ComponentActivity() {

    private val viewModel: LauncherViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // A home screen has nowhere to go "back" to; swallow the back button.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = viewModel.onQueryChange("")
        })

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFF38BDF8),
                    surface = Color(0xCC0F172A),
                ),
            ) {
                HomeScreen(viewModel)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Pick up apps installed/removed while we were in the background.
        viewModel.refreshApps()
    }
}
