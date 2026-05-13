package com.xian.nate

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.xian.nate.translation.TranslationManager
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Simple view setup for initialization test
        val textView = TextView(this).apply {
            text = "NATE: Neural Analysis & Translation Engine\nInitializing..."
            textSize = 18f
            setPadding(32, 32, 32, 32)
        }
        setContentView(textView)

        initChaquopy()
        
        val manager = TranslationManager()
        
        lifecycleScope.launch {
            val testResult = manager.translate("你好")
            val nuance = manager.getNuance("卷", false)
            
            textView.text = "NATE Engine Ready\n\nTest Result: $testResult\n\nNuance: $nuance"
            Log.d("NATE", "Engine Initialized: $testResult")
        }
    }

    private fun initChaquopy() {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }
}
