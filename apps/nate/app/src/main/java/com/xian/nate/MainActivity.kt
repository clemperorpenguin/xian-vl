/*
 * MAGE Companion — Android OCR and local dictionary client.
 * Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)
 */

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
