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

package com.xian.nate.translation

import com.chaquo.python.Python
import com.xian.nate.api.LemonadeService
import com.xian.nate.api.TranslationRequest

class TranslationManager(
    private val lemonadeService: LemonadeService? = null
) {
    private val py = Python.getInstance()
    private val engine = py.getModule("translation_engine").callAttr("get_engine")

    fun setModel(modelName: String): String {
        return engine.callAttr("set_model", modelName).toString()
    }

    suspend fun translate(text: String, remoteEnabled: Boolean = false): String {
        return if (remoteEnabled && lemonadeService != null) {
            try {
                val response = lemonadeService.translate(TranslationRequest(text))
                response.translatedText
            } catch (e: Exception) {
                // Fallback to local
                engine.callAttr("translate", text).toString()
            }
        } else {
            engine.callAttr("translate", text).toString()
        }
    }

    suspend fun getNuance(text: String, remoteEnabled: Boolean = false): String {
        return if (remoteEnabled && lemonadeService != null) {
            try {
                val response = lemonadeService.translate(TranslationRequest(text, style = "nuance"))
                response.nuance ?: "No nuance provided by remote."
            } catch (e: Exception) {
                engine.callAttr("analyze_nuance", text).toString()
            }
        } else {
            engine.callAttr("analyze_nuance", text).toString()
        }
    }
}
