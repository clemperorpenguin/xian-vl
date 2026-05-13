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
