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

package com.xian.nate.ocr

import android.graphics.Bitmap
import com.chaquo.python.Python
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.ByteArrayOutputStream

class PaddleOCR : OCREngine {
    private val py = Python.getInstance()
    private val engineModule = py.getModule("ocr_engine")
    private val engine = engineModule.callAttr("get_engine")
    private val gson = Gson()

    override suspend fun processImage(bitmap: Bitmap): List<OCRResult> {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream) // JPEG is faster
        val byteArray = stream.toByteArray()
        
        return try {
            val jsonResult = engine.callAttr("detect_and_recognize", byteArray).toString()
            val type = object : TypeToken<List<Map<String, Any>>>() {}.type
            val rawList: List<Map<String, Any>> = gson.fromJson(jsonResult, type)
            
            rawList.map { res ->
                val box = (res["box"] as List<*>).map { (it as Number).toInt() }
                OCRResult(
                    text = res["text"].toString(),
                    confidence = (res["confidence"] as Number).toFloat(),
                    boundingBox = box
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }
}
