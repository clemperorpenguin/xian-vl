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
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import kotlinx.coroutines.tasks.await

class MLKitOCR : OCREngine {
    private val recognizer = TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())

    override suspend fun processImage(bitmap: Bitmap): List<OCRResult> {
        val image = InputImage.fromBitmap(bitmap, 0)
        return try {
            val result = recognizer.process(image).await()
            result.textBlocks.map { block ->
                OCRResult(
                    text = block.text,
                    confidence = 1.0f,
                    boundingBox = block.boundingBox?.let { listOf(it.left, it.top, it.right, it.bottom) } ?: listOf(0,0,0,0)
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
}
