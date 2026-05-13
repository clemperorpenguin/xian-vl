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
