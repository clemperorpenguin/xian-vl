package com.xian.nate.ocr

import android.graphics.Bitmap

interface OCREngine {
    suspend fun processImage(bitmap: Bitmap): List<OCRResult>
}

data class OCRResult(
    val text: String,
    val confidence: Float,
    val boundingBox: List<Int>
)
