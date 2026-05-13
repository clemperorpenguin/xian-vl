package com.xian.nate.ocr

import android.graphics.Bitmap
import android.util.Base64
import com.xian.nate.api.LemonadeService
import com.xian.nate.api.OCRRequest
import java.io.ByteArrayOutputStream

class LemonadeOCR(private val service: LemonadeService) : OCREngine {
    override suspend fun processImage(bitmap: Bitmap): List<OCRResult> {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
        val base64 = Base64.encodeToString(stream.toByteArray(), Base64.DEFAULT)
        
        return try {
            val response = service.performRemoteOCR(OCRRequest(base64))
            response.results.map { 
                OCRResult(it.text, it.confidence, listOf(0,0,0,0)) 
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
}
