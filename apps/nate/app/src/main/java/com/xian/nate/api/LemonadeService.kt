package com.xian.nate.api

import retrofit2.http.Body
import retrofit2.http.POST

interface LemonadeService {
    @POST("ocr/qwen")
    suspend fun performRemoteOCR(@Body request: OCRRequest): OCRResponse

    @POST("translate/omni")
    suspend fun translate(@Body request: TranslationRequest): TranslationResponse
}

data class OCRRequest(val imageBase64: String)
data class OCRResponse(val results: List<RemoteOCRResult>)
data class RemoteOCRResult(val text: String, val confidence: Float)

data class TranslationRequest(
    val text: String,
    val style: String = "general",
    val remoteModel: String = "Qwen3.5-9B"
)
data class TranslationResponse(val translatedText: String, val nuance: String?)
