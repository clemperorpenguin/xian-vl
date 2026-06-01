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
