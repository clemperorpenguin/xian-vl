/*
 * NATE — Lemonade-powered camera translator (Android).
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

import com.google.gson.annotations.SerializedName
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

/**
 * The OpenAI-compatible surface NATE uses against a Lemonade Server. No host
 * models are involved — OCR and translation both run on the Lemonade node via
 * a vision-capable chat model selected by [NateOmniRouter].
 */
interface LemonadeService {
    /** Model discovery. ``show_all=true`` surfaces hidden Omni bundles. */
    @GET("v1/models")
    suspend fun models(@Query("show_all") showAll: Boolean = true): ModelsResponse

    /** Vision + text inference. */
    @POST("v1/chat/completions")
    suspend fun chat(@Body request: ChatRequest): ChatResponse
}

// ── Discovery ────────────────────────────────────────────────────────────────
data class ModelsResponse(val data: List<ModelInfo> = emptyList())

data class ModelInfo(
    val id: String,
    val labels: List<String>? = null,
    val recipe: String? = null,
)

// ── Chat request ─────────────────────────────────────────────────────────────
data class ChatRequest(
    val model: String,
    val messages: List<ChatMessage>,
    @SerializedName("max_tokens") val maxTokens: Int = 3072,
    val temperature: Double = 0.1,
    val stream: Boolean = false,
)

/**
 * ``content`` is intentionally [Any]: a plain string for system/text messages,
 * or a ``List<ContentPart>`` for multimodal (text + image) messages. Gson
 * serializes by runtime type, and omits null [ContentPart] fields, so each part
 * emits only ``text`` or only ``image_url`` — exactly the OpenAI vision shape.
 */
data class ChatMessage(val role: String, val content: Any)

data class ContentPart(
    val type: String,
    val text: String? = null,
    @SerializedName("image_url") val imageUrl: ImageUrl? = null,
)

data class ImageUrl(val url: String)

// ── Chat response ────────────────────────────────────────────────────────────
data class ChatResponse(val choices: List<Choice> = emptyList())

data class Choice(
    val message: ResponseMessage,
    @SerializedName("finish_reason") val finishReason: String? = null,
)

/**
 * Reasoning models (e.g. Gemma vision) place chain-of-thought in
 * ``reasoning_content`` and the answer in ``content``; we only ever read
 * ``content``.
 */
data class ResponseMessage(
    val content: String? = null,
    @SerializedName("reasoning_content") val reasoningContent: String? = null,
)
