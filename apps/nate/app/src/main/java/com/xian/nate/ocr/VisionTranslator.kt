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

package com.xian.nate.ocr

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Base64
import com.xian.nate.api.ChatMessage
import com.xian.nate.api.ChatRequest
import com.xian.nate.api.ContentPart
import com.xian.nate.api.ImageUrl
import com.xian.nate.api.LemonadeService
import com.xian.nate.api.NateOmniRouter
import org.json.JSONArray
import java.io.ByteArrayOutputStream

/**
 * Runs OCR + translation in a single Lemonade vision call and returns located,
 * translated [TextRegion]s ready for inpainting.
 *
 * The contract (verified against a live node): the model returns a JSON array of
 * `{"box":[x1,y1,x2,y2],"original":...,"translated":...}` with box coordinates
 * normalized 0–1000 over image width/height. Reasoning models wrap the array in
 * ```json fences and keep chain-of-thought in a separate `reasoning_content`
 * field, so we strip fences and parse the first array found in `content`.
 */
class VisionTranslator(
    private val service: LemonadeService,
    private val router: NateOmniRouter,
) {
    suspend fun translate(bitmap: Bitmap, targetLang: String): List<TextRegion> {
        val model = router.visionModel()
            ?: throw IllegalStateException(
                "No vision-capable model found on the Lemonade server. " +
                    "Install a model labelled \"vision\" (e.g. a Gemma or Qwen-VL build).",
            )

        val request = ChatRequest(
            model = model,
            messages = listOf(
                ChatMessage("system", systemPrompt(targetLang)),
                ChatMessage(
                    "user",
                    listOf(
                        ContentPart(
                            type = "text",
                            text = "OCR every text region and translate it to $targetLang. " +
                                "Respond with the JSON array only.",
                        ),
                        ContentPart(
                            type = "image_url",
                            imageUrl = ImageUrl("data:image/png;base64,${bitmap.toBase64Png()}"),
                        ),
                    ),
                ),
            ),
        )

        val content = service.chat(request).choices.firstOrNull()?.message?.content.orEmpty()
        return parseRegions(content, bitmap.width, bitmap.height)
    }

    private fun systemPrompt(targetLang: String): String =
        "You are an OCR and translation engine for a single image. " +
            "Detect every distinct text region. Return ONLY a JSON array, no markdown, no prose. " +
            "Each element must be {\"box\":[x1,y1,x2,y2],\"original\":\"<source text>\"," +
            "\"translated\":\"<$targetLang translation>\"}. " +
            "box coordinates are integers normalized to 0-1000 relative to image width and height " +
            "(x along width, y along height). Preserve the reading order. " +
            "If there is no text, return []."

    /** Strip code fences and parse the first JSON array in the response. */
    internal fun parseRegions(raw: String, width: Int, height: Int): List<TextRegion> {
        val start = raw.indexOf('[')
        val end = raw.lastIndexOf(']')
        if (start < 0 || end <= start) return emptyList()

        val arr = try {
            JSONArray(raw.substring(start, end + 1))
        } catch (e: Exception) {
            return emptyList()
        }

        val regions = ArrayList<TextRegion>(arr.length())
        for (i in 0 until arr.length()) {
            val obj = arr.optJSONObject(i) ?: continue
            val box = obj.optJSONArray("box") ?: continue
            if (box.length() < 4) continue

            val x1 = box.optDouble(0) / 1000.0 * width
            val y1 = box.optDouble(1) / 1000.0 * height
            val x2 = box.optDouble(2) / 1000.0 * width
            val y2 = box.optDouble(3) / 1000.0 * height
            val rect = Rect(
                minOf(x1, x2).toInt().coerceIn(0, width),
                minOf(y1, y2).toInt().coerceIn(0, height),
                maxOf(x1, x2).toInt().coerceIn(0, width),
                maxOf(y1, y2).toInt().coerceIn(0, height),
            )
            if (rect.width() <= 0 || rect.height() <= 0) continue

            val translated = obj.optString("translated").ifBlank { obj.optString("original") }
            regions.add(TextRegion(rect, obj.optString("original"), translated))
        }
        return regions
    }
}

private fun Bitmap.toBase64Png(): String {
    val stream = ByteArrayOutputStream()
    compress(Bitmap.CompressFormat.PNG, 100, stream)
    return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
}
