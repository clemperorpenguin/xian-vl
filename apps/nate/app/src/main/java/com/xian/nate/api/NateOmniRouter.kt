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

/**
 * A lightweight, client-side model router — the Kotlin counterpart of the
 * core ``xian.OmniModelRouter``. NATE does its own discovery and picks a
 * **vision-labelled** model itself, rather than trusting the server's Omni
 * bundle: on some nodes the "Lite" Omni has no vision LLM component and refuses
 * image input, while a discrete vision model (e.g. ``Gemma-4-31B-it-GGUF``,
 * label ``vision``) handles OCR correctly. Choosing the model here is what makes
 * the camera-translate path reliable across heterogeneous Lemonade nodes.
 */
class NateOmniRouter(private val service: LemonadeService) {

    private var labelToId: Map<String, String> = emptyMap()
    private var allIds: List<String> = emptyList()

    /** Query the node and build a label → model-id routing table. */
    suspend fun discover() {
        val data = service.models().data
        val map = LinkedHashMap<String, String>()
        for (m in data) {
            val labels = m.labels ?: emptyList()
            for (label in labels) map.putIfAbsent(label, m.id)

            // Heuristic fallbacks for nodes that don't label vision explicitly.
            val lid = m.id.lowercase()
            if ("vision" in labels || "vl" in lid || "-vl" in lid || "omni" in lid) {
                map.putIfAbsent("vision", m.id)
            }
            if ("qwen" in lid || "gemma" in lid || "llama" in lid || "mistral" in lid) {
                map.putIfAbsent("chat", m.id)
            }
        }
        labelToId = map
        allIds = data.map { it.id }
    }

    /** The model used for OCR + in-image translation, or null if none found. */
    fun visionModel(): String? = labelToId["vision"]

    /** A plain text/chat model, for non-image fallbacks. */
    fun textModel(): String? =
        labelToId["tool-calling"] ?: labelToId["chat"] ?: allIds.firstOrNull()

    val discoveredIds: List<String> get() = allIds
}
