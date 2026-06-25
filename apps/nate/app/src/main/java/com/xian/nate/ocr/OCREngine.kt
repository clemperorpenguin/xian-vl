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

import android.graphics.Rect

/**
 * One detected text region: its pixel rectangle in the source image, the source
 * text, and the translation. Bounding boxes arrive from the model normalized to
 * 0–1000 and are converted to pixels by [VisionTranslator].
 */
data class TextRegion(
    val rect: Rect,
    val original: String,
    val translated: String,
)
