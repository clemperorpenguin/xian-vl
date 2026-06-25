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

package com.xian.nate.render

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.text.TextPaint
import com.xian.nate.ocr.TextRegion
import kotlin.math.max

/**
 * Renders translated text over the source image in the style of Google
 * Translate's camera mode: each source region is covered with its sampled
 * background colour, then the translation is drawn fitted to the box. This is an
 * MVP "flat fill" inpaint — not generative — which is robust and fast on-device.
 */
object Inpainter {

    fun render(source: Bitmap, regions: List<TextRegion>): Bitmap {
        val out = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val fill = Paint(Paint.ANTI_ALIAS_FLAG)
        val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG)

        for (region in regions) {
            if (region.translated.isBlank()) continue
            val bg = sampleBackground(source, region.rect)

            // 1. Erase the original text by flooding the box with the background.
            fill.color = bg
            canvas.drawRect(region.rect, fill)

            // 2. Draw the translation in a contrasting colour, fitted to the box.
            textPaint.color = if (luminance(bg) > 0.55) Color.BLACK else Color.WHITE
            drawFittedText(canvas, textPaint, region.translated, region.rect)
        }
        return out
    }

    /** Average the pixels in a thin band just outside the box as the fill colour. */
    private fun sampleBackground(bmp: Bitmap, rect: Rect): Int {
        val band = max(2, rect.height() / 6)
        var r = 0L; var g = 0L; var b = 0L; var n = 0L
        fun accumulate(x: Int, y: Int) {
            if (x < 0 || y < 0 || x >= bmp.width || y >= bmp.height) return
            val c = bmp.getPixel(x, y)
            r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++
        }
        var x = rect.left
        while (x < rect.right) {
            for (d in 1..band) {
                accumulate(x, rect.top - d)
                accumulate(x, rect.bottom + d)
            }
            x += max(1, rect.width() / 32)
        }
        var y = rect.top
        while (y < rect.bottom) {
            for (d in 1..band) {
                accumulate(rect.left - d, y)
                accumulate(rect.right + d, y)
            }
            y += max(1, rect.height() / 16)
        }
        if (n == 0L) return Color.WHITE
        return Color.rgb((r / n).toInt(), (g / n).toInt(), (b / n).toInt())
    }

    private fun luminance(c: Int): Double =
        (0.299 * Color.red(c) + 0.587 * Color.green(c) + 0.114 * Color.blue(c)) / 255.0

    /** Pick the largest text size that fits the box width, then centre it. */
    private fun drawFittedText(canvas: Canvas, paint: TextPaint, text: String, rect: Rect) {
        var size = rect.height() * 0.78f
        paint.textSize = size
        val maxWidth = rect.width() * 0.96f
        while (size > 6f && paint.measureText(text) > maxWidth) {
            size -= 1f
            paint.textSize = size
        }
        val fm = paint.fontMetrics
        val baseline = rect.exactCenterY() - (fm.ascent + fm.descent) / 2f
        val x = rect.exactCenterX() - paint.measureText(text) / 2f
        canvas.drawText(text, x, baseline, paint)
    }
}
