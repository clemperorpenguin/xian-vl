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

package com.xian.nate.util

import android.content.Context

/** Tiny SharedPreferences-backed config store. */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("nate", Context.MODE_PRIVATE)

    var nodeUrl: String
        get() = prefs.getString(KEY_URL, DEFAULT_URL) ?: DEFAULT_URL
        set(v) = prefs.edit().putString(KEY_URL, v).apply()

    var targetLang: String
        get() = prefs.getString(KEY_LANG, DEFAULT_LANG) ?: DEFAULT_LANG
        set(v) = prefs.edit().putString(KEY_LANG, v).apply()

    companion object {
        // Default points at the dev node; edit it in-app for your own Lemonade host.
        const val DEFAULT_URL = "http://192.168.0.183:13305"
        const val DEFAULT_LANG = "English"
        val LANGUAGES = listOf(
            "English", "Chinese", "Japanese", "Korean", "Spanish",
            "French", "German", "Russian", "Arabic", "Hindi", "Vietnamese",
        )
        private const val KEY_URL = "node_url"
        private const val KEY_LANG = "target_lang"
    }
}
