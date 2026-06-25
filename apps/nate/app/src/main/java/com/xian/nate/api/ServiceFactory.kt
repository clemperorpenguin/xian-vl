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

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/** Builds a [LemonadeService] for a given node URL. */
object ServiceFactory {

    /**
     * Normalize a user-entered Lemonade URL to a Retrofit base ending in ``/``.
     * Accepts ``http://host:13305``, ``…/v1``, or ``…/v1/`` and trims back to the
     * host root so the ``v1/...`` endpoint paths resolve correctly.
     */
    fun normalizeBaseUrl(raw: String): String {
        var s = raw.trim().ifBlank { "http://192.168.0.183:13305" }
        if (!s.startsWith("http://") && !s.startsWith("https://")) s = "http://$s"
        s = s.trimEnd('/')
        if (s.endsWith("/v1")) s = s.dropLast(3)
        return "$s/"
    }

    fun create(nodeUrl: String): LemonadeService {
        // Vision inference on a large model can take tens of seconds; allow for it.
        val client = OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()

        return Retrofit.Builder()
            .baseUrl(normalizeBaseUrl(nodeUrl))
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(LemonadeService::class.java)
    }
}
