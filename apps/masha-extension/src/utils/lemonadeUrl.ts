/*
 * Masha — Browser extension selection translator.
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

/** Normalize Lemonade OpenAI base URL to include exactly one ``/v1`` suffix. */
export function normalizeLemonadeBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'http://localhost:13305/v1';
  }
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}

/** True when the user should be reminded that HTTP to a non-loopback host is unsafe on untrusted networks. */
export function shouldWarnHttpToNonLoopback(url: string): boolean {
  try {
    const u = new URL(normalizeLemonadeBaseUrl(url));
    if (u.protocol !== 'http:') {
      return false;
    }
    const host = u.hostname.toLowerCase();
    return host !== 'localhost' && host !== '127.0.0.1' && host !== '[::1]' && host !== '::1';
  } catch {
    return false;
  }
}

