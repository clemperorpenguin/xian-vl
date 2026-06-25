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

/*
 * Browser-free core — see core/constants.ts. The HTTP call is parameterized on
 * a `fetch`-shaped function so this module has no platform dependency at all.
 */

import { normalizeLemonadeBaseUrl } from '../utils/lemonadeUrl';
import { DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE } from './constants';
import { buildTranslationMessages, PromptOptions } from './prompt';

/** Mirror of MAGE's reasoning-tag stripping (xian/pipeline.py THINK_*_PAT). */
const THINK_TAGS = /<think>[\s\S]*?<\/think>/gi;
const THINK_OPEN = /<think>[\s\S]*$/i;

/** Characters MAGE trims from the edges of a parsed translation. */
const EDGE_TRIM = /^[\s"'`*•\-]+|[\s"'`*•\-]+$/g;

/** Remove reasoning tags and wrapping punctuation from a model response. */
export function cleanResponse(raw: string): string {
  let cleaned = raw.replace(THINK_TAGS, '');
  cleaned = cleaned.replace(THINK_OPEN, ''); // unmatched/streaming-truncated <think>
  return cleaned.replace(EDGE_TRIM, '').trim();
}

/** Minimal subset of the global `fetch` we depend on (keeps core testable). */
export type FetchFn = (
  input: string,
  init: { method: string; headers: Record<string, string>; body: string },
) => Promise<{ ok: boolean; statusText: string; json(): Promise<any> }>;

export interface TranslateConfig {
  /** Lemonade base URL (with or without ``/v1`` — normalized internally). */
  serverUrl: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

/**
 * Translate a selection via an OpenAI-compatible Lemonade endpoint.
 * Shaped identically to MAGE's chat call so the Python core stays a drop-in.
 */
export async function translate(
  fetchFn: FetchFn,
  config: TranslateConfig,
  opts: PromptOptions,
): Promise<string> {
  const base = normalizeLemonadeBaseUrl(config.serverUrl);
  const endpoint = `${base}/chat/completions`;

  const payload = {
    model: config.model || DEFAULT_MODEL,
    messages: buildTranslationMessages(opts),
    max_tokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
    temperature: config.temperature ?? DEFAULT_TEMPERATURE,
    stream: false,
  };

  const response = await fetchFn(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Lemonade API error: ${response.statusText || 'request failed'}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  const cleaned = cleanResponse(typeof content === 'string' ? content : '');
  if (!cleaned) {
    throw new Error('Empty translation returned');
  }
  return cleaned;
}
