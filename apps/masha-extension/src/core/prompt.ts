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
 * Browser-free core — see core/constants.ts.
 *
 * Text-translation prompt. Mirrors the framing of MAGE's
 * ``Pipeline.create_prompt`` (packages/xian-vl/src/xian/pipeline.py): terse,
 * "Direct translation into {target}", output ONLY the translation, keep
 * reasoning brief, optional style terms. The OCR/3-part-layout rules from MAGE
 * are dropped here because this path translates text, not images. The key
 * addition over Google Translate is the page-context block, fed strictly as
 * reference so the model disambiguates meaning, pronouns, and terminology.
 */

import { MAX_CONTEXT_CHARS } from './constants';

export interface ChatMessage {
  role: 'system' | 'user';
  content: string;
}

export interface PromptOptions {
  /** The exact text the user selected — the only thing to be translated. */
  selection: string;
  /** Surrounding page text (title + section), reference only. May be empty. */
  context?: string;
  /** ``"Auto"`` to let the model detect, or an explicit language name. */
  sourceLang: string;
  targetLang: string;
  /** Optional stylistic register terms (mirrors MAGE ``styles``). */
  styles?: string[];
}

/** Collapse whitespace and clamp context to the configured budget. */
export function clampContext(context: string, max: number = MAX_CONTEXT_CHARS): string {
  const collapsed = context.replace(/\s+/g, ' ').trim();
  return collapsed.length > max ? collapsed.slice(0, max) + '…' : collapsed;
}

/** Build the OpenAI-style chat messages for a selection translation. */
export function buildTranslationMessages(opts: PromptOptions): ChatMessage[] {
  const fromClause = opts.sourceLang && opts.sourceLang !== 'Auto' ? ` from ${opts.sourceLang}` : '';
  const styleContext =
    opts.styles && opts.styles.length > 0
      ? ` Optionally use ${opts.styles.join(', ')} terms if it does not compromise accuracy.`
      : '';

  const systemPrompt =
    `You are MASHA, a highly precise${fromClause ? ` ${opts.sourceLang} →` : ''} ${opts.targetLang} translation engine.\n` +
    `Translate the user's SELECTION${fromClause} into ${opts.targetLang}.\n` +
    `RULES:\n` +
    `- Output ONLY the translation. No quotes, no romanization, no explanations, no conversational filler.\n` +
    `- Produce a direct, faithful translation that preserves the original tone, register, and inline formatting.\n` +
    `- Use the PAGE CONTEXT ONLY as reference to disambiguate meaning, pronouns, gender, honorifics, and terminology. ` +
    `Do NOT translate the context. Translate ONLY the text under SELECTION.\n` +
    `- If the selection is already in ${opts.targetLang}, return it unchanged.${styleContext}\n` +
    `- Keep any reasoning extremely brief; do not narrate your process.`;

  const context = opts.context ? clampContext(opts.context) : '';
  const userPrompt = context
    ? `PAGE CONTEXT (reference only — do not translate):\n${context}\n\nSELECTION (translate this):\n${opts.selection}`
    : `SELECTION (translate this):\n${opts.selection}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}
