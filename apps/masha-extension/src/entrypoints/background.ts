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

import { normalizeLemonadeBaseUrl } from '../utils/lemonadeUrl';

const DEFAULT_LEMONADE_URL = 'http://localhost:13305/v1';

async function getLemonadeUrl(): Promise<string> {
  const result = await chrome.storage.local.get(['serverUrl']);
  return normalizeLemonadeBaseUrl(result.serverUrl || DEFAULT_LEMONADE_URL);
}

async function translateText(text: string, targetLanguage: string = 'English'): Promise<string> {
  const url = await getLemonadeUrl();
  const endpoint = `${url}/chat/completions`;
  
  const systemPrompt = `You are MASHA, a highly capable text translator. Translate the following text to ${targetLanguage}. Maintain the original context, tone, and formatting. Return ONLY the translated text without any conversational filler or explanations.`;
  
  const payload = {
    model: 'xian-vl',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: text }
    ],
    max_tokens: 1024,
    temperature: 0.3
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!response.ok) throw new Error(`Lemonade API Error: ${response.statusText}`);
  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;
  return (typeof content === 'string' ? content : '').trim();
}

export default defineBackground(() => {
  console.log('MASHA background script initialized.');

  chrome.runtime.onMessage.addListener((message: any, sender: chrome.runtime.MessageSender, sendResponse: (response?: any) => void) => {
    if (sender.id !== chrome.runtime.id) {
      sendResponse({ success: false, error: 'Invalid sender' });
      return;
    }

    if (message.type === 'TRANSLATE_TEXT') {
      translateText(message.payload.text, message.payload.targetLanguage || 'English')
        .then(translation => sendResponse({ success: true, translation }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true; // Keep channel open for async response
    }
  });
});
