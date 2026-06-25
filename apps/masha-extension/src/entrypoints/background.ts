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
 * Background service worker (WXT). Owns the platform pieces that must live off
 * the page: the context-menu trigger and the Lemonade HTTP call. Routing the
 * fetch through here (not the content script) is what lets it reach a local
 * http:// Lemonade node from an https:// page without mixed-content blocking —
 * the same reason it works in Firefox as well as Chrome.
 */

import { getConfig } from '../utils/config';
import { translate } from '../core/translator';
import { SelectionContext } from '../platform/bridge';

const MENU_ID = 'masha-translate';

/** Fired by the content script: a captured selection to translate. */
interface TranslateMessage {
  type: 'MASHA_TRANSLATE';
  payload: SelectionContext;
}

async function handleTranslate(payload: SelectionContext): Promise<string> {
  const config = await getConfig();
  return translate(fetch as any, config, {
    selection: payload.text,
    context: payload.context,
    sourceLang: config.sourceLang,
    targetLang: config.targetLang,
    styles: config.styles,
  });
}

export default defineBackground(() => {
  // Register the right-click trigger (shown only when text is selected).
  chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.removeAll(() => {
      chrome.contextMenus.create({
        id: MENU_ID,
        title: 'Translate with MASHA',
        contexts: ['selection'],
      });
    });
  });

  // Context-menu click → ask the page's content script to capture + render.
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId !== MENU_ID || !tab?.id) return;
    chrome.tabs.sendMessage(tab.id, { type: 'MASHA_TRIGGER' }).catch(() => {
      // No content script (e.g. chrome:// page or not yet injected) — ignore.
    });
  });

  // Content script asks us to run the model call.
  chrome.runtime.onMessage.addListener(
    (message: TranslateMessage, sender, sendResponse) => {
      if (sender.id !== chrome.runtime.id) {
        sendResponse({ success: false, error: 'Invalid sender' });
        return;
      }
      if (message.type === 'MASHA_TRANSLATE') {
        handleTranslate(message.payload)
          .then((translation) => sendResponse({ success: true, translation }))
          .catch((error) => sendResponse({ success: false, error: String(error?.message || error) }));
        return true; // async response
      }
    },
  );
});
