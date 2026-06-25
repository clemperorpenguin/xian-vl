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
 * Content script (WXT). Realizes the page-facing half of PlatformBridge:
 * selection + page-context capture, the non-destructive overlay, and
 * replace-in-place for editable inputs. The model call itself is delegated to
 * the background worker (see background.ts).
 */

import { SelectionContext } from '../platform/bridge';

const BLOCK_SELECTOR = 'p, li, blockquote, td, th, article, section, main, div';

export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    let activeOverlay: HTMLElement | null = null;

    /** The editable element holding the current selection, if any. */
    function editableTarget(): HTMLInputElement | HTMLTextAreaElement | null {
      const el = document.activeElement;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
        const input = el as HTMLInputElement | HTMLTextAreaElement;
        if (input.selectionStart !== null && input.selectionStart !== input.selectionEnd) {
          return input;
        }
      }
      return null;
    }

    /** Walk up from a node to the nearest meaningful block container. */
    function nearestBlock(node: Node | null): Element | null {
      let el = node instanceof Element ? node : node?.parentElement ?? null;
      while (el && el !== document.body) {
        if (el.matches?.(BLOCK_SELECTOR)) return el;
        el = el.parentElement;
      }
      return document.body;
    }

    /** Capture the selection plus surrounding page context (title + section). */
    function getSelectionContext(): SelectionContext | null {
      const input = editableTarget();
      if (input) {
        const text = input.value.substring(input.selectionStart!, input.selectionEnd!).trim();
        if (!text) return null;
        return { text, context: input.value.trim(), isEditable: true };
      }

      const selection = window.getSelection();
      const text = selection?.toString().trim() || '';
      if (!text) return null;

      let context = document.title ? `${document.title}\n` : '';
      if (selection && selection.rangeCount > 0) {
        const block = nearestBlock(selection.getRangeAt(0).commonAncestorContainer);
        const blockText = (block as HTMLElement)?.innerText?.trim() || '';
        if (blockText && blockText !== text) context += blockText;
      }
      return { text, context: context.trim(), isEditable: false };
    }

    function removeOverlay() {
      activeOverlay?.remove();
      activeOverlay = null;
    }

    /** Anchor a freshly created box near the current selection/caret. */
    function anchorBox(box: HTMLElement) {
      const selection = window.getSelection();
      let left = window.scrollX + 80;
      let top = window.scrollY + 80;
      if (selection && selection.rangeCount > 0) {
        const rect = selection.getRangeAt(0).getBoundingClientRect();
        if (rect.width || rect.height) {
          left = rect.left + window.scrollX;
          top = rect.bottom + window.scrollY + 8;
        }
      }
      box.style.left = `${Math.max(8, Math.min(left, window.innerWidth - 360))}px`;
      box.style.top = `${Math.max(8, top)}px`;
    }

    function baseBox(): HTMLElement {
      removeOverlay();
      const box = document.createElement('div');
      activeOverlay = box;
      box.style.cssText = [
        'position:absolute',
        'z-index:2147483647',
        'max-width:340px',
        'padding:12px 14px',
        'background:#0f172a',
        'color:#f8fafc',
        'border:1px solid #334155',
        'border-radius:12px',
        'box-shadow:0 10px 30px -10px rgba(0,0,0,0.6)',
        'font:13px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
      ].join(';');
      anchorBox(box);
      // Dismiss on outside click.
      setTimeout(() => {
        document.addEventListener(
          'mousedown',
          (e) => {
            if (activeOverlay && !activeOverlay.contains(e.target as Node)) removeOverlay();
          },
          { once: true },
        );
      }, 0);
      document.body.appendChild(box);
      return box;
    }

    function showLoading() {
      const box = baseBox();
      box.innerHTML =
        '<div style="display:flex;align-items:center;gap:8px;color:#cbd5e1">' +
        '<span style="width:14px;height:14px;border:2px solid rgba(255,255,255,0.25);' +
        'border-top-color:#a855f7;border-radius:50%;display:inline-block;' +
        'animation:masha-spin 0.6s linear infinite"></span> Translating…</div>' +
        '<style>@keyframes masha-spin{to{transform:rotate(360deg)}}</style>';
    }

    function showError(message: string) {
      const box = baseBox();
      box.style.borderColor = '#ef4444';
      box.textContent = `⚠️ ${message}`;
    }

    function showOverlay(selection: string, translation: string) {
      const box = baseBox();

      const body = document.createElement('div');
      body.textContent = translation;
      body.style.whiteSpace = 'pre-wrap';

      const bar = document.createElement('div');
      bar.style.cssText =
        'margin-top:10px;padding-top:8px;border-top:1px solid #1e293b;' +
        'display:flex;gap:14px;font-size:11px;color:#94a3b8';

      let showingOriginal = false;
      const toggle = document.createElement('button');
      const copy = document.createElement('button');
      for (const b of [toggle, copy]) {
        b.style.cssText = 'background:none;border:none;color:#a5b4fc;cursor:pointer;padding:0;font:inherit';
      }
      toggle.textContent = 'Show original';
      toggle.onclick = () => {
        showingOriginal = !showingOriginal;
        body.textContent = showingOriginal ? selection : translation;
        toggle.textContent = showingOriginal ? 'Show translation' : 'Show original';
      };
      copy.textContent = 'Copy';
      copy.onclick = () => {
        navigator.clipboard?.writeText(translation).then(() => {
          copy.textContent = 'Copied!';
          setTimeout(() => (copy.textContent = 'Copy'), 1200);
        });
      };

      bar.append(toggle, copy);
      box.append(body, bar);
    }

    /** Replace the selection inside an editable input (composition mode). */
    function replaceSelection(text: string) {
      const input = editableTarget();
      if (!input) return;
      const { selectionStart: start, selectionEnd: end, value } = input;
      input.value = value.substring(0, start!) + text + value.substring(end!);
      input.selectionStart = input.selectionEnd = start! + text.length;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      input.focus();
    }

    async function runTranslation() {
      const captured = getSelectionContext();
      if (!captured) {
        showError('Select some text first.');
        return;
      }
      showLoading();
      try {
        const response = await chrome.runtime.sendMessage({
          type: 'MASHA_TRANSLATE',
          payload: captured,
        });
        if (!response?.success) {
          showError(response?.error || 'Translation failed.');
          return;
        }
        if (captured.isEditable) {
          replaceSelection(response.translation);
          removeOverlay();
        } else {
          showOverlay(captured.text, response.translation);
        }
      } catch {
        showError('Could not reach the extension background.');
      }
    }

    // Trigger relayed from the background context-menu click.
    chrome.runtime.onMessage.addListener((message: { type?: string }) => {
      if (message?.type === 'MASHA_TRIGGER') runTranslation();
    });
  },
});
