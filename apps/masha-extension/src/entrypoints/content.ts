
export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    console.log('MASHA content script loaded.');

    let activeBtn: HTMLButtonElement | null = null;
    let lastMouseUpEvent: MouseEvent | null = null;

    // Track mouseup coordinates as a fallback position
    document.addEventListener('mouseup', (e) => {
      lastMouseUpEvent = e;
      
      // Delay slightly to allow the browser selection to update
      setTimeout(() => {
        handleSelection();
      }, 10);
    });

    document.addEventListener('mousedown', (e) => {
      // Remove button if user clicks elsewhere
      if (activeBtn && !activeBtn.contains(e.target as Node)) {
        removeButton();
      }
    });

    function removeButton() {
      if (activeBtn) {
        activeBtn.remove();
        activeBtn = null;
      }
    }

    function handleSelection() {
      let selectedText = '';
      let isInput = false;
      const activeEl = document.activeElement;

      // 1. Detect selection within an input or textarea
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
        const input = activeEl as HTMLInputElement | HTMLTextAreaElement;
        const start = input.selectionStart;
        const end = input.selectionEnd;
        if (start !== null && end !== null && start !== end) {
          selectedText = input.value.substring(start, end).trim();
          isInput = true;
        }
      }

      // 2. Standard DOM selection
      if (!selectedText) {
        const selection = window.getSelection();
        selectedText = selection?.toString().trim() || '';
      }

      if (!selectedText) {
        return; // No text selected
      }

      createFloatingButton(selectedText, isInput, activeEl);
    }

    function createFloatingButton(text: string, isInput: boolean, activeEl: Element | null) {
      removeButton(); // Remove previous button if any

      const btn = document.createElement('button');
      activeBtn = btn;

      // Premium visual styling
      btn.style.position = 'absolute';
      btn.style.zIndex = '999999';
      btn.style.background = 'linear-gradient(135deg, #6366f1, #a855f7)';
      btn.style.color = '#ffffff';
      btn.style.border = 'none';
      btn.style.borderRadius = '9999px';
      btn.style.padding = '8px 16px';
      btn.style.fontFamily = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      btn.style.fontSize = '13px';
      btn.style.fontWeight = '600';
      btn.style.cursor = 'pointer';
      btn.style.boxShadow = '0 10px 25px -5px rgba(99, 102, 241, 0.4), 0 8px 10px -6px rgba(168, 85, 247, 0.3)';
      btn.style.display = 'flex';
      btn.style.alignItems = 'center';
      btn.style.gap = '6px';
      btn.style.transition = 'transform 0.15s ease, opacity 0.15s ease, background 0.2s ease';
      btn.style.transform = 'scale(0.95)';
      btn.style.opacity = '0';
      
      // Inject self-contained spinner animation
      const styleTag = document.createElement('style');
      styleTag.innerHTML = `
        @keyframes masha-spin {
          to { transform: rotate(360deg); }
        }
        .masha-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid rgba(255,255,255,0.3);
          border-radius: 50%;
          border-top-color: #ffffff;
          animation: masha-spin 0.6s linear infinite;
        }
      `;
      btn.appendChild(styleTag);

      const iconSpan = document.createElement('span');
      iconSpan.innerText = '✨';
      const textSpan = document.createElement('span');
      textSpan.innerText = 'Translate & Replace';
      btn.appendChild(iconSpan);
      btn.appendChild(textSpan);

      // Positioning logic
      let left = 0;
      let top = 0;

      const selection = window.getSelection();
      if (!isInput && selection && selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          left = rect.left + window.scrollX + (rect.width / 2) - 80;
          top = rect.top + window.scrollY - 40;
        }
      }

      // Fallback to cursor location (for text inputs / textareas)
      if (left <= 0 || top <= 0) {
        if (lastMouseUpEvent) {
          left = lastMouseUpEvent.pageX - 80;
          top = lastMouseUpEvent.pageY - 45;
        } else {
          left = 100;
          top = 100;
        }
      }

      // Keep button on screen
      left = Math.max(10, Math.min(left, window.innerWidth - 180));
      top = Math.max(10, top);

      btn.style.left = `${left}px`;
      btn.style.top = `${top}px`;

      // Smooth hover styles
      btn.onmouseover = () => {
        btn.style.transform = 'scale(1.03)';
        btn.style.background = 'linear-gradient(135deg, #4f46e5, #9333ea)';
      };
      btn.onmouseout = () => {
        btn.style.transform = 'scale(1)';
        btn.style.background = 'linear-gradient(135deg, #6366f1, #a855f7)';
      };

      btn.onclick = async (e) => {
        e.stopPropagation();
        e.preventDefault();

        // 1. Set Loading state
        btn.disabled = true;
        btn.style.background = '#4b5563';
        btn.style.boxShadow = 'none';
        btn.style.cursor = 'not-allowed';
        textSpan.innerText = 'Translating...';
        iconSpan.innerHTML = '<div class="masha-spinner"></div>';

        try {
          const response = await chrome.runtime.sendMessage({
            type: 'TRANSLATE_TEXT',
            payload: { text }
          });

          if (response && response.success && response.translation) {
            replaceText(response.translation, isInput, activeEl);
            removeButton();
          } else {
            showErrorState(response?.error || 'Translation failed');
          }
        } catch (err) {
          showErrorState('Connection error');
        }
      };

      document.body.appendChild(btn);

      // Trigger appearance animation
      requestAnimationFrame(() => {
        btn.style.transform = 'scale(1)';
        btn.style.opacity = '1';
      });

      function showErrorState(msg: string) {
        console.error('MASHA error:', msg);
        btn.style.background = '#ef4444';
        btn.style.boxShadow = '0 10px 25px -5px rgba(239, 68, 68, 0.4)';
        textSpan.innerText = 'Error';
        iconSpan.innerText = '⚠️';
        
        setTimeout(() => {
          if (activeBtn === btn) {
            removeButton();
          }
        }, 2000);
      }
    }

    function replaceText(translatedText: string, isInput: boolean, activeEl: Element | null) {
      if (isInput && activeEl) {
        const input = activeEl as HTMLInputElement | HTMLTextAreaElement;
        const start = input.selectionStart;
        const end = input.selectionEnd;
        if (start !== null && end !== null) {
          const val = input.value;
          input.value = val.substring(0, start) + translatedText + val.substring(end);
          input.selectionStart = input.selectionEnd = start + translatedText.length;
          
          // Send inputs so modern frontend frameworks update their bounds/state
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
          input.focus();
        }
      } else {
        const selection = window.getSelection();
        if (selection && selection.rangeCount > 0) {
          const range = selection.getRangeAt(0);
          range.deleteContents();
          const textNode = document.createTextNode(translatedText);
          range.insertNode(textNode);
          selection.removeAllRanges();
        }
      }
    }
  }
});
