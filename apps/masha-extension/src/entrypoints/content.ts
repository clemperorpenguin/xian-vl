import { defineContentScript } from 'wxt/sandbox';

export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    console.log('MASHA content script injected.');

    let imageModeActive = false;
    let hoveredImage: HTMLImageElement | null = null;
    let imageOverlay: HTMLElement | null = null;

    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === 'START_PAGE_TRANSLATION') {
        runPageTranslation().catch(console.error);
        sendResponse({ success: true });
      } else if (message.type === 'TOGGLE_IMAGE_MODE') {
        imageModeActive = !imageModeActive;
        console.log(`MASHA: Image selection mode ${imageModeActive ? 'ON' : 'OFF'}`);
        if (!imageModeActive && imageOverlay) {
          imageOverlay.remove();
          imageOverlay = null;
        }
        sendResponse({ success: true, active: imageModeActive });
      } else if (message.type === 'GATHER_LORE_DATA') {
        const loreData = {
          title: document.title,
          url: window.location.href,
          context: lastContext,
          translated_content: document.body.innerText,
          metadata: {
            extracted_at: new Date().toISOString(),
            description: document.querySelector('meta[name="description"]')?.getAttribute('content') || ''
          }
        };
        sendResponse({ success: true, data: loreData });
      }
    });

    let lastContext = 'General Website';

    async function runPageTranslation() {
      console.log('MASHA: Starting full page translation...');
      
      const metadata = {
        title: document.title,
        url: window.location.href,
        description: document.querySelector('meta[name="description"]')?.getAttribute('content') || '',
        snippet: document.body.innerText.substring(0, 400)
      };

      console.log('MASHA: Analyzing context...');
      const contextRes = await chrome.runtime.sendMessage({
        type: 'ANALYZE_CONTEXT',
        payload: metadata
      });
      
      lastContext = (contextRes && contextRes.success) ? contextRes.context : 'General Website';
      console.log(`MASHA: Determined context -> ${lastContext}`);

      const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode: (node) => {
            if (!node.nodeValue?.trim()) return NodeFilter.FILTER_REJECT;
            const parent = node.parentElement;
            if (parent && ['SCRIPT', 'STYLE', 'NOSCRIPT', 'CODE'].includes(parent.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
          }
        }
      );

      const textNodes: Text[] = [];
      let currentNode;
      while (currentNode = walker.nextNode()) {
        textNodes.push(currentNode as Text);
      }

      console.log(`MASHA: Found ${textNodes.length} text nodes to translate.`);

      const CHUNK_SIZE = 15;
      for (let i = 0; i < textNodes.length; i += CHUNK_SIZE) {
        const chunk = textNodes.slice(i, i + CHUNK_SIZE);
        const texts = chunk.map(n => n.nodeValue!.trim());
        
        try {
          const transRes = await chrome.runtime.sendMessage({
            type: 'TRANSLATE_BATCH',
            payload: { texts, context: lastContext, targetLanguage: 'English' }
          });

          if (transRes && transRes.success && Array.isArray(transRes.translations)) {
            chunk.forEach((node, idx) => {
              const replacement = transRes.translations[idx];
              if (!replacement) return;
              const raw = node.nodeValue!;
              const needle = texts[idx];
              // Replace every occurrence of the trimmed segment (String.replace only hits once).
              node.nodeValue = needle ? raw.split(needle).join(replacement) : raw;
            });
          }
        } catch (e) {
          console.error('Batch translation error', e);
        }
      }
      console.log('MASHA: Page translation complete.');
    }

    let popupContainer: HTMLElement | null = null;

    function removePopup() {
      if (popupContainer) {
        popupContainer.remove();
        popupContainer = null;
      }
    }

    function createPopup(rect: DOMRect, text: string) {
      removePopup(); // Ensure only one popup exists

      popupContainer = document.createElement('div');
      popupContainer.style.position = 'absolute';
      popupContainer.style.left = `${rect.left + window.scrollX}px`;
      popupContainer.style.top = `${rect.bottom + window.scrollY + 10}px`;
      popupContainer.style.zIndex = '999999';
      popupContainer.style.fontFamily = 'sans-serif';
      popupContainer.style.backgroundColor = '#1f2937';
      popupContainer.style.color = '#ffffff';
      popupContainer.style.padding = '8px 12px';
      popupContainer.style.borderRadius = '8px';
      popupContainer.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
      popupContainer.style.display = 'flex';
      popupContainer.style.flexDirection = 'column';
      popupContainer.style.gap = '8px';
      popupContainer.style.maxWidth = '300px';
      popupContainer.style.fontSize = '14px';
      popupContainer.style.lineHeight = '1.5';
      popupContainer.style.transition = 'all 0.2s ease';

      const translateBtn = document.createElement('button');
      translateBtn.innerText = '✨ Translate';
      translateBtn.style.backgroundColor = '#3b82f6';
      translateBtn.style.color = 'white';
      translateBtn.style.border = 'none';
      translateBtn.style.padding = '6px 12px';
      translateBtn.style.borderRadius = '6px';
      translateBtn.style.cursor = 'pointer';
      translateBtn.style.fontWeight = 'bold';
      
      translateBtn.onmouseover = () => translateBtn.style.backgroundColor = '#2563eb';
      translateBtn.onmouseout = () => translateBtn.style.backgroundColor = '#3b82f6';

      translateBtn.onclick = async (e) => {
        e.stopPropagation(); // prevent document click from closing it
        translateBtn.innerText = 'Translating...';
        translateBtn.style.opacity = '0.7';
        translateBtn.disabled = true;

        try {
          const response = await chrome.runtime.sendMessage({
            type: 'TRANSLATE_TEXT',
            payload: { text, targetLanguage: 'English' }
          });

          if (response && response.success) {
            popupContainer!.innerHTML = '';
            
            const title = document.createElement('div');
            title.innerText = 'MASHA Translation';
            title.style.fontSize = '12px';
            title.style.fontWeight = 'bold';
            title.style.color = '#9ca3af';
            title.style.marginBottom = '4px';

            const result = document.createElement('div');
            result.innerText = response.translation;
            
            popupContainer!.appendChild(title);
            popupContainer!.appendChild(result);
          } else {
            translateBtn.innerText = 'Error!';
            translateBtn.style.backgroundColor = '#ef4444';
          }
        } catch (err) {
          console.error(err);
          translateBtn.innerText = 'Network Error';
          translateBtn.style.backgroundColor = '#ef4444';
        }
      };

      popupContainer.appendChild(translateBtn);
      document.body.appendChild(popupContainer);
    }

    document.addEventListener('mouseup', (e) => {
      // Don't trigger if clicking inside the popup
      if (popupContainer && popupContainer.contains(e.target as Node)) {
        return;
      }

      setTimeout(() => {
        const selection = window.getSelection();
        const text = selection?.toString().trim();

        if (text && text.length > 0 && selection && selection.rangeCount > 0) {
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          createPopup(rect, text);
        } else {
          removePopup();
        }
      }, 10);
    });

    // Remove popup if clicking away completely
    document.addEventListener('mousedown', (e) => {
      if (popupContainer && !popupContainer.contains(e.target as Node)) {
        setTimeout(() => {
           const selection = window.getSelection();
           if (!selection?.toString().trim()) {
              removePopup();
           }
        }, 10);
      }
    });

    // --- Image Selection Mode Logic ---
    document.addEventListener('mouseover', (e) => {
      if (!imageModeActive) return;
      const target = e.target as HTMLElement;
      if (target.tagName === 'IMG') {
        hoveredImage = target as HTMLImageElement;
        showImageOverlay(hoveredImage);
      }
    });

    function showImageOverlay(img: HTMLImageElement) {
      if (!imageOverlay) {
        imageOverlay = document.createElement('div');
        imageOverlay.style.position = 'absolute';
        imageOverlay.style.backgroundColor = 'rgba(139, 92, 246, 0.4)'; // Purple tint
        imageOverlay.style.border = '3px solid #8b5cf6';
        imageOverlay.style.cursor = 'pointer';
        imageOverlay.style.zIndex = '999998';
        imageOverlay.style.display = 'flex';
        imageOverlay.style.alignItems = 'center';
        imageOverlay.style.justifyContent = 'center';
        imageOverlay.style.transition = 'all 0.2s';
        
        const btn = document.createElement('button');
        btn.innerText = '🖼️ Translate Image';
        btn.style.padding = '8px 16px';
        btn.style.backgroundColor = '#1f2937';
        btn.style.color = 'white';
        btn.style.border = 'none';
        btn.style.borderRadius = '8px';
        btn.style.fontWeight = 'bold';
        btn.style.cursor = 'pointer';
        
        imageOverlay.appendChild(btn);
        document.body.appendChild(imageOverlay);

        imageOverlay.onclick = async (e) => {
          e.stopPropagation();
          e.preventDefault();
          if (!hoveredImage) return;

          btn.innerText = 'Translating...';
          btn.style.opacity = '0.7';

          try {
            const res = await chrome.runtime.sendMessage({
              type: 'TRANSLATE_IMAGE',
              payload: { imageUrl: hoveredImage.src, targetLanguage: 'English' }
            });
            if (res && res.success) {
              hoveredImage.src = res.translatedUrl;
              btn.innerText = 'Done!';
              btn.style.backgroundColor = '#10b981';
              setTimeout(() => {
                if (imageOverlay) imageOverlay.remove();
                imageOverlay = null;
              }, 1000);
            } else {
              btn.innerText = 'Error';
              btn.style.backgroundColor = '#ef4444';
            }
          } catch (err) {
            btn.innerText = 'Error';
            btn.style.backgroundColor = '#ef4444';
          }
        };
      }

      const rect = img.getBoundingClientRect();
      imageOverlay.style.left = `${rect.left + window.scrollX}px`;
      imageOverlay.style.top = `${rect.top + window.scrollY}px`;
      imageOverlay.style.width = `${rect.width}px`;
      imageOverlay.style.height = `${rect.height}px`;
    }
  },
});
