import { defineBackground } from 'wxt/sandbox';

import { normalizeLemonadeBaseUrl, isAllowedImageTranslationUrl } from '../utils/lemonadeUrl';

const DEFAULT_LEMONADE_URL = 'http://localhost:13305/v1';

async function getLemonadeUrl(): Promise<string> {
  const result = await chrome.storage.local.get(['serverUrl']);
  return normalizeLemonadeBaseUrl(result.serverUrl || DEFAULT_LEMONADE_URL);
}

// IndexedDB Caching for Images
const initDB = () => {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open('MashaImageCache', 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('images');
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

async function getCachedImage(url: string): Promise<string | null> {
  const db = await initDB();
  return new Promise((resolve) => {
    const transaction = db.transaction('images', 'readonly');
    const store = transaction.objectStore('images');
    const request = store.get(url);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => resolve(null);
  });
}

async function setCachedImage(url: string, dataUrl: string): Promise<void> {
  const db = await initDB();
  return new Promise((resolve) => {
    const transaction = db.transaction('images', 'readwrite');
    const store = transaction.objectStore('images');
    store.put(dataUrl, url);
    transaction.oncomplete = () => resolve();
  });
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

async function analyzeContext(metadata: any): Promise<string> {
  const url = await getLemonadeUrl();
  const endpoint = `${url}/chat/completions`;
  const systemPrompt = `Analyze the following webpage metadata and provide a concise 2-5 word description of its genre and context (e.g., "Chinese E-Commerce Site", "Japanese Cooking Blog"). Return ONLY the description.`;
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'xian-vl',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: JSON.stringify(metadata) }
      ],
      max_tokens: 50,
      temperature: 0.1
    })
  });

  if (!response.ok) throw new Error(`Lemonade API Error: ${response.statusText}`);
  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;
  return (typeof content === 'string' ? content : '').trim();
}

async function batchTranslateText(texts: string[], context: string, targetLanguage: string = 'English'): Promise<string[]> {
  const url = await getLemonadeUrl();
  const endpoint = `${url}/chat/completions`;
  const systemPrompt = `You are an expert translator. The context of this text is: "${context}". Translate the following JSON array of strings to ${targetLanguage}. Return ONLY a valid JSON array of translated strings in the exact same order.`;
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'xian-vl',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: JSON.stringify(texts) }
      ],
      max_tokens: 2048,
      temperature: 0.2
    })
  });

  if (!response.ok) throw new Error(`Lemonade API Error: ${response.statusText}`);
  const data = await response.json();
  const raw = data.choices?.[0]?.message?.content;
  if (typeof raw !== 'string') {
    return texts;
  }
  try {
    return JSON.parse(raw.trim());
  } catch (e) {
    return texts;
  }
}

async function translateImage(imageUrl: string, targetLanguage: string = 'English'): Promise<string> {
  if (!isAllowedImageTranslationUrl(imageUrl)) {
    throw new Error('Unsupported image URL scheme (allowed: http, https, data, blob)');
  }
  const cached = await getCachedImage(imageUrl);
  if (cached) return cached;

  const url = await getLemonadeUrl();
  const endpoint = `${url}/images/edit`; // Lemonade specialized endpoint

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'xian-vl-vision',
      image: imageUrl,
      prompt: `Translate any text in this image to ${targetLanguage} while preserving the original style and background via inpainting.`,
      response_format: 'b64_json'
    })
  });

  if (!response.ok) throw new Error(`Lemonade API Error: ${response.statusText}`);

  const data = await response.json();
  const b64 = data.data?.[0]?.b64_json;
  if (typeof b64 !== 'string') {
    throw new Error('Lemonade API returned no image data');
  }
  const base64Image = `data:image/png;base64,${b64}`;
  await setCachedImage(imageUrl, base64Image); // Cache the result
  return base64Image;
}

export default defineBackground(() => {
  console.log('MASHA background script initialized.');

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (sender.id !== chrome.runtime.id) {
      sendResponse({ success: false, error: 'Invalid sender' });
      return;
    }

    if (message.type === 'TRANSLATE_TEXT') {
      translateText(message.payload.text, message.payload.targetLanguage || 'English')
        .then(translation => sendResponse({ success: true, translation }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
    }

    if (message.type === 'ANALYZE_CONTEXT') {
      analyzeContext(message.payload)
        .then(context => sendResponse({ success: true, context }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
    }

    if (message.type === 'TRANSLATE_BATCH') {
      batchTranslateText(message.payload.texts, message.payload.context, message.payload.targetLanguage || 'English')
        .then(translations => sendResponse({ success: true, translations }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
    }

    if (message.type === 'TRANSLATE_IMAGE') {
      translateImage(message.payload.imageUrl, message.payload.targetLanguage || 'English')
        .then(translatedUrl => sendResponse({ success: true, translatedUrl }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
    }
  });
});
