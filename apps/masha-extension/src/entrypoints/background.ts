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
