/**
 * Xian Browser Extension — Content script.
 *
 * Listens for translate_selection messages from the popup and sends
 * the selected text to Lemonade for translation.
 */

const LEMONADE_URL = "http://localhost:13305/v1";

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.action === "translate_selection") {
    const selectedText = window.getSelection()?.toString().trim();
    if (!selectedText) {
      showToast("Select some text first!");
      return;
    }
    translateText(selectedText, msg.sourceLang, msg.targetLang);
  }
});

async function translateText(text, sourceLang, targetLang) {
  try {
    const res = await fetch(`${LEMONADE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "",
        messages: [
          {
            role: "system",
            content: `You are a translator. Translate the following ${sourceLang} text to ${targetLang}. Only output the translation, nothing else.`,
          },
          { role: "user", content: text },
        ],
        max_tokens: 1024,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    const translation = data.choices?.[0]?.message?.content ?? "Translation failed";
    showOverlay(translation);
  } catch (err) {
    showToast(`Translation error: ${err.message}`);
  }
}

function showOverlay(text) {
  const existing = document.getElementById("xian-overlay");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "xian-overlay";
  overlay.textContent = text;
  Object.assign(overlay.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    maxWidth: "400px",
    padding: "16px 20px",
    background: "rgba(26, 26, 46, 0.95)",
    color: "#e0e0e0",
    borderRadius: "12px",
    fontSize: "14px",
    lineHeight: "1.5",
    boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
    border: "1px solid rgba(102, 126, 234, 0.3)",
    zIndex: "999999",
    fontFamily: '"Inter", system-ui, sans-serif',
    backdropFilter: "blur(12px)",
    cursor: "pointer",
  });

  overlay.addEventListener("click", () => overlay.remove());
  document.body.appendChild(overlay);

  setTimeout(() => overlay.remove(), 15000);
}

function showToast(msg) {
  const toast = document.createElement("div");
  toast.textContent = msg;
  Object.assign(toast.style, {
    position: "fixed",
    top: "20px",
    right: "20px",
    padding: "10px 16px",
    background: "#f87171",
    color: "white",
    borderRadius: "8px",
    fontSize: "13px",
    zIndex: "999999",
  });
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
