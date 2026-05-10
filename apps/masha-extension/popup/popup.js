/**
 * Xian Browser Extension — Popup controller.
 *
 * Checks Lemonade health on open and dispatches translate requests
 * to the background service worker.
 */

const LEMONADE_URL = "http://localhost:13305/v1";

const statusEl = document.getElementById("status");
const translateBtn = document.getElementById("translate-btn");

async function checkHealth() {
  try {
    const res = await fetch(`${LEMONADE_URL}/health`);
    if (res.ok) {
      statusEl.textContent = "✓ Lemonade connected";
      statusEl.classList.add("connected");
      translateBtn.disabled = false;
    } else {
      statusEl.textContent = "✗ Lemonade not responding";
      statusEl.classList.add("error");
    }
  } catch {
    statusEl.textContent = "✗ Cannot reach Lemonade";
    statusEl.classList.add("error");
  }
}

translateBtn.addEventListener("click", () => {
  const sourceLang = document.getElementById("source-lang").value;
  const targetLang = document.getElementById("target-lang").value;

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]) return;
    
    const url = tabs[0].url || "";
    if (url.startsWith("chrome://") || url.startsWith("about:") || url.startsWith("chrome-extension://")) {
      statusEl.textContent = "⚠ Cannot translate on this page";
      return;
    }

    chrome.tabs.sendMessage(tabs[0].id, {
      action: "translate_selection",
      sourceLang,
      targetLang,
    });
  });
});

checkHealth();
