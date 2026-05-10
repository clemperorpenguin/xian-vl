/**
 * Xian Browser Extension — Background service worker.
 *
 * Handles extension lifecycle events and could be extended for
 * context menu integration or badge updates.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log("Xian Translator extension installed.");
});
