/** Normalize Lemonade OpenAI base URL to include exactly one ``/v1`` suffix. */
export function normalizeLemonadeBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'http://localhost:13305/v1';
  }
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}

/** True when the user should be reminded that HTTP to a non-loopback host is unsafe on untrusted networks. */
export function shouldWarnHttpToNonLoopback(url: string): boolean {
  try {
    const u = new URL(normalizeLemonadeBaseUrl(url));
    if (u.protocol !== 'http:') {
      return false;
    }
    const host = u.hostname.toLowerCase();
    return host !== 'localhost' && host !== '127.0.0.1' && host !== '[::1]' && host !== '::1';
  } catch {
    return false;
  }
}

/** Image ``src`` values the background script may forward to Lemonade for vision/inpainting. */
export function isAllowedImageTranslationUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    return (
      u.protocol === 'http:' ||
      u.protocol === 'https:' ||
      u.protocol === 'data:' ||
      u.protocol === 'blob:'
    );
  } catch {
    return false;
  }
}
