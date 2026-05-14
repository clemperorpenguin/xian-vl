/** Normalize Lemonade OpenAI base URL to include exactly one ``/v1`` suffix. */
export function normalizeLemonadeBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'http://localhost:13305/v1';
  }
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}
