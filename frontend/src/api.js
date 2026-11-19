const ENV_HOST = String(import.meta.env.VITE_API_HOST || '')
  .trim()
  .replace(/\/$/, '');

const HOST_CANDIDATES = [...new Set([ENV_HOST, '', 'http://127.0.0.1:8000', 'http://localhost:8000'])];

let resolvedHost;
let resolvePromise;

export function parseApiError(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload === 'string') return payload;
  const detail = payload.detail ?? payload.message ?? payload.error;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === 'string' ? item : item?.msg || JSON.stringify(item)))
      .join('; ');
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail);
  return fallback;
}

export function normalizeDocuments(raw) {
  const list = Array.isArray(raw) ? raw : Array.isArray(raw?.documents) ? raw.documents : [];
  return list
    .map((doc) => {
      if (typeof doc === 'string') {
        const ext = doc.split('.').pop()?.toLowerCase() || 'unknown';
        return { source: doc, type: ext };
      }
      const source = doc.source || doc.filename || doc.name;
      if (!source) return null;
      const ext = (doc.type || source.split('.').pop() || 'unknown').toLowerCase();
      return {
        source,
        type: ext,
        total_pages: doc.total_pages,
        slide_count: doc.slide_count,
        indexed: doc.indexed,
      };
    })
    .filter(Boolean);
}

export async function readJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function probeHost(host) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(`${host}/api/status`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function resolveApiHost() {
  if (resolvedHost !== undefined) return resolvedHost;
  if (!resolvePromise) {
    resolvePromise = (async () => {
      for (const host of HOST_CANDIDATES) {
        if (await probeHost(host)) {
          resolvedHost = host;
          return host;
        }
      }
      resolvedHost = ENV_HOST;
      return resolvedHost;
    })();
  }
  return resolvePromise;
}

export async function apiFetch(path, options) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const host = await resolveApiHost();
  try {
    return await fetch(`${host}${normalizedPath}`, options);
  } catch (error) {
    resolvedHost = undefined;
    resolvePromise = undefined;
    const retryHost = await resolveApiHost();
    if (retryHost === host) throw error;
    return fetch(`${retryHost}${normalizedPath}`, options);
  }
}
