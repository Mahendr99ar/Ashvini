import type { CatalogueObject, Conjunction, Health, Manoeuvre, ModelCard, SpaceWeather } from "./types";

export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:8000";

let apiKey = "";
try {
  apiKey = sessionStorage.getItem("ashvini-key") ?? "";
} catch {
  apiKey = "";
}

export function setApiKey(key: string): void {
  apiKey = key.trim();
  try {
    if (apiKey) sessionStorage.setItem("ashvini-key", apiKey);
    else sessionStorage.removeItem("ashvini-key");
  } catch {
    /* private windows refuse storage; the key then lives for this page only */
  }
}

export function hasApiKey(): boolean {
  return apiKey.length > 0;
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 25000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (apiKey) headers["X-API-Key"] = apiKey;
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, headers, signal: ctrl.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error((body as { detail?: string }).detail ?? `${res.status} ${res.statusText}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request<Health>("/api/health", {}, 60000),
  catalogue: () => request<{ count: number; objects: CatalogueObject[] }>("/api/catalogue"),
  conjunctions: (status = "open") => request<Conjunction[]>(`/api/conjunctions?status=${status}`),
  conjunction: (id: string) => request<Conjunction>(`/api/conjunctions/${id}`),
  spaceweather: () => request<SpaceWeather>("/api/spaceweather"),
  manoeuvres: () => request<Manoeuvre[]>("/api/manoeuvres"),
  models: () => request<{ serving: string; cards: ModelCard[] }>("/api/models"),
  review: (id: string, decision: string, note: string) =>
    request<Conjunction>(`/api/conjunctions/${id}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, note }),
    }),
  ask: (question: string) =>
    request<{ answer: string; mode: string; refused: boolean }>("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  cdmUrl: (id: string) => `${API_URL}/api/conjunctions/${id}/cdm`,
  streamUrl: () => `${API_URL}/api/stream`,
};

/** A saved snapshot, used while the free API instance wakes up or if it cannot be reached. */
export async function loadSample(): Promise<{ catalogue: CatalogueObject[]; conjunctions: Conjunction[]; weather: SpaceWeather; savedAt: string }> {
  const res = await fetch(`${import.meta.env.BASE_URL}sample.json`);
  return res.json();
}
