// ============================================================
// QazTriber — client-side session history (localStorage)
// Бэкенд не имеет эндпоинта списка задач, поэтому храним локально.
// ============================================================

import type { Job } from "./api";

export type SourceType = "mic" | "file" | "dictaphone";
export type SessionStatus = Job["status"];

export interface SessionRecord {
  id: string;
  createdAt: number;
  sourceType: SourceType;
  originalFilename: string | null;
  durationMs: number | null;
  status: SessionStatus;
  transcriptPreview: string | null;
  transcript: string | null;
  modelUsed: "220m" | "600m";
  expectedLanguage: "kazakh" | "russian" | "mixed";
  errorMessage: string | null;
  progress: number;
  stage: string;
  displayName: string | null;
}

const KEY = "qaztriber.sessions.v1";
const PREFS_KEY = "qaztriber.prefs.v1";

export interface Prefs {
  lang: "ru" | "kz";
  theme: "light" | "dark" | "system";
  defaultModel: "220m" | "600m";
  dictationEnabled: boolean;
  dictationHotkey: string;
  dictationTrigger: "hold" | "toggle";
  dictationInsertMode: "type" | "paste" | "clipboard";
}

const DEFAULT_PREFS: Prefs = {
  lang: "ru",
  theme: "dark",
  defaultModel: "220m",
  dictationEnabled: false,
  dictationHotkey: "alt+shift+KeyD",
  dictationTrigger: "hold",
  dictationInsertMode: "type",
};

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try { return JSON.parse(raw) as T; } catch { return fallback; }
}

export function loadSessions(): SessionRecord[] {
  if (typeof localStorage === "undefined") return [];
  return safeParse<SessionRecord[]>(localStorage.getItem(KEY), []);
}

function saveSessions(list: SessionRecord[]): void {
  if (typeof localStorage === "undefined") return;
  try { localStorage.setItem(KEY, JSON.stringify(list)); } catch { /* quota */ }
}

export function upsertSession(rec: SessionRecord): SessionRecord[] {
  const list = loadSessions();
  const idx = list.findIndex((s) => s.id === rec.id);
  if (idx >= 0) list[idx] = { ...list[idx], ...rec };
  else list.unshift(rec);
  saveSessions(list);
  return list;
}

export function patchSession(id: string, patch: Partial<SessionRecord>): SessionRecord[] {
  const list = loadSessions();
  const idx = list.findIndex((s) => s.id === id);
  if (idx >= 0) {
    list[idx] = { ...list[idx], ...patch };
    saveSessions(list);
  }
  return list;
}

export function removeSessions(ids: string[]): SessionRecord[] {
  const set = new Set(ids);
  const list = loadSessions().filter((s) => !set.has(s.id));
  saveSessions(list);
  return list;
}

export function clearSessions(): void {
  saveSessions([]);
}

export function renameSession(id: string, name: string | null): SessionRecord[] {
  return patchSession(id, { displayName: name && name.trim() ? name.trim() : null });
}

export function defaultDisplayName(index: number, lang: "ru" | "kz"): string {
  return lang === "kz" ? `Сессия ${index + 1}` : `Сессия ${index + 1}`;
}

export function loadPrefs(): Prefs {
  return { ...DEFAULT_PREFS, ...safeParse<Partial<Prefs>>(localStorage.getItem(PREFS_KEY), {}) };
}

export function savePrefs(prefs: Prefs): void {
  if (typeof localStorage === "undefined") return;
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* quota */ }
}

export function formatTime(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(bytes > 1024 * 1024 * 1024 ? 1 : 0)} МБ`;
}

export function formatDate(ts: number, lang: "ru" | "kz"): string {
  const d = new Date(ts);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  const yest = new Date(now); yest.setDate(now.getDate() - 1);
  const isYest = d.toDateString() === yest.toDateString();
  const time = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  const dateStr = d.toLocaleDateString(lang === "kz" ? "kk-KZ" : "ru-RU", { day: "2-digit", month: "2-digit", year: "numeric" });
  if (isToday) return `${lang === "kz" ? "Бүгін" : "Сегодня"}, ${time}`;
  if (isYest) return `${lang === "kz" ? "Кеше" : "Вчера"}, ${time}`;
  return `${dateStr}, ${time}`;
}

export interface ServerSession {
  id: string;
  status: string;  // "completed" | "interrupted" | "failed" | "active"
  created_at: number;
  filename: string | null;
  model: string | null;
  expected_language: string | null;
  has_result: boolean;
  has_source: boolean;
  duration_seconds: number | null;
  error: string | null;
}

export async function syncSessionsFromServer(): Promise<SessionRecord[]> {
  try {
    const response = await fetch("http://127.0.0.1:8765/api/sessions");
    if (!response.ok) return loadSessions();
    const serverSessions: ServerSession[] = await response.json();

    const local = loadSessions();
    const localIds = new Set(local.map(s => s.id));

    // Add sessions from server that are not in localStorage
    for (const server of serverSessions) {
      if (!localIds.has(server.id)) {
        // New session from disk (recovery after restart)
        local.unshift({
          id: server.id,
          createdAt: server.created_at * 1000,  // convert to ms
          sourceType: "file",  // unknown, default to file
          originalFilename: server.filename,
          durationMs: server.duration_seconds ? server.duration_seconds * 1000 : null,
          status: server.status === "active" ? "queued" : server.status as SessionStatus,
          transcriptPreview: null,
          transcript: null,
          modelUsed: (server.model as "220m" | "600m") || "220m",
          expectedLanguage: (server.expected_language as "kazakh" | "russian" | "mixed") || "mixed",
          errorMessage: server.error,
          progress: 0,
          stage: "",
          displayName: null,
        });
      }
    }

    saveSessions(local);
    return local;
  } catch {
    return loadSessions();  // fallback to localStorage
  }
}

export function sourceIcon(source: SourceType): string {
  switch (source) {
    case "mic": return "mic";
    case "file": return "description";
    case "dictaphone": return "graphic_eq";
  }
}
