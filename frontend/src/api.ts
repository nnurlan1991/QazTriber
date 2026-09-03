import { invoke } from "@tauri-apps/api/core";
import { openUrl as openerOpenUrl, revealItemInDir as openerReveal } from "@tauri-apps/plugin-opener";

export type ModelsStoragePath = { path: string; exists: boolean };

export type LogEntry = {
  timestamp: string;
  level: string;
  message: string;
};

export type Model = {
  id: "220m" | "600m";
  title: string;
  gigaam_name: string;
  parameters: string;
  description: string;
  cached: boolean;
  storage_path: string | null;
  size_bytes: number;
};

export type StageStatus = {
  name: "audio_preparation" | "model_download" | "model_load" | "transcription" | "merging" | "text_postprocessing" | "done";
  status: "pending" | "in_progress" | "completed" | "failed";
  progress: number;
  detail: string;
};

export type Job = {
  id: string;
  status: "queued" | "preparing" | "loading_model" | "transcribing" | "paused" | "completed" | "failed" | "cancelled";
  progress: number;
  stage: string;
  error: string | null;
  error_code: string | null;
  model: "220m" | "600m";
  expected_language: "kazakh" | "russian" | "mixed";
  filename: string;
  stages: StageStatus[];
};

export type Result = { id: string; text: string; model: "220m" | "600m"; expected_language: "kazakh" | "russian" | "mixed"; duration_seconds: number | null };
export type ModelDownloadStatus = {
  model_id: "220m" | "600m";
  status: "pending" | "downloading" | "completed" | "failed";
  progress: number;
};

export type Preload = { status: "idle" | "downloading" | "completed" | "failed" | "cancelled" | "paused"; progress: number; stage: string; error: string | null; error_code: string | null; models: ModelDownloadStatus[] };

export type SystemInfo = {
  device: string;
  cpu_count: number;
  cpu_brand: string;
  memory_gb: number;
  os: string;
  arch: string;
  speed_multiplier_220m: number;
  speed_multiplier_600m: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Локальный сервер вернул ошибку.");
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const getModels = () => request<Model[]>("/api/models");
export const getModelsStoragePath = () => request<ModelsStoragePath>("/api/models/storage-path");

export type SessionFileInfo = { name: string; size: number; kind: string };
export type SessionFiles = { directory: string; files: SessionFileInfo[] };

/** Fetch directory path and file listing for a session on disk. */
export const getSessionFiles = (jobId: string) => request<SessionFiles>(`/api/sessions/${jobId}/files`);
export const deleteModel = (modelId: Model["id"]) => request<void>(`/api/models/${modelId}`, { method: "DELETE" });
export const getResult = (jobId: string) => request<Result>(`/api/transcriptions/${jobId}/result`);
export const cancelJob = (jobId: string) => request<Job>(`/api/transcriptions/${jobId}/cancel`, { method: "POST" });
export const pauseJob = (jobId: string) => request<Job>(`/api/transcriptions/${jobId}/pause`, { method: "POST" });
export const resumeJob = (jobId: string) => request<Job>(`/api/transcriptions/${jobId}/resume`, { method: "POST" });
export const deleteJob = (jobId: string) => request<void>(`/api/transcriptions/${jobId}`, { method: "DELETE" });
export const getPreload = () => request<Preload>("/api/models/preload");
export const startPreload = (models?: string[]) => request<Preload>("/api/models/preload", {
  method: "POST",
  body: JSON.stringify({ models: models ?? null }),
  headers: { "Content-Type": "application/json" },
});

/**
 * Opens a folder in the native file manager (Finder / Explorer).
 * Tries multiple methods since no single one works reliably across all
 * Tauri 2 + WKWebView setups:
 *  1. revealItemInDir (plugin-opener) — opens Finder with item highlighted
 *  2. openUrl with file:// scheme — opens directory in Finder
 *  3. Custom Tauri command (fallback, requires local IPC)
 * Returns true if any method succeeded.
 */
export async function openFolder(path: string): Promise<boolean> {
  // Method 1: revealItemInDir — opens parent folder with item highlighted
  try {
    await openerReveal(path);
    return true;
  } catch {
    // Continue to next method
  }

  // Method 2: openUrl with file:// scheme
  try {
    const fileUrl = `file://${encodeURIComponent(path).replace(/%2F/g, "/")}`;
    await openerOpenUrl(fileUrl);
    return true;
  } catch {
    // Continue to next method
  }

  // Method 3: custom Tauri command (fallback — only works with local IPC)
  try {
    return await invoke<boolean>("open_folder", { path });
  } catch {
    return false;
  }
}

/**
 * Opens a URL in the default system browser.
 * Uses tauri-plugin-opener's openUrl which handles all platform quirks.
 * Returns true if opened, false if all methods failed.
 */
export async function openUrl(url: string): Promise<boolean> {
  // Method 1: tauri-plugin-opener (preferred — handles WKWebView quirk)
  try {
    await openerOpenUrl(url);
    return true;
  } catch {
    // Plugin failed — try fallbacks
  }

  // Method 2: custom Tauri command
  try {
    return await invoke<boolean>("open_url", { url });
  } catch {
    // IPC failed
  }

  // Method 3: <a target="_blank"> click
  try {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return true;
  } catch {
    // Fallback failed too
  }

  // Method 4: window.open (may work in dev browser, blocked in WKWebView)
  try {
    window.open(url, "_blank");
    return true;
  } catch {
    return false;
  }
}

/**
 * Saves text content to a .txt file in the Downloads directory and opens it
 * with the default application (TextEdit on macOS, Notepad on Windows).
 * Returns the saved file path, or null if not in Tauri context.
 */
export async function saveAndOpenTxt(content: string, filename: string): Promise<string | null> {
  // Method 1: Tauri custom command
  try {
    return await invoke<string>("save_and_open_txt", { content, filename });
  } catch {
    // IPC failed — try fallback
  }

  // Method 2: Browser download (opens in browser, not in default editor)
  try {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.endsWith(".txt") ? filename : `${filename}.txt`;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch {
    // Last resort: ignore
  }
  return null;
}
export const getSystemInfo = () => request<SystemInfo>("/api/system");
export const isFirstLaunch = () => request<{ first_launch: boolean }>("/api/first-launch");
export const markInitialized = () => request<{ initialized: boolean }>("/api/first-launch/initialize", { method: "POST" });
export const cancelPreload = () => request<Preload>("/api/models/preload/cancel", { method: "POST" });
export const getJob = (jobId: string) => request<Job>(`/api/transcriptions/${jobId}`);
const LOG_LINE_RE = /^(\S+)\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s+(.*)$/;

function parseLogLine(line: string): LogEntry {
  const m = LOG_LINE_RE.exec(line);
  if (m) return { timestamp: m[1], level: m[2], message: m[3] };
  return { timestamp: "", level: "INFO", message: line };
}

export const getLogs = async (tail = 100, level = "INFO"): Promise<LogEntry[]> => {
  const data = await request<{ lines: string[]; total_in_file: number; returned: number }>(
    `/api/logs?tail=${tail}&level=${level}`
  );
  return data.lines.map(parseLogLine);
};

export async function createJob(file: File, model: Model["id"], expectedLanguage: Job["expected_language"], start: number, end: number): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("expected_language", expectedLanguage);
  form.append("start_seconds", String(start));
  if (end > start) form.append("end_seconds", String(end));
  return request<Job>("/api/transcriptions", { method: "POST", body: form });
}

const TERMINAL_STATUSES = new Set<Job["status"]>(["completed", "failed", "cancelled"]);

/**
 * Polling вместо SSE — надёжнее, нет ложных onerror при закрытии стрима.
 * Опрашивает /api/transcriptions/{id} каждые 500мс до терминального статуса.
 */
export function watchJob(jobId: string, onJob: (job: Job) => void, onError: () => void): () => void {
  let stopped = false;
  let timer: number | null = null;

  async function poll() {
    if (stopped) return;
    try {
      const job = await getJob(jobId);
      if (stopped) return;
      onJob(job);
      if (TERMINAL_STATUSES.has(job.status)) {
        stopped = true;
        return;
      }
    } catch {
      if (!stopped) {
        onError();
        stopped = true;
      }
      return;
    }
    if (!stopped) {
      timer = window.setTimeout(poll, 500);
    }
  }

  poll();

  return () => {
    stopped = true;
    if (timer) window.clearTimeout(timer);
  };
}
