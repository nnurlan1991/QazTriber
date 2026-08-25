import { useState, useEffect } from "react";
import { deleteJob, getResult, getJob, pauseJob, resumeJob, cancelJob, openUrl, saveAndOpenTxt, openFolder, getSessionFiles } from "../api";
import { useApp } from "../store";
import { Icon } from "../icons";
import { Modal } from "../components/Modal";
import { ProgressBar } from "../components/ProgressBar";
import { StatusBadge } from "../components/StatusBadge";
import { Waveform } from "../components/Waveform";
import { formatDate, formatTime, sourceIcon } from "../storage";
import { notifyJobComplete } from "../notifications";

export function SessionView() {
  const { t, lang, currentSessionId, sessions, patchSession, removeSessions, navigate, setPendingRerun, setError } = useApp();
  const session = sessions.find((s) => s.id === currentSessionId) ?? null;
  const [editing, setEditing] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [text, setText] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sessionDir, setSessionDir] = useState<string | null>(null);

  // Load transcript: prefer local, else fetch from API
  if (session && text === null && session.transcript) {
    setText(session.transcript);
  }

  // Live polling for active sessions
  useEffect(() => {
    if (!session) return;

    const activeStatuses = ["queued", "preparing", "loading_model", "transcribing", "paused"];
    if (!activeStatuses.includes(session.status)) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      if (cancelled || !session) return;
      try {
        const job = await getJob(session.id);
        if (cancelled) return;

        patchSession(session.id, {
          status: job.status,
          progress: job.progress,
          stage: job.stages?.find((s) => s.status === "in_progress")?.detail ?? session.stage,
        });

        // Fire notification on terminal status
        if (job.status === "completed" || job.status === "failed") {
          notifyJobComplete(
            job.id, job.status,
            job.status === "completed" ? t("notif.completedTitle") : t("notif.failedTitle"),
            job.status === "completed" ? t("notif.completedBody") : t("notif.failedBody"),
          );
        }

        // If completed, fetch result text
        if (job.status === "completed") {
          try {
            const r = await getResult(session.id);
            if (!cancelled) {
              patchSession(session.id, {
                transcript: r.text,
                transcriptPreview: r.text.slice(0, 120),
                durationMs: r.duration_seconds != null ? Math.round(r.duration_seconds * 1000) : null,
              });
              setText(r.text);
            }
          } catch { /* result not ready yet */ }
          return;
        }

        // Stop polling on terminal status
        if (!activeStatuses.includes(job.status)) return;

        timer = setTimeout(poll, 500);
      } catch {
        // Network error — retry after 2s
        if (!cancelled) timer = setTimeout(poll, 2000);
      }
    }

    timer = setTimeout(poll, 500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.id, patchSession]);

  if (!session) {
    return (
      <div className="content narrow scroll">
        <div className="empty">
          <Icon name="audio_file" size={48} />
          <h2 className="h2">{t("session.empty")}</h2>
          <button className="btn btn-gold mt-2" onClick={() => navigate("home")}>{t("session.new")}</button>
        </div>
      </div>
    );
  }

  const defaultName = `${t("nav.session")} ${session.id.slice(-4)}`;
  const display = session.displayName ?? defaultName;
  const isActive = ["queued", "preparing", "loading_model", "transcribing", "paused"].includes(session.status);

  async function copyText() {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function refreshFromApi() {
    if (!session) return;
    try {
      const r = await getResult(session.id);
      setText(r.text);
      patchSession(session.id, { status: "completed", transcript: r.text, transcriptPreview: r.text.slice(0, 120), durationMs: r.duration_seconds != null ? Math.round(r.duration_seconds * 1000) : null });
    } catch { /* job may be gone after backend restart */ }
  }

  function startRename() {
    setNameDraft(session?.displayName ?? "");
    setEditing(true);
  }

  async function handleGemini() {
    if (!text) return;
    const GEMINI_URL = "https://gemini.google.com/gem/1JBgcHx9CZmalO7WdJlQ2JBl9yUgAnKde?usp=sharing";
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard may be blocked — continue anyway, user can copy manually
    }
    // Open in default browser via Tauri (window.open is blocked in WKWebView)
    await openUrl(GEMINI_URL);
  }

  async function handleDownloadTxt() {
    if (!text || !session) return;
    // Build a readable filename from session display name or ID
    const baseName = (session.displayName || session.originalFilename || session.id.slice(-8))
      .replace(/\.[^.]+$/, ""); // strip extension if present
    await saveAndOpenTxt(text, baseName);
  }

  async function handleListenAudio() {
    if (!session) return;
    if (showAudioPlayer) {
      // Toggle off
      setShowAudioPlayer(false);
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }
      return;
    }
    try {
      const response = await fetch(`/api/transcriptions/${session.id}/source`);
      if (!response.ok) {
        setError(`${t("session.audioMissing")} (HTTP ${response.status})`);
        return;
      }
      const blob = await response.blob();
      if (blob.size === 0) {
        setError(t("session.audioMissing"));
        return;
      }
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setShowAudioPlayer(true);
    } catch (err) {
      setError(`${t("session.audioMissing")}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleRerun() {
    if (!session) return;
    try {
      // Fetch the source audio file
      const response = await fetch(`/api/transcriptions/${session.id}/source`);
      if (!response.ok) {
        setError(`${t("session.audioMissing")} (HTTP ${response.status})`);
        return;
      }
      const blob = await response.blob();
      if (blob.size === 0) {
        setError(t("session.audioMissing"));
        return;
      }
      const filename = session.originalFilename || `${session.id.slice(-8)}.webm`;
      const file = new File([blob], filename, { type: blob.type || "audio/wav" });
      // Pass to HomeView via store
      setPendingRerun({
        file,
        model: session.modelUsed,
        expectedLanguage: session.expectedLanguage,
        sourceType: session.sourceType,
        originalFilename: filename,
      });
      navigate("home");
    } catch (err) {
      setError(`${t("session.audioMissing")}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleOpenFolder() {
    if (!session) return;
    // Fetch directory path from API (works even after backend restart)
    try {
      const info = await getSessionFiles(session.id);
      if (!info.directory) {
        setError(t("session.openFolderError"));
        return;
      }
      setSessionDir(info.directory);
      const opened = await openFolder(info.directory);
      if (!opened) {
        // Fallback: copy path to clipboard so user can open manually
        try {
          await navigator.clipboard.writeText(info.directory);
        } catch {
          // Clipboard may be blocked
        }
        setError(t("session.openFolderError"));
      }
    } catch {
      setError(t("session.openFolderError"));
    }
  }

  // Cleanup audio URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  function saveRename() {
    if (session) patchSession(session.id, { displayName: nameDraft.trim() || null });
    setEditing(false);
  }

  async function doDelete() {
    if (!session) return;
    try { await deleteJob(session.id).catch(() => undefined); } catch { /* ignore */ }
    removeSessions([session.id]);
    setConfirmDelete(false);
    navigate("history");
  }

  return (
    <div className="content narrow scroll">
      {/* Header card */}
      <section className="card pad mb-6">
        <div className="row-flex between gap-4 mb-4">
          <div className="row-flex gap-3 min-w-0">
            <button className="icon-btn" onClick={() => navigate("history")}><Icon name="arrow_back" size={20} /></button>
            <span className="row-icon"><Icon name={sourceIcon(session.sourceType)} size={20} /></span>
            <div className="min-w-0">
              {editing ? (
                <input className="input" autoFocus value={nameDraft} onChange={(e) => setNameDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") saveRename(); if (e.key === "Escape") setEditing(false); }} style={{ maxWidth: 360 }} />
              ) : (
                <h1 className="h1" style={{ cursor: "pointer" }} onClick={startRename} title={t("session.rename")}>{display}</h1>
              )}
              <div className="row-flex gap-3 mt-2"><StatusBadge status={session.status} /><span className="mono faint" style={{ fontSize: 11 }}>{formatDate(session.createdAt, lang)}</span></div>
            </div>
          </div>
          <div className="row-flex gap-1">
            {session.status === "transcribing" && (
              <button className="icon-btn" onClick={() => pauseJob(session.id).then((j) => patchSession(session.id, { status: j.status })).catch(() => {})} title={t("home.pause")}><Icon name="pause" size={20} /></button>
            )}
            {session.status === "paused" && (
              <button className="icon-btn" onClick={() => resumeJob(session.id).then((j) => patchSession(session.id, { status: j.status })).catch(() => {})} title={t("home.resume")}><Icon name="play_arrow" size={20} /></button>
            )}
            {isActive && session.status !== "paused" && (
              <button className="icon-btn danger" onClick={() => cancelJob(session.id).then((j) => patchSession(session.id, { status: j.status })).catch(() => {})} title={t("history.stop")}><Icon name="stop" size={20} /></button>
            )}
            <button className="icon-btn" onClick={startRename} title={t("session.rename")}><Icon name="edit" size={20} /></button>
            <button className="icon-btn danger" onClick={() => setConfirmDelete(true)} title={t("session.delete")}><Icon name="delete" size={20} /></button>
          </div>
        </div>
        {editing && (
          <div className="row-flex gap-2 mt-2">
            <button className="btn btn-gold sm" onClick={saveRename}><Icon name="check" size={16} />{t("common.save")}</button>
            <button className="btn btn-ghost sm" onClick={() => setEditing(false)}>{t("common.cancel")}</button>
          </div>
        )}
      </section>

      {/* Active progress */}
      {isActive && (
        <section className="card pad mb-6 gold-edge">
          <div className="row-flex between mb-3">
            <div className="row-flex gap-3"><Icon name="autorenew" size={20} /><strong>{session.stage}</strong></div>
            <span className="mono faint">{Math.round(session.progress * 100)}%</span>
          </div>
          <ProgressBar value={session.progress} />
        </section>
      )}

      {/* Metadata + waveform */}
      <section className="card pad mb-6">
        <div className="row-flex between mb-4"><h2 className="h3">{t("session.metadata")}</h2></div>
        <div className="meta-grid">
          <div className="meta-row"><span className="meta-key">{t("session.date")}</span><span className="meta-val">{formatDate(session.createdAt, lang)}</span></div>
          <div className="meta-row"><span className="meta-key">{t("session.duration")}</span><span className="meta-val">{session.durationMs ? formatTime(session.durationMs / 1000) : "—"}</span></div>
          <div className="meta-row"><span className="meta-key">{t("session.source")}</span><span className="meta-val">{t(`source.${session.sourceType}`)}</span></div>
          <div className="meta-row"><span className="meta-key">{t("session.model")}</span><span className="meta-val">{session.modelUsed.toUpperCase()}</span></div>
          <div className="meta-row"><span className="meta-key">{t("home.language")}</span><span className="meta-val">{t(`lang.${session.expectedLanguage}`)}</span></div>
          <div className="meta-row"><span className="meta-key">{t("session.status")}</span><StatusBadge status={session.status} /></div>
        </div>
        {session.durationMs && (
          <div className="mt-4">
            <Waveform count={56} seed={session.id.charCodeAt(0) + session.id.length} height={36} progress={0} />
          </div>
        )}
      </section>

      {/* Quick actions for completed sessions: rerun, listen, open folder */}
      {!isActive && (
        <section className="card pad mb-6">
          <div className="row-flex between mb-4">
            <h2 className="h3">{t("session.sessionFiles")}</h2>
          </div>
          <div className="row-flex gap-2" style={{ flexWrap: "wrap" }}>
            <button className="btn btn-soft sm" onClick={handleRerun} title={t("session.rerunHint")}>
              <Icon name="refresh" size={16} />
              {t("session.rerun")}
            </button>
            <button
              className={`btn sm ${showAudioPlayer ? "btn-gold" : "btn-soft"}`}
              onClick={handleListenAudio}
              title={t("session.listenHint")}
            >
              <Icon name={showAudioPlayer ? "pause" : "play_arrow"} size={16} />
              {t("session.listen")}
            </button>
            <button className="btn btn-ghost sm" onClick={handleOpenFolder} title={sessionDir ?? t("session.openFolder")}>
              <Icon name="folder_open" size={16} />
              {t("session.openFolder")}
            </button>
          </div>

          {/* Inline audio player */}
          {showAudioPlayer && audioUrl && (
            <div className="mt-4" style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
              <audio
                controls
                src={audioUrl}
                style={{ width: "100%", height: 38 }}
                autoPlay
              />
              <div className="faint" style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 6 }}>
                <Icon name="graphic_eq" size={14} />
                <span>{t("session.audio")} · {session.originalFilename || session.id.slice(-8)}</span>
              </div>
            </div>
          )}

          {/* Show session directory path (collapsible) */}
          {sessionDir && (
            <div className="mt-3" style={{ fontSize: 11, color: "var(--on-surface-variant)", fontFamily: "var(--font-mono)", wordBreak: "break-all", padding: "var(--sp-2) var(--sp-3)", background: "var(--overlay-bg)", borderRadius: "var(--r-sm)", border: "1px solid var(--border-soft)" }}>
              {sessionDir}
            </div>
          )}
        </section>
      )}

      {/* Transcript */}
      <section className="card pad mb-6">
        <div className="row-flex between mb-4">
          <h2 className="h3">{t("session.transcript")}</h2>
          <div className="row-flex gap-2">
            <button className="btn btn-soft sm" onClick={copyText} disabled={!text}><Icon name={copied ? "check" : "content_copy"} size={16} />{copied ? t("session.copied") : t("session.copy")}</button>
            <button className="btn btn-soft sm" onClick={handleGemini} disabled={!text} title={t("session.geminiHint")}><Icon name="bolt" size={16} />{t("session.gemini")}</button>
            <button className="btn btn-ghost sm" onClick={handleDownloadTxt} disabled={!text}><Icon name="download" size={16} />{t("session.download")}</button>
          </div>
        </div>
        <textarea
          className="transcript"
          value={text ?? ""}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => text !== null && patchSession(session.id, { transcript: text, transcriptPreview: text.slice(0, 120) })}
          placeholder={t("session.placeholder")}
          style={{ minHeight: 220 }}
        />
        {session.errorMessage && <div className="notice error mt-4"><Icon name="error" size={18} /><span>{session.errorMessage}</span></div>}
      </section>

      {confirmDelete && (
        <Modal title={t("session.delete")} danger confirmLabel={t("common.delete")} onClose={() => setConfirmDelete(false)} onConfirm={doDelete}>
          <p>{t("session.deleteConfirm")}</p>
        </Modal>
      )}
    </div>
  );
}
