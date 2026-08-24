import { useEffect, useRef, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { useApp } from "../store";
import { LANGS } from "../i18n";
import { Icon } from "../icons";
import { Logo } from "../Logo";
import { getLogs, type LogEntry } from "../api";

export function SettingsView() {
  const { t, lang, setLang, prefs, setPrefs, models } = useApp();

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logLevel, setLogLevel] = useState<string>("INFO");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  // ponytail: версия берётся из tauri.conf.json через Tauri API — === версия GitHub релиза (тег v{version}), без сетевого запроса, работает офлайн
  const [appVersion, setAppVersion] = useState<string>("");
  const versionClicksRef = useRef(0);
  const versionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => setAppVersion(""));
  }, []);

  function handleVersionClick() {
    versionClicksRef.current += 1;
    if (versionTimerRef.current) clearTimeout(versionTimerRef.current);
    versionTimerRef.current = setTimeout(() => { versionClicksRef.current = 0; }, 1500);
    if (versionClicksRef.current >= 5) {
      versionClicksRef.current = 0;
      setShowDebug((prev) => !prev);
    }
  }

  async function fetchLogs() {
    setLoadingLogs(true);
    try {
      const entries = await getLogs(100, logLevel);
      setLogs(entries);
    } catch {
      setLogs([]);
    } finally {
      setLoadingLogs(false);
    }
  }

  useEffect(() => {
    fetchLogs();
  }, [logLevel]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, logLevel]);

  const themes: { id: "light" | "dark" | "system"; labelKey: string; icon: string }[] = [
    { id: "dark", labelKey: "settings.theme.dark", icon: "dark_mode" },
    { id: "light", labelKey: "settings.theme.light", icon: "light_mode" },
    { id: "system", labelKey: "settings.theme.system", icon: "brightness_auto" },
  ];

  return (
    <div className="content narrow scroll">
      {/* Appearance */}
      <section className="card pad mb-6">
        <div className="row-flex gap-3 mb-4"><Icon name="palette" size={22} /><h2 className="h2">{t("settings.appearance")}</h2></div>
        <div className="meta-row"><span className="meta-key">{t("settings.theme")}</span>
          <div className="segmented">
            {themes.map((th) => (
              <button key={th.id} className={prefs.theme === th.id ? "active" : ""} onClick={() => setPrefs({ theme: th.id })}>
                <Icon name={th.icon} size={16} /> {t(th.labelKey)}
              </button>
            ))}
          </div>
        </div>
        <div className="meta-row"><span className="meta-key">{t("settings.language")}</span>
          <div className="segmented">
            {LANGS.map((l) => (
              <button key={l.id} className={lang === l.id ? "active" : ""} onClick={() => setLang(l.id)}>{l.label}</button>
            ))}
          </div>
        </div>
      </section>

      {/* Recognition */}
      <section className="card pad mb-6">
        <div className="row-flex gap-3 mb-4"><Icon name="memory" size={22} /><h2 className="h2">{t("settings.recognition")}</h2></div>
        <div className="meta-row"><span className="meta-key">{t("settings.defaultModel")}</span>
          <div className="segmented">
            {models.map((m) => (
              <button key={m.id} className={prefs.defaultModel === m.id ? "active" : ""} onClick={() => setPrefs({ defaultModel: m.id })}>
                {m.parameters}
              </button>
            ))}
          </div>
        </div>
        <div className="meta-row"><span className="meta-key">{t("settings.engine")}</span><span className="meta-val mono" style={{ fontSize: 12 }}>{t("settings.engineValue")}</span></div>
      </section>

      {/* About */}
      <section className="card pad mb-6">
        <div className="row-flex gap-4 mb-4">
          <Logo size={56} variant="mark" />
          <div className="stack" style={{ gap: 2 }}>
            <h2 className="h2 gold-text">{t("app.name")}</h2>
            <span className="mono faint" style={{ fontSize: 11 }}>{t("app.tagline")} · {t("app.version")} {appVersion || "—"}</span>
          </div>
        </div>
        <p className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>{t("settings.aboutDesc")}</p>
        <div className="divider" />
        <div className="meta-row"><span className="meta-key">{t("settings.version")}</span><span className="meta-val mono" style={{ cursor: "pointer", userSelect: "none" }} onClick={handleVersionClick}>{appVersion || "—"}</span></div>
        <div className="meta-row"><span className="meta-key">{t("settings.locale")}</span><span className="meta-val mono">{lang === "kz" ? "kk-KZ" : "ru-RU"}</span></div>
        <div className="meta-row"><span className="meta-key">{t("settings.engine")}</span><span className="meta-val mono" style={{ fontSize: 12 }}>ИИ модель для распознавания</span></div>
      </section>

      {/* Links */}
      <section className="card pad mb-6">
        <div className="row-flex gap-3 mb-4"><Icon name="globe" size={22} /><h2 className="h2">{t("settings.links")}</h2></div>
        <a className="row" href="https://qaztribber.aidi-lab.kz" target="_blank" rel="noreferrer">
          <span className="row-icon"><Icon name="language" size={18} /></span>
          <div className="row-body"><div className="row-title">{t("settings.website")}</div><div className="row-preview">qaztribber.aidi-lab.kz</div></div>
          <Icon name="open_in_new" size={18} />
        </a>
      </section>

      {/* Debug Panel (hidden — click version 5x to toggle) */}
      {showDebug && (
      <section className="card pad mb-6">
        <div className="row-flex between mb-4">
          <h2 className="h3">{t("settings.debug")}</h2>
          <div className="row-flex gap-2">
            <select className="input sm" value={logLevel} onChange={(e) => setLogLevel(e.target.value)} style={{ width: 120 }}>
              <option value="ALL">ALL</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </select>
            <button className="btn btn-soft sm" onClick={fetchLogs} disabled={loadingLogs}>
              <Icon name="refresh" size={16} />{t("settings.refresh")}
            </button>
            <label className="row-flex gap-1" style={{ fontSize: 13 }}>
              <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
              {t("settings.autoRefresh")}
            </label>
          </div>
        </div>
        <div className="log-panel" style={{ maxHeight: 400, overflow: "auto", background: "var(--bg-elev, rgba(0,0,0,0.2))", borderRadius: 8, padding: 12 }}>
          {logs.length === 0 ? (
            <div className="faint" style={{ textAlign: "center", padding: 20 }}>{t("settings.noLogs")}</div>
          ) : (
            logs.map((entry, i) => (
              <div key={i} className="log-entry" style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 12, marginBottom: 4, display: "flex", gap: 8 }}>
                <span className="faint" style={{ minWidth: 80 }}>{entry.timestamp}</span>
                <span style={{ minWidth: 60, color: entry.level === "ERROR" ? "var(--status-error, #ef4444)" : entry.level === "WARNING" ? "var(--status-warn, #f59e0b)" : "var(--text-muted, inherit)" }}>{entry.level}</span>
                <span style={{ flex: 1, wordBreak: "break-word" }}>{entry.message}</span>
              </div>
            ))
          )}
        </div>
      </section>
      )}
    </div>
  );
}
