import { useState, useEffect, useCallback } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { AppProvider, useApp } from "./store";
import { AuthProvider, useAuth } from "./lib/auth";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { DownloadStatusBar } from "./components/DownloadStatusBar";
import { Icon } from "./icons";
import { HomeView } from "./views/HomeView";
import { HistoryView } from "./views/HistoryView";
import { SessionView } from "./views/SessionView";
import { ModelsView } from "./views/ModelsView";
import { SettingsView } from "./views/SettingsView";
import { AuthView } from "./views/AuthView";
import { PendingApprovalView } from "./views/PendingApprovalView";
import { OnboardingModal } from "./components/OnboardingModal";
import { DownloadProgressModal } from "./components/DownloadProgressModal";
import { isFirstLaunch, markInitialized, startPreload } from "./api";

function GlobalError() {
  const { error, setError, t } = useApp();
  if (!error) return null;
  return (
    <div style={{ position: "fixed", bottom: 20, right: 20, zIndex: 200, maxWidth: 380 }}>
      <div className="notice error" style={{ boxShadow: "0 18px 40px rgba(0,0,0,0.5)" }}>
        <Icon name="error" size={20} />
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 12 }}>{t("common.error")}</div>
          <div style={{ fontSize: 12, opacity: 0.9 }}>{error}</div>
        </div>
        <button className="icon-btn" style={{ width: 28, height: 28 }} onClick={() => setError(null)}><Icon name="close" size={16} /></button>
      </div>
    </div>
  );
}

function ViewRouter() {
  const { view } = useApp();
  switch (view) {
    case "home": return <HomeView />;
    case "history": return <HistoryView />;
    case "session": return <SessionView />;
    case "models": return <ModelsView />;
    case "settings": return <SettingsView />;
    default: return <HomeView />;
  }
}

type SidecarStatus = "connected" | "restarting" | "unreachable" | "failed";

interface DictationEventPayload {
  kind: string;
  error_code?: string | null;
  detail?: string | null;
}

interface DictationStatus {
  enabled: boolean;
  hotkey: string;
  trigger: "hold" | "toggle";
  insertMode: "type" | "paste" | "clipboard";
  model: "220m" | "600m";
  language: string;
  recording: boolean;
  processing: boolean;
}

/** Мост frontend ↔ Rust-движок диктовки: применение настроек, тосты, синк с треем. */
function DictationBridge() {
  const { prefs, setPrefs, setError, t } = useApp();
  const { dictationEnabled, dictationHotkey, dictationTrigger, dictationInsertMode, defaultModel } = prefs;

  // Применяем настройки к Rust-движку при каждом изменении (и на старте приложения).
  useEffect(() => {
    invoke("dictation_configure", {
      enabled: dictationEnabled,
      hotkey: dictationHotkey,
      trigger: dictationTrigger,
      insertMode: dictationInsertMode,
      model: defaultModel,
      language: "mixed",
    }).catch((e) => {
      const message = String(e);
      if (message.startsWith("[hotkey_register_failed]")) {
        setError(t("dictation.hotkeyConflict"));
      } else {
        setError(`${t("dictation.configureError")}: ${message}`);
      }
      // Синк с реальным состоянием движка, а не слепой откат:
      // при неудачной смене хоткея диктовка остаётся включённой.
      invoke<DictationStatus>("dictation_status")
        .then((status) => setPrefs({ dictationEnabled: status.enabled }))
        .catch(() => {});
    });
    // t намеренно не в зависимостях — чтобы не пережигать конфиг при смене языка.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictationEnabled, dictationHotkey, dictationTrigger, dictationInsertMode, defaultModel]);

  // События движка: ошибки — тост; переключение из трея — синк prefs.
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    listen<DictationEventPayload>("dictation-event", (event) => {
      const { kind, error_code: code, detail } = event.payload;
      if (kind === "error") {
        if (code === "hotkey_register_failed") {
          setError(t("dictation.hotkeyConflict"));
        } else {
          setError(detail || t("common.error"));
        }
      } else if (kind === "enabled-changed") {
        // Переключили из трея — синхронизируем настройки (это не наша apply, т.к. enabled расходится).
        invoke<DictationStatus>("dictation_status").then((status) => {
          if (status.enabled !== prefs.dictationEnabled) {
            setPrefs({ dictationEnabled: status.enabled });
          }
        }).catch(() => {});
      }
    }).then((fn) => { unlisten = fn; }).catch(() => {
      // Не в Tauri-контексте (dev-браузер)
    });
    return () => { if (unlisten) unlisten(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefs.dictationEnabled]);

  return null;
}

function Shell() {
  const [sidecarStatus, setSidecarStatus] = useState<SidecarStatus | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const { setError, preload, t } = useApp();
  const { state: authState } = useAuth();

  useEffect(() => {
    isFirstLaunch()
      .then(({ first_launch }) => {
        if (first_launch) setShowOnboarding(true);
      })
      .catch(() => {
        // Silently ignore — onboarding is not critical
      });
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | null = null;

    listen("sidecar-status", (event: { payload: string }) => {
      const status = event.payload as SidecarStatus;
      setSidecarStatus(status);
    }).then((fn) => {
      unlisten = fn;
    }).catch(() => {
      // Not in Tauri context (dev browser) — ignore
    });

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const handleDownloadModels = useCallback(async (models: string[]) => {
    try {
      await markInitialized();
      setShowOnboarding(false);
      await startPreload(models);
      // Don't open the download modal — let the DownloadStatusBar on the
      // main screen show progress. User can click it for details.
    } catch (reason) {
      setError((reason as Error).message);
    }
  }, [setError]);

  const handleSkip = useCallback(async () => {
    try {
      await markInitialized();
    } catch {
      // Silently ignore
    }
    setShowOnboarding(false);
  }, []);

  return (
    <div className="app">
      {/* Auth gate: block UI until approved */}
      {authState === "loading" && (
        <div className="auth-screen">
          <div className="auth-card" style={{ textAlign: "center" }}>
            <div className="spinner" style={{ width: 32, height: 32, margin: "0 auto 16px" }} />
            <p className="muted">Загрузка…</p>
          </div>
        </div>
      )}
      {authState === "unauthenticated" && <AuthView />}
      {authState === "pending" && <PendingApprovalView />}
      {authState === "approved" && (
        <>
      {/* Sidecar status banner */}
      {sidecarStatus && sidecarStatus !== "connected" && (
        <div
          className={`sidecar-banner sidecar-banner-${sidecarStatus}`}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            zIndex: 1000,
            padding: "8px 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            fontSize: 14,
            fontWeight: 500,
            background: sidecarStatus === "restarting"
              ? "rgba(247, 189, 72, 0.15)"
              : "rgba(239, 68, 68, 0.15)",
            color: sidecarStatus === "restarting"
              ? "var(--status-warn, #f59e0b)"
              : "var(--status-error, #ef4444)",
            borderBottom: `1px solid ${
              sidecarStatus === "restarting"
                ? "rgba(247, 189, 72, 0.3)"
                : "rgba(239, 68, 68, 0.3)"
            }`,
            backdropFilter: "blur(8px)",
          }}
        >
          {sidecarStatus === "restarting" && <span className="spinner" style={{ width: 14, height: 14 }} />}
          <span>
            {sidecarStatus === "restarting" && t("sidecar.restarting")}
            {sidecarStatus === "unreachable" && t("sidecar.unreachable")}
            {sidecarStatus === "failed" && t("sidecar.failed")}
          </span>
        </div>
      )}
      <Sidebar />
      <main
        className="main"
        style={{ paddingTop: sidecarStatus && sidecarStatus !== "connected" ? 40 : 0 }}
      >
        <DownloadStatusBar onOpenDetails={() => setShowDownloadModal(true)} />
        <TopBar onOpenDownloadModal={() => setShowDownloadModal(true)} />
        <ViewRouter />
      </main>
      <DictationBridge />
      <GlobalError />
      {showOnboarding && (
        <OnboardingModal
          onDownloadModels={handleDownloadModels}
          onSkip={handleSkip}
        />
      )}
      {showDownloadModal && preload && (preload.status === "downloading" || preload.status === "paused" || preload.status === "completed" || preload.status === "failed" || preload.status === "cancelled") && (
        <DownloadProgressModal onClose={() => setShowDownloadModal(false)} />
      )}
        </>
      )}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AuthProvider>
        <Shell />
      </AuthProvider>
    </AppProvider>
  );
}
