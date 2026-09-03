import { useRef, useEffect, useState } from "react";
import { useApp } from "../store";
import { Icon } from "../icons";
import { ProgressBar } from "./ProgressBar";
import type { StageStatus } from "../api";

const STAGE_ICON: Record<StageStatus["status"], string> = {
  pending: "radio_button_unchecked",
  in_progress: "autorenew",
  completed: "check_circle",
  failed: "error",
};

const STAGE_ORDER: StageStatus["name"][] = [
  "audio_preparation",
  "model_download",
  "model_load",
  "transcription",
  "merging",
  "text_postprocessing",
  "done",
];

/**
 * Tracks elapsed seconds for a stage. When the stage is in_progress,
 * a 500ms interval drives re-renders; elapsed grows in real-time.
 * When the stage transitions to completed (or failed), the final
 * elapsed value is frozen via useState.
 */
function useElapsed(stage: StageStatus): number {
  const startRef = useRef<number>(Date.now());
  const [, forceUpdate] = useState(0);
  const [frozenElapsed, setFrozenElapsed] = useState(0);
  const prevStatusRef = useRef<StageStatus["status"]>(stage.status);

  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    prevStatusRef.current = stage.status;

    if (stage.status === "in_progress") {
      // Reset start time when entering in_progress
      if (prevStatus !== "in_progress") {
        startRef.current = Date.now();
        setFrozenElapsed(0);
      }
      const timer = window.setInterval(() => forceUpdate((n) => n + 1), 500);
      return () => {
        window.clearInterval(timer);
      };
    } else if (stage.status === "completed" || stage.status === "failed") {
      // Freeze elapsed at the moment of transition
      if (prevStatus === "in_progress") {
        setFrozenElapsed(Math.floor((Date.now() - startRef.current) / 1000));
      }
    } else {
      setFrozenElapsed(0);
    }
  }, [stage.status]);

  if (stage.status === "in_progress") {
    return Math.floor((Date.now() - startRef.current) / 1000);
  }
  return frozenElapsed;
}

/**
 * Computes ETA for the transcription stage based on progress rate.
 * ETA = (elapsed / progress) * (1 - progress)
 */
function useEta(stage: StageStatus, elapsedSec: number): number | null {
  if (stage.name !== "transcription") return null;
  if (stage.status !== "in_progress") return null;
  if (stage.progress <= 0 || stage.progress >= 1) return null;
  if (elapsedSec < 2) return null;

  const eta = (elapsedSec / stage.progress) * (1 - stage.progress);
  return Math.max(1, Math.round(eta));
}

function formatSeconds(seconds: number): string {
  return String(seconds);
}

function StageRow({ stage }: { stage: StageStatus }) {
  const { t } = useApp();
  const elapsedSec = useElapsed(stage);
  const etaSec = useEta(stage, elapsedSec);

  const statusIcon = STAGE_ICON[stage.status];
  const isInProgress = stage.status === "in_progress";
  const isCompleted = stage.status === "completed";
  const isFailed = stage.status === "failed";

  const iconColor = isCompleted
    ? "var(--status-done)"
    : isFailed
      ? "var(--status-err)"
      : isInProgress
        ? "var(--gold-shimmer)"
        : "var(--outline)";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--sp-2)",
        height: 32,
        fontSize: 12,
        color: isInProgress ? "var(--on-surface)" : isFailed ? "var(--status-err)" : "var(--on-surface-variant)",
      }}
    >
      {/* Icon */}
      <span style={{ flexShrink: 0, lineHeight: 0, animation: isInProgress ? "qzt-spin 1s linear infinite" : undefined }}>
        <Icon name={statusIcon} size={18} style={{ color: iconColor }} />
      </span>

      {/* Stage name */}
      <span
        style={{
          flex: "0 0 140px",
          fontWeight: isInProgress ? 600 : 400,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {t(`stage.${stage.name}`)}
      </span>

      {/* Progress bar + detail */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
        {isInProgress && (
          <div style={{ width: "100%" }}>
            <ProgressBar value={stage.progress} />
          </div>
        )}
        {stage.detail && isInProgress && (
          <span
            style={{
              fontSize: 10,
              color: "var(--on-surface-variant)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {stage.detail}
          </span>
        )}
      </div>

      {/* Elapsed / ETA */}
      <span
        style={{
          flexShrink: 0,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          fontWeight: 500,
          color: "var(--on-surface-variant)",
          textAlign: "right",
          minWidth: 90,
        }}
      >
        {isInProgress && elapsedSec > 0 && (
          <>
            {etaSec !== null && stage.name === "transcription" && (
              <span style={{ color: "var(--gold-shimmer)", marginRight: 6 }}>
                {t("stage.eta", { seconds: etaSec })}
              </span>
            )}
            <span>{t("stage.elapsed", { seconds: formatSeconds(elapsedSec) })}</span>
          </>
        )}
        {isCompleted && elapsedSec > 0 && (
          <span style={{ color: "var(--status-done)" }}>
            {t("stage.elapsed", { seconds: formatSeconds(elapsedSec) })}
          </span>
        )}
      </span>
    </div>
  );
}

interface MultiStageProgressBarProps {
  stages: StageStatus[];
  className?: string;
}

export function MultiStageProgressBar({ stages, className = "" }: MultiStageProgressBarProps) {
  const stageMap = new Map<StageStatus["name"], StageStatus>();
  for (const s of stages) {
    stageMap.set(s.name, s);
  }

  return (
    <>
      <style>{`@keyframes qzt-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
      <div className={className} style={{ display: "flex", flexDirection: "column", gap: "var(--sp-1)" }}>
      {STAGE_ORDER.map((name) => {
        const stage = stageMap.get(name);
        return (
          <StageRow
            key={name}
            stage={
              stage ?? {
                name,
                status: "pending",
                progress: 0,
                detail: "",
              }
            }
          />
        );
      })}
      </div>
    </>
  );
}
