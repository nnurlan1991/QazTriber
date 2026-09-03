from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import settings
from ..schemas import JobStatus
from .audio import run_ffmpeg, wav_duration_seconds
from .gigaam import GigaAMService
from .punct_restore import PunctRestoreService

logger = logging.getLogger(__name__)


_DEFAULT_STAGES: list[dict] = [
    {"name": "audio_preparation", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "model_download", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "model_load", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "transcription", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "merging", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "text_postprocessing", "status": "pending", "progress": 0.0, "detail": ""},
    {"name": "done", "status": "pending", "progress": 0.0, "detail": ""},
]


@dataclass
class Job:
    id: str
    model: str
    expected_language: str
    filename: str
    directory: Path
    source_path: Path
    start_seconds: float | None
    end_seconds: float | None
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    stage: str = "В очереди"
    error: str | None = None
    error_code: str | None = None
    text: str | None = None
    duration_seconds: float | None = None
    cancelled: bool = False
    paused: bool = False
    last_progress_time: float = 0.0
    revision: int = 0
    condition: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()))
    stages: list[dict] = field(default_factory=lambda: [dict(s) for s in _DEFAULT_STAGES])

    def update(self, status: JobStatus, stage: str, progress: float) -> None:
        with self.condition:
            self.status = status
            self.stage = stage
            self.progress = max(0.0, min(1.0, progress))
            self.revision += 1
            self.condition.notify_all()

    def update_stage(self, name: str, status: str, progress: float, detail: str = "") -> None:
        with self.condition:
            for stage in self.stages:
                if stage["name"] == name:
                    stage["status"] = status
                    stage["progress"] = max(0.0, min(1.0, progress))
                    stage["detail"] = detail
                    break
            self.revision += 1
            self.condition.notify_all()


class JobManager:
    def __init__(self, jobs_dir: Path, gigaam: GigaAMService, punct_restorer: PunctRestoreService | None = None):
        self.jobs_dir = jobs_dir
        self.gigaam = gigaam
        self.punct_restorer = punct_restorer
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._run, name="qaztriber-asr", daemon=True)
        self._worker.start()

    def create(self, model: str, expected_language: str, filename: str, start_seconds: float | None, end_seconds: float | None) -> Job:
        job_id = str(uuid.uuid4())
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory / "source"
        job = Job(job_id, model, expected_language, filename, directory, source_path, start_seconds, end_seconds)
        with self._lock:
            self._jobs[job_id] = job
        # Persist job metadata to disk
        metadata = {
            "id": job_id,
            "model": model,
            "expected_language": expected_language,
            "filename": filename,
            "created_at": time.time(),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        }
        try:
            (directory / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return job

    def enqueue(self, job: Job) -> None:
        """Ставим задачу в очередь только после полной записи upload-файла."""
        with self._wake:
            self._queue.append(job.id)
            self._wake.notify()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        with job.condition:
            job.cancelled = True
            if job.status == JobStatus.queued:
                job.status = JobStatus.cancelled
                job.stage = "Отменено пользователем"
                job.error_code = "cancelled"
                job.revision += 1
                job.condition.notify_all()
        return job

    def pause(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status not in {JobStatus.transcribing, JobStatus.queued}:
            return None
        with job.condition:
            job.paused = True
            job.update(JobStatus.paused, "Пауза", job.progress)
        return job

    def resume(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status != JobStatus.paused:
            return None
        with job.condition:
            job.paused = False
            job.last_progress_time = time.time()  # Reset timeout clock on resume
            job.update(JobStatus.transcribing, "Продолжение…", job.progress)
            job.condition.notify_all()
        return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {JobStatus.preparing, JobStatus.loading_model, JobStatus.transcribing, JobStatus.paused}:
                return False
            self._jobs.pop(job_id)
        shutil.rmtree(job.directory, ignore_errors=True)
        return True

    def _next_job(self) -> Job:
        with self._wake:
            while True:
                while not self._queue:
                    self._wake.wait()
                job_id = self._queue.pop(0)
                job = self._jobs.get(job_id)
                if job and not job.cancelled:
                    return job

    def _run(self) -> None:
        while True:
            job = self._next_job()
            try:
                self._process(job)
            except InterruptedError:
                job.error_code = "cancelled"
                job.update(JobStatus.cancelled, "Отменено пользователем", job.progress)
            except Exception as error:  # User receives a safe message via API.
                job.error = str(error)
                if "transcription_timeout" in str(error):
                    job.error_code = "transcription_timeout"
                    job.update(JobStatus.failed, "Таймаут расшифровки: нет прогресса 10 минут", job.progress)
                else:
                    error_lower = str(error).lower()
                    if "checksum" in error_lower or "контрольная сумма" in error_lower:
                        job.error_code = "checksum_mismatch"
                    elif "скачать" in error_lower or "загрузк" in error_lower or "download" in error_lower:
                        job.error_code = "model_download_failed"
                    elif "ffmpeg" in error_lower or "аудио" in error_lower or "audio" in error_lower:
                        job.error_code = "audio_preparation_failed"
                    elif "load_model" in error_lower:
                        job.error_code = "model_load_failed"
                    else:
                        job.error_code = "unknown_error"
                    job.update(JobStatus.failed, "Ошибка обработки", job.progress)
                # Mark any in_progress stage as failed
                with job.condition:
                    for stage in job.stages:
                        if stage["status"] == "in_progress":
                            stage["status"] = "failed"
                # Persist failure status to disk
                metadata_path = job.directory / "metadata.json"
                if metadata_path.is_file():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        metadata["status"] = "failed"
                        metadata["error"] = str(error)
                        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                    except (OSError, json.JSONDecodeError):
                        pass

    def _process(self, job: Job) -> None:
        if job.cancelled:
            raise InterruptedError

        # Stage: audio_preparation
        job.update(JobStatus.preparing, "Подготовка аудио…", 0.03)
        job.update_stage("audio_preparation", "in_progress", 0.0, "Подготовка аудио…")
        wav_path = job.directory / "normalized.wav"
        run_ffmpeg(job.source_path, wav_path, job.start_seconds, job.end_seconds)
        job.duration_seconds = wav_duration_seconds(wav_path)
        job.update_stage("audio_preparation", "completed", 1.0, "Аудио готово")
        if job.cancelled:
            raise InterruptedError

        def report(stage: str, progress: float) -> None:
            # Check pause — wait until resumed
            while job.paused and not job.cancelled:
                with job.condition:
                    job.condition.wait(timeout=1.0)
            if job.cancelled:
                raise InterruptedError
            job.last_progress_time = time.time()
            status = JobStatus.loading_model if progress < 0.3 else JobStatus.transcribing
            job.update(status, stage, progress)
            # Map gigaam progress to appropriate stage
            if progress < 0.22:
                job.update_stage("model_download", "in_progress", progress / 0.22, "Скачивание модели…")
            elif progress < 0.30:
                job.update_stage("model_download", "completed", 1.0, "Модель скачана")
                job.update_stage("model_load", "in_progress", (progress - 0.22) / 0.08, "Загрузка модели в память…")
            elif progress < 0.97:
                job.update_stage("model_load", "completed", 1.0, "Модель загружена")
                job.update_stage("transcription", "in_progress", (progress - 0.30) / 0.67, stage)
            else:
                job.update_stage("transcription", "completed", 1.0, "Распознавание завершено")
                job.update_stage("merging", "in_progress", (progress - 0.97) / 0.03, "Объединение результата…")

        job.last_progress_time = time.time()
        timed_out = False
        stop_monitor = threading.Event()

        def timeout_monitor() -> None:
            nonlocal timed_out
            while not job.cancelled and not stop_monitor.is_set():
                if job.paused:
                    time.sleep(5)
                    continue
                elapsed = time.time() - job.last_progress_time
                if elapsed > settings.transcription_timeout:
                    timed_out = True
                    job.cancelled = True
                    return
                time.sleep(5)

        monitor = threading.Thread(target=timeout_monitor, name=f"qaztriber-timeout-{job.id}", daemon=True)
        monitor.start()

        text: str | None = None
        try:
            text = self.gigaam.transcribe(
                job.model,
                wav_path,
                job.directory / "chunks",
                report,
                lambda: job.cancelled,
            )
        except InterruptedError:
            # If timeout triggered the cancellation, convert to RuntimeError
            if timed_out:
                raise RuntimeError("transcription_timeout") from None
            raise
        finally:
            stop_monitor.set()  # Stop monitor

        if job.cancelled:
            if timed_out:
                raise RuntimeError("transcription_timeout")
            raise InterruptedError
        job.update_stage("merging", "completed", 1.0, "Результат объединён")

        # L1.3: выгружаем GigaAM перед punct-restore — обе модели в RAM не нужны.
        # L1.4: освобождаем кэш MPS/CUDA после тяжёлой ASR-модели.
        try:
            self.gigaam.unload()
        except Exception:
            pass

        # Stage: text_postprocessing — восстановление пунктуации и регистра.
        # Это улучшение поверх сырого текста, не критичный путь: если модель
        # недоступна или упала — отдаём пользователю сырой текст, а не валим job.
        job.update_stage("text_postprocessing", "in_progress", 0.0, "Расстановка пунктуации…")
        final_text = text
        if self.punct_restorer is not None:
            try:
                final_text = self.punct_restorer.restore(text)
                job.update_stage("text_postprocessing", "completed", 1.0, "Пунктуация восстановлена")
            except Exception as error:
                logger.warning("Punct-restore failed for job %s, falling back to raw text: %s", job.id, error)
                job.update_stage("text_postprocessing", "failed", 1.0, "Пропущено (модель недоступна)")
        else:
            job.update_stage("text_postprocessing", "completed", 1.0, "Пропущено (сервис не настроен)")

        # Выгружаем punct-модель сразу после использования — точно как GigaAM
        if self.punct_restorer is not None:
            try:
                self.punct_restorer.unload()
            except Exception:
                pass

        job.update_stage("done", "completed", 1.0, "Готово")
        job.text = final_text
        job.update(JobStatus.completed, "Готово", 1.0)
        # Persist transcription.txt to disk so it survives backend restart.
        # Сырой текст (до пунктуации) тоже сохраняем — пригодится для отладки
        # и для пересчёта другой версией punct-restore модели без повторного ASR.
        try:
            (job.directory / "transcription_raw.txt").write_text(text, encoding="utf-8")
            (job.directory / "transcription.txt").write_text(final_text, encoding="utf-8")
        except OSError:
            pass
        # Update metadata.json with completion status
        metadata_path = job.directory / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["status"] = "completed"
                metadata["duration_seconds"] = job.duration_seconds
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass
