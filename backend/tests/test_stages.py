from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import soundfile as sf

from backend.app.schemas import JobStatus, JobResponse
from backend.app.services.jobs import Job, JobManager


class FakeGigaAM:
    """Simulates gigaam.transcribe() with a controllable report callback."""

    def __init__(self) -> None:
        self.report_calls: list[tuple[str, float]] = []

    def transcribe(self, model, wav_path, chunks_dir, report, cancelled):
        assert not cancelled()
        report("Скачивание модели…", 0.22)
        report("Загрузка модели 220M в память…", 0.30)
        report("Распознавание фрагмента 1/1…", 0.60)
        report("Объединение результата…", 0.97)
        return "Сәлем әлем"


class BlockingGigaAM:
    """Simulates a stuck transcription — no progress, waits for cancellation."""

    def transcribe(self, model, wav_path, chunks_dir, report, cancelled):
        # Report initial progress once (sets last_progress_time in Job)
        report("Загрузка модели 220M в память…", 0.30)
        # Simulate a single chunk iteration that checks cancelled
        sample_index = 0
        while not cancelled():
            # Sleep until cancelled (test mocks time to accelerate timeout)
            time.sleep(0.05)
            # Report no meaningful progress
            report(f"Распознавание фрагмента {sample_index + 1}/1…", 0.31 + 0.01 * sample_index)
            sample_index += 1
            if sample_index > 100:
                break
        raise InterruptedError("Задача отменена пользователем.")


class TestStagesTracking:
    """Test cases for multi-stage progress tracking."""

    # -----------------------------------------------------------------
    # Test 1: Job starts with 6 stages, all pending
    # -----------------------------------------------------------------
    def test_stages_initialized_as_pending(self) -> None:
        job = self._make_job()
        assert len(job.stages) == 7, f"expected 7 stages, got {len(job.stages)}"

        expected_names = [
            "audio_preparation", "model_download", "model_load",
            "transcription", "merging", "text_postprocessing", "done",
        ]
        for i, stage in enumerate(job.stages):
            assert stage["name"] == expected_names[i], f"stage[{i}] name mismatch"
            assert stage["status"] == "pending", f"stage[{i}] should be pending"
            assert stage["progress"] == 0.0
            assert stage["detail"] == ""

    # -----------------------------------------------------------------
    # Test 2: update_stage() updates correct stage
    # -----------------------------------------------------------------
    def test_update_stage_modifies_correct_stage(self) -> None:
        job = self._make_job()
        job.update_stage("transcription", "in_progress", 0.5, "Распознавание…")

        transcription = job.stages[3]
        assert transcription["name"] == "transcription"
        assert transcription["status"] == "in_progress"
        assert transcription["progress"] == 0.5
        assert transcription["detail"] == "Распознавание…"

    # -----------------------------------------------------------------
    # Test 3: update_stage() doesn't affect other stages
    # -----------------------------------------------------------------
    def test_update_stage_does_not_affect_others(self) -> None:
        job = self._make_job()
        job.update_stage("transcription", "in_progress", 0.8, "detail")

        for i, stage in enumerate(job.stages):
            if stage["name"] == "transcription":
                assert stage["status"] == "in_progress", f"transcription should be in_progress"
            else:
                assert stage["status"] == "pending", f"stage[{i}] should still be pending"

    # -----------------------------------------------------------------
    # Test 4: Stage progress clamped to [0, 1]
    # -----------------------------------------------------------------
    def test_stage_progress_clamped(self) -> None:
        job = self._make_job()

        job.update_stage("audio_preparation", "in_progress", -0.5, "")
        assert job.stages[0]["progress"] == 0.0, "negative progress should clamp to 0"

        job.update_stage("audio_preparation", "in_progress", 2.5, "")
        assert job.stages[0]["progress"] == 1.0, "progress >1 should clamp to 1"

        job.update_stage("audio_preparation", "completed", 0.75, "")
        assert job.stages[0]["progress"] == 0.75, "valid progress should pass through"

    # -----------------------------------------------------------------
    # Test 5: JobResponse serializes stages correctly
    # -----------------------------------------------------------------
    def test_job_response_includes_stages(self) -> None:
        from backend.app.api.transcriptions import job_response

        job = self._make_job()
        job.update_stage("audio_preparation", "completed", 1.0, "Аудио готово")
        job.update_stage("model_download", "completed", 1.0, "")
        job.update_stage("model_load", "in_progress", 0.5, "Загрузка…")

        response = job_response(job)

        assert isinstance(response, JobResponse)
        assert len(response.stages) == 7

        # Verify structure of a completed stage
        audio_stage = response.stages[0]
        assert audio_stage.name == "audio_preparation"
        assert audio_stage.status == "completed"
        assert audio_stage.progress == 1.0
        assert audio_stage.detail == "Аудио готово"

        # Model download should be completed
        assert response.stages[1].status == "completed"

        # Model load should be in_progress
        assert response.stages[2].status == "in_progress"
        assert response.stages[2].progress == 0.5

        # Remaining stages should be pending
        for i in range(3, 7):
            assert response.stages[i].status == "pending", f"stage[{i}] should be pending"

    # -----------------------------------------------------------------
    # Test 6: _process() updates stages in correct order
    # -----------------------------------------------------------------
    def test_process_updates_stages_in_order(self, tmp_path) -> None:
        fake_gigaam = FakeGigaAM()
        manager = JobManager(tmp_path / "jobs", fake_gigaam)

        job = manager.create("220m", "mixed", "sample.wav", None, None)
        sf.write(job.source_path, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
        manager.enqueue(job)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and job.status not in {JobStatus.completed, JobStatus.failed}:
            time.sleep(0.05)

        assert job.status == JobStatus.completed, job.error
        assert job.text == "Сәлем әлем"

        # Verify all 7 stages reached a non-pending state
        expected_names = [
            "audio_preparation", "model_download", "model_load",
            "transcription", "merging", "text_postprocessing", "done",
        ]
        for stage_name in expected_names:
            matching = [s for s in job.stages if s["name"] == stage_name]
            assert len(matching) == 1, f"stage {stage_name} not found"
            assert matching[0]["status"] == "completed", (
                f"stage '{stage_name}' should be completed, got '{matching[0]['status']}'"
            )

        # Verify order preserved
        actual_names = [s["name"] for s in job.stages]
        assert actual_names == expected_names, f"stage order changed: {actual_names}"

        assert manager.delete(job.id)

    # -----------------------------------------------------------------
    # Test 7: Transcription timeout sets job as failed
    # -----------------------------------------------------------------
    def test_transcription_timeout_fails_job(self, tmp_path) -> None:
        """If no progress for the configured timeout period, job fails with timeout error."""
        from backend.app.services import jobs as jobs_module

        blocking_gigaam = BlockingGigaAM()
        manager = JobManager(tmp_path / "jobs", blocking_gigaam)

        job = manager.create("220m", "mixed", "sample.wav", None, None)
        sf.write(job.source_path, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
        manager.enqueue(job)

        # Use a short timeout (1 second) to make the test fast
        # Mock time.time to accelerate: every call returns +10 seconds
        fake_time_start = [0.0]

        def fake_time() -> float:
            fake_time_start[0] += 10.0
            return fake_time_start[0]

        # Disable time.sleep so the monitor thread loop runs instantly
        def fake_sleep(seconds: float) -> None:
            pass

        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.transcription_timeout = 1

        with patch.object(jobs_module, "settings", mock_settings):
            with patch.object(jobs_module.time, "time", fake_time):
                with patch.object(jobs_module.time, "sleep", fake_sleep):
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline and job.status not in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                        time.sleep(0.05)

        assert job.status == JobStatus.failed, (
            f"expected failed status, got {job.status.value} — error: {job.error}"
        )
        assert "transcription_timeout" in str(job.error), (
            f"expected timeout error, got: {job.error}"
        )
        assert "Таймаут" in job.stage, (
            f"expected timeout message in stage, got: {job.stage}"
        )

        assert manager.delete(job.id)

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _make_job() -> Job:
        from pathlib import Path

        return Job(
            id="test-job-id",
            model="220m",
            expected_language="mixed",
            filename="test.wav",
            directory=Path("/tmp/test-job-dir"),
            source_path=Path("/tmp/test-job-dir/source"),
            start_seconds=None,
            end_seconds=None,
        )
