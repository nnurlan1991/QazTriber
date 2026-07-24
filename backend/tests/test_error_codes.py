"""Tests for error_code mapping in JobResponse and PreloadResponse."""

from backend.app.schemas import JobResponse, JobStatus, PreloadResponse, PreloadStatus


class TestJobResponseErrorCodes:
    """Verify JobResponse accepts and serializes error_code."""

    def test_job_response_includes_error_code_field(self):
        job = JobResponse(
            id="test-id",
            status=JobStatus.failed,
            progress=0.5,
            stage="Ошибка",
            error="Something went wrong",
            error_code="unknown_error",
            model="220m",
            expected_language="mixed",
            filename="test.wav",
        )
        data = job.model_dump(mode="json")
        assert data["error_code"] == "unknown_error"
        assert data["error"] == "Something went wrong"

    def test_job_response_error_code_none_by_default(self):
        job = JobResponse(
            id="test-id",
            status=JobStatus.queued,
            progress=0.0,
            stage="В очереди",
            model="220m",
            expected_language="mixed",
            filename="test.wav",
        )
        data = job.model_dump(mode="json")
        assert data["error_code"] is None
        assert data["error"] is None

    def test_job_response_transcription_timeout_code(self):
        job = JobResponse(
            id="test-id",
            status=JobStatus.failed,
            progress=0.5,
            stage="Таймаут расшифровки",
            error="Таймаут расшифровки: нет прогресса 10 минут",
            error_code="transcription_timeout",
            model="220m",
            expected_language="mixed",
            filename="test.wav",
        )
        data = job.model_dump(mode="json")
        assert data["error_code"] == "transcription_timeout"

    def test_job_response_cancelled_code(self):
        job = JobResponse(
            id="test-id",
            status=JobStatus.cancelled,
            progress=0.0,
            stage="Отменено пользователем",
            error="Задача отменена пользователем.",
            error_code="cancelled",
            model="220m",
            expected_language="mixed",
            filename="test.wav",
        )
        data = job.model_dump(mode="json")
        assert data["error_code"] == "cancelled"


class TestPreloadResponseErrorCodes:
    """Verify PreloadResponse accepts and serializes error_code."""

    def test_preload_response_includes_error_code_field(self):
        preload = PreloadResponse(
            status=PreloadStatus.failed,
            progress=0.3,
            stage="Ошибка загрузки",
            error="Download failed",
            error_code="model_download_failed",
        )
        data = preload.model_dump(mode="json")
        assert data["error_code"] == "model_download_failed"
        assert data["error"] == "Download failed"

    def test_preload_response_error_code_none_by_default(self):
        preload = PreloadResponse(
            status=PreloadStatus.idle,
            progress=0.0,
            stage="Модели ещё не подготовлены",
        )
        data = preload.model_dump(mode="json")
        assert data["error_code"] is None
        assert data["error"] is None

    def test_preload_response_checksum_mismatch_code(self):
        preload = PreloadResponse(
            status=PreloadStatus.failed,
            progress=0.5,
            stage="Не удалось подготовить модели",
            error="контрольная сумма не совпала; файл не сохранён",
            error_code="checksum_mismatch",
        )
        data = preload.model_dump(mode="json")
        assert data["error_code"] == "checksum_mismatch"

    def test_preload_response_unknown_error_code(self):
        preload = PreloadResponse(
            status=PreloadStatus.failed,
            progress=0.2,
            stage="Не удалось подготовить модели",
            error="Something unexpected",
            error_code="unknown_error",
        )
        data = preload.model_dump(mode="json")
        assert data["error_code"] == "unknown_error"


class TestErrorCodeMappingLogic:
    """Verify the error_code classification logic for various error strings."""

    @staticmethod
    def _classify_job_error(error: str) -> str:
        """Replicates the classification logic from jobs.py _run() handler.

        Order matters — matches jobs.py exactly:
        1. transcription_timeout (checked before lower())
        2. checksum / контрольная сумма
        3. скачать / загрузк / download
        4. ffmpeg / аудио / audio
        5. load_model (NOT gigaam)
        6. unknown_error
        """
        if "transcription_timeout" in error:
            return "transcription_timeout"
        error_lower = error.lower()
        if "checksum" in error_lower or "контрольная сумма" in error_lower:
            return "checksum_mismatch"
        if "скачать" in error_lower or "загрузк" in error_lower or "download" in error_lower:
            return "model_download_failed"
        if "ffmpeg" in error_lower or "аудио" in error_lower or "audio" in error_lower:
            return "audio_preparation_failed"
        if "load_model" in error_lower:
            return "model_load_failed"
        return "unknown_error"

    @staticmethod
    def _classify_preload_error(error: str) -> str:
        """Replicatess the classification logic from gigaam.py _run() handler."""
        error_lower = error.lower()
        if "checksum" in error_lower or "контрольная сумма" in error_lower:
            return "checksum_mismatch"
        if "download" in error_lower or "timeout" in error_lower or "url" in error_lower:
            return "model_download_failed"
        return "unknown_error"

    def test_transcription_timeout_error(self):
        assert self._classify_job_error("transcription_timeout") == "transcription_timeout"

    def test_ffmpeg_error(self):
        assert self._classify_job_error("FFmpeg не смог прочитать аудиофайл.") == "audio_preparation_failed"
        assert self._classify_job_error("Встроенный FFmpeg не найден. Переустановите QazTriber из актуального установщика.") == "audio_preparation_failed"

    def test_audio_preparation_error(self):
        assert self._classify_job_error("Audio preparation failed") == "audio_preparation_failed"

    def test_checksum_error(self):
        assert self._classify_job_error("контрольная сумма не совпала") == "checksum_mismatch"
        assert self._classify_job_error("checksum mismatch") == "checksum_mismatch"

    def test_download_error(self):
        assert self._classify_job_error("Download failed") == "model_download_failed"
        assert self._classify_job_error("Failed to download") == "model_download_failed"

    def test_download_russian_keywords(self):
        assert self._classify_job_error("Не удалось скачать модель 220M: network error") == "model_download_failed"
        assert self._classify_job_error("Таймаут загрузки модели 220M: timeout") == "model_download_failed"

    def test_url_not_classified_as_download(self):
        """'url' alone does NOT trigger model_download_failed — only Russian + English download keywords."""
        assert self._classify_job_error("url fetch error") == "unknown_error"

    def test_model_load_error(self):
        assert self._classify_job_error("load_model failed") == "model_load_failed"
        assert self._classify_job_error("Failed to load_model: PyTorch error") == "model_load_failed"

    def test_gigaam_without_download_or_load_keywords_is_unknown(self):
        """'model error' without download or load_model keywords → unknown_error."""
        assert self._classify_job_error("model error") == "unknown_error"
        assert self._classify_job_error("Model initialization failed") == "unknown_error"

    def test_unknown_error(self):
        assert self._classify_job_error("Something unexpected") == "unknown_error"
        assert self._classify_job_error("") == "unknown_error"

    def test_preload_checksum_mismatch(self):
        assert self._classify_preload_error("контрольная сумма не совпала; файл не сохранён") == "checksum_mismatch"

    def test_preload_download_failed(self):
        assert self._classify_preload_error("Download timeout") == "model_download_failed"
        assert self._classify_preload_error("url not reachable") == "model_download_failed"

    def test_preload_unknown(self):
        assert self._classify_preload_error("Some other error") == "unknown_error"
