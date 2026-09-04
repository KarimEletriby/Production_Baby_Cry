"""
Tests for api.validators — audio upload validation.

Covers:
  - Missing filename
  - Unsupported format
  - File too small / too large
  - Audio too short / too long
  - Valid audio passes all checks
"""
import io
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import AsyncMock, MagicMock

from api.validators import validate_upload, get_audio_info, ACCEPTED_FORMATS
from fastapi import HTTPException


# =========================================================
# Helper to create a mock UploadFile
# =========================================================
def make_upload_file(
    content: bytes,
    filename: str = "test.wav",
) -> MagicMock:
    """Create a mock UploadFile with given content and filename."""
    mock = AsyncMock()
    mock.filename = filename
    mock.read = AsyncMock(return_value=content)
    return mock


def make_wav_bytes(duration_s: float, sr: int = 16000) -> bytes:
    """Generate WAV bytes of a given duration."""
    n_samples = int(sr * duration_s)
    signal = np.random.default_rng(42).standard_normal(n_samples).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# =========================================================
# Tests
# =========================================================
class TestValidateUpload:
    """Tests for the validate_upload function."""

    @pytest.mark.asyncio
    async def test_missing_filename_raises(self):
        """Reject uploads with no filename."""
        upload = make_upload_file(b"fake audio", filename="")
        # filename="" is falsy but not None
        upload.filename = ""
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "MISSING_FILENAME"

    @pytest.mark.asyncio
    async def test_none_filename_raises(self):
        """Reject uploads with None filename."""
        upload = make_upload_file(b"fake audio", filename="test.wav")
        upload.filename = None
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "MISSING_FILENAME"

    @pytest.mark.asyncio
    async def test_unsupported_format_raises(self):
        """Reject unsupported file extensions."""
        upload = make_upload_file(b"not audio" * 200, filename="test.txt")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "UNSUPPORTED_FORMAT"

    @pytest.mark.asyncio
    async def test_file_too_small_raises(self):
        """Reject files smaller than minimum size."""
        upload = make_upload_file(b"tiny", filename="test.wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "FILE_TOO_SMALL"

    @pytest.mark.asyncio
    async def test_file_too_large_raises(self):
        """Reject files larger than maximum size (25 MB)."""
        large_data = b"\x00" * (26 * 1024 * 1024)  # 26 MB
        upload = make_upload_file(large_data, filename="test.wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 413
        assert exc_info.value.detail["error_code"] == "FILE_TOO_LARGE"

    @pytest.mark.asyncio
    async def test_audio_too_short_raises(self):
        """Reject audio shorter than 0.5s."""
        short_wav = make_wav_bytes(0.2)
        upload = make_upload_file(short_wav, filename="short.wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "AUDIO_TOO_SHORT"

    @pytest.mark.asyncio
    async def test_audio_too_long_raises(self):
        """Reject audio longer than 30s."""
        long_wav = make_wav_bytes(31.0)
        upload = make_upload_file(long_wav, filename="long.wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "AUDIO_TOO_LONG"

    @pytest.mark.asyncio
    async def test_invalid_audio_content_raises(self):
        """Reject files that look like audio by extension but are not parseable."""
        fake_wav = b"RIFF" + b"\x00" * 2000  # Looks like WAV header but invalid
        upload = make_upload_file(fake_wav, filename="corrupt.wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error_code"] == "INVALID_AUDIO_FORMAT"

    @pytest.mark.asyncio
    async def test_valid_audio_passes(self):
        """Valid 2-second WAV should pass all checks."""
        wav_bytes = make_wav_bytes(2.0)
        upload = make_upload_file(wav_bytes, filename="valid.wav")
        result = await validate_upload(upload)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestGetAudioInfo:
    """Tests for the get_audio_info helper."""

    def test_returns_correct_metadata(self):
        """get_audio_info should return duration, sample rate, channels, format, size."""
        wav_bytes = make_wav_bytes(2.0, sr=16000)
        info = get_audio_info(wav_bytes)

        assert "duration_s" in info
        assert abs(info["duration_s"] - 2.0) < 0.1
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["format"] == "WAV"
        assert info["size_kb"] > 0


class TestAcceptedFormats:
    """Verify the accepted format set is correct."""

    def test_wav_accepted(self):
        assert "wav" in ACCEPTED_FORMATS

    def test_mp3_accepted(self):
        assert "mp3" in ACCEPTED_FORMATS

    def test_ogg_accepted(self):
        assert "ogg" in ACCEPTED_FORMATS

    def test_exe_not_accepted(self):
        assert "exe" not in ACCEPTED_FORMATS
