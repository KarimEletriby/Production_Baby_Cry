"""
Audio validation — runs BEFORE inference to catch bad inputs early.
Prevents crashes from malformed files, oversized uploads, etc.
"""
import io
from typing import Tuple

import soundfile as sf
from fastapi import HTTPException, UploadFile, status


# ---- Limits ----
MAX_FILE_SIZE_MB    = 25       # 25 MB max upload
MIN_FILE_SIZE_BYTES = 1000     # 1 KB min (filter empty files)
MIN_DURATION_SEC    = 0.5      # need at least half a second
MAX_DURATION_SEC    = 30       # block extremely long uploads
ACCEPTED_FORMATS    = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}


async def validate_upload(audio: UploadFile) -> bytes:
    """
    Read and validate an uploaded audio file.
    Returns the raw bytes if valid, raises HTTPException otherwise.

    Performs:
      1. Filename extension check
      2. Size limits (min and max)
      3. Format parsing (using soundfile.info)
      4. Duration limits
    """
    # ---- 1. Filename / extension check ----
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "MISSING_FILENAME",
                "error_message_ar": "يرجى رفع ملف صوتي صحيح.",
                "error_message_en": "Missing filename in upload.",
            },
        )

    ext = audio.filename.lower().rsplit(".", 1)[-1] if "." in audio.filename else ""
    if ext not in ACCEPTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "UNSUPPORTED_FORMAT",
                "error_message_ar": f"صيغة الملف غير مدعومة. الصيغ المتاحة: {', '.join(ACCEPTED_FORMATS)}",
                "error_message_en": f"Unsupported format '{ext}'. Accepted: {ACCEPTED_FORMATS}",
            },
        )

    # ---- 2. Read bytes (respecting size limit) ----
    audio_bytes = await audio.read()

    if len(audio_bytes) < MIN_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "FILE_TOO_SMALL",
                "error_message_ar": "الملف صغير جداً أو فارغ.",
                "error_message_en": f"File too small ({len(audio_bytes)} bytes, min {MIN_FILE_SIZE_BYTES}).",
            },
        )

    if len(audio_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error_code": "FILE_TOO_LARGE",
                "error_message_ar": f"الملف كبير جداً. الحد الأقصى {MAX_FILE_SIZE_MB} ميجابايت.",
                "error_message_en": f"File too large ({len(audio_bytes)/1e6:.1f} MB, max {MAX_FILE_SIZE_MB} MB).",
            },
        )

    # ---- 3. Parse audio metadata BEFORE full decode ----
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_AUDIO_FORMAT",
                "error_message_ar": "ملف الصوت غير صالح أو تالف.",
                "error_message_en": f"Cannot parse audio: {type(e).__name__}: {str(e)}",
            },
        )

    # ---- 4. Duration limits ----
    if info.duration < MIN_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "AUDIO_TOO_SHORT",
                "error_message_ar": f"الصوت قصير جداً. الحد الأدنى {MIN_DURATION_SEC} ثانية.",
                "error_message_en": f"Audio too short ({info.duration:.2f}s, min {MIN_DURATION_SEC}s).",
            },
        )

    if info.duration > MAX_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "AUDIO_TOO_LONG",
                "error_message_ar": f"الصوت طويل جداً. الحد الأقصى {MAX_DURATION_SEC} ثانية.",
                "error_message_en": f"Audio too long ({info.duration:.2f}s, max {MAX_DURATION_SEC}s).",
            },
        )

    return audio_bytes


def get_audio_info(audio_bytes: bytes) -> dict:
    """Return audio metadata for logging."""
    info = sf.info(io.BytesIO(audio_bytes))
    return {
        "duration_s": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "size_kb": round(len(audio_bytes) / 1024, 1),
    }