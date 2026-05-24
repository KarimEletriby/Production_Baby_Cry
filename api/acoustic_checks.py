"""
Acoustic sanity checks — verify audio plausibly contains a baby cry
BEFORE running the HuBERT inference.

Cheap to compute, catches obvious non-cry inputs like silence, noise,
or pure tones that might fool the deep model.
"""
from dataclasses import dataclass
from typing import Optional
import io

import numpy as np
import librosa
import soundfile as sf


# =========================================================
# Thresholds — tuned for baby cry detection at 16 kHz
# =========================================================

# Minimum RMS energy (audio loudness). Below this = effectively silent.
MIN_RMS_THRESHOLD = 0.010    # ~ -40 dBFS

# Voiced frequency band where baby cries concentrate energy
CRY_FREQ_MIN_HZ   = 250
CRY_FREQ_MAX_HZ   = 3000
MIN_CRY_BAND_RATIO = 0.40   # at least 40% of total energy in this band

# Spectral flux: measure of how much spectrum changes over time
MIN_SPECTRAL_FLUX = 0.015    # below = constant tone / hum

# Zero-crossing rate: cries are mid-range
MIN_ZCR = 0.02
MAX_ZCR = 0.35

# Minimum signal-to-silence ratio
MIN_VOICED_RATIO = 0.20      # at least 20% of frames must be voiced (loud enough)


@dataclass
class AcousticCheckResult:
    """Result of acoustic sanity check on an audio clip."""
    passed: bool
    reason_code: Optional[str] = None     # machine-readable code (None if passed)
    reason_ar: Optional[str] = None       # Arabic explanation
    reason_en: Optional[str] = None       # English explanation
    measurements: Optional[dict] = None   # raw measurements for logging


def acoustic_sanity_check(
    audio_bytes: bytes,
    target_sr: int = 16000,
) -> AcousticCheckResult:
    """
    Run all acoustic sanity checks on audio bytes.

    Returns AcousticCheckResult with passed=True if the audio plausibly
    contains a cry-like signal, otherwise passed=False with a reason.
    """
    # ---- Load audio ----
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    except Exception as e:
        return AcousticCheckResult(
            passed=False,
            reason_code="DECODE_FAILED",
            reason_ar="تعذر قراءة ملف الصوت.",
            reason_en=f"Decode failed: {e}",
        )

    # Mono + resample
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # ---- Compute features ----
    measurements = {}

    # 1. RMS energy
    rms = float(np.sqrt(np.mean(wav ** 2)))
    measurements['rms'] = round(rms, 4)
    measurements['rms_db'] = round(20 * np.log10(rms + 1e-10), 1)

    if rms < MIN_RMS_THRESHOLD:
        return AcousticCheckResult(
            passed=False,
            reason_code="TOO_QUIET",
            reason_ar="الصوت هادئ جداً أو صامت. هل ميكروفون الجهاز يعمل؟",
            reason_en=f"Audio is too quiet (RMS={rms:.4f}, threshold={MIN_RMS_THRESHOLD}).",
            measurements=measurements,
        )

    # 2. Voiced frame ratio (fraction of frames with enough energy)
    frame_length = 2048
    hop_length = 512
    frame_rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    voiced_frames = (frame_rms > MIN_RMS_THRESHOLD).sum()
    voiced_ratio = voiced_frames / max(len(frame_rms), 1)
    measurements['voiced_ratio'] = round(float(voiced_ratio), 3)

    if voiced_ratio < MIN_VOICED_RATIO:
        return AcousticCheckResult(
            passed=False,
            reason_code="MOSTLY_SILENCE",
            reason_ar="معظم الصوت هادئ. تأكد من تسجيل صوت البكاء بوضوح.",
            reason_en=f"Mostly silence (only {voiced_ratio*100:.0f}% voiced).",
            measurements=measurements,
        )

    # 3. Frequency band energy ratio (cry band: 250-3000 Hz)
    stft = np.abs(librosa.stft(wav, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Energy in cry band vs total
    cry_band_mask = (freqs >= CRY_FREQ_MIN_HZ) & (freqs <= CRY_FREQ_MAX_HZ)
    cry_band_energy = stft[cry_band_mask].sum()
    total_energy = stft.sum() + 1e-10
    cry_band_ratio = float(cry_band_energy / total_energy)
    measurements['cry_band_ratio'] = round(cry_band_ratio, 3)

    if cry_band_ratio < MIN_CRY_BAND_RATIO:
        return AcousticCheckResult(
            passed=False,
            reason_code="WRONG_FREQUENCY_PROFILE",
            reason_ar="الصوت لا يحتوي على ترددات صوت بكاء الأطفال المتوقعة.",
            reason_en=(
                f"Energy in cry band ({CRY_FREQ_MIN_HZ}-{CRY_FREQ_MAX_HZ} Hz) "
                f"is only {cry_band_ratio*100:.0f}%, expected >{MIN_CRY_BAND_RATIO*100:.0f}%."
            ),
            measurements=measurements,
        )

    # 4. Spectral flux (variation over time)
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=hop_length)
    spectral_flux = float(np.mean(onset_env))
    measurements['spectral_flux'] = round(spectral_flux, 4)

    if spectral_flux < MIN_SPECTRAL_FLUX:
        return AcousticCheckResult(
            passed=False,
            reason_code="CONSTANT_TONE",
            reason_ar="الصوت ثابت ولا يتغير. قد يكون صوت آلة أو نغمة ثابتة.",
            reason_en=f"Spectral flux too low ({spectral_flux:.4f}) — likely a constant tone or hum.",
            measurements=measurements,
        )

    # 5. Zero-crossing rate (mid-range for vocal sounds)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(wav)[0]))
    measurements['zcr'] = round(zcr, 4)

    if zcr < MIN_ZCR:
        return AcousticCheckResult(
            passed=False,
            reason_code="TOO_SMOOTH",
            reason_ar="نمط الصوت غير متوقع لبكاء طفل.",
            reason_en=f"ZCR too low ({zcr:.4f}) — atypical for voiced sounds.",
            measurements=measurements,
        )

    if zcr > MAX_ZCR:
        return AcousticCheckResult(
            passed=False,
            reason_code="TOO_NOISY",
            reason_ar="الصوت يحتوي على ضوضاء كثيرة. حاول التسجيل في مكان أهدأ.",
            reason_en=f"ZCR too high ({zcr:.4f}) — likely white noise.",
            measurements=measurements,
        )

    # ---- All checks passed ----
    return AcousticCheckResult(
        passed=True,
        reason_code=None,
        reason_ar=None,
        reason_en=None,
        measurements=measurements,
    )