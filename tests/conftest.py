"""
Shared pytest fixtures for Baby Cry Classifier tests.

Provides:
  - Audio generation helpers (silence, white noise, sine tones, cry-like signals)
  - Mock classifier for API integration tests
  - FastAPI test client
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Audio generation helpers
# =========================================================
SAMPLE_RATE = 16000


def generate_audio_bytes(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    fmt: str = "WAV",
) -> bytes:
    """Convert a numpy signal to in-memory WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def silence_audio() -> bytes:
    """0.8s of silence — should fail acoustic checks (TOO_QUIET)."""
    signal = np.zeros(int(SAMPLE_RATE * 0.8), dtype=np.float32)
    return generate_audio_bytes(signal)


@pytest.fixture
def white_noise_audio() -> bytes:
    """2s of white noise — should fail acoustic checks (TOO_NOISY or WRONG_FREQUENCY_PROFILE)."""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(SAMPLE_RATE * 2).astype(np.float32) * 0.3
    return generate_audio_bytes(signal)


@pytest.fixture
def constant_tone_audio() -> bytes:
    """2s of a constant 440 Hz sine tone — should fail acoustic checks (CONSTANT_TONE or WRONG_FREQUENCY_PROFILE)."""
    t = np.linspace(0, 2, SAMPLE_RATE * 2, endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    return generate_audio_bytes(signal)


@pytest.fixture
def cry_like_audio() -> bytes:
    """
    2s of a signal that resembles a baby cry for acoustic check purposes:
    - Energy in 250–3000 Hz band
    - Moderate RMS
    - Varying spectral content (not constant)
    - Mid-range ZCR
    """
    rng = np.random.default_rng(123)
    t = np.linspace(0, 2, SAMPLE_RATE * 2, endpoint=False, dtype=np.float32)

    # Mix of frequencies in the cry band (400–2500 Hz) with modulation
    signal = np.zeros_like(t)
    for freq in [400, 600, 900, 1200, 1800, 2200]:
        # Amplitude modulation to create variation (spectral flux)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * (3 + rng.random() * 4) * t)
        signal += mod * 0.08 * np.sin(2 * np.pi * freq * t + rng.random() * 2 * np.pi)

    # Add slight noise for naturalness
    signal += rng.standard_normal(len(t)).astype(np.float32) * 0.01

    # Normalize to moderate RMS
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 0:
        signal = signal * (0.1 / rms)

    return generate_audio_bytes(signal)


@pytest.fixture
def short_audio() -> bytes:
    """0.2s audio — below minimum duration (0.5s)."""
    signal = np.random.default_rng(1).standard_normal(int(SAMPLE_RATE * 0.2)).astype(np.float32) * 0.1
    return generate_audio_bytes(signal)


@pytest.fixture
def valid_wav_2s() -> bytes:
    """2s of mid-frequency signal — valid format, valid duration."""
    t = np.linspace(0, 2, SAMPLE_RATE * 2, endpoint=False, dtype=np.float32)
    signal = 0.1 * np.sin(2 * np.pi * 800 * t)
    return generate_audio_bytes(signal)


# =========================================================
# Mock classifier for API tests
# =========================================================
def _make_mock_classifier():
    """Create a mock BabyCryClassifier that returns realistic results."""
    mock = MagicMock()
    mock.device = "cpu"
    mock.predict.return_value = {
        "audio_duration_s": 2.0,
        "stage1_prediction": "baby_cry",
        "stage1_confidence": 0.95,
        "stage1_all_probs": {"baby_cry": 0.95, "not_baby_cry": 0.05},
        "stage2_prediction": "needs",
        "stage2_confidence": 0.78,
        "stage2_all_probs": {
            "burping": 0.05,
            "needs": 0.78,
            "physical_pain": 0.12,
            "scared": 0.05,
        },
    }
    return mock


@pytest.fixture
def mock_classifier():
    """A mock BabyCryClassifier with realistic return values."""
    return _make_mock_classifier()


@pytest.fixture
def test_client(mock_classifier):
    """
    FastAPI TestClient with mocked classifier.
    Uses httpx-based async client via FastAPI's test utilities.
    """
    from fastapi.testclient import TestClient

    # Patch the classifier in api.main before importing app
    with patch("api.main.classifier", mock_classifier):
        from api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client
