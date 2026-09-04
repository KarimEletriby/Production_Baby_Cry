"""
End-to-end integration tests for the API.
Uses a mock classifier to test the API logic without loading real models.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Fixtures (test_client, audio fixtures) come from conftest.py


class TestHealthEndpoint:
    def test_health_public(self, test_client):
        """Health endpoint should be accessible without API key."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "metrics" in data


class TestAuthMiddleware:
    def test_predict_requires_auth(self, test_client):
        """Predict endpoint should reject requests without API key when auth is enabled."""
        # Patch API_KEY at the module level so the middleware enforces it
        with patch("api.main.API_KEY", "test_secret_key"):
            files = {"audio": ("test.wav", b"\x00" * 2000, "audio/wav")}
            response = test_client.post("/predict", files=files)
            assert response.status_code == 401
            assert response.json()["error_code"] == "UNAUTHORIZED"

    def test_predict_rejects_invalid_key(self, test_client):
        """Predict endpoint should reject wrong API key."""
        with patch("api.main.API_KEY", "test_secret_key"):
            files = {"audio": ("test.wav", b"\x00" * 2000, "audio/wav")}
            response = test_client.post(
                "/predict",
                headers={"X-API-Key": "wrong_key"},
                files=files,
            )
            assert response.status_code == 401
            assert response.json()["error_code"] == "UNAUTHORIZED"

    def test_predict_accepts_correct_key(self, test_client, valid_wav_2s):
        """Predict endpoint should accept correct API key."""
        with patch("api.main.API_KEY", "test_secret_key"):
            with patch("api.main.acoustic_sanity_check") as mock_check:
                mock_check.return_value.passed = True
                mock_check.return_value.measurements = {"mock": 1}

                files = {"audio": ("test.wav", valid_wav_2s, "audio/wav")}
                response = test_client.post(
                    "/predict",
                    headers={"X-API-Key": "test_secret_key"},
                    files=files,
                )
                assert response.status_code == 200
                assert response.json()["success"] is True

    def test_no_auth_when_key_not_configured(self, test_client, valid_wav_2s):
        """When API_KEY is None, no authentication is required."""
        with patch("api.main.API_KEY", None):
            with patch("api.main.acoustic_sanity_check") as mock_check:
                mock_check.return_value.passed = True
                mock_check.return_value.measurements = {"mock": 1}

                files = {"audio": ("test.wav", valid_wav_2s, "audio/wav")}
                response = test_client.post("/predict", files=files)
                assert response.status_code == 200


class TestPredictEndpoint:
    def test_predict_valid_audio(self, test_client, valid_wav_2s):
        """Valid audio upload should return full prediction."""
        files = {"audio": ("test.wav", valid_wav_2s, "audio/wav")}

        with patch("api.main.acoustic_sanity_check") as mock_check:
            mock_check.return_value.passed = True
            mock_check.return_value.measurements = {"mock": 1}

            response = test_client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "request_id" in data
        assert data["stage1"]["is_baby_cry"] is True
        assert data["stage2"]["type"] == "needs"
        assert data["stage2"]["confidence"] == 0.78

    def test_predict_fails_acoustic_check(self, test_client, silence_audio):
        """Silent audio should fail acoustic check and return 400."""
        files = {"audio": ("silence.wav", silence_audio, "audio/wav")}

        response = test_client.post("/predict", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "TOO_QUIET"

    def test_predict_validation_error(self, test_client):
        """File too short should fail early validation."""
        files = {"audio": ("short.wav", b"too short", "audio/wav")}

        response = test_client.post("/predict", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] in ["FILE_TOO_SMALL", "INVALID_AUDIO_FORMAT"]

    def test_predict_response_has_medical_disclaimer(self, test_client, valid_wav_2s):
        """Response must always include the medical disclaimer."""
        files = {"audio": ("test.wav", valid_wav_2s, "audio/wav")}

        with patch("api.main.acoustic_sanity_check") as mock_check:
            mock_check.return_value.passed = True
            mock_check.return_value.measurements = {"mock": 1}

            response = test_client.post("/predict", files=files)

        data = response.json()
        assert "medical_disclaimer_ar" in data
        assert len(data["medical_disclaimer_ar"]) > 0
        assert "تنبيه مهم" in data["medical_disclaimer_ar"]
