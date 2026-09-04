"""
FastAPI backend for Arabic Baby Cry Classifier — PRODUCTION VERSION.

Improvements over Phase 0:
  ✅ Audio validation before inference
  ✅ Acoustic sanity checks (rejects silence, noise, constant tones)
  ✅ Rate limiting (30/min, 500/hour per IP)
  ✅ Structured logging (JSON in production)
  ✅ Request ID tracking
  ✅ Health endpoint with live metrics
  ✅ Graceful error handling
  ✅ Model warm-up on startup
  ✅ Inference time tracking

Phase 1 hardening:
  ✅ CORS restricted to configured origins
  ✅ API key authentication
  ✅ Lifespan context manager (replaces deprecated on_event)
  ✅ Inference timeout (30s default)
  ✅ Enhanced low-confidence warnings for burping class
"""
import asyncio
import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from inference.inference import BabyCryClassifier
from api.content import (
    STAGE1_CONTENT, STAGE2_CONTENT,
    MEDICAL_DISCLAIMER_AR,
    confidence_label_ar, confidence_advice_ar,
)
from api.schemas import (
    PredictionResponse, Stage1Result, Stage2Result,
    AudioInfo, ErrorResponse, HealthResponse,
)
from api.validators import validate_upload, get_audio_info
from api.rate_limit import limiter, rate_limit_exceeded_handler
from api.logging_config import configure_logging
from api.metrics import metrics
from api.acoustic_checks import acoustic_sanity_check


# =========================================================
# Configuration
# =========================================================
ENV = os.getenv("ENV", "development")
logger = configure_logging(env=ENV)

# CORS — restricted to configured origins (comma-separated env var)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080",
).split(",")

# API Key authentication — set API_KEY env var to enable
API_KEY = os.getenv("API_KEY", None)

# Inference timeout in seconds
INFERENCE_TIMEOUT_S = int(os.getenv("INFERENCE_TIMEOUT_S", "30"))

# Paths to skip authentication (always public)
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


# =========================================================
# Model loading + warm-up (lifespan)
# =========================================================
PROJECT_DIR = Path(__file__).parent.parent
classifier: BabyCryClassifier = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global classifier
    logger.info("server_starting", env=ENV)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading_models", device=device)
    t0 = time.time()

    classifier = BabyCryClassifier(
        stage1_ckpt=str(PROJECT_DIR / "models" / "stage1_binary.ckpt"),
        stage2_ckpts=[
            str(PROJECT_DIR / "models" / "stage2_4class_epoch6.ckpt"),
            str(PROJECT_DIR / "models" / "stage2_4class_epoch7.ckpt"),
        ],
        config_dir=str(PROJECT_DIR / "config"),
        device=device,
    )

    # Warm-up: run a dummy inference to JIT-compile + warm caches
    logger.info("warming_up_model")
    dummy_audio = np.zeros(80000, dtype=np.float32)  # 5 seconds of silence at 16kHz
    try:
        classifier.predict((16000, dummy_audio))
        logger.info("warmup_complete", load_time_seconds=round(time.time() - t0, 1))
    except Exception as e:
        logger.error("warmup_failed", error=str(e), exc_info=True)

    logger.info("server_ready", device=device)

    yield  # App is running

    # Shutdown
    logger.info("server_shutting_down")
    classifier = None


# =========================================================
# App setup
# =========================================================
app = FastAPI(
    title="Baby Cry Classifier API — صنف بكاء طفلك",
    description="واجهة برمجية لتصنيف بكاء الأطفال باستخدام الذكاء الاصطناعي",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — restricted to configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# =========================================================
# API Key Authentication Middleware
# =========================================================
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Check for valid API key in X-API-Key header.
    Skips authentication for public paths and when API_KEY is not configured.
    """
    # Skip if no API key is configured (development mode)
    if API_KEY is None:
        return await call_next(request)

    # Skip public endpoints
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # Check API key
    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != API_KEY:
        metrics.inc("auth_failures")
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error_code": "UNAUTHORIZED",
                "error_message_ar": "مفتاح API غير صالح أو مفقود. أضف الـ header: X-API-Key",
                "error_message_en": "Invalid or missing API key. Provide X-API-Key header.",
            },
        )

    return await call_next(request)


# =========================================================
# Endpoints
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    """Arabic landing page."""
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>صنف بكاء طفلك — API</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 720px;
               margin: 40px auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }
        .card { background: white; padding: 40px; border-radius: 16px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
        h1 { margin: 0 0 12px; }
        .subtitle { color: #7f8c8d; font-size: 16px; margin-bottom: 24px; }
        .endpoint { background: #f8fbfd; padding: 16px; border-radius: 8px;
                    border-right: 4px solid #3498db; margin: 12px 0; }
        .method { display: inline-block; padding: 2px 10px; border-radius: 4px;
                  font-weight: bold; font-size: 13px; margin-left: 8px; }
        .get { background: #d5f5e3; color: #27ae60; }
        .post { background: #fdebd0; color: #d68910; }
        code { background: #ecf0f1; padding: 2px 6px; border-radius: 4px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>👶 صنف بكاء طفلك</h1>
        <p class="subtitle">واجهة برمجية للتعرف على بكاء الأطفال وتصنيفه</p>

        <div class="endpoint">
          <span class="method get">GET</span> <code>/health</code>
          <p>التحقق من حالة الخدمة والإحصائيات</p>
        </div>

        <div class="endpoint">
          <span class="method post">POST</span> <code>/predict</code>
          <p>تحليل ملف صوتي وتصنيف البكاء (30 طلب / دقيقة)</p>
        </div>

        <div class="endpoint">
          <span class="method get">GET</span> <code>/docs</code>
          <p>وثائق API التفاعلية</p>
        </div>

        <p style="margin-top:30px; color:#888; font-size:13px;">
          🔒 يتم معالجة الصوت في الوقت الفعلي ولا يتم تخزينه.
        </p>
      </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with live metrics."""
    return HealthResponse(
        status="ok",
        models_loaded=(classifier is not None),
        device=(classifier.device if classifier else "none"),
        metrics=metrics.snapshot(),
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(
    request: Request,
    audio: UploadFile = File(..., description="ملف صوتي (WAV/MP3/OGG/FLAC/M4A)"),
):
    """
    Analyze an audio file and classify the cry.
    Rate limit: 30 requests/minute per IP.
    """
    metrics.inc("requests_total")
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"

    log = logger.bind(request_id=request_id, client_ip=client_ip)

    if classifier is None:
        log.error("model_not_loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "MODEL_NOT_LOADED",
                "error_message_ar": "النموذج لم يتم تحميله بعد. حاول مرة أخرى بعد قليل.",
                "error_message_en": "Models not loaded yet",
            },
        )

    t_start = time.time()

    # ---- 1. Validate audio (catches 90% of common errors) ----
    try:
        audio_bytes = await validate_upload(audio)
        info = get_audio_info(audio_bytes)
        log.info("audio_validated", **info)

    except HTTPException as exc:
        metrics.inc("validation_errors")
        log.warning("validation_failed", detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={**exc.detail, "request_id": request_id, "success": False},
        )

    # ---- 1b. Acoustic sanity check (rejects obvious non-cries before inference) ----
    check_result = acoustic_sanity_check(audio_bytes)

    if not check_result.passed:
        metrics.inc("acoustic_rejections")
        log.warning(
            "acoustic_check_failed",
            reason_code=check_result.reason_code,
            measurements=check_result.measurements,
        )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "request_id": request_id,
                "error_code": check_result.reason_code,
                "error_message_ar": check_result.reason_ar,
                "error_message_en": check_result.reason_en,
                "measurements": check_result.measurements,
            },
        )

    log.info("acoustic_check_passed", **check_result.measurements)

    # ---- 2. Decode audio ----
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        metrics.inc("validation_errors")
        log.error("audio_decode_failed", error=str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "request_id": request_id,
                "error_code": "DECODE_ERROR",
                "error_message_ar": "تعذر قراءة ملف الصوت.",
                "error_message_en": f"Audio decode error: {str(e)}",
            },
        )

    # ---- 3. Run inference with timeout ----
    t_infer = time.time()
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, classifier.predict, (sr, wav)),
            timeout=INFERENCE_TIMEOUT_S,
        )
        inference_ms = (time.time() - t_infer) * 1000
        metrics.add_inference_time(inference_ms)
    except asyncio.TimeoutError:
        metrics.inc("inference_errors")
        log.error("inference_timeout", timeout_s=INFERENCE_TIMEOUT_S)
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "request_id": request_id,
                "error_code": "INFERENCE_TIMEOUT",
                "error_message_ar": f"تجاوز وقت التحليل الحد المسموح ({INFERENCE_TIMEOUT_S} ثانية). حاول مرة أخرى.",
                "error_message_en": f"Inference timed out after {INFERENCE_TIMEOUT_S}s.",
            },
        )
    except Exception as e:
        metrics.inc("inference_errors")
        log.error("inference_failed", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "request_id": request_id,
                "error_code": "INFERENCE_ERROR",
                "error_message_ar": "حدث خطأ أثناء تحليل الصوت.",
                "error_message_en": f"Inference error: {type(e).__name__}",
            },
        )

    # ---- 4. Build response ----
    s1_label = result["stage1_prediction"]
    s1_conf = result["stage1_confidence"]
    s1_content = STAGE1_CONTENT[s1_label]

    stage1 = Stage1Result(
        is_baby_cry=(s1_label == "baby_cry"),
        emoji=s1_content["emoji"],
        title_ar=s1_content["title_ar"],
        description_ar=s1_content["description_ar"],
        confidence=s1_conf,
        confidence_label_ar=confidence_label_ar(s1_conf),
    )

    stage2 = None
    if result["stage2_prediction"]:
        s2_label = result["stage2_prediction"]
        s2_conf = result["stage2_confidence"]
        s2_content = STAGE2_CONTENT[s2_label]

        # Enhanced confidence advice — extra warning for burping class
        advice = confidence_advice_ar(s2_conf, cry_type=s2_label)

        stage2 = Stage2Result(
            type=s2_label,
            emoji=s2_content["emoji"],
            name_ar=s2_content["name_ar"],
            definition_ar=s2_content["definition_ar"],
            physical_signs_ar=s2_content["physical_signs_ar"],
            tips_ar=s2_content["tips_ar"],
            warning_ar=s2_content["warning_ar"],
            confidence=s2_conf,
            confidence_label_ar=confidence_label_ar(s2_conf),
            confidence_advice_ar=advice,
            all_probabilities=result["stage2_all_probs"],
        )
        metrics.record_prediction(s2_label)
    else:
        metrics.record_prediction("not_baby_cry")

    metrics.inc("predictions_success")
    processing_ms = int((time.time() - t_start) * 1000)

    log.info(
        "prediction_success",
        stage1_label=s1_label,
        stage1_conf=round(s1_conf, 3),
        stage2_label=result.get("stage2_prediction"),
        stage2_conf=round(result.get("stage2_confidence") or 0, 3),
        processing_ms=processing_ms,
        inference_ms=round(inference_ms, 1),
    )

    return PredictionResponse(
        success=True,
        request_id=request_id,
        audio_info=AudioInfo(
            duration_seconds=info["duration_s"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            format=info["format"],
            size_kb=info["size_kb"],
        ),
        acoustic_check=check_result.measurements,
        stage1=stage1,
        stage2=stage2,
        medical_disclaimer_ar=MEDICAL_DISCLAIMER_AR,
        processing_time_ms=processing_ms,
    )