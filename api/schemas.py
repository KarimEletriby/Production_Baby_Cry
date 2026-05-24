"""
Pydantic schemas for API responses.
Defines the exact JSON structure that mobile apps will receive.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Stage1Result(BaseModel):
    """Stage 1 — cry detection result."""
    is_baby_cry: bool = Field(..., description="هل الصوت بكاء طفل؟")
    emoji: str
    title_ar: str
    description_ar: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label_ar: str = Field(..., description="مستوى الثقة بالعربية")


class Stage2Result(BaseModel):
    """Stage 2 — cry type classification result."""
    type: str = Field(..., description="نوع البكاء (scared/needs/physical_pain/burping)")
    emoji: str
    name_ar: str = Field(..., description="اسم النوع بالعربية")
    definition_ar: str = Field(..., description="تعريف نوع البكاء")
    physical_signs_ar: List[str] = Field(..., description="العلامات الجسدية")
    tips_ar: List[str] = Field(..., description="نصائح للأم")
    warning_ar: str = Field(..., description="متى تستشيرين الطبيب")
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label_ar: str
    confidence_advice_ar: str = Field("", description="نصيحة إضافية بناءً على مستوى الثقة")
    all_probabilities: Dict[str, float] = Field(..., description="احتمالية كل نوع")


class AudioInfo(BaseModel):
    """Audio file metadata."""
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    size_kb: float


class PredictionResponse(BaseModel):
    """Full prediction response — what the mobile app receives."""
    success: bool = True
    request_id: str
    audio_info: AudioInfo
    acoustic_check: Dict[str, Any] = Field(
        default_factory=dict,
        description="Acoustic sanity check measurements (rms, voiced_ratio, cry_band_ratio, etc.)"
    )
    stage1: Stage1Result
    stage2: Optional[Stage2Result] = Field(
        None,
        description="موجود فقط إذا تم اكتشاف بكاء طفل في المرحلة الأولى"
    )
    medical_disclaimer_ar: str
    processing_time_ms: int = Field(..., description="وقت المعالجة بالميلي ثانية")


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error_code: str
    error_message_ar: str
    error_message_en: str
    request_id: Optional[str] = None
    measurements: Optional[Dict[str, Any]] = Field(
        None,
        description="Acoustic measurements (when error is from acoustic check)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    models_loaded: bool
    device: str
    version: str = "1.0.0"
    metrics: Dict[str, Any]