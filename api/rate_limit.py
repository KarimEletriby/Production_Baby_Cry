"""
Rate limiting to prevent abuse.
Default: 30 requests per minute per client IP.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse


# ---- Limiter instance (uses client IP as identifier) ----
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/minute", "500/hour"],
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom error handler for rate limit violations."""
    return JSONResponse(
        status_code=429,
        content={
            "error_code": "RATE_LIMIT_EXCEEDED",
            "error_message_ar": "تم تجاوز الحد المسموح من الطلبات. الرجاء المحاولة بعد قليل.",
            "error_message_en": f"Rate limit exceeded: {exc.detail}. Try again later.",
            "retry_after_seconds": 60,
        },
        headers={"Retry-After": "60"},
    )