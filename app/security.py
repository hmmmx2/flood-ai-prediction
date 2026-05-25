import os

from fastapi import Header, HTTPException, status


def is_production() -> bool:
    return os.getenv("ENVIRONMENT", os.getenv("NODE_ENV", "development")).lower() == "production"


def require_ai_service_key(x_ai_service_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("AI_SERVICE_API_KEY", "")
    if not expected:
        if is_production():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service key is not configured",
            )
        return
    if x_ai_service_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid AI service key",
        )
