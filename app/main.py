from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.model import load_model
from app.routers import predict, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Flood AI Prediction API",
    description="XGBoost-based flood risk prediction for the Sarawak flood monitoring system",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENABLE_AI_DOCS", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_AI_DOCS", "false").lower() == "true" else None,
    openapi_url="/openapi.json" if os.getenv("ENABLE_AI_DOCS", "false").lower() == "true" else None,
    lifespan=lifespan,
)

environment = os.getenv("ENVIRONMENT", os.getenv("NODE_ENV", "development")).lower()
origins_raw = os.getenv("ALLOWED_ORIGINS")
if origins_raw:
    origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
elif environment == "production":
    origins = []
else:
    origins = ["*"]

if environment == "production" and not os.getenv("AI_SERVICE_API_KEY"):
    raise RuntimeError("AI_SERVICE_API_KEY must be set in production")
if environment == "production" and os.getenv("ENABLE_AI_DOCS", "false").lower() == "true":
    raise RuntimeError("ENABLE_AI_DOCS must not be true in production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Flood AI Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
