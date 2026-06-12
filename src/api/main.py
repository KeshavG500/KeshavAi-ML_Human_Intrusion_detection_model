"""
main.py
───────
FastAPI application entrypoint.
Mounts all routers, sets up lifespan (startup/shutdown),
configures middleware, and initialises shared resources.
"""

from __future__ import annotations

import time
import torch
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routers import alerts, detection, stream, zones
from src.api.schemas import HealthResponse
from src.inference.alert_engine import AlertEngine
from src.inference.pipeline import InferencePipeline
from src.inference.zone_manager import ZoneManager
from src.models.detector import IntrusionDetector
from src.models.tracker import ByteTracker

# ──────────────────────────── App State ──────────────────────────────────

class AppState:
    pipeline: Optional[InferencePipeline] = None
    redis: Optional[aioredis.Redis] = None
    start_time: float = 0.0


state = AppState()


# ──────────────────────────── Lifespan ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → Shutdown."""
    state.start_time = time.time()
    logger.info("🚀 Intrusion Detection API starting up...")

    # ── Redis connection ─────────────────────────────────────────────────
    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        state.redis = await aioredis.from_url(redis_url, decode_responses=True)
        await state.redis.ping()
        logger.success(f"✅ Redis connected: {redis_url}")
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}). Alerts won't stream via pub/sub.")
        state.redis = None

    # ── Load model & build pipeline ──────────────────────────────────────
    weights = os.getenv("DETECTOR_WEIGHTS", "yolov8m.pt")
    device  = os.getenv("INFERENCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    zones_path = os.getenv("ZONES_CONFIG_PATH", "data/zones/zones_config.json")

    try:
        detector = IntrusionDetector(weights=weights, device=device)
        tracker  = ByteTracker()
        zone_mgr = ZoneManager(config_path=zones_path)
        alert_eng = AlertEngine(
            redis_client=state.redis,
            redis_channel=os.getenv("REDIS_CHANNEL", "intrusion_alerts"),
        )
        state.pipeline = InferencePipeline(
            detector=detector,
            tracker=tracker,
            zone_manager=zone_mgr,
            alert_engine=alert_eng,
        )
        logger.success("✅ Inference pipeline ready.")
    except Exception as e:
        logger.error(f"❌ Pipeline init failed: {e}")
        state.pipeline = None

    # Expose state to routers via app
    app.state.pipeline = state.pipeline
    app.state.redis    = state.redis
    app.state.start_time = state.start_time

    yield  # ── Running ────────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down Intrusion Detection API...")
    if state.redis:
        await state.redis.close()
    logger.info("Shutdown complete.")


# ──────────────────────────── App Init ───────────────────────────────────

app = FastAPI(
    title="Human Intrusion Detection API",
    description=(
        "Real-time AI-powered intrusion detection system. "
        "Detects zone entry, boundary crossings, multiple intruders, "
        "and vehicle intrusions using YOLOv8 + ByteTrack."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ──────────────────────────── Middleware ──────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────── Prometheus ─────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ──────────────────────────── Static Files ───────────────────────────────

frames_dir = Path("logs/frames")
frames_dir.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(frames_dir)), name="frames")

# ──────────────────────────── Routers ────────────────────────────────────

app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(stream.router,    prefix="/api/v1", tags=["Streaming"])
app.include_router(zones.router,     prefix="/api/v1", tags=["Zones"])
app.include_router(alerts.router,    prefix="/api/v1", tags=["Alerts"])

# ──────────────────────────── Core Endpoints ─────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Human Intrusion Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    uptime = time.time() - request.app.state.start_time
    return HealthResponse(
        status="healthy" if request.app.state.pipeline else "degraded",
        version="1.0.0",
        uptime_seconds=round(uptime, 1),
        model_loaded=request.app.state.pipeline is not None,
        gpu_available=torch.cuda.is_available(),
        active_zones=(
            len(request.app.state.pipeline.zone_manager.zones)
            if request.app.state.pipeline else 0
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs."},
    )
