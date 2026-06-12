"""
schemas.py
──────────
Pydantic v2 request / response schemas for the FastAPI layer.
Exactly mirrors the output spec from the project brief.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────── Enums ───────────────────────────────────────

class AlertSeverity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    ZONE_ENTRY         = "Zone Entry"
    BOUNDARY_BREACH    = "Boundary Breach"
    MULTIPLE_INTRUDERS = "Multiple Intruders"
    VEHICLE_INTRUSION  = "Vehicle Intrusion"


class EventType(str, Enum):
    INTRUSION_DETECTED = "Intrusion Detected"
    BOUNDARY_BREACH    = "Boundary Breach"
    ALL_CLEAR          = "All Clear"
    VEHICLE_DETECTED   = "Vehicle Detected"


# ──────────────────────────── Detection Schemas ───────────────────────────

class DetectionOut(BaseModel):
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    center: List[int] = Field(..., description="[cx, cy] centroid")


# ──────────────────────────── Alert Schemas ───────────────────────────────

class AlertOut(BaseModel):
    """
    Primary alert response — matches the output spec from the project brief:
    {
        "zone": "Storage Area",
        "personCount": 3
    }
    """
    alert_id: str
    zone: str = Field(..., example="Storage Area")
    zone_id: str
    camera_id: str = Field(..., example="CAM_03")
    personCount: int = Field(..., ge=0, example=3)
    vehicleCount: int = Field(default=0, ge=0)
    event: EventType = Field(..., example="Intrusion Detected")
    alert_type: str = Field(..., example="Multiple Intruders")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.94)
    severity: AlertSeverity
    timestamp: str = Field(..., example="2026-06-06T07:14:22Z")
    bboxes: List[List[int]]
    track_ids: List[int] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None


# ──────────────────────────── Frame Processing Schemas ────────────────────

class FrameDetectResponse(BaseModel):
    """Response from POST /detect (single frame)."""
    camera_id: str
    frame_id: int
    timestamp: float
    inference_ms: float
    personCount: int
    vehicleCount: int
    detections: List[DetectionOut]
    alerts: List[AlertOut]


class StreamFrameOut(BaseModel):
    """WebSocket frame event."""
    camera_id: str
    frame_id: int
    personCount: int
    vehicleCount: int
    fps: float
    detections: List[DetectionOut]
    alerts: List[AlertOut]


# ──────────────────────────── Zone Management Schemas ─────────────────────

class BoundaryLine(BaseModel):
    name: str
    start: List[int] = Field(..., min_length=2, max_length=2)
    end: List[int]   = Field(..., min_length=2, max_length=2)


class RestrictedHours(BaseModel):
    start: str = Field(..., example="22:00", pattern=r"^\d{2}:\d{2}$")
    end: str   = Field(..., example="06:00", pattern=r"^\d{2}:\d{2}$")


class ZoneCreate(BaseModel):
    zone_id: str
    name: str = Field(..., example="Storage Area")
    camera_id: str = Field(..., example="CAM_01")
    polygon: List[List[int]] = Field(
        ..., description="List of [x, y] polygon vertices",
        example=[[100, 200], [500, 200], [500, 600], [100, 600]]
    )
    restricted_hours: Optional[RestrictedHours] = None
    person_threshold: int = Field(default=1, ge=1)
    vehicle_intrusion: bool = True
    enabled: bool = True
    boundary_lines: List[BoundaryLine] = Field(default_factory=list)


class ZoneOut(ZoneCreate):
    is_restricted_now: bool


# ──────────────────────────── Alert Query Schemas ─────────────────────────

class AlertListResponse(BaseModel):
    total: int
    alerts: List[AlertOut]
    stats: dict


# ──────────────────────────── Auth Schemas ────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    model_loaded: bool
    gpu_available: bool
    active_zones: int
