"""
alert_engine.py
───────────────
Rule-based alert engine that consumes zone evaluation results
and enriches them with severity scoring, deduplication, and
downstream dispatch (Redis pub/sub, webhook, database).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from enum import Enum
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


# ──────────────────────────── Enums & Constants ───────────────────────────

class AlertSeverity(str, Enum):
    LOW      = "LOW"       # Single person during off-hours
    MEDIUM   = "MEDIUM"    # Multiple persons / boundary breach
    HIGH     = "HIGH"      # Vehicle intrusion / server room
    CRITICAL = "CRITICAL"  # Multiple alert types combined


SEVERITY_MAP = {
    "Zone Entry":          AlertSeverity.LOW,
    "Boundary Breach":     AlertSeverity.MEDIUM,
    "Multiple Intruders":  AlertSeverity.MEDIUM,
    "Vehicle Intrusion":   AlertSeverity.HIGH,
}


# ──────────────────────────── Alert Model ─────────────────────────────────

class Alert:
    """Fully enriched alert ready for storage / dispatch."""

    def __init__(self, raw: dict, frame: Optional[np.ndarray] = None):
        self.alert_id: str = str(uuid.uuid4())
        self.zone: str = raw["zone"]
        self.zone_id: str = raw["zone_id"]
        self.camera_id: str = raw["camera_id"]
        self.person_count: int = raw["personCount"]
        self.vehicle_count: int = raw.get("vehicleCount", 0)
        self.event: str = raw["event"]
        self.alert_type: str = raw["alert_type"]
        self.confidence: float = raw["confidence"]
        self.timestamp: str = raw["timestamp"]
        self.bboxes: List[List[int]] = raw["bboxes"]
        self.track_ids: List[int] = raw.get("track_ids", [])
        self.severity: AlertSeverity = self._compute_severity()
        self.thumbnail_b64: Optional[str] = None
        self.thumbnail_url: Optional[str] = None

        if frame is not None:
            self.thumbnail_b64 = self._encode_thumbnail(frame, raw["bboxes"])

    def _compute_severity(self) -> AlertSeverity:
        types = self.alert_type.split(" | ")
        if len(types) >= 3:
            return AlertSeverity.CRITICAL
        max_sev = AlertSeverity.LOW
        order = [AlertSeverity.LOW, AlertSeverity.MEDIUM,
                 AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        for t in types:
            s = SEVERITY_MAP.get(t.strip(), AlertSeverity.LOW)
            if order.index(s) > order.index(max_sev):
                max_sev = s
        if self.vehicle_count > 0 and self.person_count >= 2:
            return AlertSeverity.CRITICAL
        return max_sev

    @staticmethod
    def _encode_thumbnail(frame: np.ndarray, bboxes: List[List[int]]) -> str:
        """Crop region of interest from frame and encode as base64 JPEG."""
        if not bboxes:
            return ""
        # Merge all bboxes into one crop
        xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
        ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
        x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
        x2, y2 = min(frame.shape[1], max(xs) + 20), min(frame.shape[0], max(ys) + 20)
        crop = frame[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "zone": self.zone,
            "zone_id": self.zone_id,
            "camera_id": self.camera_id,
            "personCount": self.person_count,
            "vehicleCount": self.vehicle_count,
            "event": self.event,
            "alert_type": self.alert_type,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "bboxes": self.bboxes,
            "track_ids": self.track_ids,
            "thumbnail_url": self.thumbnail_url,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ──────────────────────────── Alert Engine ────────────────────────────────

class AlertEngine:
    """
    Processes raw zone alerts, enriches them, and dispatches to:
      - Redis Pub/Sub (real-time streaming to WebSocket clients)
      - PostgreSQL (persistent alert log)
      - HTTP Webhooks (external integrations)
      - In-memory ring buffer (last N alerts for REST API)
    """

    def __init__(
        self,
        redis_client=None,
        redis_channel: str = "intrusion_alerts",
        max_buffer: int = 1000,
        frames_save_dir: str = "logs/frames",
    ):
        self.redis = redis_client
        self.redis_channel = redis_channel
        self.max_buffer = max_buffer
        self.frames_save_dir = frames_save_dir

        self._alert_buffer: List[Alert] = []
        self._webhooks: List[str] = []
        self._callbacks: List[Callable] = []
        self._total_alerts = 0

    def register_webhook(self, url: str) -> None:
        self._webhooks.append(url)
        logger.info(f"Webhook registered: {url}")

    def register_callback(self, fn: Callable) -> None:
        """Register a sync/async callback invoked on every alert."""
        self._callbacks.append(fn)

    async def process(
        self,
        raw_alerts: List[dict],
        frame: Optional[np.ndarray] = None,
    ) -> List[Alert]:
        """
        Convert raw zone alert dicts into enriched Alert objects
        and dispatch them to all configured channels.

        Args:
            raw_alerts : Output from ZoneManager.evaluate()
            frame      : Current video frame (for thumbnail generation)

        Returns:
            List of dispatched Alert objects.
        """
        dispatched = []
        for raw in raw_alerts:
            alert = Alert(raw, frame=frame)
            alert.thumbnail_url = self._save_thumbnail_to_disk(alert, frame)

            self._buffer_alert(alert)
            await self._dispatch(alert)

            self._total_alerts += 1
            dispatched.append(alert)

            logger.warning(
                f"🔔 Alert dispatched | id={alert.alert_id[:8]}... "
                f"| severity={alert.severity.value} | zone={alert.zone}"
            )

        return dispatched

    def _buffer_alert(self, alert: Alert) -> None:
        self._alert_buffer.append(alert)
        if len(self._alert_buffer) > self.max_buffer:
            self._alert_buffer.pop(0)

    def _save_thumbnail_to_disk(
        self, alert: Alert, frame: Optional[np.ndarray]
    ) -> Optional[str]:
        if frame is None or not alert.bboxes:
            return None
        import os
        from pathlib import Path
        save_dir = Path(self.frames_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{alert.alert_id}.jpg"
        filepath = save_dir / filename
        # Draw bboxes on thumbnail
        thumb = frame.copy()
        for bbox in alert.bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(thumb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(filepath), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"/frames/{filename}"

    async def _dispatch(self, alert: Alert) -> None:
        """Concurrently dispatch to Redis and webhooks."""
        tasks = []

        # Redis Pub/Sub
        if self.redis:
            tasks.append(self._publish_redis(alert))

        # Webhooks
        for url in self._webhooks:
            tasks.append(self._call_webhook(url, alert))

        # Registered callbacks
        for cb in self._callbacks:
            if asyncio.iscoroutinefunction(cb):
                tasks.append(cb(alert))
            else:
                cb(alert)   # sync callbacks run inline

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _publish_redis(self, alert: Alert) -> None:
        try:
            await self.redis.publish(self.redis_channel, alert.to_json())
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

    async def _call_webhook(self, url: str, alert: Alert) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json=alert.to_dict())
        except Exception as e:
            logger.error(f"Webhook {url} failed: {e}")

    # ── Query Interface ───────────────────────────────────────────────────

    def get_recent_alerts(
        self, limit: int = 50, camera_id: Optional[str] = None
    ) -> List[dict]:
        alerts = self._alert_buffer[-limit:]
        if camera_id:
            alerts = [a for a in alerts if a.camera_id == camera_id]
        return [a.to_dict() for a in reversed(alerts)]

    def get_stats(self) -> dict:
        return {
            "total_alerts": self._total_alerts,
            "buffered_alerts": len(self._alert_buffer),
            "registered_webhooks": len(self._webhooks),
        }
