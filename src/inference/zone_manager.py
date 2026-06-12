"""
zone_manager.py
───────────────
Polygon-based zone management with:
  • Zone entry detection (point-in-polygon via Shapely)
  • Boundary line crossing detection (segment intersection)
  • Time-restricted zone enforcement
  • Per-zone alert cooldown logic
"""

from __future__ import annotations

import json
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from shapely.geometry import LineString, Point, Polygon


# ──────────────────────────── Zone Data Model ─────────────────────────────

class Zone:
    """Represents a single surveillance zone."""

    def __init__(self, cfg: dict):
        self.zone_id: str = cfg["zone_id"]
        self.name: str = cfg["name"]
        self.camera_id: str = cfg["camera_id"]
        self.enabled: bool = cfg.get("enabled", True)
        self.person_threshold: int = cfg.get("person_threshold", 1)
        self.vehicle_intrusion: bool = cfg.get("vehicle_intrusion", True)

        # Build Shapely polygon
        pts = cfg["polygon"]
        self.polygon = Polygon(pts)
        self.polygon_pts = np.array(pts, dtype=np.int32)

        # Restricted hours
        hours = cfg.get("restricted_hours", {})
        self.restricted_start: Optional[dtime] = self._parse_time(hours.get("start"))
        self.restricted_end: Optional[dtime] = self._parse_time(hours.get("end"))

        # Boundary lines
        self.boundary_lines: List[dict] = cfg.get("boundary_lines", [])

        # Alert cooldown tracking
        self._last_alert_time: float = 0.0
        self._active_track_ids: Set[int] = set()

    @staticmethod
    def _parse_time(t_str: Optional[str]) -> Optional[dtime]:
        if t_str is None:
            return None
        h, m = map(int, t_str.split(":"))
        return dtime(h, m)

    def is_restricted_now(self) -> bool:
        """Return True if current local time falls in the restricted window."""
        if self.restricted_start is None or self.restricted_end is None:
            return True   # Always restricted if no hours set
        now = datetime.now().time()
        # Handle overnight windows (e.g. 22:00 → 06:00)
        if self.restricted_start > self.restricted_end:
            return now >= self.restricted_start or now <= self.restricted_end
        return self.restricted_start <= now <= self.restricted_end

    def contains_point(self, x: int, y: int) -> bool:
        """Check if pixel coordinate (x, y) is inside this zone polygon."""
        return self.polygon.contains(Point(x, y))

    def cooldown_expired(self, cooldown_seconds: float = 10.0) -> bool:
        return (time.time() - self._last_alert_time) >= cooldown_seconds

    def mark_alerted(self) -> None:
        self._last_alert_time = time.time()


# ──────────────────────────── Zone Manager ────────────────────────────────

class ZoneManager:
    """
    Loads zone configurations and evaluates each Detection / FrameResult
    against all active zones.

    Detection conditions checked:
      1. Zone Entry          — centroid inside polygon
      2. Boundary Crossing   — movement path crosses a boundary line
      3. Restricted Hours    — alert only if time window is active
      4. Multiple Intruders  — personCount >= zone.person_threshold
      5. Vehicle Intrusion   — vehicle detected inside zone (if enabled)
    """

    def __init__(
        self,
        config_path: str,
        cooldown_seconds: float = 10.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.zones: Dict[str, Zone] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Zone config not found: {path}. Starting with no zones.")
            return
        with open(path, "r") as f:
            configs = json.load(f)
        for cfg in configs:
            z = Zone(cfg)
            self.zones[z.zone_id] = z
        logger.info(f"Loaded {len(self.zones)} zones from {path}")

    def add_zone(self, zone_config: dict) -> Zone:
        """Dynamically add a new zone at runtime."""
        z = Zone(zone_config)
        self.zones[z.zone_id] = z
        logger.info(f"Zone added: {z.name} ({z.zone_id})")
        return z

    def remove_zone(self, zone_id: str) -> None:
        if zone_id in self.zones:
            del self.zones[zone_id]

    def evaluate(
        self,
        detections,         # List[Detection]
        camera_id: str,
        trajectory_history: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Evaluate a list of detections against all zones for the given camera.

        Args:
            detections         : List[Detection] from FrameResult.
            camera_id          : Camera to filter zones by.
            trajectory_history : Dict[track_id → List[(x,y)]] for crossing check.

        Returns:
            List of alert dicts (may be empty if no intrusions detected).
        """
        alerts = []
        camera_zones = [z for z in self.zones.values()
                        if z.camera_id == camera_id and z.enabled]

        for zone in camera_zones:
            # Check time restriction
            if not zone.is_restricted_now():
                continue

            zone_persons = []
            zone_vehicles = []

            for det in detections:
                cx, cy = det.center

                # ── Condition 1: Zone Entry ──────────────────────────────
                if not zone.contains_point(cx, cy):
                    continue

                if det.class_name == "person":
                    zone_persons.append(det)
                elif det.class_name == "vehicle" and zone.vehicle_intrusion:
                    zone_vehicles.append(det)

            # ── Condition 4: Multiple Intruders ─────────────────────────
            person_count = len(zone_persons)
            vehicle_count = len(zone_vehicles)

            if person_count == 0 and vehicle_count == 0:
                continue

            if not zone.cooldown_expired(self.cooldown_seconds):
                continue

            # Determine alert type
            alert_types = []
            if person_count >= zone.person_threshold and zone_persons:
                if person_count >= 2:
                    alert_types.append("Multiple Intruders")
                else:
                    alert_types.append("Zone Entry")

            if vehicle_count > 0:
                alert_types.append("Vehicle Intrusion")

            # ── Condition 2: Boundary Crossing ───────────────────────────
            if trajectory_history:
                for boundary in zone.boundary_lines:
                    b_start = tuple(boundary["start"])
                    b_end = tuple(boundary["end"])
                    crossing_ids = self._check_boundary_crossing(
                        trajectory_history, b_start, b_end
                    )
                    if crossing_ids:
                        alert_types.append("Boundary Breach")
                        break

            if not alert_types:
                continue

            # Build alert payload
            all_dets = zone_persons + zone_vehicles
            alert = {
                "zone": zone.name,
                "zone_id": zone.zone_id,
                "camera_id": camera_id,
                "personCount": person_count,
                "vehicleCount": vehicle_count,
                "event": "Intrusion Detected",
                "alert_type": " | ".join(alert_types),
                "confidence": round(
                    max(d.confidence for d in all_dets), 4
                ) if all_dets else 0.0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bboxes": [list(d.bbox) for d in all_dets],
                "track_ids": [d.track_id for d in all_dets if d.track_id],
                "restricted_hours_active": True,
            }
            alerts.append(alert)
            zone.mark_alerted()
            logger.warning(
                f"🚨 ALERT [{zone.name}] {alert['alert_type']} | "
                f"persons={person_count} vehicles={vehicle_count}"
            )

        return alerts

    @staticmethod
    def _check_boundary_crossing(
        trajectory_history: Dict[int, List[Tuple[int, int]]],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
    ) -> List[int]:
        """
        Check which track IDs have crossed the given boundary line.
        Uses Shapely LineString intersection.
        """
        boundary = LineString([line_start, line_end])
        crossed = []
        for track_id, positions in trajectory_history.items():
            if len(positions) < 2:
                continue
            # Only check the most recent movement segment
            path = LineString([positions[-2], positions[-1]])
            if path.intersects(boundary):
                crossed.append(track_id)
        return crossed

    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Draw zone overlays on the frame for visualization."""
        import cv2
        vis = frame.copy()
        for zone in self.zones.values():
            if zone.camera_id != camera_id or not zone.enabled:
                continue
            color = (0, 165, 255) if zone.is_restricted_now() else (128, 128, 128)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [zone.polygon_pts], color)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.polylines(vis, [zone.polygon_pts], True, color, 2)
            # Zone label
            cx = int(np.mean(zone.polygon_pts[:, 0]))
            cy = int(np.mean(zone.polygon_pts[:, 1]))
            cv2.putText(
                vis, zone.name, (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            # Boundary lines
            for boundary in zone.boundary_lines:
                pt1 = tuple(boundary["start"])
                pt2 = tuple(boundary["end"])
                cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
        return vis
