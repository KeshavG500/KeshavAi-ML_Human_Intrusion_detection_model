"""
pipeline.py
───────────
Full end-to-end real-time inference pipeline.
Ties together: FrameExtractor → Detector → Tracker → ZoneManager → AlertEngine

Usage:
    pipeline = InferencePipeline.from_config("src/training/config.yaml")
    await pipeline.run_stream(source="rtsp://camera_ip/stream", camera_id="CAM_01")
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import cv2
import numpy as np
import yaml
from loguru import logger

from src.inference.alert_engine import AlertEngine
from src.inference.zone_manager import ZoneManager
from src.models.detector import IntrusionDetector, FrameResult
from src.models.tracker import ByteTracker
from src.preprocessing.frame_extractor import FrameExtractor


class InferencePipeline:
    """
    Master pipeline that orchestrates the full detection → alert flow.

    Architecture:
        Camera → FrameExtractor (threaded buffer)
               → IntrusionDetector (YOLOv8 GPU inference)
               → ByteTracker (multi-object ID assignment)
               → ZoneManager (polygon + time logic)
               → AlertEngine (enrich, thumbnail, Redis dispatch)
    """

    def __init__(
        self,
        detector: IntrusionDetector,
        tracker: ByteTracker,
        zone_manager: ZoneManager,
        alert_engine: AlertEngine,
        target_fps: int = 15,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        self.detector = detector
        self.tracker = tracker
        self.zone_manager = zone_manager
        self.alert_engine = alert_engine
        self.target_fps = target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Per-track trajectory history for boundary crossing checks
        self._trajectory_history: Dict[int, List] = {}
        self._frame_count = 0
        self._start_time = 0.0

    @classmethod
    def from_config(
        cls,
        config_path: str,
        redis_client=None,
    ) -> "InferencePipeline":
        """Construct pipeline from a YAML config file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        detector = IntrusionDetector(
            weights=cfg["model"].get("pretrained_weights", "yolov8m.pt"),
            device=cfg["inference"]["device"],
            conf_threshold=cfg["inference"]["conf_threshold"],
            iou_threshold=cfg["inference"]["iou_threshold"],
            img_size=cfg["model"]["input_size"],
            half=cfg["inference"]["half_precision"],
        )
        tracker = ByteTracker(
            track_thresh=cfg["tracking"]["track_thresh"],
            match_thresh=cfg["tracking"]["match_thresh"],
            track_buffer=cfg["tracking"]["track_buffer"],
            frame_rate=cfg["tracking"]["frame_rate"],
        )
        zone_manager = ZoneManager(
            config_path=cfg["alert"].get(
                "zones_config_path", "data/zones/zones_config.json"
            ),
            cooldown_seconds=cfg["alert"]["cooldown_seconds"],
        )
        alert_engine = AlertEngine(
            redis_client=redis_client,
            frames_save_dir=cfg["alert"].get("frames_save_dir", "logs/frames"),
        )
        return cls(
            detector=detector,
            tracker=tracker,
            zone_manager=zone_manager,
            alert_engine=alert_engine,
            target_fps=cfg["preprocessing"]["target_fps"],
            frame_width=cfg["model"]["input_size"],
            frame_height=cfg["model"]["input_size"],
        )

    # ──────────────────────────── Public API ──────────────────────────────

    async def run_stream(
        self,
        source: str | int,
        camera_id: str = "CAM_00",
    ) -> None:
        """
        Continuously process a camera stream until interrupted.
        Handles RTSP reconnection automatically.
        """
        logger.info(f"Starting stream | source={source} | camera={camera_id}")
        self._start_time = time.time()

        while True:
            extractor = FrameExtractor(
                source=source,
                target_fps=self.target_fps,
                width=self.frame_width,
                height=self.frame_height,
                camera_id=camera_id,
            )
            try:
                extractor.start()
                async for result, alerts in self._process_frames(extractor, camera_id):
                    pass    # results are dispatched internally
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user.")
                break
            except Exception as e:
                logger.error(f"[{camera_id}] Stream error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)
            finally:
                extractor.stop()

    async def process_single_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM_00",
    ) -> dict:
        """
        Process a single frame (for REST API /detect endpoint).
        Returns a dict with detections and alerts.
        """
        result = self.detector.detect(frame, camera_id=camera_id)
        result = self.tracker.update(result)
        self._update_trajectories(result)

        raw_alerts = self.zone_manager.evaluate(
            detections=result.detections,
            camera_id=camera_id,
            trajectory_history=self._trajectory_history,
        )
        alerts = await self.alert_engine.process(raw_alerts, frame=frame)

        return {
            "camera_id": camera_id,
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "inference_ms": result.inference_ms,
            "personCount": result.person_count,
            "vehicleCount": result.vehicle_count,
            "detections": [d.to_dict() for d in result.detections],
            "alerts": [a.to_dict() for a in alerts],
        }

    async def process_frames_generator(
        self,
        source: str | int,
        camera_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields processed frame results.
        Used by the WebSocket streaming endpoint.
        """
        extractor = FrameExtractor(
            source=source,
            target_fps=self.target_fps,
            camera_id=camera_id,
        )
        extractor.start()
        try:
            async for result, alerts in self._process_frames(extractor, camera_id):
                yield {
                    "camera_id": camera_id,
                    "frame_id": result.frame_id,
                    "personCount": result.person_count,
                    "vehicleCount": result.vehicle_count,
                    "alerts": [a.to_dict() for a in alerts],
                    "detections": [d.to_dict() for d in result.detections],
                    "fps": self._current_fps(),
                }
        finally:
            extractor.stop()

    # ──────────────────────────── Internals ───────────────────────────────

    async def _process_frames(
        self,
        extractor: FrameExtractor,
        camera_id: str,
    ):
        """Core async frame processing loop."""
        loop = asyncio.get_event_loop()

        for frame, timestamp in extractor.frames():
            # Run heavy GPU inference in a thread pool to not block async loop
            result: FrameResult = await loop.run_in_executor(
                None,
                lambda f=frame, ts=timestamp: self.detector.detect(f, camera_id, ts),
            )
            result = self.tracker.update(result)
            self._update_trajectories(result)

            raw_alerts = self.zone_manager.evaluate(
                detections=result.detections,
                camera_id=camera_id,
                trajectory_history=self._trajectory_history,
            )
            alerts = await self.alert_engine.process(raw_alerts, frame=frame)
            self._frame_count += 1

            yield result, alerts

    def _update_trajectories(self, result: FrameResult) -> None:
        """Maintain per-track trajectory history for boundary crossing checks."""
        for det in result.detections:
            if det.track_id is None:
                continue
            if det.track_id not in self._trajectory_history:
                self._trajectory_history[det.track_id] = []
            self._trajectory_history[det.track_id].append(det.center)
            # Prune old history
            if len(self._trajectory_history[det.track_id]) > 50:
                self._trajectory_history[det.track_id].pop(0)

        # Prune tracks that no longer appear (keep last 200 track IDs)
        active_ids = {d.track_id for d in result.detections if d.track_id}
        if len(self._trajectory_history) > 200:
            old_ids = [
                k for k in self._trajectory_history if k not in active_ids
            ]
            for k in old_ids[:50]:
                del self._trajectory_history[k]

    def _current_fps(self) -> float:
        elapsed = time.time() - self._start_time
        return round(self._frame_count / elapsed, 1) if elapsed > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "total_frames": self._frame_count,
            "current_fps": self._current_fps(),
            "avg_inference_ms": self.detector.avg_inference_ms,
            "tracked_objects": len(self.tracker.get_confirmed_tracks()),
            "alert_stats": self.alert_engine.get_stats(),
        }
