"""
detector.py
───────────
YOLOv8 / YOLOv9 detection wrapper.
Supports PyTorch inference (.pt) and ONNX Runtime (.onnx).
Output: list of Detection(bbox, conf, class_id, class_name)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO


# ──────────────────────────── Data Classes ────────────────────────────────

@dataclass
class Detection:
    """Single object detection result."""
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "center": list(self.center),
        }


@dataclass
class FrameResult:
    """All detections for a single frame."""
    frame_id: int
    timestamp: float
    camera_id: str
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.class_name == "person")

    @property
    def persons(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "person"]


# ──────────────────────────── Detector Class ──────────────────────────────

# Person ONLY — no vehicle detection
# (vehicle detection removed per project spec)
COCO_PERSON_ID = 0

CLASS_MAP = {
    COCO_PERSON_ID: "person",
}


class IntrusionDetector:
    """
    YOLOv8/v9 detector optimized for real-time intrusion detection.

    Usage:
        detector = IntrusionDetector("weights/yolov8m.pt", device="cuda")
        result = detector.detect(frame_bgr, camera_id="CAM_01")
    """

    def __init__(
        self,
        weights: str = "yolov8m.pt",
        device: str = "cuda",
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        half: bool = True,
    ):
        self.weights = Path(weights)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half = half and self.device != "cpu"

        self._model: Optional[YOLO] = None
        self._frame_count = 0
        self._total_inference_ms = 0.0

        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading detector: {self.weights} → device={self.device}")
        self._model = YOLO(str(self.weights))
        if self.half:
            self._model.model.half()
        # Warm-up pass
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self._model.predict(
            dummy, device=self.device,
            conf=self.conf_threshold, verbose=False
        )
        logger.success(f"Detector ready | FP16={self.half} | conf={self.conf_threshold}")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM_00",
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """
        Run detection on a single BGR frame.

        Args:
            frame       : BGR numpy array (H, W, 3).
            camera_id   : Camera identifier tag.
            timestamp   : Unix timestamp (auto-set if None).

        Returns:
            FrameResult with all detected objects.
        """
        if timestamp is None:
            timestamp = time.time()

        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            classes=list(CLASS_MAP.keys()),  # only detect relevant classes
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._parse_results(results)

        self._frame_count += 1
        self._total_inference_ms += inference_ms

        return FrameResult(
            frame_id=self._frame_count,
            timestamp=timestamp,
            camera_id=camera_id,
            detections=detections,
            inference_ms=round(inference_ms, 2),
        )

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_id: str = "CAM_00",
    ) -> List[FrameResult]:
        """Run detection on a batch of frames (more GPU efficient)."""
        t0 = time.perf_counter()
        results_batch = self._model.predict(
            frames,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            classes=list(CLASS_MAP.keys()),
        )
        total_ms = (time.perf_counter() - t0) * 1000
        per_frame_ms = total_ms / len(frames)

        ts = time.time()
        output = []
        for i, res in enumerate(results_batch):
            self._frame_count += 1
            output.append(FrameResult(
                frame_id=self._frame_count,
                timestamp=ts + i / 15.0,
                camera_id=camera_id,
                detections=self._parse_results([res]),
                inference_ms=round(per_frame_ms, 2),
            ))
        return output

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in CLASS_MAP:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=CLASS_MAP[cls_id],
                ))
        return detections

    @property
    def avg_inference_ms(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return round(self._total_inference_ms / self._frame_count, 2)

    def draw_detections(
        self,
        frame: np.ndarray,
        result: FrameResult,
        draw_conf: bool = True,
    ) -> np.ndarray:
        """Render bounding boxes on frame (for visualization/debug)."""
        vis = frame.copy()
        COLORS = {"person": (0, 255, 0), "vehicle": (0, 0, 255)}

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.class_name, (255, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if det.track_id is not None:
                label += f" #{det.track_id}"
            if draw_conf:
                label += f" {det.confidence:.2f}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(
                vis, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
        return vis

    def export_onnx(self, output_path: str = "weights/model.onnx") -> str:
        """Export the loaded model to ONNX format for deployment."""
        logger.info("Exporting model to ONNX...")
        path = self._model.export(
            format="onnx",
            imgsz=self.img_size,
            half=self.half,
            dynamic=False,
            simplify=True,
            opset=17,
        )
        logger.success(f"ONNX exported → {path}")
        return str(path)
