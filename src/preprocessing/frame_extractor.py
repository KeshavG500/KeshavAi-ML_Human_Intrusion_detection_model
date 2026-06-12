"""
frame_extractor.py
──────────────────
Extracts frames from RTSP streams, video files, or webcam feeds.
Supports multi-threaded buffered capture for real-time inference.
"""

import cv2
import queue
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
from loguru import logger


class FrameExtractor:
    """
    High-performance frame extractor supporting:
    - RTSP / RTMP camera streams
    - Local video files (.mp4, .avi, .mkv)
    - USB / webcam devices (index 0, 1, ...)
    - HTTP MJPEG streams

    Uses a background reader thread with a bounded queue so the
    main inference thread always gets the *latest* frame, not a
    stale one from a slow capture loop.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".ts"}

    def __init__(
        self,
        source: str | int,
        target_fps: int = 15,
        width: int = 1280,
        height: int = 720,
        queue_size: int = 4,
        camera_id: str = "CAM_00",
    ):
        """
        Args:
            source      : RTSP URL, file path, or webcam index.
            target_fps  : Desired output FPS (frames are dropped to match).
            width       : Resize width. 0 = keep original.
            height      : Resize height. 0 = keep original.
            queue_size  : Max frames held in the internal buffer.
            camera_id   : Logical name for this camera source.
        """
        self.source = source
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.camera_id = camera_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._dropped = 0

    # ──────────────────────────── Public API ──────────────────────────────

    def start(self) -> "FrameExtractor":
        """Open the capture device and start background reader thread."""
        self._cap = self._open_capture()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"FrameReader-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[{self.camera_id}] Started capture | source={self.source} | "
            f"fps={self.target_fps} | resolution={self.width}x{self.height}"
        )
        return self

    def stop(self) -> None:
        """Signal reader thread to stop and release the capture device."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info(
            f"[{self.camera_id}] Stopped | frames={self._frame_count} "
            f"dropped={self._dropped}"
        )

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Returns the latest frame as (frame_bgr, timestamp).
        Returns None if the stream ended or buffer is empty.
        """
        try:
            return self._queue.get(timeout=2.0)
        except queue.Empty:
            return None

    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Generator yielding (frame, timestamp) tuples until stream ends."""
        while not self._stop_event.is_set():
            result = self.read()
            if result is None:
                break
            yield result

    @property
    def native_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def native_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    # ──────────────────────────── Internals ───────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(
                f"[{self.camera_id}] Cannot open source: {self.source}"
            )
        # RTSP buffer settings — minimise latency
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
        return cap

    def _reader_loop(self) -> None:
        native_fps = self.native_fps
        skip_ratio = max(1, round(native_fps / self.target_fps))
        interval = 1.0 / self.target_fps
        last_time = time.monotonic()
        local_idx = 0

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"[{self.camera_id}] Stream ended / read error.")
                self._stop_event.set()
                break

            local_idx += 1
            # Drop frames to hit target FPS
            if local_idx % skip_ratio != 0:
                self._dropped += 1
                continue

            # Resize if requested
            if self.width > 0 and self.height > 0:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )

            timestamp = time.time()
            # Drop oldest frame if queue is full (always keep latest)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass

            self._queue.put((frame, timestamp))
            self._frame_count += 1

            # Throttle to not spin too fast
            elapsed = time.monotonic() - last_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.monotonic()


class VideoFileExtractor:
    """
    Simple batch extractor for offline training — extracts frames from
    video files at a fixed interval and saves them to disk.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        sample_every_n: int = 5,
        max_frames: Optional[int] = None,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.sample_every_n = sample_every_n
        self.max_frames = max_frames

    def extract(self) -> int:
        """Extract frames and save as JPEG. Returns count of saved frames."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        idx = 0

        logger.info(
            f"Extracting from {self.video_path.name} "
            f"({total} frames, every {self.sample_every_n})"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.sample_every_n == 0:
                stem = self.video_path.stem
                out_path = self.output_dir / f"{stem}_{idx:07d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
                if self.max_frames and saved >= self.max_frames:
                    break
            idx += 1

        cap.release()
        logger.success(f"Saved {saved} frames → {self.output_dir}")
        return saved
