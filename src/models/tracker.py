"""
tracker.py
──────────
Multi-object tracker wrapping ByteTrack and DeepSORT.
Assigns persistent track IDs to Detection objects across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.models.detector import Detection, FrameResult


# ──────────────────────────── ByteTrack Implementation ────────────────────

@dataclass
class TrackState:
    TENTATIVE = 0   # First seen, not yet confirmed
    CONFIRMED = 1   # Tracked for min_hits frames
    LOST = 2        # Not seen for track_buffer frames


@dataclass
class Track:
    """Represents a single tracked object."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    class_id: int
    confidence: float
    state: int = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: List[Tuple[int, int]] = None

    def __post_init__(self):
        self.trajectory = []

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def update_trajectory(self) -> None:
        self.trajectory.append(self.center)
        if len(self.trajectory) > 50:   # keep last 50 positions
            self.trajectory.pop(0)


class ByteTracker:
    """
    Simplified ByteTrack-style multi-object tracker.

    ByteTrack uses two-stage association:
      1. High-confidence detections → confirmed tracks (IoU match)
      2. Low-confidence detections → lost tracks (IoU match)

    Reference: https://arxiv.org/abs/2110.06864
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
        frame_rate: int = 15,
    ):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, frame_result: FrameResult) -> FrameResult:
        """
        Associate detections with existing tracks and return frame_result
        with track_id populated on each Detection.
        """
        detections = frame_result.detections

        # Separate by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets  = [d for d in detections if d.confidence < self.high_thresh
                     and d.confidence >= self.track_thresh]

        active_tracks = list(self._tracks.values())

        # ── Stage 1: Match high-conf detections to tracks ────────────────
        matched_high, unmatched_tracks, unmatched_dets = self._associate(
            high_dets, active_tracks, self.match_thresh
        )

        for track_id, det in matched_high:
            track = self._tracks[track_id]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.time_since_update = 0
            track.hits += 1
            track.age += 1
            if track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED
            track.update_trajectory()
            det.track_id = track_id

        # ── Stage 2: Try to recover lost tracks with low-conf dets ───────
        lost_tracks = [self._tracks[tid] for tid in unmatched_tracks
                       if self._tracks[tid].state != TrackState.TENTATIVE]
        matched_low, still_lost, _ = self._associate(
            low_dets, lost_tracks, self.match_thresh * 0.7
        )

        for track_id, det in matched_low:
            track = self._tracks[track_id]
            track.bbox = det.bbox
            track.time_since_update = 0
            track.age += 1
            track.update_trajectory()
            det.track_id = track_id

        # ── Create new tracks for unmatched high-conf detections ─────────
        matched_det_ids = {id(det) for _, det in matched_high + matched_low}
        for det in high_dets:
            if id(det) not in matched_det_ids:
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    confidence=det.confidence,
                )
                new_track.update_trajectory()
                self._tracks[self._next_id] = new_track
                det.track_id = self._next_id
                self._next_id += 1

        # ── Age out stale tracks ─────────────────────────────────────────
        to_delete = []
        for tid, track in self._tracks.items():
            if track.time_since_update == 0:
                continue
            track.time_since_update += 1
            track.age += 1
            if track.time_since_update > self.max_time_lost:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # Increment time_since_update for unmatched tracks
        for tid in unmatched_tracks:
            if tid in self._tracks:
                self._tracks[tid].time_since_update += 1

        return frame_result

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[Track],
        iou_threshold: float,
    ) -> Tuple[List, List[int], List]:
        """
        Greedy IoU-based assignment (Hungarian can replace for better accuracy).
        Returns: (matched_pairs, unmatched_track_ids, unmatched_dets)
        """
        if not detections or not tracks:
            track_ids = [t.track_id for t in tracks]
            return [], track_ids, detections

        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)

        matched_pairs = []
        unmatched_track_ids = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        # Greedy matching by descending IoU
        while True:
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] < iou_threshold:
                break
            matched_pairs.append((tracks[i].track_id, detections[j]))
            if i in unmatched_track_ids:
                unmatched_track_ids.remove(i)
            if j in unmatched_dets:
                unmatched_dets.remove(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        unmatched_track_ids = [tracks[i].track_id for i in unmatched_track_ids]
        unmatched_dets = [detections[j] for j in unmatched_dets]
        return matched_pairs, unmatched_track_ids, unmatched_dets

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.is_confirmed]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
