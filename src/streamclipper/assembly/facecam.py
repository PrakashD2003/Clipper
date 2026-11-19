"""Facecam detection using OpenCV Haar Cascades with corner filtering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..models import FacecamRegion

logger = logging.getLogger(__name__)


def _is_in_corner(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> bool:
    """Check if a bounding box is located in one of the 4 screen corners.

    A bounding box is defined as in a corner if its center point lies within the
    outer 30% horizontal margins (x < 0.3*W or x > 0.7*W) and outer 35% vertical
    margins (y < 0.35*H or y > 0.65*H).
    """
    if frame_w <= 0 or frame_h <= 0 or w <= 0 or h <= 0:
        return False

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    in_left_or_right = center_x < frame_w * 0.3 or center_x > frame_w * 0.7
    in_top_or_bottom = center_y < frame_h * 0.35 or center_y > frame_h * 0.65

    return bool(in_left_or_right and in_top_or_bottom)


class FacecamDetector:
    """Detects facecam overlay position in gaming streams using OpenCV Haar Cascades."""

    def __init__(
        self,
        config: Config | None = None,
        min_area_ratio: float | None = None,
    ) -> None:
        self.config = config

        # Configurable detection parameters
        if min_area_ratio is not None:
            self.min_area_ratio = float(min_area_ratio)
        elif config is not None:
            self.min_area_ratio = float(config.get("facecam.min_area_ratio", 0.002))
        else:
            self.min_area_ratio = 0.015

        if config is not None:
            self.max_area_ratio: float = float(config.get("facecam.max_area_ratio", 0.25))
            self.min_detection_ratio: float = float(config.get("facecam.min_detection_ratio", 0.30))
            self.padding_w_ratio: float = float(config.get("facecam.padding_w_ratio", 1.8))
            self.padding_h_ratio: float = float(config.get("facecam.padding_h_ratio", 1.5))
            self.num_samples: int = int(config.get("facecam.num_samples", 50))
        else:
            self.max_area_ratio = 0.25
            self.min_detection_ratio = 0.30
            self.padding_w_ratio = 1.8
            self.padding_h_ratio = 1.5
            self.num_samples = 50

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            logger.warning("Haar cascade classifier failed to load from %s", cascade_path)

    @staticmethod
    def _is_in_corner(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> bool:
        """Class-accessible static method for corner checking."""
        return _is_in_corner(x, y, w, h, frame_w, frame_h)

    def _is_valid_area(self, w: int, h: int, frame_w: int, frame_h: int) -> bool:
        """Verify face detection bounding box area is between 1.5% and 25% of frame area."""
        if frame_w <= 0 or frame_h <= 0 or w <= 0 or h <= 0:
            return False
        total_area = frame_w * frame_h
        face_area = w * h
        return bool(self.min_area_ratio * total_area <= face_area <= self.max_area_ratio * total_area)

    @staticmethod
    def _get_corner_quadrant(center_x: float, center_y: float, frame_w: int, frame_h: int) -> str:
        """Identify which quadrant (top_left, top_right, bottom_left, bottom_right) a point is in."""
        h_side = "left" if center_x < frame_w * 0.5 else "right"
        v_side = "top" if center_y < frame_h * 0.5 else "bottom"
        return f"{v_side}_{h_side}"

    def _calculate_padded_box(
        self,
        med_x: float,
        med_y: float,
        med_w: float,
        med_h: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int]:
        """Compute padded bounding box centered on median face and clamped to frame boundaries."""
        center_x = med_x + med_w / 2.0
        center_y = med_y + med_h / 2.0

        padded_w = med_w * self.padding_w_ratio
        padded_h = med_h * self.padding_h_ratio

        raw_x1 = center_x - padded_w / 2.0
        raw_y1 = center_y - padded_h / 2.0
        raw_x2 = raw_x1 + padded_w
        raw_y2 = raw_y1 + padded_h

        clamped_x1 = max(0.0, min(float(frame_w), raw_x1))
        clamped_y1 = max(0.0, min(float(frame_h), raw_y1))
        clamped_x2 = max(0.0, min(float(frame_w), raw_x2))
        clamped_y2 = max(0.0, min(float(frame_h), raw_y2))

        final_x = int(round(clamped_x1))
        final_y = int(round(clamped_y1))
        final_x2 = int(round(clamped_x2))
        final_y2 = int(round(clamped_y2))

        final_w = max(0, final_x2 - final_x)
        final_h = max(0, final_y2 - final_y)

        return final_x, final_y, final_w, final_h

    def detect(self, video_path: Path | str, num_samples: int | None = None) -> FacecamRegion | None:
        """Detect facecam position in video by sampling frames evenly across its duration.

        Args:
            video_path: Path to video segment file.
            num_samples: Number of frames to sample evenly across video (defaults to config facecam.num_samples, 50).

        Returns:
            FacecamRegion with padded coordinates if consistent corner face detected in >= 30%
            of sampled frames, else None.
        """
        path = Path(video_path)
        if not path.is_file():
            logger.warning("Video file not found or not a file: %s", video_path)
            return None

        if self.cascade.empty():
            logger.warning("Haar cascade classifier is empty; cannot detect facecam.")
            return None

        requested_samples = self.num_samples if num_samples is None else num_samples
        if requested_samples <= 0:
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning("OpenCV could not open video: %s", video_path)
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0:
                logger.warning("Video reports 0 frames: %s", video_path)
                return None

            # Sample evenly across the video
            sample_count = min(requested_samples, total_frames)
            if sample_count <= 1:
                frame_indices = [0]
            else:
                frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()

            corner_detections_by_quadrant: dict[str, list[tuple[int, int, int, int]]] = {
                "top_left": [],
                "top_right": [],
                "bottom_left": [],
                "bottom_right": [],
            }

            sampled_frames_read = 0

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                sampled_frames_read += 1
                if frame_w <= 0 or frame_h <= 0:
                    frame_h, frame_w = frame.shape[:2]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20),
                )

                # Group valid corner faces detected in this frame by quadrant
                frame_quadrant_faces: dict[str, list[tuple[int, int, int, int]]] = {}
                for (fx, fy, fw, fh) in detected_faces:
                    # Filter 1: Size check (1.5% - 25% of frame area)
                    if not self._is_valid_area(fw, fh, frame_w, frame_h):
                        continue

                    # Filter 2: Corner placement check
                    if not self._is_in_corner(fx, fy, fw, fh, frame_w, frame_h):
                        continue

                    # Assign to quadrant
                    center_x = fx + fw / 2.0
                    center_y = fy + fh / 2.0
                    quad = self._get_corner_quadrant(center_x, center_y, frame_w, frame_h)
                    frame_quadrant_faces.setdefault(quad, []).append((int(fx), int(fy), int(fw), int(fh)))

                # To prevent double-counting in a single frame, record the primary detection per quadrant
                for quad, faces in frame_quadrant_faces.items():
                    # Pick largest face in that quadrant for the frame
                    faces.sort(key=lambda b: b[2] * b[3], reverse=True)
                    corner_detections_by_quadrant[quad].append(faces[0])

            effective_samples = sampled_frames_read if sampled_frames_read > 0 else sample_count
            if effective_samples <= 0:
                return None

            # Find the dominant corner
            best_quad, best_detections = max(
                corner_detections_by_quadrant.items(),
                key=lambda item: len(item[1]),
            )

            # Require consistent corner detections in at least 30% of sampled frames
            required_count = effective_samples * self.min_detection_ratio
            if len(best_detections) < required_count:
                logger.info(
                    "No consistent facecam detected (best quadrant %s had %d detections, required >= %.1f from %d samples).",
                    best_quad,
                    len(best_detections),
                    required_count,
                    effective_samples,
                )
                return None

            # Compute median bounding box from valid corner detections
            boxes = np.array(best_detections)
            med_x = float(np.median(boxes[:, 0]))
            med_y = float(np.median(boxes[:, 1]))
            med_w = float(np.median(boxes[:, 2]))
            med_h = float(np.median(boxes[:, 3]))

            # Add webcam padding clamped to frame boundaries
            pad_x, pad_y, pad_w, pad_h = self._calculate_padded_box(
                med_x, med_y, med_w, med_h, frame_w, frame_h
            )

            if pad_w <= 0 or pad_h <= 0:
                return None

            confidence = float(len(best_detections)) / float(effective_samples)
            logger.info(
                "Facecam detected in %s corner: (%d, %d, %d, %d) with confidence %.2f (%d/%d frames).",
                best_quad,
                pad_x,
                pad_y,
                pad_w,
                pad_h,
                confidence,
                len(best_detections),
                effective_samples,
            )

            return FacecamRegion(
                x=pad_x,
                y=pad_y,
                w=pad_w,
                h=pad_h,
                confidence=round(confidence, 2),
            )

        finally:
            cap.release()
