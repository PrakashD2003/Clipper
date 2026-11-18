"""Unit tests for Package 4.1: FacecamDetector and corner detection logic."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest

from streamclipper.assembly.facecam import FacecamDetector, _is_in_corner
from streamclipper.config import Config
from streamclipper.models import FacecamRegion


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def draw_synthetic_face(frame: np.ndarray, center_x: int, center_y: int, radius: int = 70) -> None:
    """Draw a face pattern with eyes and mouth recognizable by Haar Cascade."""
    # Head circle
    cv2.circle(frame, (center_x, center_y), radius, (220, 220, 220), -1)
    # Eyes
    eye_offset_x = int(radius * 0.35)
    eye_offset_y = int(radius * 0.20)
    eye_radius = max(2, int(radius * 0.16))
    cv2.circle(frame, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, (40, 40, 40), -1)
    cv2.circle(frame, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (40, 40, 40), -1)
    # Mouth ellipse (smile)
    mouth_offset_y = int(radius * 0.40)
    cv2.ellipse(
        frame,
        (center_x, center_y + mouth_offset_y),
        (int(radius * 0.35), int(radius * 0.12)),
        0,
        0,
        180,
        (40, 40, 40),
        -1,
    )


def create_synthetic_video(
    video_path: Path | str,
    width: int = 1280,
    height: int = 720,
    total_frames: int = 20,
    fps: float = 30.0,
    face_positions: list[tuple[int, int] | None] | None = None,
) -> None:
    """Generate a temporary MP4 video file with synthetic frames and optional face drawing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        for i in range(total_frames):
            frame = np.full((height, width, 3), 128, dtype=np.uint8)
            if face_positions:
                pos = face_positions[i % len(face_positions)]
                if pos is not None:
                    cx, cy = pos
                    draw_synthetic_face(frame, cx, cy, radius=70)
            out.write(frame)
    finally:
        out.release()


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestCornerDetection:
    """Tests for _is_in_corner function and method across corners and edge cases."""

    def test_is_in_corner_all_four_corners(self) -> None:
        """Verify all 4 screen corners are correctly detected as in corner."""
        w_frame, h_frame = 1920, 1080
        box_w, box_h = 200, 200

        # Top-left corner: center_x = 150 (< 576), center_y = 150 (< 378)
        assert _is_in_corner(50, 50, box_w, box_h, w_frame, h_frame) is True

        # Top-right corner: center_x = 1750 (> 1344), center_y = 150 (< 378)
        assert _is_in_corner(1650, 50, box_w, box_h, w_frame, h_frame) is True

        # Bottom-left corner: center_x = 150 (< 576), center_y = 900 (> 702)
        assert _is_in_corner(50, 800, box_w, box_h, w_frame, h_frame) is True

        # Bottom-right corner: center_x = 1750 (> 1344), center_y = 900 (> 702)
        assert _is_in_corner(1650, 800, box_w, box_h, w_frame, h_frame) is True

    def test_is_in_corner_center_rejected(self) -> None:
        """Verify center of frame is rejected by corner check."""
        w_frame, h_frame = 1920, 1080
        # Dead center: center_x = 960, center_y = 540
        assert _is_in_corner(860, 440, 200, 200, w_frame, h_frame) is False

    def test_is_in_corner_edge_midpoints_rejected(self) -> None:
        """Verify top-center, bottom-center, left-middle, and right-middle are rejected."""
        w_frame, h_frame = 1920, 1080
        box_w, box_h = 200, 200

        # Top-center (center_x = 960, center_y = 150): horizontal is in middle
        assert _is_in_corner(860, 50, box_w, box_h, w_frame, h_frame) is False

        # Bottom-center (center_x = 960, center_y = 950): horizontal is in middle
        assert _is_in_corner(860, 850, box_w, box_h, w_frame, h_frame) is False

        # Left-middle (center_x = 150, center_y = 540): vertical is in middle
        assert _is_in_corner(50, 440, box_w, box_h, w_frame, h_frame) is False

        # Right-middle (center_x = 1750, center_y = 540): vertical is in middle
        assert _is_in_corner(1650, 440, box_w, box_h, w_frame, h_frame) is False

    def test_is_in_corner_boundary_conditions(self) -> None:
        """Verify strict inequality on the 30% horizontal and 35% vertical boundary lines."""
        w_frame, h_frame = 1000, 1000
        # Horizontal boundaries are 300 and 700. Vertical boundaries are 350 and 650.
        # Exactly on x=300 (not < 300):
        assert _is_in_corner(250, 50, 100, 100, w_frame, h_frame) is False  # center_x = 300
        # Just inside (< 300):
        assert _is_in_corner(240, 50, 100, 100, w_frame, h_frame) is True   # center_x = 290

        # Exactly on y=350 (not < 350):
        assert _is_in_corner(50, 300, 100, 100, w_frame, h_frame) is False  # center_y = 350
        # Just inside (< 350):
        assert _is_in_corner(50, 290, 100, 100, w_frame, h_frame) is True   # center_y = 340

    def test_is_in_corner_zero_or_negative_dimensions(self) -> None:
        """Verify handling of invalid dimensions returns False without crashing."""
        assert _is_in_corner(10, 10, 50, 50, 0, 1080) is False
        assert _is_in_corner(10, 10, 50, 50, 1920, -100) is False
        assert _is_in_corner(10, 10, 0, 50, 1920, 1080) is False
        assert _is_in_corner(10, 10, 50, -5, 1920, 1080) is False

    def test_is_in_corner_callable_via_class_and_instance(self) -> None:
        """Verify _is_in_corner is accessible via module, staticmethod, and instance."""
        detector = FacecamDetector()
        assert _is_in_corner(50, 50, 200, 200, 1920, 1080) is True
        assert FacecamDetector._is_in_corner(50, 50, 200, 200, 1920, 1080) is True
        assert detector._is_in_corner(50, 50, 200, 200, 1920, 1080) is True


class TestAreaFiltering:
    """Tests for size/area verification (between 1.5% and 25% of frame area)."""

    def test_valid_area_within_range(self) -> None:
        """Verify area between 1.5% and 25% passes area filter."""
        detector = FacecamDetector()
        # 1920x1080 = 2,073,600.
        # 5% area = 103,680 -> 320x324
        assert detector._is_valid_area(320, 324, 1920, 1080) is True
        # 10% area = 207,360 -> 450x460
        assert detector._is_valid_area(450, 460, 1920, 1080) is True

    def test_area_below_minimum_rejected(self) -> None:
        """Verify area below 1.5% is rejected (background faces / noise)."""
        detector = FacecamDetector()
        # 1% area = 20,736 -> 144x144
        assert detector._is_valid_area(144, 144, 1920, 1080) is False
        # Tiny 30x30 face
        assert detector._is_valid_area(30, 30, 1920, 1080) is False

    def test_area_above_maximum_rejected(self) -> None:
        """Verify area above 25% is rejected (close-up fullscreen gameplay / cinematics)."""
        detector = FacecamDetector()
        # 30% area = 622,080 -> 800x777
        assert detector._is_valid_area(800, 777, 1920, 1080) is False
        # 50% area
        assert detector._is_valid_area(1000, 1036, 1920, 1080) is False

    def test_area_exact_boundaries(self) -> None:
        """Verify exact 1.5% and 25% boundaries pass."""
        detector = FacecamDetector()
        w_frame, h_frame = 1000, 1000  # Total area = 1,000,000
        # 1.5% = 15,000
        assert detector._is_valid_area(150, 100, w_frame, h_frame) is True
        # Just below 1.5% (14,900)
        assert detector._is_valid_area(149, 100, w_frame, h_frame) is False
        # 25% = 250,000
        assert detector._is_valid_area(500, 500, w_frame, h_frame) is True
        # Just above 25% (251,000)
        assert detector._is_valid_area(502, 500, w_frame, h_frame) is False

    def test_area_zero_or_negative_dimensions(self) -> None:
        """Verify non-positive dimensions return False."""
        detector = FacecamDetector()
        assert detector._is_valid_area(0, 100, 1920, 1080) is False
        assert detector._is_valid_area(100, -50, 1920, 1080) is False
        assert detector._is_valid_area(100, 100, 0, 1080) is False


class TestMedianAndPaddingClamping:
    """Tests for median bounding box calculation and boundary-clamped padding."""

    def test_padded_box_centered_and_scaled(self) -> None:
        """Verify padding expands by 1.8x width and 1.5x height, centered on median."""
        detector = FacecamDetector()
        # Median box: x=100, y=100, w=200, h=200 in 1920x1080
        # Center = (200, 200). Padded w = 360, Padded h = 300.
        # x1 = 200 - 180 = 20, y1 = 200 - 150 = 50.
        x, y, w, h = detector._calculate_padded_box(100, 100, 200, 200, 1920, 1080)
        assert x == 20
        assert y == 50
        assert w == 360
        assert h == 300

    def test_padded_box_clamps_to_top_left_boundaries(self) -> None:
        """Verify padding clamping prevents negative coordinates near top-left edge."""
        detector = FacecamDetector()
        # Median box very close to (0, 0): x=20, y=20, w=200, h=200
        # Center = (120, 120). Padded w = 360, h = 300.
        # Raw x1 = 120 - 180 = -60 -> clamped to 0. Raw x2 = -60 + 360 = 300.
        # Raw y1 = 120 - 150 = -30 -> clamped to 0. Raw y2 = -30 + 300 = 270.
        x, y, w, h = detector._calculate_padded_box(20, 20, 200, 200, 1920, 1080)
        assert x == 0
        assert y == 0
        assert w == 300
        assert h == 270
        assert x + w <= 1920
        assert y + h <= 1080

    def test_padded_box_clamps_to_bottom_right_boundaries(self) -> None:
        """Verify padding clamping prevents overflow near bottom-right edge."""
        detector = FacecamDetector()
        # Median box near bottom-right: x=1800, y=950, w=200, h=200 in 1920x1080
        # Center = (1900, 1050). Padded w = 360, h = 300.
        # Raw x1 = 1900 - 180 = 1720, Raw x2 = 1720 + 360 = 2080 -> clamped to 1920.
        # Raw y1 = 1050 - 150 = 900, Raw y2 = 900 + 300 = 1200 -> clamped to 1080.
        x, y, w, h = detector._calculate_padded_box(1800, 950, 200, 200, 1920, 1080)
        assert x == 1720
        assert y == 900
        assert x + w == 1920
        assert y + h == 1080

    def test_custom_padding_ratios(self, tmp_path: Path) -> None:
        """Verify custom padding ratios configured via Config are applied."""
        cfg_path = tmp_path / "custom_padding.yaml"
        cfg_path.write_text(
            "facecam:\n"
            "  padding_w_ratio: 2.0\n"
            "  padding_h_ratio: 2.0\n",
            encoding="utf-8",
        )
        cfg = Config(user_config_path=cfg_path)
        detector = FacecamDetector(cfg)
        # Center = (200, 200). Padded w = 400, h = 400.
        # x1 = 200 - 200 = 0, y1 = 200 - 200 = 0.
        x, y, w, h = detector._calculate_padded_box(100, 100, 200, 200, 1920, 1080)
        assert x == 0
        assert y == 0
        assert w == 400
        assert h == 400


class TestThresholdAndConsistency:
    """Tests for the 30% consistency requirement and quadrant clustering."""

    def test_detection_below_30_percent_returns_none(self, tmp_path: Path) -> None:
        """Verify that when only 20% of frames have a facecam (< 30%), detect returns None."""
        video_path = tmp_path / "low_consistency.mp4"
        # 10 frames total: 2 frames with face (20%), 8 frames blank
        positions: list[tuple[int, int] | None] = [
            (180, 150), (180, 150),
            None, None, None, None, None, None, None, None,
        ]
        create_synthetic_video(video_path, total_frames=10, face_positions=positions)

        detector = FacecamDetector()
        result = detector.detect(video_path, num_samples=10)
        assert result is None

    def test_detection_at_or_above_30_percent_succeeds(self, tmp_path: Path) -> None:
        """Verify that when 40% of frames have a facecam (>= 30%), detect succeeds."""
        video_path = tmp_path / "sufficient_consistency.mp4"
        # 10 frames total: 4 frames with face (40%), 6 frames blank
        positions: list[tuple[int, int] | None] = [
            (180, 150), (180, 150), (180, 150), (180, 150),
            None, None, None, None, None, None,
        ]
        create_synthetic_video(video_path, total_frames=10, face_positions=positions)

        detector = FacecamDetector()
        result = detector.detect(video_path, num_samples=10)
        assert result is not None
        assert isinstance(result, FacecamRegion)
        assert result.confidence == 0.40

    def test_scattered_corner_detections_return_none(self, tmp_path: Path) -> None:
        """Verify that detections scattered across different corners with none reaching 30% return None."""
        video_path = tmp_path / "scattered_corners.mp4"
        # 10 frames total: 1 in top-left, 1 in top-right, 1 in bottom-left, 1 in bottom-right, 6 blank
        # Total corner detections = 4 (40%), but no single corner has >= 3 (30%)
        positions: list[tuple[int, int] | None] = [
            (180, 150),   # top-left
            (1100, 150),  # top-right
            (180, 570),   # bottom-left
            (1100, 570),  # bottom-right
            None, None, None, None, None, None,
        ]
        create_synthetic_video(video_path, total_frames=10, face_positions=positions)

        detector = FacecamDetector()
        result = detector.detect(video_path, num_samples=10)
        assert result is None


class TestEndToEndSyntheticVideo:
    """End-to-end tests using real synthetic MP4 video files and OpenCV Haar Cascade detection."""

    def test_detect_facecam_top_left(self, tmp_path: Path) -> None:
        """Verify facecam in top-left corner is detected and padded correctly."""
        video_path = tmp_path / "stream_top_left.mp4"
        create_synthetic_video(
            video_path,
            total_frames=10,
            face_positions=[(180, 150)],
        )

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)

        assert region is not None
        assert isinstance(region, FacecamRegion)
        assert region.confidence == 1.0
        # Verify region coordinates are in top-left quadrant of 1280x720
        assert region.x < 1280 * 0.3
        assert region.y < 720 * 0.35
        assert region.x + region.w <= 1280
        assert region.y + region.h <= 720

    def test_detect_facecam_top_right(self, tmp_path: Path) -> None:
        """Verify facecam in top-right corner is detected."""
        video_path = tmp_path / "stream_top_right.mp4"
        create_synthetic_video(
            video_path,
            total_frames=10,
            face_positions=[(1100, 150)],
        )

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)

        assert region is not None
        assert region.confidence == 1.0
        assert (region.x + region.w / 2) > 1280 * 0.7
        assert (region.y + region.h / 2) < 720 * 0.35

    def test_detect_facecam_bottom_left(self, tmp_path: Path) -> None:
        """Verify facecam in bottom-left corner is detected."""
        video_path = tmp_path / "stream_bottom_left.mp4"
        create_synthetic_video(
            video_path,
            total_frames=10,
            face_positions=[(180, 570)],
        )

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)

        assert region is not None
        assert region.confidence == 1.0
        assert (region.x + region.w / 2) < 1280 * 0.3
        assert (region.y + region.h / 2) > 720 * 0.65

    def test_detect_facecam_bottom_right(self, tmp_path: Path) -> None:
        """Verify facecam in bottom-right corner is detected."""
        video_path = tmp_path / "stream_bottom_right.mp4"
        create_synthetic_video(
            video_path,
            total_frames=10,
            face_positions=[(1100, 570)],
        )

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)

        assert region is not None
        assert region.confidence == 1.0
        assert (region.x + region.w / 2) > 1280 * 0.7
        assert (region.y + region.h / 2) > 720 * 0.65

    def test_detect_stream_without_facecam_returns_none(self, tmp_path: Path) -> None:
        """Verify streams without any facecam return None."""
        video_path = tmp_path / "no_facecam.mp4"
        create_synthetic_video(video_path, total_frames=10, face_positions=None)

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)
        assert region is None

    def test_detect_in_game_center_face_returns_none(self, tmp_path: Path) -> None:
        """Verify in-game character faces in center are ignored and return None."""
        video_path = tmp_path / "center_character.mp4"
        # Face at dead center (640, 360)
        create_synthetic_video(video_path, total_frames=10, face_positions=[(640, 360)])

        detector = FacecamDetector()
        region = detector.detect(video_path, num_samples=10)
        assert region is None

    def test_60s_video_processes_under_15_seconds(self, tmp_path: Path) -> None:
        """Acceptance Criterion: processes a 60-second video segment in under 15 seconds."""
        video_path = tmp_path / "long_segment.mp4"
        # 60s at 30fps = 1800 frames
        create_synthetic_video(
            video_path,
            total_frames=1800,
            fps=30.0,
            face_positions=[(180, 150)],
        )

        detector = FacecamDetector()
        t0 = time.time()
        region = detector.detect(video_path, num_samples=50)
        elapsed = time.time() - t0

        assert region is not None
        assert elapsed < 15.0, f"Detection took {elapsed:.2f}s, exceeding 15s limit"


class TestRobustnessAndEdgeCases:
    """Tests for edge cases, missing files, configuration, and data models."""

    def test_non_existent_file_returns_none(self, tmp_path: Path) -> None:
        """Verify missing video file returns None gracefully without raising."""
        detector = FacecamDetector()
        result = detector.detect(tmp_path / "does_not_exist.mp4")
        assert result is None

    def test_zero_or_negative_sample_count_returns_none(self, tmp_path: Path) -> None:
        """Verify num_samples <= 0 returns None."""
        video_path = tmp_path / "sample.mp4"
        create_synthetic_video(video_path, total_frames=5)
        detector = FacecamDetector()
        assert detector.detect(video_path, num_samples=0) is None
        assert detector.detect(video_path, num_samples=-5) is None

    def test_corrupt_video_file_returns_none(self, tmp_path: Path) -> None:
        """Verify corrupt/empty file returns None gracefully."""
        corrupt_path = tmp_path / "corrupt.mp4"
        corrupt_path.write_bytes(b"not a valid video stream data")
        detector = FacecamDetector()
        result = detector.detect(corrupt_path)
        assert result is None

    def test_config_overrides_respected(self, tmp_path: Path) -> None:
        """Verify custom Config overrides are loaded correctly into FacecamDetector."""
        cfg_path = tmp_path / "custom_facecam.yaml"
        cfg_path.write_text(
            "facecam:\n"
            "  min_area_ratio: 0.02\n"
            "  max_area_ratio: 0.20\n"
            "  min_detection_ratio: 0.50\n"
            "  padding_w_ratio: 2.2\n"
            "  padding_h_ratio: 1.7\n"
            "  num_samples: 30\n",
            encoding="utf-8",
        )
        cfg = Config(user_config_path=cfg_path)
        detector = FacecamDetector(cfg)
        assert detector.min_area_ratio == 0.02
        assert detector.max_area_ratio == 0.20
        assert detector.min_detection_ratio == 0.50
        assert detector.padding_w_ratio == 2.2
        assert detector.padding_h_ratio == 1.7
        assert detector.num_samples == 30

    def test_detect_uses_configured_num_samples_by_default(self, tmp_path: Path) -> None:
        """Verify detect(video_path) respects facecam.num_samples when num_samples is omitted."""
        video_path = tmp_path / "ten_frames.mp4"
        create_synthetic_video(video_path, total_frames=12, face_positions=[(180, 150)])

        cfg_path = tmp_path / "zero_samples.yaml"
        cfg_path.write_text("facecam:\n  num_samples: 0\n", encoding="utf-8")
        detector = FacecamDetector(Config(user_config_path=cfg_path))
        assert detector.detect(video_path) is None

        # Explicit num_samples argument still overrides config
        assert detector.detect(video_path, num_samples=6) is not None

    def test_facecam_region_serialization_round_trip(self) -> None:
        """Verify FacecamRegion serializes to dict and reconstructs identically."""
        region = FacecamRegion(x=50, y=80, w=350, h=280, confidence=0.85)
        data = region.to_dict()
        assert data == {
            "x": 50,
            "y": 80,
            "w": 350,
            "h": 280,
            "confidence": 0.85,
        }
        reconstructed = FacecamRegion.from_dict(data)
        assert reconstructed == region
