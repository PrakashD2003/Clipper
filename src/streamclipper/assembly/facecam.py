"""Facecam detection: YuNet (Haar fallback), whole-frame stable search, webcam-rect expansion, profiles."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..models import FacecamRegion

logger = logging.getLogger(__name__)

YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"


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
    """Detects facecam overlay position in gaming streams (YuNet, Haar fallback)."""

    def __init__(
        self,
        config: Config | None = None,
        min_area_ratio: float | None = None,
        use_yunet: bool = True,
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
            self.min_detection_ratio: float = float(config.get("facecam.min_detection_ratio", 0.6))
            self.padding_w_ratio: float = float(config.get("facecam.padding_w_ratio", 1.8))
            self.padding_h_ratio: float = float(config.get("facecam.padding_h_ratio", 1.5))
            self.num_samples: int = int(config.get("facecam.num_samples", 50))
        else:
            self.max_area_ratio = 0.25
            self.min_detection_ratio = 0.6
            self.padding_w_ratio = 1.8
            self.padding_h_ratio = 1.5
            self.num_samples = 50

        self.max_center_jitter: float = (
            float(config.get("facecam.max_center_jitter", 0.05)) if config is not None else 0.05
        )
        self.yunet: Any = self._load_yunet() if use_yunet else None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            logger.warning("Haar cascade classifier failed to load from %s", cascade_path)
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        if self.profile_cascade.empty():
            logger.warning("Profile-face cascade failed to load from %s", profile_path)

    @staticmethod
    def _load_yunet() -> Any:
        """Create the YuNet detector, or None if cv2/model is unavailable (Haar fallback)."""
        model = Path(__file__).resolve().parent.parent / "assets" / YUNET_MODEL_NAME
        if not hasattr(cv2, "FaceDetectorYN") or not model.is_file():
            logger.info("YuNet unavailable; using Haar cascade for facecam detection.")
            return None
        try:
            return cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 5000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YuNet failed to load (%s); using Haar cascade.", exc)
            return None

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

    # ------------------------------------------------------------------
    # Face detection primitives
    # ------------------------------------------------------------------

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a BGR frame. YuNet when available, else Haar cascades."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, faces = self.yunet.detect(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("YuNet detect failed (%s)", exc)
                return []
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]

        if self.cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = list(self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)))
        if not self.profile_cascade.empty():
            found.extend(
                list(self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)))
            )
        return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in found]

    @staticmethod
    def _sample_frames(paths: list[Path], num_samples: int) -> list[np.ndarray]:
        """Read ``num_samples`` frames spread across all given segment files."""
        infos: list[tuple[Path, int]] = []
        for p in paths:
            if not p.is_file():
                logger.warning("Video file not found or not a file: %s", p)
                continue
            cap = cv2.VideoCapture(str(p))
            try:
                if not cap.isOpened():
                    logger.warning("OpenCV could not open video: %s", p)
                    continue
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            if n > 0:
                infos.append((p, n))
        if not infos:
            return []

        total = sum(n for _, n in infos)
        remaining_budget = min(num_samples, total)
        remaining_total = total
        frames: list[np.ndarray] = []
        for p, n in infos:
            share = max(1, int(round(remaining_budget * n / remaining_total))) if remaining_total else 0
            share = min(share, n, remaining_budget)
            remaining_budget -= share
            remaining_total -= n
            if share <= 0:
                continue
            idxs = [0] if share == 1 else np.linspace(0, n - 1, share, dtype=int).tolist()
            cap = cv2.VideoCapture(str(p))
            try:
                for idx in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frames.append(frame)
            finally:
                cap.release()
        if not frames:
            return []
        # Normalize mismatched frame sizes to the first frame's size.
        h0, w0 = frames[0].shape[:2]
        return [f if f.shape[:2] == (h0, w0) else cv2.resize(f, (w0, h0)) for f in frames]

    # ------------------------------------------------------------------
    # B3: expand face box to webcam rectangle
    # ------------------------------------------------------------------

    def _expand_to_webcam_rect(
        self,
        frames: list[np.ndarray],
        face: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int] | None:
        """Find the webcam rectangle around a face via stable edge search; None if not found."""
        if not frames:
            return None
        edges = None
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            e = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
            e = cv2.dilate(e, np.ones((3, 3), np.uint8))
            edges = e if edges is None else cv2.bitwise_and(edges, e)  # only edges present in every frame
        if edges is None:
            return None

        fx, fy, fw, fh = (int(round(v)) for v in face)
        fx2, fy2 = fx + fw, fy + fh
        max_w, max_h = int(0.45 * frame_w), int(0.45 * frame_h)
        thresh = 0.6

        def line_col(x: int) -> float:
            seg = edges[max(0, fy):min(frame_h, fy2), x]
            return float(np.count_nonzero(seg)) / max(1, seg.size)

        def line_row(y: int) -> float:
            seg = edges[y, max(0, fx):min(frame_w, fx2)]
            return float(np.count_nonzero(seg)) / max(1, seg.size)

        gap_x, gap_y = max(2, int(fw * 0.15)), max(2, int(fh * 0.15))

        def search(start: int, stop: int, step: int, density: Any) -> int | None:
            for pos in range(start, stop, step):
                if density(pos) >= thresh:
                    return pos
            return None

        left = search(min(frame_w - 1, fx - gap_x), max(-1, fx2 - max_w - 1), -1, line_col)
        right = search(max(0, fx2 + gap_x), min(frame_w, fx + max_w), 1, line_col)
        top = search(min(frame_h - 1, fy - gap_y), max(-1, fy2 - max_h - 1), -1, line_row)
        bottom = search(max(0, fy2 + gap_y), min(frame_h, fy + max_h), 1, line_row)

        # A side with no border is acceptable only if the frame edge is close (corner webcam).
        if left is None and fx <= fw * 0.6:
            left = 0
        if right is None and frame_w - fx2 <= fw * 0.6:
            right = frame_w - 1
        if top is None and fy <= fh * 0.6:
            top = 0
        if bottom is None and frame_h - fy2 <= fh * 0.6:
            bottom = frame_h - 1
        if left is None or right is None or top is None or bottom is None:
            return None

        x, y = int(left), int(top)
        w, h = int(right - left + 1), int(bottom - top + 1)
        if w <= fw or h <= fh or w > max_w or h > max_h:
            return None
        return x, y, w, h

    # ------------------------------------------------------------------
    # B4: profiles
    # ------------------------------------------------------------------

    def _profiles_path(self) -> Path:
        configured = self.config.get("facecam.profiles_path", "") if self.config is not None else ""
        if configured:
            return Path(str(configured))
        cache_dir = self.config.get("general.cache_dir", "./.cache") if self.config is not None else "./.cache"
        return Path(str(cache_dir)) / "facecam_profiles.json"

    def _load_profiles(self) -> dict[str, Any]:
        path = self._profiles_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_profile(self, streamer: str, region: FacecamRegion, frame_w: int, frame_h: int) -> None:
        path = self._profiles_path()
        profiles = self._load_profiles()
        profiles[streamer] = [
            round(region.x / frame_w, 5),
            round(region.y / frame_h, 5),
            round(region.w / frame_w, 5),
            round(region.h / frame_h, 5),
        ]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Could not save facecam profile to %s: %s", path, exc)

    @staticmethod
    def _norm_to_region(norm: Any, frame_w: int, frame_h: int, confidence: float) -> FacecamRegion | None:
        try:
            nx, ny, nw, nh = (float(v) for v in norm)
        except (TypeError, ValueError):
            return None
        x, y = max(0, int(round(nx * frame_w))), max(0, int(round(ny * frame_h)))
        w, h = int(round(nw * frame_w)), int(round(nh * frame_h))
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w <= 0 or h <= 0:
            return None
        return FacecamRegion(x=x, y=y, w=w, h=h, confidence=confidence)

    def _face_ratio_inside(self, frames: list[np.ndarray], region: FacecamRegion) -> float:
        """Fraction of frames with a face whose centre lies inside ``region``."""
        if not frames:
            return 0.0
        hits = 0
        for f in frames:
            for (fx, fy, fw, fh) in self._detect_faces(f):
                cx, cy = fx + fw / 2.0, fy + fh / 2.0
                if region.x <= cx <= region.x + region.w and region.y <= cy <= region.y + region.h:
                    hits += 1
                    break
        return hits / len(frames)

    # ------------------------------------------------------------------
    # Public API (C5)
    # ------------------------------------------------------------------

    def detect(
        self,
        video_path: Path | str | list[Path | str],
        num_samples: int | None = None,
        *,
        streamer: str | None = None,
    ) -> FacecamRegion | None:
        """Detect the facecam region.

        Order: ``facecam.overrides[streamer]`` -> stored profile (re-validated, face in >= 40% of
        samples) -> fresh whole-frame detection (saved to the profile store on success).
        """
        if isinstance(video_path, (list, tuple)):
            paths = [Path(p) for p in video_path]
        else:
            paths = [Path(video_path)]

        requested = self.num_samples if num_samples is None else num_samples
        if requested <= 0:
            return None

        frames = self._sample_frames(paths, requested)
        if not frames:
            return None
        frame_h, frame_w = frames[0].shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        if streamer:
            overrides = self.config.get("facecam.overrides", {}) if self.config is not None else {}
            if isinstance(overrides, dict) and streamer in overrides:
                region = self._norm_to_region(overrides[streamer], frame_w, frame_h, 1.0)
                if region is not None:
                    logger.info("Using facecam override for %s: %s", streamer, region)
                    return region

            stored = self._load_profiles().get(streamer)
            if stored is not None:
                region = self._norm_to_region(stored, frame_w, frame_h, 1.0)
                if region is not None:
                    ratio = self._face_ratio_inside(frames, region)
                    if ratio >= 0.4:
                        region.confidence = round(ratio, 2)
                        logger.info("Using stored facecam profile for %s (%.0f%% present).", streamer, ratio * 100)
                        return region
                    logger.info("Stored facecam profile for %s failed re-validation (%.0f%%).", streamer, ratio * 100)

        region = self._detect_fresh(frames, frame_w, frame_h)
        if region is not None and streamer:
            self._save_profile(streamer, region, frame_w, frame_h)
        return region

    def _detect_fresh(self, frames: list[np.ndarray], frame_w: int, frame_h: int) -> FacecamRegion | None:
        """Whole-frame stable-face search over sampled frames."""
        sample_count = len(frames)
        clusters: list[dict[str, Any]] = []
        tol_x, tol_y = 0.10 * frame_w, 0.10 * frame_h

        for fi, frame in enumerate(frames):
            faces = [b for b in self._detect_faces(frame) if self._is_valid_area(b[2], b[3], frame_w, frame_h)]
            faces.sort(key=lambda b: b[2] * b[3], reverse=True)
            for (fx, fy, fw, fh) in faces:
                cx, cy = fx + fw / 2.0, fy + fh / 2.0
                best: int | None = None
                best_d: float | None = None
                for ci, c in enumerate(clusters):
                    if c["last_frame"] == fi:
                        continue
                    mx = float(np.mean([p[0] for p in c["centers"]]))
                    my = float(np.mean([p[1] for p in c["centers"]]))
                    if abs(cx - mx) <= tol_x and abs(cy - my) <= tol_y:
                        d = abs(cx - mx) + abs(cy - my)
                        if best_d is None or d < best_d:
                            best, best_d = ci, d
                if best is None:
                    clusters.append({"centers": [], "boxes": [], "frames": [], "last_frame": -1})
                    best = len(clusters) - 1
                c = clusters[best]
                c["centers"].append((cx, cy))
                c["boxes"].append((fx, fy, fw, fh))
                c["frames"].append(fi)
                c["last_frame"] = fi

        if not clusters:
            logger.info("No face candidates found for facecam detection.")
            return None

        best_c = max(clusters, key=lambda c: len(c["boxes"]))
        count = len(best_c["boxes"])
        required = sample_count * self.min_detection_ratio
        if count < required:
            logger.info(
                "No consistent facecam (best cluster %d detections, required >= %.1f from %d samples).",
                count, required, sample_count,
            )
            return None

        centers = np.array(best_c["centers"], dtype=float)
        jitter = float(max(np.std(centers[:, 0]) / frame_w, np.std(centers[:, 1]) / frame_h))
        if jitter > self.max_center_jitter:
            logger.info("Facecam candidate rejected: centre jitter %.3f > %.3f.", jitter, self.max_center_jitter)
            return None

        boxes = np.array(best_c["boxes"], dtype=float)
        med = (
            float(np.median(boxes[:, 0])),
            float(np.median(boxes[:, 1])),
            float(np.median(boxes[:, 2])),
            float(np.median(boxes[:, 3])),
        )

        support = [frames[i] for i in best_c["frames"]][:6]
        rect = self._expand_to_webcam_rect(support, med, frame_w, frame_h)
        if rect is None:
            rect = self._calculate_padded_box(med[0], med[1], med[2], med[3], frame_w, frame_h)
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return None

        confidence = count / float(sample_count)
        logger.info(
            "Facecam detected: (%d, %d, %d, %d) confidence %.2f (%d/%d frames).",
            x, y, w, h, confidence, count, sample_count,
        )
        return FacecamRegion(x=x, y=y, w=w, h=h, confidence=round(confidence, 2))
