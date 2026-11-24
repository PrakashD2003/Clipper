"""9:16 Video Composer using MoviePy v2 for YouTube Shorts assembly."""

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, VideoFileClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip
from PIL import Image, ImageFilter

from ..config import Config
from ..models import FacecamRegion, WordTimestamp
from .layout import Box, LayoutGeometry, compute_layout

logger = logging.getLogger(__name__)


def remap_timeline_for_beats(
    words: list[WordTimestamp],
    chat_burst: list[dict[str, Any]],
    peak_time: float,
    beats: list[dict[str, Any]],
) -> tuple[list[WordTimestamp], list[dict[str, Any]], float | None]:
    """Remap stream-absolute word timestamps, chat_burst messages, and peak_time
    onto the contiguous stitched timeline (starting at 0.0) produced by multi-beat cuts.

    Words and chat messages falling inside cut dead air are dropped; timestamps inside
    beat k are shifted by cumulative_offset - beat_start.
    """
    if not beats:
        return list(words), list(chat_burst), (peak_time if peak_time > 0.0 else None)

    valid_beats: list[tuple[float, float]] = []
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        b_start = float(beat.get("start", 0.0))
        b_end = float(beat.get("end", 0.0))
        if b_end > b_start:
            valid_beats.append((b_start, b_end))

    remapped_words: list[WordTimestamp] = []
    remapped_chat: list[dict[str, Any]] = []
    remapped_peak: float | None = None

    cum_offset = 0.0
    for i, (b_start, b_end) in enumerate(valid_beats):
        b_dur = b_end - b_start
        is_last_beat = i == len(valid_beats) - 1

        # Remap words overlapping this beat
        for w in words:
            if isinstance(w, dict):
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))
                w_text = str(w.get("text") or w.get("word") or "")
                w_conf = float(w.get("confidence") or w.get("score") or 0.0)
            else:
                w_start = float(w.start)
                w_end = float(w.end)
                w_text = str(w.text or "")
                w_conf = getattr(w, "confidence", 0.0)

            if w_end > b_start and w_start < b_end:
                s = max(b_start, w_start)
                e = min(b_end, w_end)
                if e > s:
                    shifted_start = round(s - b_start + cum_offset, 3)
                    shifted_end = round(e - b_start + cum_offset, 3)
                    remapped_words.append(
                        WordTimestamp(
                            text=w_text,
                            start=shifted_start,
                            end=shifted_end,
                            confidence=w_conf,
                        )
                    )

        # Remap chat messages
        for msg in chat_burst:
            if not isinstance(msg, dict):
                continue
            raw_time = msg.get("offset_seconds")
            if raw_time is None:
                raw_time = msg.get("timestamp")
            if raw_time is None:
                raw_time = msg.get("time")
            if raw_time is None:
                raw_time = msg.get("offset")
            if raw_time is None or isinstance(raw_time, bool):
                continue
            try:
                msg_time = float(raw_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(msg_time):
                continue

            is_in_beat = (
                (b_start <= msg_time <= b_end)
                if is_last_beat
                else (b_start <= msg_time < b_end)
            )
            if is_in_beat:
                new_time = round(msg_time - b_start + cum_offset, 3)
                new_msg = dict(msg)
                if "offset_seconds" in new_msg or (
                    "timestamp" not in new_msg
                    and "time" not in new_msg
                    and "offset" not in new_msg
                ):
                    new_msg["offset_seconds"] = new_time
                if "timestamp" in new_msg:
                    new_msg["timestamp"] = new_time
                if "time" in new_msg:
                    new_msg["time"] = new_time
                if "offset" in new_msg:
                    new_msg["offset"] = new_time
                remapped_chat.append(new_msg)

        # Remap peak_time if inside this beat
        if peak_time > 0.0 and remapped_peak is None:
            is_peak_in_beat = (
                (b_start <= peak_time <= b_end)
                if is_last_beat
                else (b_start <= peak_time < b_end)
            )
            if is_peak_in_beat:
                remapped_peak = round(peak_time - b_start + cum_offset, 3)

        cum_offset += b_dur

    return remapped_words, remapped_chat, remapped_peak


def remap_timestamps(
    words: list[WordTimestamp] | list[dict[str, Any]],
    sub_clips: list[dict[str, Any]] | None = None,
    chat_burst: list[dict[str, Any]] | None = None,
    peak_time: float = 0.0,
    *,
    beats: list[dict[str, Any]] | None = None,
) -> tuple[list[WordTimestamp], list[dict[str, Any]], float | None]:
    """Helper mapping word timestamps, chat messages, and peak_time onto concatenated story timeline.

    Alias/wrapper for remap_timeline_for_beats().
    """
    effective_beats = beats if beats is not None else (sub_clips or [])
    return remap_timeline_for_beats(
        words=words,  # type: ignore[arg-type]
        chat_burst=chat_burst or [],
        peak_time=peak_time,
        beats=effective_beats,
    )


class VideoComposer:
    def __init__(self, config: Config):
        self._config = config
        res = config.get("video.resolution", [1080, 1920])
        self.width = int(res[0])
        self.height = int(res[1])
        self.fps = int(config.get("video.fps", 30))
        self.punch_in_zoom = float(config.get("transformation.punch_in_zoom", 1.20))
        self.zoom_duration = float(
            config.get("transformation.zoom_duration_seconds", 3.5)
        )
        raw_fit = str(config.get("video.game_fit", "cinema_split") or "cinema_split").strip().lower()
        self.game_fit = raw_fit if raw_fit in {"full", "balanced", "full_bleed", "cinema_split"} else "cinema_split"
        try:
            self.blur_fill_game_scale = float(config.get("video.blur_fill_game_scale", 1.15))
        except (TypeError, ValueError):
            self.blur_fill_game_scale = 1.15

    def _resolve_layout(
        self, layout_kind: str | None, facecam: FacecamRegion | None
    ) -> LayoutGeometry:
        kind = layout_kind or ("stacked" if facecam is not None else "blur_fill")
        if facecam is None and kind in {"stacked", "reaction"}:
            kind = "blur_fill"  # nothing to put in the facecam box
        return compute_layout(
            kind,
            self.game_fit,
            self.width,
            self.height,
            blur_game_scale=self.blur_fill_game_scale,
        )

    def compose(
        self,
        video_path: Path | str | Any,
        facecam: FacecamRegion | None,
        peak_offset: float | None = None,
        trim_range: tuple[float, float] | None = None,
        face_center_x: float | None = None,
        layout_kind: str | None = None,
    ) -> CompositeVideoClip:
        """Create a 9:16 vertical CompositeVideoClip using MoviePy v2 methods, optionally trimming download buffer.

        ``layout_kind`` is "stacked" | "blur_fill" | "reaction"; None picks stacked with a facecam
        else blur_fill. ``face_center_x`` is unused (kept for API compatibility).
        """
        opened_by_us = isinstance(video_path, (str, Path))
        clip = VideoFileClip(str(video_path)) if opened_by_us else video_path
        all_sources = [clip]
        applied_trim_start = 0.0
        try:
            effective_peak_offset = peak_offset
            if trim_range is not None:
                t_start = max(0.0, float(trim_range[0]))
                clip_dur = float(getattr(clip, "duration", 0.0) or 0.0)
                t_end = (
                    min(clip_dur, float(trim_range[1]))
                    if clip_dur > 0.0
                    else float(trim_range[1])
                )
                if t_end > t_start:
                    clip = clip.subclipped(t_start, t_end)
                    applied_trim_start = t_start
                    if effective_peak_offset is not None:
                        effective_peak_offset = max(0.0, effective_peak_offset - t_start)
            layout = self._resolve_layout(layout_kind, facecam)
            result = self._compose_layout(clip, facecam, layout)
            if effective_peak_offset is not None and self.punch_in_zoom > 1.0:
                result = self.apply_spike_zoom(result, effective_peak_offset)
        except Exception:
            if opened_by_us:
                for c in all_sources:
                    try:
                        c.close()
                    except Exception as close_exc:
                        logger.debug("Failed to close source clip: %s", close_exc)
            raise
        result._source_clip = all_sources[0]  # ensure Windows file handles can be closed cleanly
        result._source_clips = all_sources
        # Seconds actually trimmed from the source head (0.0 when trim_range was skipped),
        # so callers can map stream timestamps onto the composed timeline exactly.
        result._trim_start = applied_trim_start
        return result

    @staticmethod
    def _crop_to_aspect(
        clip: Any,
        x1: int | float,
        y1: int | float,
        x2: int | float,
        y2: int | float,
        target_ratio: float,
        expand_width: bool = False,
    ) -> Any:
        """Crop or expand a region to match target_ratio (width/height) around its center without stretching."""
        x1 = max(0, min(int(x1), max(0, clip.w - 1)))
        y1 = max(0, min(int(y1), max(0, clip.h - 1)))
        x2 = max(x1 + 1, min(int(x2), clip.w))
        y2 = max(y1 + 1, min(int(y2), clip.h))

        reg_w = max(1, x2 - x1)
        reg_h = max(1, y2 - y1)
        current_ratio = reg_w / reg_h

        if current_ratio > target_ratio:
            new_w = max(1, int(round(reg_h * target_ratio)))
            cx = (x1 + x2) // 2
            nx1 = max(0, cx - new_w // 2)
            nx2 = nx1 + new_w
            if nx2 > clip.w:
                nx2 = clip.w
                nx1 = max(0, nx2 - new_w)
            return clip.cropped(x1=nx1, y1=y1, x2=nx2, y2=y2)

        if expand_width:
            # Expand width horizontally to preserve full face height (no forehead/chin crop)
            new_w = max(1, int(round(reg_h * target_ratio)))
            if new_w <= clip.w:
                cx = (x1 + x2) // 2
                nx1 = max(0, cx - new_w // 2)
                nx2 = nx1 + new_w
                if nx2 > clip.w:
                    nx2 = clip.w
                    nx1 = max(0, nx2 - new_w)
                return clip.cropped(x1=nx1, y1=y1, x2=nx2, y2=y2)
            else:
                new_h = max(1, int(round(clip.w / target_ratio)))
                cy = (y1 + y2) // 2
                ny1 = max(0, cy - new_h // 2)
                ny2 = min(clip.h, ny1 + new_h)
                return clip.cropped(x1=0, y1=ny1, x2=clip.w, y2=ny2)

        new_h = max(1, int(round(reg_w / target_ratio)))
        cy = (y1 + y2) // 2
        ny1 = max(0, cy - new_h // 2)
        ny2 = ny1 + new_h
        if ny2 > clip.h:
            ny2 = clip.h
            ny1 = max(0, ny2 - new_h)
        return clip.cropped(x1=x1, y1=ny1, x2=x2, y2=ny2)

    def _game_layer(self, clip: Any, box: Box, centre_crop: bool = False) -> Any:
        """Scale the source frame into ``box``: centre crop without distortion (full_bleed),
        4:3 centre crop (balanced), or 16:9 full frame with optional centre crop (full).
        """
        src = clip
        if self.game_fit == "full_bleed":
            target_ratio = box.w / box.h
            if clip.w / clip.h > target_ratio:
                crop_w = max(1, min(clip.w, int(round(clip.h * target_ratio))))
                x1 = (clip.w - crop_w) // 2
                src = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
            else:
                crop_h = max(1, min(clip.h, int(round(clip.w / target_ratio))))
                y1 = (clip.h - crop_h) // 2
                src = clip.cropped(x1=0, y1=y1, x2=clip.w, y2=y1 + crop_h)
        elif self.game_fit in {"balanced", "cinema_split"}:
            target_ratio = box.w / box.h
            if clip.w / clip.h > target_ratio:
                crop_w = min(clip.w, int(round(clip.h * target_ratio)))
                x1 = (clip.w - crop_w) // 2
                src = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
            else:
                crop_h = min(clip.h, int(round(clip.w / target_ratio)))
                y1 = (clip.h - crop_h) // 2
                src = clip.cropped(x1=0, y1=y1, x2=clip.w, y2=y1 + crop_h)
        elif centre_crop:
            target_ratio = box.w / box.h
            if clip.w / clip.h > target_ratio + 0.01:
                crop_w = max(1, min(clip.w, int(round(clip.h * target_ratio))))
                x1 = (clip.w - crop_w) // 2
                src = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
        return src.resized((box.w, box.h)).with_position((box.x, box.y))

    def _blur_background(self, clip: Any) -> Any:
        """Game frame scaled to cover the canvas, Gaussian-blurred and darkened (~0.5)."""
        target_ratio = self.width / self.height
        if clip.w / clip.h > target_ratio:
            crop_w = max(1, min(clip.w, int(round(clip.h * target_ratio))))
            x1 = (clip.w - crop_w) // 2
            cover = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
        else:
            crop_h = max(1, min(clip.h, int(round(clip.w / target_ratio))))
            y1 = (clip.h - crop_h) // 2
            cover = clip.cropped(x1=0, y1=y1, x2=clip.w, y2=y1 + crop_h)

        # Blur at quarter resolution (cheap), then upscale to the full canvas.
        small = cover.resized((max(1, self.width // 4), max(1, self.height // 4)))

        def _blur_dark(frame: np.ndarray) -> np.ndarray:
            img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=8)
            )
            return (np.asarray(img, dtype=np.float32) * 0.5).astype(np.uint8)

        return (
            small.image_transform(_blur_dark)
            .resized((self.width, self.height))
            .with_position((0, 0))
        )

    def _compose_layout(
        self,
        clip: Any,
        facecam: FacecamRegion | None,
        layout: LayoutGeometry,
        peak_offset: float | None = None,
    ) -> CompositeVideoClip:
        """Compose one clip into ``layout`` (stacked / reaction / blur_fill) on a black canvas."""
        layers: list[Any] = []
        if (
            layout.kind in {"blur_fill", "reaction"}
            or layout.facecam_box is None
            or facecam is None
            or (layout.kind == "stacked" and self.game_fit in {"full", "balanced", "cinema_split"})
        ):
            layers.append(self._blur_background(clip))
        if layout.facecam_box is not None and facecam is not None:
            fb = layout.facecam_box
            face_clip = (
                self._crop_to_aspect(
                    clip,
                    facecam.x,
                    facecam.y,
                    facecam.x + facecam.w,
                    facecam.y + facecam.h,
                    target_ratio=fb.w / fb.h,
                    expand_width=True,
                )
                .resized((fb.w, fb.h))
                .with_position((fb.x, fb.y))
            )
            layers.append(face_clip)
        layers.append(
            self._game_layer(clip, layout.game_box, centre_crop=layout.kind == "blur_fill")
        )

        comp = CompositeVideoClip(layers, size=(self.width, self.height))
        if clip.audio is not None:
            comp = comp.with_audio(clip.audio)
        comp._layout = layout
        if peak_offset is not None and self.punch_in_zoom > 1.0:
            comp = self.apply_spike_zoom(comp, peak_offset)
        return comp

    def _compose_with_facecam(
        self,
        clip: Any,
        facecam: FacecamRegion,
        peak_offset: float | None = None,
    ) -> CompositeVideoClip:
        """Stacked layout: facecam on top, full game below (kept as a thin wrapper)."""
        return self._compose_layout(
            clip, facecam, self._resolve_layout("stacked", facecam), peak_offset=peak_offset
        )

    @staticmethod
    def _detect_face_center_x(clip: Any) -> float | None:
        """Sample a few frames to detect face horizontal center for face-aware Layout B cropping."""
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return None

            dur = getattr(clip, "duration", None)
            if dur and dur > 0.5:
                sample_times = [0.1 * dur, 0.5 * dur, 0.9 * dur]
            else:
                sample_times = [0.0]

            detected_centers: list[float] = []
            for t in sample_times:
                try:
                    frame = clip.get_frame(t)
                except Exception as frame_exc:
                    logger.debug("Face-detect frame read at %.2fs failed: %s", t, frame_exc)
                    continue
                if frame is None or len(frame.shape) < 2:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    detected_centers.append(float(fx + fw / 2.0))

            if detected_centers:
                return float(np.median(detected_centers))
            return None
        except Exception as detect_exc:
            logger.warning("Face detection failed; using centered crop: %s", detect_exc)
            return None

    def _compose_gameplay_only(
        self, clip: Any, face_center_x: float | None = None
    ) -> CompositeVideoClip:
        """blur_fill layout (``face_center_x`` is unused; kept for API compatibility)."""
        return self._compose_layout(clip, None, self._resolve_layout("blur_fill", None))

    def compose_story_beats(
        self,
        segment_paths: list[Path | str | Any],
        facecam: FacecamRegion | None,
        beat_trim_ranges: list[tuple[float, float]] | None = None,
        peak_offset: float | None = None,
        face_center_x: float | None = None,
        layout_kind: str | None = None,
    ) -> CompositeVideoClip:
        """Trim any download buffer padding per beat, compose each beat into 9:16,
        stitch via concatenate_videoclips(..., method='compose'), optionally apply spike zoom,
        and track all underlying VideoFileClips in result._source_clips.

        CRITICAL (MoviePy 2.2.1): Clamp each (t_start, t_end) to [0.0, beat_clip.duration]
        and guard t_end > t_start before calling beat_clip.subclipped(t_start, t_end),
        because MoviePy 2.2.1 raises ValueError if end_time > clip.duration.
        """
        if not segment_paths:
            raise ValueError("No segment paths provided to compose_story_beats")

        if len(segment_paths) == 1:
            return self.compose(
                segment_paths[0],
                facecam,
                peak_offset=peak_offset,
                face_center_x=face_center_x,
                layout_kind=layout_kind,
            )

        layout = self._resolve_layout(layout_kind, facecam)
        all_source_clips: list[Any] = []
        opened_clips: list[Any] = []
        beat_clips: list[CompositeVideoClip] = []
        beat_durations: list[float] = []

        try:
            cum_beat_offset = 0.0
            for i, item in enumerate(segment_paths):
                if isinstance(item, (str, Path)):
                    raw_clip = VideoFileClip(str(item))
                    opened_clips.append(raw_clip)
                else:
                    raw_clip = item
                all_source_clips.append(raw_clip)

                if beat_trim_ranges and i < len(beat_trim_ranges):
                    t_start = max(0.0, float(beat_trim_ranges[i][0]))
                    t_end = min(float(raw_clip.duration), float(beat_trim_ranges[i][1]))
                    if t_end > t_start:
                        trimmed = raw_clip.subclipped(t_start, t_end)
                    else:
                        logger.warning(
                            "Beat %d trim range invalid after clamping (start=%.3f, end=%.3f, "
                            "duration=%.3f); using untrimmed clip",
                            i,
                            t_start,
                            t_end,
                            float(raw_clip.duration),
                        )
                        trimmed = raw_clip
                else:
                    trimmed = raw_clip

                trimmed_dur = float(trimmed.duration) if getattr(trimmed, "duration", None) else 0.0
                composed_beat = self._compose_layout(trimmed, facecam, layout)
                beat_clips.append(composed_beat)
                beat_durations.append(trimmed_dur)
                cum_beat_offset += trimmed_dur

            result = concatenate_videoclips(beat_clips, method="compose")
            result._layout = layout
            if peak_offset is not None and self.punch_in_zoom > 1.0:
                result = self.apply_spike_zoom(result, peak_offset)
        except Exception:
            for opened in opened_clips:
                try:
                    opened.close()
                except Exception as close_exc:
                    logger.debug("Failed to close opened beat clip: %s", close_exc)
            raise

        result._source_clips = all_source_clips
        result._source_clip = all_source_clips[0]
        result._beat_durations = beat_durations
        return result

    def apply_spike_zoom(
        self, clip: CompositeVideoClip, peak_offset: float
    ) -> CompositeVideoClip:
        """Apply a center punch-in zoom window around peak_offset, preserving audio and _source_clips."""
        if self.punch_in_zoom <= 1.0 or peak_offset is None or peak_offset < 0.0:
            return clip
        if clip.duration is None or clip.duration <= 0.0:
            return clip

        z_start = max(0.0, float(peak_offset) - 0.5)
        z_end = min(float(clip.duration), z_start + self.zoom_duration)
        if z_end <= z_start or z_start >= clip.duration:
            return clip

        layout = getattr(clip, "_layout", None)
        w = getattr(clip, "w", self.width) or self.width
        h = getattr(clip, "h", self.height) or self.height
        if layout is not None:
            gb = layout.game_box
            zw, zh = gb.w, gb.h
        else:
            zw, zh = w, h
        crop_w = max(1, int(zw / self.punch_in_zoom))
        crop_h = max(1, int(zh / self.punch_in_zoom))
        x1 = (zw - crop_w) // 2
        y1 = (zh - crop_h) // 2

        slices: list[Any] = []
        if z_start > 0.0:
            slices.append(clip.subclipped(0.0, z_start))

        if layout is not None:
            # Zoom inside game_box only; facecam, hooks band and caption band stay locked.
            gb = layout.game_box
            base = clip.subclipped(z_start, z_end)
            inner = (
                base.cropped(x1=gb.x, y1=gb.y, x2=gb.x + gb.w, y2=gb.y + gb.h)
                .cropped(x1=x1, y1=y1, x2=x1 + crop_w, y2=y1 + crop_h)
                .resized((gb.w, gb.h))
                .with_position((gb.x, gb.y))
            )
            zoomed_slice = CompositeVideoClip([base, inner], size=(w, h))
            if base.audio is not None:
                zoomed_slice = zoomed_slice.with_audio(base.audio)
            zoomed_slice._layout = layout
        else:
            zoomed_slice = (
                clip.subclipped(z_start, z_end)
                .cropped(x1=x1, y1=y1, x2=x1 + crop_w, y2=y1 + crop_h)
                .resized((w, h))
            )
        slices.append(zoomed_slice)

        if z_end < clip.duration:
            slices.append(clip.subclipped(z_end, clip.duration))

        result = concatenate_videoclips(slices, method="compose")
        if clip.audio is not None:
            result = result.with_audio(clip.audio)

        if hasattr(clip, "_source_clip"):
            result._source_clip = clip._source_clip
        if hasattr(clip, "_source_clips"):
            result._source_clips = clip._source_clips
        if layout is not None:
            result._layout = layout

        return result

    def compose_instant_replay(
        self,
        main_clip: Any,
        peak_offset: float | None,
        replay_window: float = 5.0,
        slow_mo: float = 0.5,
        badge_clip: Any | None = None,
    ) -> tuple[CompositeVideoClip, float]:
        """Assemble instant replay ending on main_clip with replay badge."""
        return compose_instant_replay(
            main_clip,
            peak_offset=peak_offset,
            replay_window=replay_window,
            slow_mo=slow_mo,
            badge_clip=badge_clip,
            config=getattr(self, "_config", None),
        )

    def compose_freeze_breakdown(
        self,
        main_clip: Any,
        peak_offset: float | None,
        freeze_duration: float = 4.0,
        zoom: float = 1.35,
    ) -> CompositeVideoClip:
        """Freeze frame at peak_offset with zoom callout and silent audio."""
        return compose_freeze_breakdown(
            main_clip,
            peak_offset=peak_offset,
            freeze_duration=freeze_duration,
            zoom=zoom,
        )

    def compose_zoom_spotlight_breakdown(
        self,
        main_clip: Any,
        source_video: str | Path | None = None,
        target_time_offset: float | None = None,
        roi: list[float] | None = None,
        duration: float = 5.7,
        zoom_scale: float = 2.2,
        ease_in_seconds: float = 1.2,
        ease_out_seconds: float = 1.0,
        badge_text: str = "🔍 EVIDENCE BREAKDOWN",
        fullscreen_takeover: bool = True,
    ) -> tuple[CompositeVideoClip, float]:
        """Assemble a high-resolution Ken Burns zoom spotlight breakdown ending on main_clip."""
        return compose_zoom_spotlight_breakdown(
            main_clip=main_clip,
            source_video=source_video,
            target_time_offset=target_time_offset,
            roi=roi,
            duration=duration,
            zoom_scale=zoom_scale,
            ease_in_seconds=ease_in_seconds,
            ease_out_seconds=ease_out_seconds,
            badge_text=badge_text,
            fullscreen_takeover=fullscreen_takeover,
            config=getattr(self, "_config", None),
        )


def compose_instant_replay(
    main_clip: Any,
    peak_offset: float | None,
    replay_window: float = 5.0,
    slow_mo: float = 0.5,
    badge_clip: Any | None = None,
    config: Config | None = None,
) -> tuple[CompositeVideoClip, float]:
    """Assemble an instant replay ending on main_clip at slow_mo speed with replay badge.

    Args:
        main_clip: The base composed vertical clip (e.g. 1080x1920).
        peak_offset: Timestamp of the climax/spike in seconds on main_clip.
                     If None or <= 0, defaults to mid-point of main_clip.
        replay_window: Duration in seconds around peak to slice (default 5.0s).
        slow_mo: Speed scale factor (default 0.5 for 2x slow motion).
        badge_clip: Optional ImageClip for replay badge. If None, default
                    '🔴 INSTANT REPLAY' badge is created and positioned at ('center', 140).

    Returns:
        (combined_clip, replay_start_time) where replay_start_time == main_clip.duration.
    """
    clip_dur = float(getattr(main_clip, "duration", 0.0) or 0.0)
    if clip_dur <= 0.0:
        return main_clip, 0.0

    peak = float(peak_offset) if (peak_offset is not None and float(peak_offset) > 0.0) else (clip_dur / 2.0)
    peak = max(0.0, min(clip_dur, peak))

    half_w = float(replay_window) / 2.0
    r_start = max(0.0, peak - half_w)
    r_end = min(clip_dur, peak + half_w)

    # If bounded by clip edge, maintain replay_window duration if clip is long enough
    if (r_end - r_start) < replay_window and clip_dur >= replay_window:
        if r_start == 0.0:
            r_end = min(clip_dur, float(replay_window))
        elif r_end == clip_dur:
            r_start = max(0.0, clip_dur - float(replay_window))

    if r_end <= r_start:
        r_start = 0.0
        r_end = clip_dur

    replay_slice = main_clip.subclipped(r_start, r_end)
    scale_factor = float(slow_mo) if slow_mo and float(slow_mo) > 0.0 else 0.5
    replay_slow = replay_slice.with_speed_scaled(scale_factor)

    w = getattr(main_clip, "w", 1080) or 1080
    h = getattr(main_clip, "h", 1920) or 1920

    if badge_clip is None:
        from .hooks import create_replay_badge

        badge = create_replay_badge(
            width=w,
            height=h,
            duration=replay_slow.duration,
            config=config,
            layout=getattr(main_clip, "_layout", None),
        )
    else:
        badge = badge_clip
        if hasattr(badge, "with_duration"):
            badge = badge.with_duration(replay_slow.duration)

    replay_comp = CompositeVideoClip([replay_slow, badge], size=(w, h))
    if replay_slow.audio is not None:
        replay_comp = replay_comp.with_audio(replay_slow.audio)

    replay_start_time = float(main_clip.duration)
    combined = concatenate_videoclips([main_clip, replay_comp], method="compose")

    # Propagate source clips for Windows resource cleanup in Exporter
    # Dedupe by identity: MoviePy 2's Clip.__eq__ decodes and compares every frame.
    sources: list[Any] = []
    seen_ids: set[int] = set()
    for c in (main_clip, replay_comp):
        candidates = list(getattr(c, "_source_clips", None) or [])
        sc = getattr(c, "_source_clip", None)
        if sc is not None:
            candidates.append(sc)
        for s in candidates:
            if id(s) not in seen_ids:
                seen_ids.add(id(s))
                sources.append(s)
    if not sources:
        sources = [main_clip]

    combined._source_clip = sources[0]
    combined._source_clips = sources
    if getattr(main_clip, "_layout", None) is not None:
        combined._layout = main_clip._layout
        replay_comp._layout = main_clip._layout

    return combined, replay_start_time


def compose_freeze_breakdown(
    main_clip: Any,
    peak_offset: float | None,
    freeze_duration: float = 4.0,
    zoom: float = 1.35,
) -> CompositeVideoClip:
    """Freeze frame at peak_offset, apply zoom callout, and re-insert into timeline.

    Args:
        main_clip: The base vertical video clip.
        peak_offset: Timestamp in seconds to freeze.
        freeze_duration: How long to hold the frozen frame (default 4.0s).
        zoom: Center zoom scale factor on the frozen callout (default 1.35).

    Returns:
        CompositeVideoClip with structure [before, freeze_zoomed, after] and silent audio
        attached to freeze to avoid MoviePy audio desync.
    """
    clip_dur = float(getattr(main_clip, "duration", 0.0) or 0.0)
    if clip_dur <= 0.0:
        return main_clip

    peak = float(peak_offset) if (peak_offset is not None and float(peak_offset) >= 0.0) else (clip_dur / 2.0)
    peak = max(0.0, min(clip_dur, peak))
    # Safe frame time clamped slightly before duration to prevent index overflow
    frame_time = min(peak, max(0.0, clip_dur - 0.04))

    frame = main_clip.get_frame(frame_time)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    fh, fw = frame.shape[:2]
    w = getattr(main_clip, "w", fw) or fw
    h = getattr(main_clip, "h", fh) or fh

    zoom_factor = float(zoom) if zoom and float(zoom) > 1.0 else 1.0
    if zoom_factor > 1.0:
        crop_w = max(1, int(fw / zoom_factor))
        crop_h = max(1, int(fh / zoom_factor))
        x1 = (fw - crop_w) // 2
        y1 = (fh - crop_h) // 2
        cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        pil_img = Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
        zoomed_frame = np.array(pil_img, dtype=np.uint8)
    else:
        if (fw, fh) != (w, h):
            pil_img = Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS)
            zoomed_frame = np.array(pil_img, dtype=np.uint8)
        else:
            zoomed_frame = frame

    freeze_dur = max(0.1, float(freeze_duration))
    freeze_clip = ImageClip(zoomed_frame).with_duration(freeze_dur)

    # Attach silent audio array to prevent MoviePy audio desync on concatenation
    audio_fps = 44100
    if main_clip.audio is not None:
        audio_fps = int(getattr(main_clip.audio, "fps", 44100) or 44100)
    n_samples = int(math.ceil(freeze_dur * audio_fps))
    silent_audio = AudioArrayClip(np.zeros((n_samples, 2), dtype=np.float32), fps=audio_fps)
    freeze_clip = freeze_clip.with_audio(silent_audio)

    slices: list[Any] = []
    if peak > 0.01:
        slices.append(main_clip.subclipped(0.0, peak))
    slices.append(freeze_clip)
    if peak < (clip_dur - 0.01):
        slices.append(main_clip.subclipped(peak, clip_dur))

    combined = concatenate_videoclips(slices, method="compose")
    # Expose where the freeze was inserted so callers can shift captions/chat after it
    # and align the breakdown voiceover with the frozen callout.
    combined._freeze_start = peak if peak > 0.01 else 0.0
    combined._freeze_duration = freeze_dur

    # Propagate source clips for Windows resource cleanup in Exporter
    sources = getattr(main_clip, "_source_clips", None)
    if sources:
        combined._source_clips = list(sources)
        combined._source_clip = getattr(main_clip, "_source_clip", sources[0])
    else:
        combined._source_clip = main_clip
        combined._source_clips = [main_clip]
    if getattr(main_clip, "_layout", None) is not None:
        combined._layout = main_clip._layout

    return combined


def compose_zoom_spotlight_breakdown(
    main_clip: Any,
    source_video: str | Path | None = None,
    target_time_offset: float | None = None,
    roi: list[float] | None = None,
    duration: float = 5.7,
    zoom_scale: float = 2.2,
    ease_in_seconds: float = 1.2,
    ease_out_seconds: float = 1.0,
    badge_text: str = "🔍 EVIDENCE BREAKDOWN",
    fullscreen_takeover: bool = True,
    config: Config | None = None,
) -> tuple[CompositeVideoClip, float]:
    """Assemble a high-resolution Ken Burns zoom spotlight breakdown ending on main_clip.

    Args:
        main_clip: The base composed vertical clip (e.g. 1080x1920).
        source_video: Path to uncompressed 1080p source video (for raw frame extraction).
        target_time_offset: Stream/clip second where the screen/document appears.
        roi: Normalized Region of Interest [ymin, xmin, ymax, xmax] in [0.0, 1.0].
        duration: Total duration of the spotlight clip (default 5.7s).
        zoom_scale: Zoom magnification factor into ROI (default 2.2x).
        ease_in_seconds: Duration of sinusoidal zoom-in ease (default 1.2s).
        ease_out_seconds: Duration of sinusoidal zoom-out ease (default 1.0s).
        badge_text: Text for on-screen breakdown badge (default "🔍 EVIDENCE BREAKDOWN").
        fullscreen_takeover: Whether to expand to full 9:16 vertical canvas (default True).
        config: Optional Config instance.

    Returns:
        (combined_clip, replay_start_time) where replay_start_time == main_clip.duration.
    """
    clip_dur = float(getattr(main_clip, "duration", 0.0) or 0.0)
    w = getattr(main_clip, "w", 1080) or 1080
    h = getattr(main_clip, "h", 1920) or 1920
    if clip_dur <= 0.0:
        return main_clip, 0.0

    duration = float(duration) if duration and float(duration) > 0.0 else 5.7

    raw_frame: np.ndarray | None = None
    target_time = (
        float(target_time_offset)
        if target_time_offset is not None and float(target_time_offset) >= 0.0
        else min(3.5, 0.25 * clip_dur)
    )

    if source_video is not None:
        src_path = Path(source_video)
        if src_path.exists():
            try:
                cap = cv2.VideoCapture(str(src_path))
                if cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target_time * 1000.0))
                        ret, bgr = cap.read()
                        if (not ret or bgr is None) and target_time > 0:
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            if total_frames > 1:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
                                ret, bgr = cap.read()
                        if ret and bgr is not None:
                            raw_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    finally:
                        cap.release()
            except Exception as e:
                logger.debug("Failed to extract frame from %s at %.2fs: %s", src_path, target_time, e)

    if raw_frame is None:
        frame_time = min(max(0.0, target_time), max(0.0, clip_dur - 0.04))
        try:
            raw_frame = main_clip.get_frame(frame_time)
        except Exception as e:
            logger.warning("Failed to extract frame from main_clip at %.2fs: %s", frame_time, e)
            raw_frame = np.zeros((h, w, 3), dtype=np.uint8)

    if raw_frame.dtype != np.uint8:
        raw_frame = np.clip(raw_frame, 0, 255).astype(np.uint8)

    src_h, src_w = raw_frame.shape[:2]

    # ROI normalization: [ymin, xmin, ymax, xmax] in 0.0 - 1.0
    valid_roi = [0.0, 0.0, 1.0, 1.0]
    if roi and len(roi) == 4:
        try:
            valid_roi = [float(x) for x in roi]
        except (ValueError, TypeError):
            valid_roi = [0.0, 0.0, 1.0, 1.0]

    ymin = max(0.0, min(1.0, min(valid_roi[0], valid_roi[2])))
    ymax = max(0.0, min(1.0, max(valid_roi[0], valid_roi[2])))
    xmin = max(0.0, min(1.0, min(valid_roi[1], valid_roi[3])))
    xmax = max(0.0, min(1.0, max(valid_roi[1], valid_roi[3])))

    # Handle degenerate point / invalid ROI with a reasonable window
    if (ymax - ymin) < 0.01:
        ymin = max(0.0, ymin - 0.15)
        ymax = min(1.0, ymax + 0.15)
    if (xmax - xmin) < 0.01:
        xmin = max(0.0, xmin - 0.15)
        xmax = min(1.0, xmax + 0.15)

    roi_cx_norm = (xmin + xmax) / 2.0
    roi_cy_norm = (ymin + ymax) / 2.0

    target_aspect = float(w) / float(h)
    src_aspect = float(src_w) / float(src_h)

    # Initial full-bleed crop box fitting the target vertical aspect ratio
    if src_aspect >= target_aspect:
        # Source is wider than target (e.g. 16:9 source in 9:16 vertical)
        crop_h_0 = float(src_h)
        crop_w_0 = crop_h_0 * target_aspect
    else:
        # Source is taller than target
        crop_w_0 = float(src_w)
        crop_h_0 = crop_w_0 / target_aspect

    # Target zoom scale
    z_target = max(1.0, float(zoom_scale)) if zoom_scale is not None and float(zoom_scale) > 0 else 2.2

    # Target crop dimensions at full zoom (maintaining target_aspect)
    crop_h_1 = max(10.0, crop_h_0 / z_target)
    crop_w_1 = crop_h_1 * target_aspect

    # Center coordinates in source pixels
    cx_0 = float(src_w) / 2.0
    cy_0 = float(src_h) / 2.0
    cx_target = roi_cx_norm * float(src_w)
    cy_target = roi_cy_norm * float(src_h)

    t_in = (
        max(0.0, min(float(ease_in_seconds), duration / 2.0))
        if ease_in_seconds is not None
        else min(1.2, duration / 2.0)
    )
    t_out = (
        max(0.0, min(float(ease_out_seconds), duration - t_in))
        if ease_out_seconds is not None
        else min(1.0, duration - t_in)
    )

    def make_frame(t: float) -> np.ndarray:
        t_clamped = max(0.0, min(duration, float(t)))
        if t_in > 0.0 and t_clamped <= t_in:
            alpha = t_clamped / t_in
            u = (1.0 - math.cos(math.pi * alpha)) / 2.0
        elif t_clamped <= (duration - t_out):
            u = 1.0
        elif t_out > 0.0:
            beta = (t_clamped - (duration - t_out)) / t_out
            u = (1.0 + math.cos(math.pi * beta)) / 2.0
        else:
            u = 1.0

        # Current crop size and center interpolated smoothly
        curr_crop_w = crop_w_0 + (crop_w_1 - crop_w_0) * u
        curr_crop_h = crop_h_0 + (crop_h_1 - crop_h_0) * u
        curr_cx = cx_0 + (cx_target - cx_0) * u
        curr_cy = cy_0 + (cy_target - cy_0) * u

        # Clamp center so crop box stays strictly inside raw_frame boundaries
        curr_cx = max(curr_crop_w / 2.0, min(float(src_w) - curr_crop_w / 2.0, curr_cx))
        curr_cy = max(curr_crop_h / 2.0, min(float(src_h) - curr_crop_h / 2.0, curr_cy))

        x1 = curr_cx - curr_crop_w / 2.0
        y1 = curr_cy - curr_crop_h / 2.0
        x2 = curr_cx + curr_crop_w / 2.0
        y2 = curr_cy + curr_crop_h / 2.0

        src_pts = np.float32([[x1, y1], [x2, y1], [x1, y2]])
        dst_pts = np.float32([[0, 0], [w, 0], [0, h]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(raw_frame, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    spotlight_clip = VideoClip(make_frame, duration=duration)
    spotlight_clip.fps = float(getattr(main_clip, "fps", 30.0) or 30.0)

    from .hooks import create_replay_badge

    badge_clip = create_replay_badge(
        text=badge_text,
        width=w,
        height=h,
        duration=duration,
        config=config,
        layout=None if fullscreen_takeover else getattr(main_clip, "_layout", None),
    )

    spotlight_comp = CompositeVideoClip([spotlight_clip, badge_clip], size=(w, h))

    audio_fps = 44100
    if main_clip.audio is not None:
        audio_fps = int(getattr(main_clip.audio, "fps", 44100) or 44100)
    n_samples = int(math.ceil(duration * audio_fps))
    silent_audio = AudioArrayClip(np.zeros((n_samples, 2), dtype=np.float32), fps=audio_fps)
    spotlight_comp = spotlight_comp.with_audio(silent_audio)

    replay_start_time = float(main_clip.duration)
    combined = concatenate_videoclips([main_clip, spotlight_comp], method="compose")

    sources: list[Any] = []
    seen_ids: set[int] = set()
    for c in (main_clip, spotlight_comp):
        candidates = list(getattr(c, "_source_clips", None) or [])
        sc = getattr(c, "_source_clip", None)
        if sc is not None:
            candidates.append(sc)
        for s in candidates:
            if id(s) not in seen_ids:
                seen_ids.add(id(s))
                sources.append(s)
    if not sources:
        sources = [main_clip]

    combined._source_clip = sources[0]
    combined._source_clips = sources
    combined._breakdown_kind = "zoom_spotlight"
    combined._replay_start = replay_start_time
    combined._spotlight_duration = duration
    if getattr(main_clip, "_layout", None) is not None:
        combined._layout = main_clip._layout

    return combined, replay_start_time
