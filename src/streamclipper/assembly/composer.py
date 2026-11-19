"""9:16 Video Composer using MoviePy v2 for YouTube Shorts assembly."""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip
from PIL import Image

from ..config import Config
from ..models import FacecamRegion, WordTimestamp

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
        res = config.get("video.resolution", [1080, 1920])
        self.width = int(res[0])
        self.height = int(res[1])
        self.fps = int(config.get("video.fps", 30))
        self.punch_in_zoom = float(config.get("transformation.punch_in_zoom", 1.20))
        self.zoom_duration = float(
            config.get("transformation.zoom_duration_seconds", 3.5)
        )

    def compose(
        self,
        video_path: Path | str | Any,
        facecam: FacecamRegion | None,
        peak_offset: float | None = None,
        trim_range: tuple[float, float] | None = None,
        face_center_x: float | None = None,
    ) -> CompositeVideoClip:
        """Create a 9:16 vertical CompositeVideoClip using MoviePy v2 methods, optionally trimming download buffer."""
        opened_by_us = isinstance(video_path, (str, Path))
        clip = VideoFileClip(str(video_path)) if opened_by_us else video_path
        all_sources = [clip]
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
                    if effective_peak_offset is not None:
                        effective_peak_offset = max(0.0, effective_peak_offset - t_start)
            if facecam is not None:
                result = self._compose_with_facecam(
                    clip, facecam, peak_offset=effective_peak_offset
                )
            else:
                result = self._compose_gameplay_only(clip, face_center_x=face_center_x)
                if effective_peak_offset is not None and self.punch_in_zoom > 1.0:
                    result = self.apply_spike_zoom(result, effective_peak_offset)
        except Exception:
            if opened_by_us:
                for c in all_sources:
                    try:
                        c.close()
                    except Exception:
                        pass
            raise
        result._source_clip = all_sources[0]  # ensure Windows file handles can be closed cleanly
        result._source_clips = all_sources
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

    def _compose_with_facecam(
        self,
        clip: Any,
        facecam: FacecamRegion,
        peak_offset: float | None = None,
    ) -> CompositeVideoClip:
        """Layout A: aspect-correct facecam on top (35%), gameplay below (65%)."""
        face_h = int(self.height * 0.35)
        game_h = self.height - face_h

        face_clip = (
            self._crop_to_aspect(
                clip,
                facecam.x,
                facecam.y,
                facecam.x + facecam.w,
                facecam.y + facecam.h,
                target_ratio=self.width / face_h,
                expand_width=True,
            )
            .resized((self.width, face_h))
            .with_position(("center", 0))
        )

        target_ratio = self.width / game_h
        if clip.w / clip.h >= target_ratio:
            crop_w = min(clip.w, int(clip.h * target_ratio))
            x1 = (clip.w - crop_w) // 2
            game_cropped = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
        else:
            crop_h = min(clip.h, int(clip.w / target_ratio))
            y1 = (clip.h - crop_h) // 2
            game_cropped = clip.cropped(x1=0, y1=y1, x2=clip.w, y2=y1 + crop_h)

        game_clip = game_cropped.resized((self.width, game_h))
        # Lock 35%/65% split line: apply spike zoom inside gameplay clip (bottom 65%)
        # so facecam (top 35%) is never cropped and the split line stays locked at face_h!
        if peak_offset is not None and self.punch_in_zoom > 1.0:
            game_clip = self.apply_spike_zoom(game_clip, peak_offset)

        game_clip = game_clip.with_position(("center", face_h))

        comp = CompositeVideoClip(
            [game_clip, face_clip], size=(self.width, self.height)
        )
        if clip.audio is not None:
            comp = comp.with_audio(clip.audio)
        return comp

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
                except Exception:
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
        except Exception:
            return None

    def _compose_gameplay_only(
        self, clip: Any, face_center_x: float | None = None
    ) -> CompositeVideoClip:
        """Layout B: face-aware crop gameplay to 9:16."""
        target_ratio = self.width / self.height
        if clip.w / clip.h >= target_ratio:
            crop_w = min(clip.w, int(clip.h * target_ratio))
            default_x1 = (clip.w - crop_w) // 2
            x1 = default_x1
            try:
                face_cx = (
                    float(face_center_x)
                    if face_center_x is not None
                    else self._detect_face_center_x(clip)
                )
                if face_cx is not None:
                    x1 = max(0, min(clip.w - crop_w, int(face_cx - crop_w // 2)))
            except Exception as e:
                logger.debug("Face-aware Layout B crop detection skipped: %s", e)
                x1 = default_x1

            cropped = clip.cropped(x1=x1, y1=0, x2=x1 + crop_w, y2=clip.h)
        else:
            crop_h = min(clip.h, int(clip.w / target_ratio))
            y1 = (clip.h - crop_h) // 2
            cropped = clip.cropped(x1=0, y1=y1, x2=clip.w, y2=y1 + crop_h)

        resized = (
            cropped.resized((self.width, self.height))
            .with_position(("center", "center"))
        )
        comp = CompositeVideoClip([resized], size=(self.width, self.height))
        if clip.audio is not None:
            comp = comp.with_audio(clip.audio)
        return comp

    def compose_story_beats(
        self,
        segment_paths: list[Path | str | Any],
        facecam: FacecamRegion | None,
        beat_trim_ranges: list[tuple[float, float]] | None = None,
        peak_offset: float | None = None,
        face_center_x: float | None = None,
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
            )

        all_source_clips: list[Any] = []
        opened_clips: list[Any] = []
        beat_clips: list[CompositeVideoClip] = []

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
                        trimmed = raw_clip
                else:
                    trimmed = raw_clip

                trimmed_dur = float(trimmed.duration) if getattr(trimmed, "duration", None) else 0.0
                beat_peak: float | None = None
                if peak_offset is not None and facecam is not None:
                    is_in_this_beat = (
                        (cum_beat_offset <= peak_offset <= (cum_beat_offset + trimmed_dur))
                        if i == len(segment_paths) - 1
                        else (cum_beat_offset <= peak_offset < (cum_beat_offset + trimmed_dur))
                    )
                    if is_in_this_beat:
                        beat_peak = peak_offset - cum_beat_offset

                composed_beat = (
                    self._compose_with_facecam(trimmed, facecam, peak_offset=beat_peak)
                    if facecam is not None
                    else self._compose_gameplay_only(trimmed, face_center_x=face_center_x)
                )
                beat_clips.append(composed_beat)
                cum_beat_offset += trimmed_dur

            result = concatenate_videoclips(beat_clips, method="compose")
            if peak_offset is not None and self.punch_in_zoom > 1.0 and facecam is None:
                result = self.apply_spike_zoom(result, peak_offset)
        except Exception:
            for opened in opened_clips:
                try:
                    opened.close()
                except Exception:
                    pass
            raise

        result._source_clips = all_source_clips
        result._source_clip = all_source_clips[0]
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

        w = getattr(clip, "w", self.width) or self.width
        h = getattr(clip, "h", self.height) or self.height
        crop_w = max(1, int(w / self.punch_in_zoom))
        crop_h = max(1, int(h / self.punch_in_zoom))
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2

        slices: list[Any] = []
        if z_start > 0.0:
            slices.append(clip.subclipped(0.0, z_start))

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


def compose_instant_replay(
    main_clip: Any,
    peak_offset: float | None,
    replay_window: float = 5.0,
    slow_mo: float = 0.5,
    badge_clip: Any | None = None,
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

        badge = create_replay_badge(width=w, height=h, duration=replay_slow.duration)
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
    sources: list[Any] = []
    for c in (main_clip, replay_comp):
        sc_list = getattr(c, "_source_clips", None)
        if sc_list:
            for s in sc_list:
                if s not in sources:
                    sources.append(s)
        sc = getattr(c, "_source_clip", None)
        if sc and sc not in sources:
            sources.append(sc)
    if not sources:
        sources = [main_clip]

    combined._source_clip = sources[0]
    combined._source_clips = sources

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

    # Propagate source clips for Windows resource cleanup in Exporter
    sources = getattr(main_clip, "_source_clips", None)
    if sources:
        combined._source_clips = list(sources)
        combined._source_clip = getattr(main_clip, "_source_clip", sources[0])
    else:
        combined._source_clip = main_clip
        combined._source_clips = [main_clip]

    return combined
