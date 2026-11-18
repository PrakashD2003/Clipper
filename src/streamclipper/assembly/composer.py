"""9:16 Video Composer using MoviePy v2 for YouTube Shorts assembly."""

import logging
import math
from pathlib import Path
from typing import Any

from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips

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
            w_start = float(w.start)
            w_end = float(w.end)
            if w_end > b_start and w_start < b_end:
                s = max(b_start, w_start)
                e = min(b_end, w_end)
                if e > s:
                    shifted_start = round(s - b_start + cum_offset, 3)
                    shifted_end = round(e - b_start + cum_offset, 3)
                    remapped_words.append(
                        WordTimestamp(
                            text=w.text,
                            start=shifted_start,
                            end=shifted_end,
                            confidence=getattr(w, "confidence", 0.0),
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
    ) -> CompositeVideoClip:
        """Create a 9:16 vertical CompositeVideoClip using MoviePy v2 methods."""
        opened_by_us = isinstance(video_path, (str, Path))
        clip = VideoFileClip(str(video_path)) if opened_by_us else video_path
        try:
            result = (
                self._compose_with_facecam(clip, facecam)
                if facecam is not None
                else self._compose_gameplay_only(clip)
            )
            if peak_offset is not None and self.punch_in_zoom > 1.0:
                result = self.apply_spike_zoom(result, peak_offset)
        except Exception:
            if opened_by_us:
                try:
                    clip.close()
                except Exception:
                    pass
            raise
        result._source_clip = clip  # ensure Windows file handles can be closed cleanly
        result._source_clips = [clip]
        return result

    @staticmethod
    def _crop_to_aspect(
        clip: Any,
        x1: int | float,
        y1: int | float,
        x2: int | float,
        y2: int | float,
        target_ratio: float,
    ) -> Any:
        """Crop a region to match target_ratio (width/height) around its center without stretching."""
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

        new_h = max(1, int(round(reg_w / target_ratio)))
        cy = (y1 + y2) // 2
        ny1 = max(0, cy - new_h // 2)
        ny2 = ny1 + new_h
        if ny2 > clip.h:
            ny2 = clip.h
            ny1 = max(0, ny2 - new_h)
        return clip.cropped(x1=x1, y1=ny1, x2=x2, y2=ny2)

    def _compose_with_facecam(
        self, clip: Any, facecam: FacecamRegion
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

        game_clip = (
            game_cropped.resized((self.width, game_h))
            .with_position(("center", face_h))
        )

        comp = CompositeVideoClip(
            [game_clip, face_clip], size=(self.width, self.height)
        )
        if clip.audio is not None:
            comp = comp.with_audio(clip.audio)
        return comp

    def _compose_gameplay_only(self, clip: Any) -> CompositeVideoClip:
        """Layout B: center-crop gameplay to 9:16."""
        target_ratio = self.width / self.height
        if clip.w / clip.h >= target_ratio:
            crop_w = min(clip.w, int(clip.h * target_ratio))
            x1 = (clip.w - crop_w) // 2
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
            return self.compose(segment_paths[0], facecam, peak_offset=peak_offset)

        all_source_clips: list[Any] = []
        opened_clips: list[Any] = []
        beat_clips: list[CompositeVideoClip] = []

        try:
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

                composed_beat = (
                    self._compose_with_facecam(trimmed, facecam)
                    if facecam is not None
                    else self._compose_gameplay_only(trimmed)
                )
                beat_clips.append(composed_beat)

            result = concatenate_videoclips(beat_clips, method="compose")
            if peak_offset is not None and self.punch_in_zoom > 1.0:
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
