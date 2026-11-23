"""5-minute context window extractor with interleaved transcript + live chat timeline."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..models import TranscriptionResult
from ..scoring.scorer import Scorer

logger = logging.getLogger(__name__)


@dataclass
class ChatBucket:
    """A 5-second chat velocity bucket with top reactions."""
    start: float
    end: float
    message_count: int
    messages_per_second: float
    is_burst: bool  # True if rate >= 3x the window average
    top_messages: list[str] = field(default_factory=list)  # Up to 5 representative messages
    top_emojis: list[str] = field(default_factory=list)


@dataclass
class TimelineSegment:
    """One chronological slice of the interleaved timeline."""
    start: float
    end: float
    streamer_speech: str  # What the streamer said (or "(silent)" / "(cutscene music)")
    chat_bucket: ChatBucket | None = None


@dataclass
class ContextWindow:
    """Complete 5-minute context window around a proposed timestamp."""
    center_time: float
    window_start: float
    window_end: float
    segments: list[TimelineSegment] = field(default_factory=list)
    total_chat_messages: int = 0
    avg_chat_rate: float = 0.0
    burst_timestamps: list[float] = field(default_factory=list)  # timestamps of chat bursts
    contact_sheets: list[Path] = field(default_factory=list)  # Sheet 1 & Sheet 2 paths
    sheet_timestamps: list[list[float]] = field(default_factory=list)

    def to_llm_text(self) -> str:
        """Format as a human/LLM-readable chronological timeline."""
        bursts_str = (
            ", ".join(f"{t:.0f}s" for t in self.burst_timestamps)
            if self.burst_timestamps
            else "none"
        )
        lines = [
            f"=== STREAM CONTEXT WINDOW [{self.window_start:.1f}s - {self.window_end:.1f}s] ===",
            f"Center: {self.center_time:.1f}s | Chat: {self.total_chat_messages} msgs "
            f"(avg {self.avg_chat_rate:.1f}/s) | Bursts at: {bursts_str}",
        ]
        if self.contact_sheets:
            sheets_info = []
            for idx, sheet_item in enumerate(self.contact_sheets, start=1):
                sheet_p = Path(sheet_item)
                ts_list = (
                    self.sheet_timestamps[idx - 1]
                    if self.sheet_timestamps and idx - 1 < len(self.sheet_timestamps)
                    else []
                )
                ts_summary = (
                    f"{ts_list[0]:.1f}s to {ts_list[-1]:.1f}s ({len(ts_list)} tiles)"
                    if ts_list
                    else ""
                )
                sheets_info.append(
                    f"Sheet {idx} ({sheet_p.name}): {ts_summary}".rstrip(" :")
                )
            if sheets_info:
                lines.append("Visual Contact Sheets: " + " | ".join(sheets_info))
            else:
                lines.append(f"Visual Contact Sheets: {len(self.contact_sheets)} sheets attached")
        lines.append("")
        for seg in self.segments:
            time_label = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
            lines.append(time_label)
            lines.append(f'  Streamer: "{seg.streamer_speech}"')
            if seg.chat_bucket:
                cb = seg.chat_bucket
                burst_tag = " [BURST]" if cb.is_burst else ""
                lines.append(
                    f"  Live Chat ({cb.messages_per_second:.1f} msgs/s{burst_tag}): "
                    f"{', '.join(repr(m) for m in cb.top_messages[:5])}"
                )
            else:
                lines.append("  Live Chat: (no messages)")
            lines.append("")
        return "\n".join(lines).strip()


BUCKET_SECONDS = 5.0
WINDOW_BEFORE = 120.0  # 2 minutes before center (Sheet 1)
WINDOW_AFTER = 60.0    # 1 minute after center (Sheet 2)


def extract_context_window(
    transcript: TranscriptionResult,
    chat_csv_path: Path | str | None,
    center_time: float,
    *,
    window_before: float = WINDOW_BEFORE,
    window_after: float = WINDOW_AFTER,
    bucket_seconds: float = BUCKET_SECONDS,
    url: str = "",
    temp_dir: Path | str | None = None,
    extract_vision: bool = True,
    contact_sheets: list[Path | str] | None = None,
    sheet_timestamps: list[list[float]] | None = None,
) -> ContextWindow:
    """Extract a context window around center_time with interleaved transcript + chat + visual sheets.

    Parameters:
        transcript: Full stream transcription with word-level timestamps.
        chat_csv_path: Path to the {video_id}_chat.csv file (or None if no chat).
        center_time: The proposed highlight timestamp (seconds from stream start).
        window_before: Seconds of context before center_time (default: 120).
        window_after: Seconds of context after center_time (default: 60).
        bucket_seconds: Chat velocity bucket size (default: 5.0).

    Returns:
        ContextWindow with interleaved timeline segments.
    """
    win_start = max(0.0, float(center_time) - float(window_before))
    win_end = float(center_time) + float(window_after)

    # Load chat data if available
    chat_df: pd.DataFrame | None = None
    if chat_csv_path is not None:
        p = Path(chat_csv_path)
        if p.is_file():
            try:
                raw_df = pd.read_csv(p)
                if "offset_seconds" in raw_df.columns:
                    raw_df["offset_seconds"] = pd.to_numeric(
                        raw_df["offset_seconds"], errors="coerce"
                    )
                elif "offset_ms" in raw_df.columns:
                    raw_df["offset_seconds"] = pd.to_numeric(
                        raw_df["offset_ms"], errors="coerce"
                    ) / 1000.0
                else:
                    raw_df["offset_seconds"] = pd.Series(dtype=float)

                raw_df = raw_df.dropna(subset=["offset_seconds"])
                chat_df = raw_df[
                    (raw_df["offset_seconds"] >= win_start)
                    & (raw_df["offset_seconds"] <= win_end)
                ]
            except Exception as exc:
                logger.warning("Failed to load chat CSV %s: %s", p, exc)
                chat_df = None

    total_chat = len(chat_df) if chat_df is not None else 0
    duration = max(0.0, win_end - win_start)
    avg_rate = total_chat / duration if duration > 0 else 0.0

    num_buckets = max(1, math.ceil(duration / bucket_seconds)) if duration > 0 else 0

    # Build chat buckets
    buckets: dict[int, ChatBucket] = {}
    if chat_df is not None and not chat_df.empty and num_buckets > 0:
        for _, row in chat_df.iterrows():
            t = float(row["offset_seconds"])
            raw_idx = int((t - win_start) / bucket_seconds)
            idx = max(0, min(num_buckets - 1, raw_idx))
            b_start = win_start + idx * bucket_seconds
            b_end = min(win_end, b_start + bucket_seconds)
            if idx not in buckets:
                buckets[idx] = ChatBucket(
                    start=b_start,
                    end=b_end,
                    message_count=0,
                    messages_per_second=0.0,
                    is_burst=False,
                    top_messages=[],
                    top_emojis=[],
                )
            buckets[idx].message_count += 1
            msg = str(row.get("message", "")).strip()
            if msg and len(buckets[idx].top_messages) < 5:
                buckets[idx].top_messages.append(msg)

        for b in buckets.values():
            dur = max(0.1, b.end - b.start)
            b.messages_per_second = b.message_count / dur
            b.is_burst = (
                avg_rate > 0
                and b.messages_per_second >= max(0.5, avg_rate * 3.0)
                and b.message_count >= 2
            )

    burst_timestamps = sorted(
        b.start for b in buckets.values() if b.is_burst
    )

    # Build timeline segments (bucket-aligned)
    segments: list[TimelineSegment] = []
    for i in range(num_buckets):
        seg_start = win_start + i * bucket_seconds
        seg_end = min(seg_start + bucket_seconds, win_end)
        if seg_end <= seg_start:
            continue

        speech = Scorer._extract_transcript_for_range(transcript, seg_start, seg_end)
        if not speech:
            speech = "(silent)"

        chat_bucket = buckets.get(i)

        # Only include segments with speech or chat activity to keep output concise
        if speech != "(silent)" or (chat_bucket and chat_bucket.message_count > 0):
            segments.append(
                TimelineSegment(
                    start=seg_start,
                    end=seg_end,
                    streamer_speech=speech,
                    chat_bucket=chat_bucket,
                )
            )

    resolved_sheets: list[Path] = []
    resolved_timestamps: list[list[float]] = []

    if contact_sheets is not None:
        resolved_sheets = [Path(p) for p in contact_sheets]
        resolved_timestamps = list(sheet_timestamps or [])
    elif url and extract_vision:
        try:
            from ..library.frames import extract_moment_contact_sheets

            t_dir = Path(temp_dir) if temp_dir is not None else Path("./temp")
            sheets = extract_moment_contact_sheets(
                url=url,
                center_time=center_time,
                temp_dir=t_dir,
                window_before=window_before,
                window_after=window_after,
            )
            if sheets:
                resolved_sheets = [s.path for s in sheets]
                resolved_timestamps = [s.timestamps for s in sheets]
            else:
                logger.warning(
                    "Contact sheet extraction returned empty for %s at %.1fs; falling back to text+chat",
                    url,
                    center_time,
                )
        except Exception as exc:
            logger.warning(
                "Contact sheet extraction failed for %s at %.1fs: %s; falling back to text+chat",
                url,
                center_time,
                exc,
            )

    return ContextWindow(
        center_time=center_time,
        window_start=win_start,
        window_end=win_end,
        segments=segments,
        total_chat_messages=total_chat,
        avg_chat_rate=avg_rate,
        burst_timestamps=burst_timestamps,
        contact_sheets=resolved_sheets,
        sheet_timestamps=resolved_timestamps,
    )
