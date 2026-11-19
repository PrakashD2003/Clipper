"""Writes a ready-to-paste posting file next to each exported clip."""
from __future__ import annotations

from pathlib import Path

from ..models import ClipMetadata, StreamInfo

AI_DISCLOSURE_REMINDER = (
    "AI voiceover: YouTube requires the 'Altered or synthetic content' label when the voice "
    "is cloned from a real person or could pass as a real person; for an obviously synthetic "
    "narrator it's optional (ticking it is always safe). Never clone the streamer's voice."
)
MUSIC_UNKNOWN_REMINDER = (
    "Music check inconclusive (streamer talks over the audio) - listen for background "
    "music before posting."
)
POSTING_CADENCE_REMINDER = "Post at most 2–3 Shorts per day"


def write_post_file(video_path: Path, metadata: ClipMetadata, stream_info: StreamInfo) -> Path:
    """Write `<clip_stem>_post.txt` with title, description, hashtags and reminders."""
    video = Path(video_path)
    streamer = metadata.streamer or stream_info.streamer or "the streamer"
    source_url = metadata.source_url or stream_info.url
    tags = " ".join(
        t if str(t).startswith("#") else f"#{t}"
        for t in (metadata.hashtags or [])
        if str(t).strip()
    )
    description_lines = [metadata.caption.strip()] if metadata.caption.strip() else []
    description_lines.append(f"Original stream by {streamer}: {source_url}")
    lines = [
        "TITLE",
        metadata.hook_title,
        "",
        "DESCRIPTION",
        *description_lines,
        "",
        "HASHTAGS",
        tags,
        "",
        "REMINDERS",
        f"- {AI_DISCLOSURE_REMINDER}",
        f"- {POSTING_CADENCE_REMINDER}",
        *(
            [f"- {MUSIC_UNKNOWN_REMINDER}"]
            if str(getattr(metadata, "music_risk", "") or "").upper() == "UNKNOWN"
            else []
        ),
        "",
    ]
    out = video.with_name(f"{video.stem}_post.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
