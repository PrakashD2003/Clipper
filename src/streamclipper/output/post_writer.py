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


def generate_clip_title_variants(metadata: ClipMetadata, stream_info: StreamInfo) -> list[str]:
    """Generate 3–5 punchy curiosity-gap title variants for YouTube Shorts A/B testing."""
    primary = (metadata.hook_title or stream_info.title or "Insane Clip").strip()
    streamer = (metadata.streamer or stream_info.streamer or "").strip()
    game = (metadata.game or stream_info.game or "GTA 6").strip()
    callout = (getattr(metadata, "callout", None) or getattr(metadata, "breakdown_callout", None) or "").strip()
    moment_type = (metadata.moment_type or "").strip().lower()

    candidates: list[str] = []

    # 1. Streamer / Game disbelief angle
    if streamer:
        candidates.append(f"{streamer} was NOT ready for this...")
        candidates.append(f"{streamer}'s reaction says it all 😭")
    else:
        candidates.append(f"Nobody was ready for this in {game}...")
        candidates.append(f"Wait... did this actually just happen in {game}?!")

    # 2. Callout / moment reframe
    if callout:
        callout_clean = callout.rstrip(".!?- ")
        candidates.append(f"{callout_clean} (Watch till end!)")
    elif moment_type:
        candidates.append(f"The most {moment_type} {game} moment ever")

    # 3. Shortened punchy hook from primary
    words = primary.split()
    if len(words) > 3:
        shortened = " ".join(words[:4]).rstrip(":-,.!")
        candidates.append(f"Wait for it... {shortened}")
    else:
        candidates.append(f"How is this even possible in {game}?!")

    # 4. Curiosity gap questions / statements
    candidates.append("You won't believe what happened next...")
    candidates.append(f"This is why everyone is talking about {game}")

    unique: list[str] = []
    seen: set[str] = {primary.lower()}
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean.lower() not in seen:
            seen.add(c_clean.lower())
            unique.append(c_clean)
        if len(unique) == 5:
            break

    fallbacks = [
        "Watch until the very end...",
        f"Did anyone else catch this in {game}?",
        f"This shouldn't be possible in {game}",
    ]
    for fb in fallbacks:
        if len(unique) >= 3:
            break
        if fb.lower() not in seen:
            seen.add(fb.lower())
            unique.append(fb)

    return unique[:5]


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
    title_variants = generate_clip_title_variants(metadata, stream_info)
    lines = [
        "TITLE",
        metadata.hook_title,
        "",
        "TITLE VARIANTS (A/B TEST)",
        *[f"- {v}" for v in title_variants],
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
