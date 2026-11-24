"""Writes a ready-to-paste posting file next to each exported clip."""
from __future__ import annotations

from pathlib import Path

from ..models import ClipMetadata, StreamInfo
from ..narration_style import BANNED_PHRASES

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
    lang = str(getattr(metadata, "narration_language", "english") or "english").strip().lower()

    def _is_banned(title: str) -> bool:
        t_low = title.lower()
        return any(p in t_low for p in BANNED_PHRASES)

    candidates: list[str] = []

    if lang == "hinglish":
        # Hinglish title variants
        if streamer:
            candidates.append(f"{streamer} ko ye bilkul expect nahi tha 😭")
            candidates.append(f"{streamer} ka reaction dekho bas!")
            candidates.append(f"{streamer} ne ye kaise kiya?!")
        else:
            candidates.append(f"{game} mein ye kya ho gaya?!")
            candidates.append(f"Bhai ye {game} mein kaise possible hai?!")
            candidates.append(f"Kisi ko expect nahi tha {game} mein ye hoga...")

        if callout:
            callout_clean = callout.rstrip(".!?- ")
            candidates.append(f"{callout_clean} Dekho!")
        elif moment_type:
            candidates.append(f"{game} ka sabse unexpected play!")

        candidates.append(f"{game} mein ye kisne dekha?")
        candidates.append(f"Aisa scene {game} mein pehle kabhi nahi dekha!")
        candidates.append(f"Bhai ye scene dekh ke hansi nahi rukegi!")
    else:
        # English title variants
        if streamer:
            candidates.append(f"{streamer} was NOT ready for this...")
            candidates.append(f"{streamer}'s reaction says it all 😭")
            candidates.append(f"How did {streamer} pull this off?!")
        else:
            candidates.append(f"Nobody was ready for this in {game}...")
            candidates.append(f"Wait... did this actually just happen in {game}?!")
            candidates.append(f"How is this even possible in {game}?!")

        if callout:
            callout_clean = callout.rstrip(".!?- ")
            candidates.append(f"{callout_clean}?!")
        elif moment_type:
            candidates.append(f"The most unexpected {game} play")

        candidates.append(f"This is why everyone is talking about {game}")
        candidates.append(f"Did anyone else catch this in {game}?")
        candidates.append(f"This shouldn't be possible in {game}")

    unique: list[str] = []
    seen: set[str] = {primary.lower()}
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean.lower() not in seen and not _is_banned(c_clean):
            seen.add(c_clean.lower())
            unique.append(c_clean)
        if len(unique) == 5:
            break

    fallbacks = (
        [
            f"Kya aapne {game} mein ye notice kiya?",
            f"{game} ka sabse zabardast scene!",
            f"{game} ka unexpected twist!",
            f"Bhai ye dekh ke shocked ho jaoge!",
        ]
        if lang == "hinglish"
        else [
            f"Did anyone else catch this in {game}?",
            f"This shouldn't be possible in {game}",
            f"Tell me this didn't just happen in {game}",
        ]
    )
    for fb in fallbacks:
        if len(unique) >= 3:
            break
        if fb.lower() not in seen and not _is_banned(fb):
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
