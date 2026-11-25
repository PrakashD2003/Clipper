"""Plan validation for topic videos.

Enforces all topic video contracts, duration bounds, shape rules,
clip constraints, Shorts separation, and narration style/grounding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex
from streamclipper.narration_style import BANNED_PHRASES, check_narration
from streamclipper.topics.ledger import TopicLedger
from streamclipper.topics.models import TopicPlan

logger = logging.getLogger(__name__)

# Phrases required in teaser outro pointing to the full long-form video
TEASER_POINTER_TERMS: tuple[str, ...] = (
    "full breakdown",
    "full video",
    "long video",
    "full investigation",
    "full episode",
    "poora video",
    "full analysis",
    "on the channel",
    "check out the channel",
    "watch the full",
    "breakdown on the channel",
    "watch full",
    "link in",
    "channel par",
    "related video",
)

# References strictly forbidden in standalone Shorts
STANDALONE_BANNED_REFERENCES: tuple[str, ...] = (
    "full breakdown",
    "full video",
    "long video",
    "on the channel",
    "full episode",
    "poora video",
    "full analysis",
    "channel par",
    "part 2",
)


def validate_plan(config: Config, plan: TopicPlan) -> list[str]:
    """Validate a TopicPlan against all architectural and editorial rules.

    Also calculates and updates plan.estimated_seconds and plan.narration_seconds.

    Returns a list of validation problem strings (empty if valid).
    """
    problems: list[str] = []

    # 1. Basic Metadata
    if not plan.game or not plan.game.strip():
        problems.append("Plan 'game' cannot be empty.")
    if not plan.topic_id or not plan.topic_id.strip():
        problems.append("Plan 'topic_id' cannot be empty.")

    valid_shapes = ("numbered", "countdown", "story")
    if plan.shape not in valid_shapes:
        problems.append(f"Invalid shape '{plan.shape}'; must be one of {valid_shapes}.")

    # 2. Shape Rotation
    if plan.game:
        ledger = TopicLedger(config)
        last_shape = ledger.last_shape(plan.game)
        if last_shape and plan.shape == last_shape:
            problems.append(
                f"Shape '{plan.shape}' was used in the previous video for game '{plan.game}'. Please rotate shape."
            )

    # 3. Points Count and Headings
    num_points = len(plan.points)
    if plan.shape in ("numbered", "countdown"):
        if not (5 <= num_points <= 7):
            problems.append(
                f"{plan.shape.capitalize()} shape requires 5–7 points, found {num_points}."
            )
    elif plan.shape == "story":
        if not (4 <= num_points <= 6):
            problems.append(
                f"Story shape requires 4–6 points, found {num_points}."
            )

    # Headings and Ordering
    if plan.shape == "numbered":
        for p in plan.points:
            if "#" not in p.heading:
                problems.append(f"Point {p.index} heading '{p.heading}' must contain a '#' chapter prefix.")
        for i in range(len(plan.points) - 1):
            if plan.points[i].index >= plan.points[i + 1].index:
                problems.append(
                    f"Numbered shape points must be in ascending index order (found {plan.points[i].index} before {plan.points[i + 1].index})."
                )
                break
    elif plan.shape == "countdown":
        for p in plan.points:
            if "#" not in p.heading:
                problems.append(f"Point {p.index} heading '{p.heading}' must contain a '#' chapter prefix.")
        for i in range(len(plan.points) - 1):
            if plan.points[i].index <= plan.points[i + 1].index:
                problems.append(
                    f"Countdown shape points must be ordered high to low (found {plan.points[i].index} before {plan.points[i + 1].index})."
                )
                break
    elif plan.shape == "story":
        for p in plan.points:
            if "#" in p.heading:
                problems.append(f"Story shape heading '{p.heading}' must not contain '#' chapter prefixes.")

    # 4. Cold Open & Outro
    if not plan.cold_open or not plan.cold_open.strip():
        problems.append("Plan cold_open cannot be empty.")
    if not plan.outro or not plan.outro.strip():
        problems.append("Plan outro cannot be empty.")

    # 5. Evidence Clips & Library Verification
    index = LibraryIndex(config)
    max_clip = float(config.get("topics.max_clip_seconds", 20))
    max_official = float(config.get("topics.max_official_clip_seconds", 10))

    # Cache looked-up sources to avoid redundant DB queries
    source_cache: dict[str, Any] = {}

    def _get_source_safe(sid: str) -> Any:
        if sid not in source_cache:
            try:
                with index._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM sources WHERE source_id = ? OR url = ?;", (sid, sid)
                    ).fetchone()
                    from streamclipper.library.models import LibrarySource
                    source_cache[sid] = LibrarySource.from_dict(dict(row)) if row else None
            except Exception:
                source_cache[sid] = None
        return source_cache[sid]

    temp_base = Path(config.get("general.temp_dir", "./temp"))
    broll_dir = temp_base / "broll"

    def _is_local_broll_clip(clip_obj: Any) -> bool:
        raw_url = str(getattr(clip_obj, "url", "") or "").strip()
        if raw_url and not raw_url.lower().startswith(("http://", "https://")):
            local_cand = raw_url[7:] if raw_url.lower().startswith("file://") else raw_url
            p = Path(local_cand)
            if p.is_file() and p.stat().st_size > 0:
                return True
        sid = str(getattr(clip_obj, "source_id", "") or "").strip()
        if sid and broll_dir.is_dir():
            import re as _re
            safe_sid = _re.sub(r"[^A-Za-z0-9_-]", "_", sid)
            for cand in broll_dir.glob(f"{safe_sid}_*"):
                if cand.is_file() and cand.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov") and cand.stat().st_size > 0:
                    return True
        return False

    for p in plan.points:
        if not (1 <= len(p.evidence) <= 2):
            problems.append(f"Point {p.index} ('{p.heading}') must have 1–2 evidence clips, found {len(p.evidence)}.")

        all_clips = list(p.evidence) + list(p.alternates)
        for clip in all_clips:
            if not clip.source_id and not _is_local_broll_clip(clip):
                problems.append(f"Point {p.index} clip missing source_id.")
                continue

            src = _get_source_safe(clip.source_id) if clip.source_id else None
            if src is None and not _is_local_broll_clip(clip):
                problems.append(f"Point {p.index} clip source '{clip.source_id}' not found in library.")

            if clip.start < 0 or clip.end <= clip.start:
                problems.append(f"Point {p.index} clip '{clip.source_id}' has invalid timestamps ({clip.start}–{clip.end}).")
                continue

            dur = clip.end - clip.start
            is_official = (clip.kind == "official") or (src is not None and getattr(src, "kind", "") == "official")

            if is_official and dur > max_official:
                problems.append(
                    f"Point {p.index} official clip '{clip.source_id}' ({dur:.1f}s) exceeds max official clip duration of {max_official:.1f}s."
                )
            elif dur > max_clip:
                problems.append(
                    f"Point {p.index} evidence clip '{clip.source_id}' ({dur:.1f}s) exceeds max clip duration of {max_clip:.1f}s."
                )

            if src is not None and getattr(src, "duration_seconds", 0) > 0:
                if clip.end > (src.duration_seconds + 1.0):
                    problems.append(
                        f"Point {p.index} clip '{clip.source_id}' end {clip.end:.1f}s exceeds source duration {src.duration_seconds}s."
                    )

    # 6. Estimate Math (Long Video)
    cold_words = len(plan.cold_open.split()) if plan.cold_open else 0
    outro_words = len(plan.outro.split()) if plan.outro else 0
    pts_words = sum(len(p.narration.split()) for p in plan.points)
    total_words = cold_words + pts_words + outro_words

    narration_sec = total_words / 2.7
    chapter_card_sec = (2.5 * len(plan.points)) if plan.shape in ("numbered", "countdown") else 0.0
    evidence_sec = sum(max(0.0, c.end - c.start) for p in plan.points for c in p.evidence)
    total_sec = narration_sec + chapter_card_sec + evidence_sec

    plan.narration_seconds = round(narration_sec, 1)
    plan.estimated_seconds = round(total_sec, 1)

    long_min = float(config.get("topics.long_min_seconds", 480))
    long_max = float(config.get("topics.long_max_seconds", 600))
    if total_sec < long_min:
        problems.append(f"Estimated video duration {total_sec:.1f}s is below minimum {long_min:.1f}s.")
    elif total_sec > long_max:
        problems.append(f"Estimated video duration {total_sec:.1f}s exceeds maximum {long_max:.1f}s.")

    min_ratio = float(config.get("topics.min_narration_ratio", 0.40))
    ratio = (narration_sec / total_sec) if total_sec > 0 else 0.0
    if ratio < min_ratio:
        problems.append(f"Narration share ({ratio:.1%}) is below minimum {min_ratio:.0%}.")

    # 7. Shorts Validation
    if len(plan.shorts) != 3:
        problems.append(f"Plan must contain exactly 3 Shorts, found {len(plan.shorts)}.")

    slots = [s.slot for s in plan.shorts]
    if sorted(slots) != ["short_a", "short_b", "teaser"]:
        problems.append(f"Shorts must have slots ['teaser', 'short_a', 'short_b'], found {slots}.")

    valid_indexes = {p.index for p in plan.points}
    short_min = float(config.get("topics.short_min_seconds", 35))
    short_max = float(config.get("topics.short_max_seconds", 180))

    teaser = next((s for s in plan.shorts if s.slot == "teaser"), None)
    short_a = next((s for s in plan.shorts if s.slot == "short_a"), None)
    short_b = next((s for s in plan.shorts if s.slot == "short_b"), None)

    for s in plan.shorts:
        if not s.point_indexes:
            problems.append(f"Short '{s.slot}' has no assigned point_indexes.")
        for idx in s.point_indexes:
            if idx not in valid_indexes:
                problems.append(f"Short '{s.slot}' references non-existent point index {idx}.")

        if s.slot == "teaser" and s.kind != "teaser":
            problems.append(f"Short 'teaser' must have kind 'teaser' (found '{s.kind}').")
        if s.slot in ("short_a", "short_b") and s.kind != "standalone":
            problems.append(f"Short '{s.slot}' must have kind 'standalone' (found '{s.kind}').")

        # Short duration estimate
        s_words = (
            len(s.narration_intro.split())
            + len(s.callout.split())
            + len(s.narration_outro.split())
        )
        s_narr = s_words / 2.7
        s_clips = [c for p in plan.points if p.index in s.point_indexes for c in p.evidence]
        s_clip_dur = sum(max(0.0, c.end - c.start) for c in s_clips)
        s_dur = s_narr + s_clip_dur

        if s_dur < short_min:
            problems.append(f"Short '{s.slot}' estimated duration {s_dur:.1f}s is below minimum {short_min:.1f}s.")
        elif s_dur > short_max:
            problems.append(f"Short '{s.slot}' estimated duration {s_dur:.1f}s exceeds maximum {short_max:.1f}s.")

    # Teaser Outro Pointer Check
    if teaser:
        t_low = teaser.narration_outro.lower()
        if not any(term in t_low for term in TEASER_POINTER_TERMS):
            problems.append(
                "Teaser outro must end with a pointer to the full video (e.g. 'Full breakdown on the channel')."
            )

    # Standalone Shorts Reference Check
    for s in (short_a, short_b):
        if s is not None:
            combined = f"{s.hook_title} {s.narration_intro} {s.callout} {s.narration_outro}".lower()
            for banned in STANDALONE_BANNED_REFERENCES:
                if banned in combined:
                    problems.append(f"Standalone Short '{s.slot}' must not reference the long video ('{banned}').")
                    break

    # short_a and short_b Overlap Detection
    if short_a is not None and short_b is not None:
        clips_a = [c for p in plan.points if p.index in short_a.point_indexes for c in p.evidence]
        clips_b = [c for p in plan.points if p.index in short_b.point_indexes for c in p.evidence]

        for ca in clips_a:
            for cb in clips_b:
                src_a = _get_source_safe(ca.source_id)
                src_b = _get_source_safe(cb.source_id)
                same_source = (
                    (ca.source_id and ca.source_id == cb.source_id)
                    or (src_a is not None and src_b is not None and src_a.source_id == src_b.source_id)
                    or (ca.url and ca.url == cb.url)
                )
                if same_source:
                    # Check range overlap: max(start) < min(end)
                    overlap_start = max(ca.start, cb.start)
                    overlap_end = min(ca.end, cb.end)
                    if overlap_start < overlap_end:
                        sid_name = (src_a.source_id if src_a else None) or ca.source_id
                        problems.append(
                            f"Shorts 'short_a' and 'short_b' share overlapping footage from source '{sid_name}' ({overlap_start:.1f}s–{overlap_end:.1f}s)."
                        )

    # 8. Narration Style and Grounding Checks
    # Helper to gather snippet text from source segments if available
    def _get_snippets_for_clips(clips: list[Any]) -> str:
        texts: list[str] = []
        for c in clips:
            if c.reason:
                texts.append(c.reason)
            src = _get_source_safe(c.source_id)
            sid = src.source_id if src else c.source_id
            # Fetch snippet or text and chat from segments in index
            try:
                with index._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT text, chat_text FROM segments WHERE source_id = ? AND start <= ? AND end >= ?",
                        (sid, c.end, c.start),
                    ).fetchall()
                    for r in rows:
                        t = str(r["text"] or "").strip()
                        if t:
                            texts.append(t)
                        ct = str(r["chat_text"] or "").strip()
                        if ct:
                            texts.append(ct)
            except Exception:
                pass
        return " ".join(texts)

    if plan.cold_open:
        for pb in check_narration({"narration_intro": plan.cold_open}):
            problems.append(f"Cold open: {pb}")

    if plan.outro:
        for pb in check_narration({"narration_outro": plan.outro}):
            problems.append(f"Outro: {pb}")

    for p in plan.points:
        p_low = p.narration.lower()
        for bp in BANNED_PHRASES:
            if bp in p_low:
                problems.append(f"Point {p.index} narration uses banned phrase '{bp}'.")

        p_snippets = _get_snippets_for_clips(p.evidence)
        for pb in check_narration({"narration_intro": p.narration}, transcript_excerpt=p_snippets):
            clean_pb = pb.replace("narration_intro", "narration")
            problems.append(f"Point {p.index}: {clean_pb}")

    for s in plan.shorts:
        s_snippets = _get_snippets_for_clips([c for p in plan.points if p.index in s.point_indexes for c in p.evidence])
        for pb in check_narration(
            {"narration_intro": s.narration_intro, "callout": s.callout, "narration_outro": s.narration_outro},
            transcript_excerpt=s_snippets,
        ):
            problems.append(f"Short '{s.slot}': {pb}")

    return problems
