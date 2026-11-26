"""Writes a ready-to-paste posting file for long-form topic videos."""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any
import urllib.parse

from streamclipper.config import Config
from streamclipper.output.post_writer import AI_DISCLOSURE_REMINDER
from streamclipper.topics.models import ShortPlan, TopicPlan
from streamclipper.topics.planner import plan_dir

logger = logging.getLogger(__name__)

STOPWORDS: frozenset[str] = frozenset({
    # Common English stopwords
    "the", "and", "for", "how", "every", "what", "this", "that", "with",
    "from", "into", "actually", "really", "all", "are", "about", "above",
    "after", "again", "against", "any", "because", "been", "before", "being",
    "below", "between", "both", "but", "can", "could", "did", "does", "doing",
    "down", "during", "each", "few", "further", "had", "has", "have", "having",
    "her", "here", "hers", "herself", "him", "himself", "his", "just", "more",
    "most", "not", "now", "off", "once", "only", "other", "our", "ours", "out",
    "over", "own", "same", "she", "should", "some", "such", "than", "then",
    "there", "these", "they", "through", "too", "under", "until", "very", "was",
    "were", "when", "where", "which", "while", "who", "whom", "why", "will",
    "you", "your", "yours", "its", "itself", "one", "two", "first", "second",
    # Common Hinglish stopwords
    "hai", "bhi", "aur", "yeh", "woh", "kya", "kyu", "kyun", "kaise", "hota",
    "karte", "wali", "wale", "wala", "mein", "par", "se", "ko", "ke", "ki",
    "ka", "kare", "karna", "nahi", "tha", "thi", "the", "ab", "sab",
})


def _format_chapters(chapters: list[tuple[str, str]] | list[str] | str) -> list[str]:
    """Normalize chapters into a list of formatted YouTube chapter strings."""
    if isinstance(chapters, str):
        return [line.strip() for line in chapters.strip().splitlines() if line.strip()]
    lines: list[str] = []
    for item in chapters:
        if isinstance(item, tuple) and len(item) == 2:
            lines.append(f"{item[0]} {item[1]}")
        elif isinstance(item, str) and item.strip():
            lines.append(item.strip())
    return lines


def _format_timestamp_url(url: str, start_seconds: float) -> str:
    """Format URL with &t= timestamp parameter."""
    clean_url = str(url or "").strip()
    t_sec = int(round(start_seconds))
    if not clean_url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(clean_url)
        query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        # Remove existing t param if any, then append new t
        filtered = [(k, v) for k, v in query_params if k != "t"]
        filtered.append(("t", f"{t_sec}s"))
        new_query = urllib.parse.urlencode(filtered)
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))
    except Exception:
        sep = "&" if "?" in clean_url else "?"
        return f"{clean_url}{sep}t={t_sec}s"


def _clean_point_label_for_title(raw_heading: str) -> str:
    """Strip countdown numbers (#1, #2) and verbose colon subtitles from point headings for A/B titles."""
    label = re.sub(r"^\s*#\d+[\s:.\-–—]*", "", str(raw_heading or "")).strip()
    if ":" in label:
        head = label.split(":", 1)[0].strip()
        if head:
            label = head
    words = label.split()
    if len(words) > 6:
        label = " ".join(words[:6])
    return label.strip(" :-–—.,!?")


def generate_long_title_variants(plan: TopicPlan) -> list[str]:
    """Generate 3–5 punchy curiosity-gap title variants for YouTube Studio A/B testing."""
    primary = (plan.video_title or plan.topic_title or "Topic Video").strip()
    game = (plan.game or "GTA 6").strip()
    topic = (plan.topic_title or primary).strip()
    core_topic = topic
    if core_topic.lower().startswith(game.lower()):
        core_topic = core_topic[len(game):].lstrip(":- ").strip()
    if not core_topic:
        core_topic = primary

    candidates: list[str] = []

    # 1. High-intrigue curiosity statement
    candidates.append(f"Rockstar Hid THIS in {game} (And Nobody Noticed)")

    # 2. Reframed from thumbnail or core topic
    if plan.thumbnail_text and plan.thumbnail_text.strip():
        thumb_clean = plan.thumbnail_text.strip().title()
        candidates.append(f"{thumb_clean}: What You Completely Missed")
    else:
        candidates.append(f"The Truth About {core_topic}")

    # 3. Point-based or numbered breakdown
    if plan.points and len(plan.points) >= 2:
        candidates.append(f"{len(plan.points)} Insane Secrets Hidden Inside {game}")
        raw_first_pt = (getattr(plan.points[0], "heading", None) or getattr(plan.points[0], "title", None) or "").strip()
        first_pt = _clean_point_label_for_title(raw_first_pt)
        if first_pt:
            candidates.append(f"Why {first_pt} Changes Everything in {game}")
    else:
        candidates.append(f"We Analyzed {core_topic} — Here's What We Found")

    # 4. Curiosity gap questions
    candidates.append(f"Did Everyone Miss This In {game}?")
    candidates.append(f"The Craziest Detail In {game} That Explains Everything")

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
        f"Wait... Did Rockstar Actually Put This in {game}?",
        f"Everything You Need To Know About {core_topic}",
        f"This Changes Everything We Know About {game}",
    ]
    for fb in fallbacks:
        if len(unique) >= 3:
            break
        if fb.lower() not in seen:
            seen.add(fb.lower())
            unique.append(fb)

    return unique[:5]


def generate_short_title_variants(plan: TopicPlan, short: ShortPlan) -> list[str]:
    """Generate 3–5 punchy curiosity-gap title variants for YouTube Shorts A/B testing."""
    primary = (short.hook_title or plan.topic_title or "Topic Short").strip()
    game = (plan.game or "GTA 6").strip()
    callout = (short.callout or "").strip()

    candidates: list[str] = []

    # 1. Callout or hook reaction
    if callout:
        callout_clean = callout.rstrip(".!?- ")
        candidates.append(f"Wait, did you see {callout_clean}?! 😱")
    else:
        candidates.append(f"Wait... did you catch this in {game}?!")

    # 2. Disbelief angle
    candidates.append(f"Rockstar really thought we wouldn't notice this in {game}...")

    # 3. Shortened punchy hook
    words = primary.split()
    if len(words) > 3:
        shortened = " ".join(words[:4]).rstrip(":-,.!")
        candidates.append(f"{shortened} (Wait for the end!)")
    else:
        candidates.append(f"This {game} detail is INSANE (Watch till end)")

    # 4. Curiosity questions
    candidates.append(f"Is this the craziest secret in {game}?")
    candidates.append(f"No way this is actually in {game} 💀")

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
        f"Did you know about this in {game}?",
        "Watch what happens at the end...",
        f"The secret nobody talks about in {game}",
    ]
    for fb in fallbacks:
        if len(unique) >= 3:
            break
        if fb.lower() not in seen:
            seen.add(fb.lower())
            unique.append(fb)

    return unique[:5]


def write_long_post(
    config: Config,
    plan: TopicPlan,
    chapters: list[tuple[str, str]] | list[str] | str,
) -> Path:
    """Write long_post.txt with title, description, chapters, sources, tags, and reminders."""
    target_dir = plan_dir(config, plan.game, plan.topic_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    post_file = target_dir / "long_post.txt"

    title = (plan.video_title or plan.topic_title or "Topic Video").strip()
    chapter_lines = _format_chapters(chapters)

    # Collect source clips and check for official footage
    sources_entries: list[str] = []
    seen_sources: set[tuple[str, int]] = set()
    has_official = False

    for point in plan.points:
        for clip in point.evidence:
            if clip.kind == "official":
                has_official = True
            key = (clip.url, int(round(clip.start)))
            if key in seen_sources or not clip.url:
                continue
            seen_sources.add(key)
            timed_url = _format_timestamp_url(clip.url, clip.start)
            if clip.kind == "official":
                raw_channel = (clip.channel or "").strip().lstrip("@")
                if raw_channel:
                    sources_entries.append(f"- Official footage ({raw_channel}): {timed_url}")
                else:
                    sources_entries.append(f"- Official {plan.game} footage: {timed_url}")
            else:
                raw_channel = (clip.channel or "").strip().lstrip("@")
                channel_str = f"@{raw_channel}" if raw_channel else "@Unknown"
                sources_entries.append(f"- {channel_str}: {timed_url}")

    # Build description lines
    desc_body = plan.description.strip() if plan.description else ""
    desc_parts: list[str] = []
    if desc_body:
        desc_parts.append(desc_body)
        desc_parts.append("")

    if chapter_lines:
        desc_parts.append("TIMESTAMPS")
        desc_parts.extend(chapter_lines)
        desc_parts.append("")

    desc_parts.append("SOURCES")
    if has_official:
        official_names: list[str] = []
        for point in plan.points:
            for clip in point.evidence:
                if clip.kind == "official":
                    raw_ch = (clip.channel or "").strip().lstrip("@")
                    if raw_ch and raw_ch not in official_names:
                        official_names.append(raw_ch)
        if official_names:
            desc_parts.append(f"Official footage: {', '.join(official_names)}")
        else:
            desc_parts.append(f"Official {plan.game} footage")
    desc_parts.extend(sources_entries)
    desc_parts.append("")

    # Tags / Hashtags: game tag + up to 3 meaningful topic tags (cap 4)
    max_hashtags = int(config.get("metadata.hashtag_count", 4) or 4)
    game_tag = "".join(ch for ch in plan.game if ch.isalnum())
    tags_list: list[str] = [f"#{game_tag}"] if game_tag else []
    seen_tags_lower: set[str] = {t.lower() for t in tags_list}

    for word in (plan.topic_title or "").split():
        if len(tags_list) >= max_hashtags:
            break
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if len(cleaned) >= 3 and cleaned.lower() not in STOPWORDS:
            candidate = f"#{cleaned}"
            if candidate.lower() not in seen_tags_lower:
                tags_list.append(candidate)
                seen_tags_lower.add(candidate.lower())

    if len(tags_list) < max_hashtags and "#gaming" not in seen_tags_lower:
        tags_list.append("#Gaming")

    title_variants = generate_long_title_variants(plan)
    lines = [
        "TITLE",
        title,
        "",
        "TITLE VARIANTS (A/B TEST)",
        *[f"- {v}" for v in title_variants],
        "",
        "DESCRIPTION",
        *desc_parts,
        "HASHTAGS",
        " ".join(tags_list),
        "",
        "REMINDERS",
        f"- {AI_DISCLOSURE_REMINDER}",
        "- Upload the thumbnail.jpg",
        "",
    ]

    post_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote long-form topic post file to %s", post_file)
    return post_file


def write_short_post(
    config: Config,
    plan: TopicPlan,
    short: ShortPlan,
    video_path: Path,
) -> Path:
    """Write <name>_post.txt for a 9:16 topic Short (teaser or standalone).

    Title, description with credits (channel + URL + timestamp), hashtags,
    AI-voice disclosure reminder, and for teasers the Related Video reminder.
    Standalone posts do not mention the long video.
    """
    video_path = Path(video_path)
    post_file = video_path.with_name(f"{video_path.stem}_post.txt")
    post_file.parent.mkdir(parents=True, exist_ok=True)

    title = (short.hook_title or plan.topic_title or "Topic Short").strip()
    is_teaser = short.slot == "teaser" or short.kind == "teaser"

    # Collect source clips for the points included in this Short
    sources_entries: list[str] = []
    seen_sources: set[tuple[str, int]] = set()
    has_official = False

    target_point_indexes = set(short.point_indexes)
    if not target_point_indexes:
        for pt in plan.points:
            if pt.evidence:
                target_point_indexes = {pt.index}
                break

    for point in plan.points:
        if target_point_indexes and point.index not in target_point_indexes:
            continue
        for clip in point.evidence:
            if clip.kind == "official":
                has_official = True
            key = (clip.url, int(round(clip.start)))
            if key in seen_sources or not clip.url:
                continue
            seen_sources.add(key)
            timed_url = _format_timestamp_url(clip.url, clip.start)
            if clip.kind == "official":
                raw_channel = (clip.channel or "").strip().lstrip("@")
                if raw_channel:
                    sources_entries.append(f"- Official footage ({raw_channel}): {timed_url}")
                else:
                    sources_entries.append(f"- Official {plan.game} footage: {timed_url}")
            else:
                raw_channel = (clip.channel or "").strip().lstrip("@")
                channel_str = f"@{raw_channel}" if raw_channel else "@Unknown"
                sources_entries.append(f"- {channel_str}: {timed_url}")

    # Build description lines
    desc_parts: list[str] = []
    if is_teaser:
        desc_parts.append(f"{title} — full breakdown on the channel!")
    else:
        desc_parts.append(title)
    desc_parts.append("")

    if sources_entries:
        desc_parts.append("SOURCES")
        if has_official:
            official_names: list[str] = []
            for point in plan.points:
                if target_point_indexes and point.index not in target_point_indexes:
                    continue
                for clip in point.evidence:
                    if clip.kind == "official":
                        raw_ch = (clip.channel or "").strip().lstrip("@")
                        if raw_ch and raw_ch not in official_names:
                            official_names.append(raw_ch)
            if official_names:
                desc_parts.append(f"Official footage: {', '.join(official_names)}")
            else:
                desc_parts.append(f"Official {plan.game} footage")
        desc_parts.extend(sources_entries)
        desc_parts.append("")

    # Tags / Hashtags: #Shorts + game tag + up to 2 meaningful topic tags (cap 4)
    max_hashtags = int(config.get("metadata.hashtag_count", 4) or 4)
    tags_list: list[str] = ["#Shorts"]
    seen_tags_lower: set[str] = {"#shorts"}

    game_tag = "".join(ch for ch in plan.game if ch.isalnum())
    if game_tag and len(tags_list) < max_hashtags:
        candidate_game = f"#{game_tag}"
        if candidate_game.lower() not in seen_tags_lower:
            tags_list.append(candidate_game)
            seen_tags_lower.add(candidate_game.lower())

    for word in (plan.topic_title or "").split():
        if len(tags_list) >= max_hashtags:
            break
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if len(cleaned) >= 3 and cleaned.lower() not in STOPWORDS:
            candidate = f"#{cleaned}"
            if candidate.lower() not in seen_tags_lower:
                tags_list.append(candidate)
                seen_tags_lower.add(candidate.lower())

    if len(tags_list) < max_hashtags and "#gaming" not in seen_tags_lower:
        tags_list.append("#Gaming")

    # Reminders
    reminders: list[str] = [
        f"- {AI_DISCLOSURE_REMINDER}",
    ]
    if is_teaser:
        reminders.append("- Set YouTube's Related Video to the long video")

    title_variants = generate_short_title_variants(plan, short)
    lines = [
        "TITLE",
        title,
        "",
        "TITLE VARIANTS (A/B TEST)",
        *[f"- {v}" for v in title_variants],
        "",
        "DESCRIPTION",
        *desc_parts,
        "HASHTAGS",
        " ".join(tags_list),
        "",
        "REMINDERS",
        *reminders,
        "",
    ]

    post_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote topic short post file to %s", post_file)
    return post_file

