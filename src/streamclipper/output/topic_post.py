"""Writes a ready-to-paste posting file for long-form topic videos."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import urllib.parse

from streamclipper.config import Config
from streamclipper.output.post_writer import AI_DISCLOSURE_REMINDER
from streamclipper.topics.models import TopicPlan
from streamclipper.topics.planner import plan_dir

logger = logging.getLogger(__name__)


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
                sources_entries.append(f"- Official footage (Rockstar Games): {timed_url}")
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
        desc_parts.append("Official footage: Rockstar Games")
    desc_parts.extend(sources_entries)
    desc_parts.append("")

    # Tags / Hashtags
    game_tag = "".join(ch for ch in plan.game if ch.isalnum())
    tags_list = [f"#{game_tag}"] if game_tag else []
    for word in (plan.topic_title or "").split():
        cleaned = "".join(ch for ch in word if ch.isalnum())
        if len(cleaned) >= 3 and f"#{cleaned}" not in tags_list:
            tags_list.append(f"#{cleaned}")
    if "#Gaming" not in tags_list:
        tags_list.append("#Gaming")

    lines = [
        "TITLE",
        title,
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
