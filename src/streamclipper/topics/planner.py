"""Topic video planner: outline generation, clip selection, script synthesis, and approval flow."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any
import uuid

from streamclipper.cache import _atomic_write_text
from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex
from streamclipper.narration_style import narration_guide
from streamclipper.topics.discover import (
    _extract_json,
    _format_time,
    _slugify,
    evaluate_topic,
)
from streamclipper.topics.ledger import TopicLedger
from streamclipper.topics.models import (
    EvidenceClip,
    ShortPlan,
    TopicCandidate,
    TopicPlan,
    TopicPoint,
)
from streamclipper.topics.validate import validate_plan

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()

SHAPE_ROTATION: tuple[str, ...] = ("numbered", "countdown", "story")

TOPIC_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "plan_id",
        "topic_id",
        "topic_title",
        "game",
        "shape",
        "cold_open",
        "points",
        "outro",
        "shorts",
    ],
    "properties": {
        "plan_id": {"type": "string"},
        "topic_id": {"type": "string"},
        "topic_title": {"type": "string"},
        "game": {"type": "string"},
        "language": {"type": "string", "enum": ["english", "hinglish"]},
        "shape": {"type": "string", "enum": ["numbered", "countdown", "story"]},
        "status": {"type": "string", "default": "draft"},
        "cold_open": {"type": "string"},
        "outro": {"type": "string"},
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "heading", "narration", "evidence"],
                "properties": {
                    "index": {"type": "integer"},
                    "heading": {"type": "string"},
                    "narration": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["source_id", "start", "end", "role"],
                            "properties": {
                                "source_id": {"type": "string"},
                                "url": {"type": "string"},
                                "channel": {"type": "string"},
                                "kind": {"type": "string"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "role": {"type": "string", "enum": ["evidence", "reaction", "broll"]},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                    "alternates": {"type": "array"},
                    "use_in": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "shorts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "slot", "point_indexes", "hook_title", "narration_intro", "callout", "narration_outro"],
                "properties": {
                    "kind": {"type": "string", "enum": ["teaser", "standalone"]},
                    "slot": {"type": "string", "enum": ["teaser", "short_a", "short_b"]},
                    "point_indexes": {"type": "array", "items": {"type": "integer"}},
                    "hook_title": {"type": "string"},
                    "narration_intro": {"type": "string"},
                    "callout": {"type": "string"},
                    "narration_outro": {"type": "string"},
                },
            },
        },
        "video_title": {"type": "string"},
        "thumbnail_text": {"type": "string"},
        "description": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Storage & Paths
# ---------------------------------------------------------------------------

def plan_dir(config: Config, game: str, topic_id: str) -> Path:
    """Return directory path for topic plan and rendered assets: <output_dir>/<game>/topics/<topic_id>/."""
    out_dir_str = config.get("general.output_dir", "./output")
    clean_game = _slugify(game) or "game"
    clean_topic = _slugify(topic_id) or "topic"
    return Path(out_dir_str) / clean_game / "topics" / clean_topic


def _index_path(config: Config) -> Path:
    """Return path to cached plan id->path index: <cache_dir>/topic_plans.json."""
    cache_dir = Path(config.get("general.cache_dir", "./.cache"))
    return cache_dir / "topic_plans.json"


def _load_index(config: Config) -> dict[str, str]:
    """Load plan id->path mapping safely."""
    path = _index_path(config)
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_index(config: Config, index: dict[str, str]) -> None:
    """Persist plan id->path mapping atomically."""
    path = _index_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(index, indent=2))


def save_plan(config: Config, plan: TopicPlan) -> Path:
    """Atomically persist a TopicPlan to plan.json and update the id index."""
    with _LOCK:
        pdir = plan_dir(config, plan.game, plan.topic_id)
        pdir.mkdir(parents=True, exist_ok=True)
        plan_file = pdir / "plan.json"

        payload = json.dumps(plan.to_dict(), indent=2)
        _atomic_write_text(plan_file, payload)

        idx = _load_index(config)
        idx[plan.plan_id] = str(plan_file.resolve())
        _save_index(config, idx)

        return plan_file


def load_plan(config: Config, plan_id: str) -> TopicPlan | None:
    """Find and load a TopicPlan by plan_id from index or by scanning <output_dir>/*/topics/*/plan.json."""
    with _LOCK:
        clean_id = str(plan_id or "").strip()
        if not clean_id:
            return None

        # 1. Check cached index
        idx = _load_index(config)
        indexed_path_str = idx.get(clean_id)
        if indexed_path_str:
            p = Path(indexed_path_str)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("plan_id") == clean_id:
                        return TopicPlan.from_dict(data)
                except (OSError, ValueError):
                    pass

        # 2. Scan <output_dir>/*/topics/*/plan.json
        out_dir_str = config.get("general.output_dir", "./output")
        out_dir = Path(out_dir_str)
        if out_dir.exists():
            for plan_file in out_dir.glob("*/topics/*/plan.json"):
                try:
                    data = json.loads(plan_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("plan_id") == clean_id:
                        # Update index
                        idx[clean_id] = str(plan_file.resolve())
                        _save_index(config, idx)
                        return TopicPlan.from_dict(data)
                except (OSError, ValueError):
                    continue

        return None


# ---------------------------------------------------------------------------
# Shape Rotation
# ---------------------------------------------------------------------------

def choose_shape(config: Config, game: str) -> str:
    """Rotate story shape among 'numbered' -> 'countdown' -> 'story', skipping ledger's last_shape."""
    ledger = TopicLedger(config)
    last_shape = ledger.last_shape(game).strip().lower()

    if not last_shape or last_shape not in SHAPE_ROTATION:
        return SHAPE_ROTATION[0]

    idx = SHAPE_ROTATION.index(last_shape)
    next_shape = SHAPE_ROTATION[(idx + 1) % len(SHAPE_ROTATION)]
    return next_shape


# ---------------------------------------------------------------------------
# Normalization Helper
# ---------------------------------------------------------------------------

def normalize_plan_clips(config: Config, plan: TopicPlan) -> TopicPlan:
    """Fill channel, url, and kind from the library for every evidence clip in the plan."""
    index = LibraryIndex(config)
    cache: dict[str, Any] = {}

    def _get_src(sid: str) -> Any:
        if sid not in cache:
            try:
                with index._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM sources WHERE source_id = ? OR url = ?;", (sid, sid)
                    ).fetchone()
                    from streamclipper.library.models import LibrarySource
                    cache[sid] = LibrarySource.from_dict(dict(row)) if row else None
            except Exception:
                cache[sid] = None
        return cache[sid]

    for p in plan.points:
        all_clips = list(p.evidence) + list(p.alternates)
        for clip in all_clips:
            if not clip.source_id:
                continue
            src = _get_src(clip.source_id)
            if src is not None:
                clip.source_id = src.source_id
                if not clip.url:
                    clip.url = src.url
                if not clip.channel:
                    clip.channel = src.channel
                if not clip.kind or clip.kind == "stream":
                    clip.kind = getattr(src, "kind", "stream")

    return plan


# ---------------------------------------------------------------------------
# Merge Patch Helper (RFC 7396)
# ---------------------------------------------------------------------------

def apply_merge_patch(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply an RFC 7396 JSON merge patch to target dictionary recursively."""
    res = copy.deepcopy(target)
    for key, value in patch.items():
        if value is None:
            res.pop(key, None)
        elif isinstance(value, dict) and isinstance(res.get(key), dict):
            res[key] = apply_merge_patch(res[key], value)
        else:
            res[key] = copy.deepcopy(value)
    return res


# ---------------------------------------------------------------------------
# Material for Writing
# ---------------------------------------------------------------------------

def planning_material(
    config: Config,
    game: str,
    topic: str | TopicCandidate,
    language: str | None = None,
) -> dict[str, Any]:
    """Evaluate topic and assemble structured material for prompt or MCP client."""
    clean_game = str(game or "").strip()
    index = LibraryIndex(config)

    # 1. Topic Candidate Evaluation
    if isinstance(topic, TopicCandidate):
        candidate = topic
    else:
        candidate = evaluate_topic(config, clean_game, title=str(topic), keywords=[])

    # 2. Gather Top ~20 Evidence Spans
    evidence_spans: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, float, float]] = set()

    for ev in candidate.evidence:
        span_key = (ev.source_id, ev.start, ev.end)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            evidence_spans.append({
                "source_id": ev.source_id,
                "url": ev.url,
                "channel": ev.channel,
                "kind": ev.kind,
                "start": ev.start,
                "end": ev.end,
                "duration": round(ev.end - ev.start, 1),
                "transcript_text": ev.snippet,
                "chat_sample": "",
                "frame_notes": "",
                "heat": ev.heat,
                "score": ev.score,
            })

    # If candidate has fewer than 20, query index for extra hits on candidate title/keywords
    if len(evidence_spans) < 20:
        queries = [candidate.title] + candidate.keywords
        for q in queries:
            if len(evidence_spans) >= 20:
                break
            if not q:
                continue
            hits = index.search(q, game=clean_game or None, limit=10)
            for h in hits:
                span_key = (h.source_id, h.start, h.end)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    evidence_spans.append({
                        "source_id": h.source_id,
                        "url": h.url,
                        "channel": h.channel,
                        "kind": h.kind,
                        "start": h.start,
                        "end": h.end,
                        "duration": round(h.end - h.start, 1),
                        "transcript_text": h.snippet,
                        "chat_sample": "",
                        "frame_notes": "",
                        "heat": h.heat,
                        "score": h.score,
                    })
                    if len(evidence_spans) >= 20:
                        break

    # Attach frame notes and chat samples from segments table where available
    for span in evidence_spans:
        try:
            with index._get_connection() as conn:
                # Query chat_text
                row = conn.execute(
                    "SELECT chat_text FROM segments WHERE source_id = ? AND start <= ? AND end >= ? LIMIT 1",
                    (span["source_id"], span["end"], span["start"]),
                ).fetchone()
                if row and row["chat_text"]:
                    span["chat_sample"] = str(row["chat_text"])[:200]

                # Query frame_notes
                fn_rows = conn.execute(
                    "SELECT note FROM frame_notes WHERE source_id = ? AND t >= ? AND t <= ? LIMIT 2",
                    (span["source_id"], span["start"], span["end"]),
                ).fetchall()
                if fn_rows:
                    span["frame_notes"] = "; ".join(str(r["note"]) for r in fn_rows)
        except Exception:
            pass

    # 3. Shape Rotation
    shape = choose_shape(config, clean_game)

    # 4. Resolved Narration Language
    if language and language.strip():
        resolved_lang = language.strip().lower()
    else:
        lang_counts: dict[str, int] = {}
        for span in evidence_spans:
            src = index.get_source(span["source_id"])
            lang = getattr(src, "language", "en") if src else "en"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        majority_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else "en"
        from streamclipper.language import resolve_narration_language
        resolved_lang = resolve_narration_language(config, None, majority_lang)

    # 5. Narration Guide
    guide = narration_guide(resolved_lang, hook_style="nobody_noticed")

    # 6. Plan Rules Summary
    rules_text = (
        "- Long video length: 480–600 seconds (8–10 minutes).\n"
        "- Long video structure: 5–7 points (or 4–6 points if story shape).\n"
        "- Chapter headings:\n"
        "  * 'numbered' shape: 5–7 points with '#N Heading' headings in ascending order.\n"
        "  * 'countdown' shape: 5–7 points with '#N Heading' headings ordered strictly high-to-low (e.g. #5 down to #1).\n"
        "  * 'story' shape: 4–6 points with descriptive headings WITHOUT '#'.\n"
        "- Narration share: >= 40% of total estimated length (narration words / 2.7).\n"
        "- Every evidence clip: <= 20s (if official trailer/reveal footage, <= 10s).\n"
        "- Exactly 3 Shorts: teaser (35-60s, ends with pointer to full video), short_a (35-60s, standalone), short_b (35-60s, standalone).\n"
        "- Footage separation: short_a and short_b must NEVER share overlapping footage.\n"
        "- Standalone Shorts must NEVER reference the full video or channel.\n"
        "- All sentences <= 14 words; energetic gamer friend explainer tone."
    )

    return {
        "game": clean_game,
        "topic_id": candidate.topic_id,
        "topic_title": candidate.title,
        "shape": shape,
        "language": resolved_lang,
        "candidate": candidate.to_dict(),
        "evidence_spans": evidence_spans,
        "narration_guide": guide,
        "rules": rules_text,
        "schema": TOPIC_PLAN_JSON_SCHEMA,
    }


# ---------------------------------------------------------------------------
# Human-Readable Plan Outline Formatter
# ---------------------------------------------------------------------------

def format_plan_outline(plan: TopicPlan) -> str:
    """Format a TopicPlan into a clean, human-readable outline with timing and clips."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"TOPIC PLAN: {plan.topic_title or plan.video_title} (ID: {plan.plan_id})")
    lines.append(f"Game: {plan.game} | Shape: {plan.shape} | Language: {plan.language} | Status: {plan.status}")
    ratio = (plan.narration_seconds / plan.estimated_seconds) if plan.estimated_seconds > 0 else 0.0
    lines.append(
        f"Estimated Duration: {_format_time(plan.estimated_seconds)} ({plan.estimated_seconds:.1f}s) | "
        f"Narration: {_format_time(plan.narration_seconds)} ({plan.narration_seconds:.1f}s, {ratio:.1%})"
    )
    lines.append("=" * 60)

    lines.append("\n[COLD OPEN]")
    lines.append(f"  {plan.cold_open or '(None)'}")

    lines.append("\n[POINTS & SCRIPT]")
    for p in plan.points:
        lines.append(f"\n* Point {p.index}: {p.heading}")
        lines.append(f"  Narration: {p.narration}")
        lines.append("  Evidence Clips:")
        for c in p.evidence:
            dur = c.end - c.start
            ch_str = f"@{c.channel}" if c.channel else c.source_id
            lines.append(f"    - [{ch_str}] {_format_time(c.start)}–{_format_time(c.end)} ({dur:.1f}s) [{c.kind}, {c.role}] {c.reason}")
        if p.alternates:
            lines.append("  Alternates:")
            for a in p.alternates:
                dur = a.end - a.start
                ch_str = f"@{a.channel}" if a.channel else a.source_id
                lines.append(f"    - [{ch_str}] {_format_time(a.start)}–{_format_time(a.end)} ({dur:.1f}s) {a.reason}")
        lines.append(f"  Used In: {', '.join(p.use_in)}")

    lines.append("\n[OUTRO]")
    lines.append(f"  {plan.outro or '(None)'}")

    lines.append("\n[SHORTS SPECIFICATION]")
    for s in plan.shorts:
        lines.append(f"\n- Slot: {s.slot} ({s.kind.upper()}) | Points: {s.point_indexes}")
        lines.append(f"  Hook Title: \"{s.hook_title}\"")
        lines.append(f"  Intro: \"{s.narration_intro}\"")
        lines.append(f"  Callout: \"{s.callout}\"")
        lines.append(f"  Outro: \"{s.narration_outro}\"")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local Planner (Ollama + Retry & Keep Best)
# ---------------------------------------------------------------------------

def plan_topic_local(
    config: Config,
    game: str,
    topic: str,
    *,
    client: Any = None,
) -> TopicPlan:
    """Draft a TopicPlan locally using Ollama, retrying up to 3 times feeding back problems.

    Keeps the attempt with the fewest validation problems, status 'draft'.
    """
    material = planning_material(config, game, topic)

    prompt_path = Path(__file__).parent.parent / "prompts" / "topic_planner.txt"
    template = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    evidence_summary_lines: list[str] = []
    for sp in material["evidence_spans"]:
        evidence_summary_lines.append(
            f"- Source: {sp['source_id']} ({sp['kind']}, @{sp['channel']}) "
            f"[{_format_time(sp['start'])}–{_format_time(sp['end'])}]: "
            f"Transcript: \"{sp['transcript_text']}\" "
            + (f"| Frame notes: {sp['frame_notes']} " if sp["frame_notes"] else "")
            + (f"| Chat: {sp['chat_sample']}" if sp["chat_sample"] else "")
        )
    evidence_summary = "\n".join(evidence_summary_lines)

    base_prompt = template.format(
        game=material["game"],
        topic=material["topic_title"],
        shape=material["shape"],
        language=material["language"],
        narration_guide=material["narration_guide"],
        evidence_summary=evidence_summary,
        plan_rules=material["rules"],
        json_schema=json.dumps(material["schema"], indent=2),
    ) if template else json.dumps(material, indent=2)

    # Initialize client if not provided
    if client is None:
        try:
            import ollama
            base_url = str(config.get("llm.base_url", "http://localhost:11434") or "http://localhost:11434")
            timeout = float(config.get("llm.timeout", 45.0))
            client = ollama.Client(host=base_url, timeout=timeout)
        except Exception as exc:
            logger.warning("Could not initialize Ollama client: %s", exc)
            client = None

    model = str(config.get("llm.model", "qwen3:8b") or "qwen3:8b")

    attempts: list[tuple[TopicPlan, list[str]]] = []
    current_prompt = base_prompt
    fresh_plan_id = f"plan_{_slugify(material['topic_title'])}_{uuid.uuid4().hex[:8]}"

    for attempt_idx in range(3):
        raw_text = ""
        if client is not None:
            try:
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": current_prompt}],
                    options={"temperature": 0.3},
                )
                if isinstance(response, dict):
                    raw_text = response.get("message", {}).get("content", "")
                else:
                    raw_text = getattr(getattr(response, "message", None), "content", "")
            except Exception as exc:
                logger.info("Ollama call failed on attempt %d: %s", attempt_idx + 1, exc)

        parsed = _extract_json(raw_text) if raw_text else None
        if not isinstance(parsed, dict):
            # Construct a minimal draft baseline plan from material
            points: list[TopicPoint] = []
            spans = material["evidence_spans"]
            num_pts = 5 if material["shape"] in ("numbered", "countdown") else 4
            for i in range(num_pts):
                idx = (num_pts - i) if material["shape"] == "countdown" else (i + 1)
                heading_prefix = f"#{idx} " if material["shape"] != "story" else ""
                sp = spans[i % len(spans)] if spans else {}
                points.append(
                    TopicPoint(
                        index=idx,
                        heading=f"{heading_prefix}Point {idx}",
                        narration="Here is the critical observation. Watch what happens right here.",
                        evidence=[
                            EvidenceClip(
                                source_id=sp.get("source_id", ""),
                                url=sp.get("url", ""),
                                channel=sp.get("channel", ""),
                                kind=sp.get("kind", "stream"),
                                start=float(sp.get("start", 0.0)),
                                end=float(min(sp.get("start", 0.0) + 15.0, sp.get("end", 15.0))),
                                role="evidence",
                                reason=sp.get("transcript_text", ""),
                            )
                        ],
                    )
                )

            plan = TopicPlan(
                plan_id=fresh_plan_id,
                topic_id=material["topic_id"],
                topic_title=material["topic_title"],
                game=material["game"],
                shape=material["shape"],
                language=material["language"],
                status="draft",
                cold_open="Check this out bro. You will not believe what we found.",
                outro="What do you think about this secret? Let me know in the comments.",
                points=points,
                shorts=[
                    ShortPlan(
                        kind="teaser",
                        slot="teaser",
                        point_indexes=[points[0].index],
                        hook_title="Unbelievable Secret",
                        narration_intro="Look at this secret entrance.",
                        callout="Watch the gate!",
                        narration_outro="Full breakdown on the channel!",
                    ),
                    ShortPlan(
                        kind="standalone",
                        slot="short_a",
                        point_indexes=[points[1].index if len(points) > 1 else points[0].index],
                        hook_title="Crazy Detail",
                        narration_intro="Look at this incredible detail.",
                        callout="Check the cargo!",
                        narration_outro="Did you catch that?",
                    ),
                    ShortPlan(
                        kind="standalone",
                        slot="short_b",
                        point_indexes=[points[2].index if len(points) > 2 else points[0].index],
                        hook_title="Secret Path",
                        narration_intro="Check out this hidden tunnel.",
                        callout="Look at the water!",
                        narration_outro="Would you try this route?",
                    ),
                ],
            )
        else:
            if not parsed.get("plan_id"):
                parsed["plan_id"] = fresh_plan_id
            if not parsed.get("game"):
                parsed["game"] = material["game"]
            if not parsed.get("topic_id"):
                parsed["topic_id"] = material["topic_id"]
            if not parsed.get("topic_title"):
                parsed["topic_title"] = material["topic_title"]
            if not parsed.get("shape"):
                parsed["shape"] = material["shape"]
            plan = TopicPlan.from_dict(parsed)

        # Normalize clips from library
        normalize_plan_clips(config, plan)
        plan.status = "draft"
        if not plan.created_at:
            plan.created_at = datetime.now(timezone.utc).isoformat()

        # Validate
        problems = validate_plan(config, plan)
        plan.problems = problems
        attempts.append((plan, problems))

        if not problems:
            # Completely valid plan found!
            save_plan(config, plan)
            return plan

        # Prepare feedback for retry
        current_prompt = (
            f"{base_prompt}\n\n"
            f"Your previous attempt had the following {len(problems)} validation problems:\n"
            + "\n".join(f"- {pb}" for pb in problems)
            + "\n\nPlease fix ALL problems and return the revised JSON object."
        )

    # Keep attempt with the fewest problems
    best_plan, best_problems = min(attempts, key=lambda att: len(att[1]))
    best_plan.problems = best_problems
    best_plan.status = "draft"
    save_plan(config, best_plan)
    return best_plan
