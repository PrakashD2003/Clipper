"""Data models for topic video discovery, evidence evaluation, and planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streamclipper.models import SerializableModel


@dataclass
class EvidenceRef(SerializableModel):
    """Reference to a specific evidence segment or frame note in the source library."""

    source_id: str = ""
    url: str = ""
    channel: str = ""
    kind: str = "stream"
    start: float = 0.0
    end: float = 0.0
    snippet: str = ""
    heat: float = 0.0
    score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRef:
        return cls(
            source_id=str(data.get("source_id") or ""),
            url=str(data.get("url") or ""),
            channel=str(data.get("channel") or ""),
            kind=str(data.get("kind") or "stream"),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            snippet=str(data.get("snippet") or ""),
            heat=float(data.get("heat") or 0.0),
            score=float(data.get("score") or 0.0),
        )


@dataclass
class TopicCandidate(SerializableModel):
    """Candidate topic video idea with evidence spans from the library."""

    topic_id: str = ""  # slug, e.g. "hidden-casino-details"
    title: str = ""
    game: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    source_count: int = 0
    channel_count: int = 0
    already_made: bool = False
    similar_to: str = ""  # ledger topic_id when already_made
    min_evidence: int = 6
    min_sources: int = 2

    @property
    def is_viable(self) -> bool:
        """A candidate is viable when it has >= min_evidence spans from >= min_sources sources."""
        return len(self.evidence) >= self.min_evidence and self.source_count >= self.min_sources

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["is_viable"] = self.is_viable
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicCandidate:
        raw_evidence = data.get("evidence") or []
        evidence_list = [
            e if isinstance(e, EvidenceRef) else EvidenceRef.from_dict(e)
            for e in raw_evidence
            if isinstance(e, (dict, EvidenceRef))
        ]
        raw_kw = data.get("keywords") or []
        if isinstance(raw_kw, str):
            kw_list = [k.strip() for k in raw_kw.split(",") if k.strip()]
        else:
            kw_list = [str(k).strip() for k in raw_kw if str(k).strip()]
        return cls(
            topic_id=str(data.get("topic_id") or ""),
            title=str(data.get("title") or ""),
            game=str(data.get("game") or ""),
            description=str(data.get("description") or ""),
            keywords=kw_list,
            evidence=evidence_list,
            source_count=int(data.get("source_count") or len({e.source_id for e in evidence_list if e.source_id})),
            channel_count=int(data.get("channel_count") or len({e.channel for e in evidence_list if e.channel})),
            already_made=bool(data.get("already_made") or False),
            similar_to=str(data.get("similar_to") or ""),
            min_evidence=int(data.get("min_evidence") or 6),
            min_sources=int(data.get("min_sources") or 2),
        )


@dataclass
class EvidenceClip(SerializableModel):
    """Specific evidence or reaction video clip slice from the source library."""

    source_id: str = ""
    url: str = ""
    channel: str = ""
    kind: str = "stream"  # stream | gameplay | official
    start: float = 0.0  # stream seconds; end-start <= 20 (official <= 10)
    end: float = 0.0
    role: str = "evidence"  # evidence | reaction | broll
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceClip:
        return cls(
            source_id=str(data.get("source_id") or ""),
            url=str(data.get("url") or ""),
            channel=str(data.get("channel") or ""),
            kind=str(data.get("kind") or "stream"),
            start=float(data.get("start") if data.get("start") is not None else 0.0),
            end=float(data.get("end") if data.get("end") is not None else 0.0),
            role=str(data.get("role") or "evidence"),
            reason=str(data.get("reason") or ""),
        )


@dataclass
class TopicPoint(SerializableModel):
    """A single beat or chapter point in the topic video outline."""

    index: int = 1
    heading: str = ""  # "#3 The hidden casino" style for numbered shapes
    narration: str = ""  # spoken before/over B-roll for this point
    evidence: list[EvidenceClip] = field(default_factory=list)  # 1-2 played at full volume
    alternates: list[EvidenceClip] = field(default_factory=list)  # swap-ins if a clip fails checks
    use_in: list[str] = field(default_factory=lambda: ["long"])  # long | teaser | short_a | short_b

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicPoint:
        raw_evidence = data.get("evidence") or []
        evidence_list = [
            e if isinstance(e, EvidenceClip) else EvidenceClip.from_dict(e)
            for e in raw_evidence
            if isinstance(e, (dict, EvidenceClip))
        ]
        raw_alts = data.get("alternates") or []
        alternates_list = [
            a if isinstance(a, EvidenceClip) else EvidenceClip.from_dict(a)
            for a in raw_alts
            if isinstance(a, (dict, EvidenceClip))
        ]
        raw_use = data.get("use_in") or ["long"]
        if isinstance(raw_use, str):
            use_in_list = [u.strip() for u in raw_use.split(",") if u.strip()]
        else:
            use_in_list = [str(u).strip() for u in raw_use if str(u).strip()]
        return cls(
            index=int(data.get("index") if data.get("index") is not None else 1),
            heading=str(data.get("heading") or ""),
            narration=str(data.get("narration") or ""),
            evidence=evidence_list,
            alternates=alternates_list,
            use_in=use_in_list or ["long"],
        )


@dataclass
class ShortPlan(SerializableModel):
    """Plan for a single 9:16 Short produced from the topic material."""

    kind: str = "teaser"  # teaser | standalone
    slot: str = "teaser"  # teaser | short_a | short_b
    point_indexes: list[int] = field(default_factory=list)
    hook_title: str = ""
    narration_intro: str = ""
    callout: str = ""
    narration_outro: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShortPlan:
        raw_pts = data.get("point_indexes") or []
        point_indexes: list[int] = []
        for p in raw_pts:
            try:
                point_indexes.append(int(p))
            except (ValueError, TypeError):
                continue
        return cls(
            kind=str(data.get("kind") or "teaser"),
            slot=str(data.get("slot") or "teaser"),
            point_indexes=point_indexes,
            hook_title=str(data.get("hook_title") or ""),
            narration_intro=str(data.get("narration_intro") or ""),
            callout=str(data.get("callout") or ""),
            narration_outro=str(data.get("narration_outro") or ""),
        )


@dataclass
class TopicPlan(SerializableModel):
    """Complete outline, script, clip selection, and output metadata for a topic video."""

    plan_id: str = ""
    topic_id: str = ""
    topic_title: str = ""
    game: str = ""
    language: str = "english"
    shape: str = "numbered"  # numbered | countdown | story
    status: str = "draft"  # draft | approved | rendering | rendered | failed
    cold_open: str = ""
    outro: str = ""
    points: list[TopicPoint] = field(default_factory=list)
    shorts: list[ShortPlan] = field(default_factory=list)
    video_title: str = ""
    thumbnail_text: str = ""
    description: str = ""
    estimated_seconds: float = 0.0
    narration_seconds: float = 0.0
    created_at: str = ""
    approved_at: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicPlan:
        raw_pts = data.get("points") or []
        points = [
            p if isinstance(p, TopicPoint) else TopicPoint.from_dict(p)
            for p in raw_pts
            if isinstance(p, (dict, TopicPoint))
        ]
        raw_shorts = data.get("shorts") or []
        shorts = [
            s if isinstance(s, ShortPlan) else ShortPlan.from_dict(s)
            for s in raw_shorts
            if isinstance(s, (dict, ShortPlan))
        ]
        raw_outputs = data.get("outputs") or {}
        outputs = {str(k): str(v) for k, v in raw_outputs.items()} if isinstance(raw_outputs, dict) else {}
        raw_problems = data.get("problems") or []
        problems = [str(pr) for pr in raw_problems] if isinstance(raw_problems, list) else []

        return cls(
            plan_id=str(data.get("plan_id") or ""),
            topic_id=str(data.get("topic_id") or ""),
            topic_title=str(data.get("topic_title") or ""),
            game=str(data.get("game") or ""),
            language=str(data.get("language") or "english"),
            shape=str(data.get("shape") or "numbered"),
            status=str(data.get("status") or "draft"),
            cold_open=str(data.get("cold_open") or ""),
            outro=str(data.get("outro") or ""),
            points=points,
            shorts=shorts,
            video_title=str(data.get("video_title") or ""),
            thumbnail_text=str(data.get("thumbnail_text") or ""),
            description=str(data.get("description") or ""),
            estimated_seconds=float(data.get("estimated_seconds") if data.get("estimated_seconds") is not None else 0.0),
            narration_seconds=float(data.get("narration_seconds") if data.get("narration_seconds") is not None else 0.0),
            created_at=str(data.get("created_at") or ""),
            approved_at=str(data.get("approved_at") or ""),
            outputs=outputs,
            problems=problems,
        )
