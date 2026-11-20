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
