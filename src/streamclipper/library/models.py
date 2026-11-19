"""Data models for the topic video source library and search index."""
from dataclasses import dataclass
from typing import Any

from streamclipper.models import SerializableModel


@dataclass
class LibrarySource(SerializableModel):
    """Source video registered and indexed in the local library."""

    source_id: str = ""
    url: str = ""
    title: str = ""
    channel: str = ""
    game: str = ""
    kind: str = "stream"  # "stream" | "gameplay" | "official"
    language: str = "en"
    duration_seconds: int = 0
    video_id: str = ""
    added_at: str = ""
    indexed_at: str = ""
    segment_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibrarySource":
        return cls(
            source_id=str(data.get("source_id") or ""),
            url=str(data.get("url") or ""),
            title=str(data.get("title") or ""),
            channel=str(data.get("channel") or ""),
            game=str(data.get("game") or ""),
            kind=str(data.get("kind") or "stream").strip().lower(),
            language=str(data.get("language") or "en"),
            duration_seconds=int(float(data.get("duration_seconds") or 0)),
            video_id=str(data.get("video_id") or ""),
            added_at=str(data.get("added_at") or ""),
            indexed_at=str(data.get("indexed_at") or ""),
            segment_count=int(float(data.get("segment_count") or 0)),
        )


@dataclass
class SearchHit(SerializableModel):
    """Search result hit representing a matching segment or merged span."""

    source_id: str = ""
    url: str = ""
    channel: str = ""
    kind: str = "stream"
    start: float = 0.0
    end: float = 0.0
    snippet: str = ""
    score: float = 0.0
    bm25: float = 0.0
    heat: float = 0.0
    loudness: float = 0.0
    chat_rate: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchHit":
        return cls(
            source_id=str(data.get("source_id") or ""),
            url=str(data.get("url") or ""),
            channel=str(data.get("channel") or ""),
            kind=str(data.get("kind") or "stream").strip().lower(),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            snippet=str(data.get("snippet") or ""),
            score=float(data.get("score") or 0.0),
            bm25=float(data.get("bm25") or 0.0),
            heat=float(data.get("heat") or 0.0),
            loudness=float(data.get("loudness") or 0.0),
            chat_rate=float(data.get("chat_rate") or 0.0),
        )
