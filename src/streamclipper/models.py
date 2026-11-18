"""Shared data models for StreamClipper across all pipeline phases."""
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Optional
import json


class Platform(str, Enum):
    YOUTUBE = "youtube"
    TWITCH = "twitch"  # future
    KICK = "kick"      # future


class SignalType(str, Enum):
    LOUDNESS = "loudness"
    CHAT_VELOCITY = "chat_velocity"
    LLM_PICK = "llm_pick"


class PipelineStatus(IntEnum):
    PENDING = 1
    DOWNLOADING = 2
    TRANSCRIBING = 3
    ANALYZING = 4
    SCORING = 5
    ASSEMBLING = 6
    COMPLETE = 7
    FAILED = 99


def _serialize_value(val: Any) -> Any:
    """Recursively convert dataclasses, Paths, and Enums into JSON-safe primitives."""
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, Enum):
        return val.value
    if is_dataclass(val) and not isinstance(val, type):
        return {f.name: _serialize_value(getattr(val, f.name)) for f in fields(val)}
    if isinstance(val, list):
        return [_serialize_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    return val


@dataclass
class SerializableModel:
    def to_dict(self) -> dict[str, Any]:
        """Convert to a recursively JSON-serializable dict."""
        return _serialize_value(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StreamInfo(SerializableModel):
    url: str
    title: str
    platform: Platform
    streamer: str
    game: str
    duration_seconds: int
    view_count: int = 0
    video_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamInfo":
        return cls(
            url=data["url"],
            title=data["title"],
            platform=Platform(data["platform"]),
            streamer=data["streamer"],
            game=data["game"],
            duration_seconds=int(data["duration_seconds"]),
            view_count=int(data.get("view_count", 0)),
            video_id=data.get("video_id", ""),
        )


@dataclass
class TimeRange(SerializableModel):
    start: float          # seconds from stream start
    end: float            # seconds from stream start
    score: float = 0.0
    signal_type: SignalType = SignalType.LOUDNESS
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeRange":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            score=float(data.get("score", 0.0)),
            signal_type=SignalType(data.get("signal_type", SignalType.LOUDNESS.value)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class WordTimestamp(SerializableModel):
    text: str
    start: float          # seconds
    end: float            # seconds
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordTimestamp":
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class TranscriptSegment(SerializableModel):
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            words=[WordTimestamp.from_dict(w) for w in data.get("words", [])],
        )


@dataclass
class TranscriptionResult(SerializableModel):
    segments: list[TranscriptSegment]
    full_text: str
    language: str = "en"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])],
            full_text=data.get("full_text", ""),
            language=data.get("language", "en"),
        )


@dataclass
class Highlight(SerializableModel):
    start: float
    end: float
    score: float                  # 1-10 from LLM
    title: str                    # suggested hook title
    reason: str                   # why this is a good clip
    best_quote: str = ""          # most memorable line
    loudness_score: float = 0.0   # from loudness detector
    chat_score: float = 0.0       # from chat analyzer
    final_score: float = 0.0      # weighted composite score
    transcript_chunk: str = ""    # the spoken words in this range
    word_timestamps: list[WordTimestamp] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Highlight":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            score=float(data["score"]),
            title=data["title"],
            reason=data["reason"],
            best_quote=data.get("best_quote", ""),
            loudness_score=float(data.get("loudness_score", 0.0)),
            chat_score=float(data.get("chat_score", 0.0)),
            final_score=float(data.get("final_score", 0.0)),
            transcript_chunk=data.get("transcript_chunk", ""),
            word_timestamps=[
                WordTimestamp.from_dict(w) for w in data.get("word_timestamps", [])
            ],
        )


@dataclass
class FacecamRegion(SerializableModel):
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FacecamRegion":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class ClipMetadata(SerializableModel):
    hook_title: str
    caption: str
    hashtags: list[str]
    streamer: str
    game: str
    source_url: str
    clip_start: float
    clip_end: float
    highlight_score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMetadata":
        return cls(
            hook_title=data["hook_title"],
            caption=data["caption"],
            hashtags=list(data.get("hashtags", [])),
            streamer=data["streamer"],
            game=data["game"],
            source_url=data["source_url"],
            clip_start=float(data["clip_start"]),
            clip_end=float(data["clip_end"]),
            highlight_score=float(data["highlight_score"]),
        )


@dataclass
class ClipResult(SerializableModel):
    video_path: Path
    metadata_path: Path
    metadata: ClipMetadata
    duration_seconds: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipResult":
        return cls(
            video_path=Path(data["video_path"]),
            metadata_path=Path(data["metadata_path"]),
            metadata=ClipMetadata.from_dict(data["metadata"]),
            duration_seconds=float(data["duration_seconds"]),
        )


@dataclass
class PipelineState(SerializableModel):
    """Tracks progress for resume capability."""
    stream_info: StreamInfo
    status: PipelineStatus = PipelineStatus.PENDING
    audio_path: Optional[Path] = None
    transcript: Optional[TranscriptionResult] = None
    loudness_spikes: list[TimeRange] = field(default_factory=list)
    chat_spikes: list[TimeRange] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    selected_highlights: list[Highlight] = field(default_factory=list)
    clips: list[ClipResult] = field(default_factory=list)
    cache_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        return cls(
            stream_info=StreamInfo.from_dict(data["stream_info"]),
            status=PipelineStatus(int(data.get("status", PipelineStatus.PENDING))),
            audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
            transcript=(
                TranscriptionResult.from_dict(data["transcript"])
                if data.get("transcript") is not None
                else None
            ),
            loudness_spikes=[
                TimeRange.from_dict(s) for s in data.get("loudness_spikes", [])
            ],
            chat_spikes=[
                TimeRange.from_dict(s) for s in data.get("chat_spikes", [])
            ],
            highlights=[
                Highlight.from_dict(h) for h in data.get("highlights", [])
            ],
            selected_highlights=[
                Highlight.from_dict(h) for h in data.get("selected_highlights", [])
            ],
            clips=[ClipResult.from_dict(c) for c in data.get("clips", [])],
            cache_dir=Path(data["cache_dir"]) if data.get("cache_dir") else None,
        )
