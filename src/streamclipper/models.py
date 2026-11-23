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
    HEATMAP = "heatmap"
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
            title=data.get("title") or "",
            platform=Platform(data["platform"]),
            streamer=data.get("streamer") or "",
            game=data.get("game") or "",
            duration_seconds=int(data.get("duration_seconds") or 0),
            view_count=int(data.get("view_count") or 0),
            video_id=data.get("video_id") or "",
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
        raw_meta = data.get("metadata")
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            score=float(data.get("score") if data.get("score") is not None else 0.0),
            signal_type=SignalType(
                data.get("signal_type") or SignalType.LOUDNESS.value
            ),
            metadata=dict(raw_meta) if isinstance(raw_meta, dict) else {},
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
            confidence=float(
                data.get("confidence") if data.get("confidence") is not None else 0.0
            ),
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
            words=[WordTimestamp.from_dict(w) for w in (data.get("words") or [])],
        )


@dataclass
class TranscriptionResult(SerializableModel):
    segments: list[TranscriptSegment]
    full_text: str
    language: str = "en"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            segments=[
                TranscriptSegment.from_dict(s) for s in (data.get("segments") or [])
            ],
            full_text=data.get("full_text") or "",
            language=data.get("language") or "en",
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
    # YPP Transformative Narrative fields (Phase 3+, default-valued for backward compat)
    moment_type: str = "hype"          # funny | fail | graphics_detail | hype | clutch | chat_chaos | lore_secret | analysis | reaction
    narration_intro: str = ""          # 3-4s tone-matched spoken setup (10-15 words)
    narration_outro: str = ""          # 3-4s tone-matched spoken punchline / callout / takeaway (10-15 words)
    context_badge: str = ""            # Short adaptive pill (e.g., "🎨 Insane Detail", "💀 Instant Karma", "🔥 Clutch Escape")
    peak_time: float = 0.0             # Exact second of loudness peak for 1.2x punch-in zoom
    chat_burst: list[dict[str, Any]] = field(default_factory=list)  # Up to 5 viewer chat messages in range
    sub_clips: list[dict[str, Any]] = field(default_factory=list)   # Optional [{"start": s, "end": e, "label": "Setup"}]
    breakdown_commentary: str = ""     # Spoken breakdown analyzing mechanics/physics/mistake during replay
    breakdown_callout: str = ""        # 2-5 word on-screen callout note
    breakdown_kind: str = "auto"       # auto | slow_mo | zoom_spotlight
    breakdown_target_time: float | None = None  # Exact stream second of visual detail
    breakdown_roi: list[float] = field(default_factory=list)  # [ymin, xmin, ymax, xmax] (0.0 - 1.0)
    breakdown_zoom: float = 2.2        # Zoom magnification factor into ROI
    breakdown_focus: str = ""          # e.g. "laptop screen", "classified file"
    music_risk: str = "CLEAN"          # CLEAN | LOW | UNKNOWN | HIGH_MUSIC_DETECTED
    music_confidence: float = 0.0      # Harmonic-percussive music confidence 0.0-1.0
    bounds_adjustment: str = ""        # How the clip range was fitted to length limits ("" = unchanged)
    callout: str = ""                  # 4-6 word voiced callout said during the build-up
    transformation_mode: str = "auto"  # auto | narrative | story | breakdown | raw
    caption_style: str = "auto"        # auto | hormozi | bounce | pop | karaoke | box | single_word | sentence

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Highlight":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            score=float(data["score"]),
            title=data.get("title") or "",
            reason=data.get("reason") or "",
            best_quote=data.get("best_quote") or "",
            loudness_score=float(
                data.get("loudness_score") if data.get("loudness_score") is not None else 0.0
            ),
            chat_score=float(
                data.get("chat_score") if data.get("chat_score") is not None else 0.0
            ),
            final_score=float(
                data.get("final_score") if data.get("final_score") is not None else 0.0
            ),
            transcript_chunk=data.get("transcript_chunk") or "",
            word_timestamps=[
                WordTimestamp.from_dict(w) for w in (data.get("word_timestamps") or [])
            ],
            moment_type=data.get("moment_type") or "hype",
            narration_intro=data.get("narration_intro") or "",
            narration_outro=data.get("narration_outro") or "",
            context_badge=data.get("context_badge") or "",
            peak_time=float(
                data.get("peak_time") if data.get("peak_time") is not None else 0.0
            ),
            chat_burst=list(data.get("chat_burst") or []),
            sub_clips=list(data.get("sub_clips") or []),
            breakdown_commentary=data.get("breakdown_commentary") or "",
            breakdown_callout=data.get("breakdown_callout") or "",
            breakdown_kind=str(data.get("breakdown_kind") or "auto"),
            breakdown_target_time=(
                float(data["breakdown_target_time"])
                if data.get("breakdown_target_time") is not None
                else None
            ),
            breakdown_roi=[float(x) for x in (data.get("breakdown_roi") or [])],
            breakdown_zoom=float(
                data.get("breakdown_zoom") if data.get("breakdown_zoom") is not None else 2.2
            ),
            breakdown_focus=str(data.get("breakdown_focus") or ""),
            music_risk=data.get("music_risk") or "CLEAN",
            music_confidence=float(
                data.get("music_confidence") if data.get("music_confidence") is not None else 0.0
            ),
            bounds_adjustment=data.get("bounds_adjustment") or "",
            callout=data.get("callout") or "",
            transformation_mode=str(data.get("transformation_mode") or data.get("mode") or "auto"),
            caption_style=str(data.get("caption_style") or "auto"),
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
            confidence=float(
                data.get("confidence") if data.get("confidence") is not None else 0.0
            ),
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
    # YPP Transformative Narrative fields (Phase 3+, default-valued for backward compat)
    moment_type: str = "hype"
    narration_intro: str = ""
    narration_outro: str = ""
    context_badge: str = ""
    transformation_mode: str = "narrative"
    breakdown_commentary: str = ""
    breakdown_callout: str = ""
    breakdown_kind: str = "auto"
    breakdown_target_time: float | None = None
    breakdown_roi: list[float] = field(default_factory=list)
    breakdown_zoom: float = 2.2
    breakdown_focus: str = ""
    music_risk: str = "CLEAN"
    music_confidence: float = 0.0
    callout: str = ""
    narration_language: str = ""
    caption_style: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMetadata":
        return cls(
            hook_title=data.get("hook_title") or "",
            caption=data.get("caption") or "",
            hashtags=list(data.get("hashtags") or []),
            streamer=data.get("streamer") or "",
            game=data.get("game") or "",
            source_url=data.get("source_url") or "",
            clip_start=float(data["clip_start"]),
            clip_end=float(data["clip_end"]),
            highlight_score=float(data["highlight_score"]),
            moment_type=data.get("moment_type") or "hype",
            narration_intro=data.get("narration_intro") or "",
            narration_outro=data.get("narration_outro") or "",
            context_badge=data.get("context_badge") or "",
            transformation_mode=data.get("transformation_mode") or "narrative",
            breakdown_commentary=data.get("breakdown_commentary") or "",
            breakdown_callout=data.get("breakdown_callout") or "",
            breakdown_kind=str(data.get("breakdown_kind") or "auto"),
            breakdown_target_time=(
                float(data["breakdown_target_time"])
                if data.get("breakdown_target_time") is not None
                else None
            ),
            breakdown_roi=[float(x) for x in (data.get("breakdown_roi") or [])],
            breakdown_zoom=float(
                data.get("breakdown_zoom") if data.get("breakdown_zoom") is not None else 2.2
            ),
            breakdown_focus=str(data.get("breakdown_focus") or ""),
            music_risk=data.get("music_risk") or "CLEAN",
            music_confidence=float(
                data.get("music_confidence") if data.get("music_confidence") is not None else 0.0
            ),
            callout=data.get("callout") or "",
            narration_language=data.get("narration_language", "") or "",
            caption_style=str(data.get("caption_style") or "auto"),
        )


@dataclass
class ClipResult(SerializableModel):
    video_path: Path
    metadata_path: Optional[Path]
    metadata: ClipMetadata
    duration_seconds: float
    quarantined: bool = False        # True when the narration gate moved the clip to the quarantine dir
    quarantine_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipResult":
        raw_meta_path = data.get("metadata_path")
        return cls(
            video_path=Path(data["video_path"]),
            metadata_path=Path(raw_meta_path) if raw_meta_path else None,
            metadata=ClipMetadata.from_dict(data["metadata"]),
            duration_seconds=float(data["duration_seconds"]),
            quarantined=bool(data.get("quarantined", False)),
            quarantine_reason=str(data.get("quarantine_reason") or ""),
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
    heatmap_spikes: list[TimeRange] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    selected_highlights: list[Highlight] = field(default_factory=list)
    clips: list[ClipResult] = field(default_factory=list)
    cache_dir: Optional[Path] = None
    analysis_complete: bool = False
    requested_max_clips: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        return cls(
            stream_info=StreamInfo.from_dict(data["stream_info"]),
            status=PipelineStatus(
                int(data.get("status") if data.get("status") is not None else PipelineStatus.PENDING)
            ),
            audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
            transcript=(
                TranscriptionResult.from_dict(data["transcript"])
                if data.get("transcript") is not None
                else None
            ),
            loudness_spikes=[
                TimeRange.from_dict(s) for s in (data.get("loudness_spikes") or [])
            ],
            chat_spikes=[
                TimeRange.from_dict(s) for s in (data.get("chat_spikes") or [])
            ],
            heatmap_spikes=[
                TimeRange.from_dict(s) for s in (data.get("heatmap_spikes") or [])
            ],
            highlights=[
                Highlight.from_dict(h) for h in (data.get("highlights") or [])
            ],
            selected_highlights=[
                Highlight.from_dict(h) for h in (data.get("selected_highlights") or [])
            ],
            clips=[ClipResult.from_dict(c) for c in (data.get("clips") or [])],
            cache_dir=Path(data["cache_dir"]) if data.get("cache_dir") else None,
            analysis_complete=bool(data.get("analysis_complete", False)),
            requested_max_clips=(
                int(data["requested_max_clips"])
                if data.get("requested_max_clips") is not None
                else None
            ),
        )
