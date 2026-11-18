"""Step-level caching for pipeline resume capability."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from .config import Config
from .models import (
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    StreamInfo,
    TimeRange,
    TranscriptionResult,
)


class StepCache:
    """Persists pipeline state after each major step for crash recovery."""

    def __init__(self, config: Config):
        self.cache_dir = Path(config.get("general.cache_dir", "./.cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, url: str) -> str:
        """Generate a stable cache key from a URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a given stream URL."""
        return self.cache_dir / f"{self.get_cache_key(url)}.json"

    def save_state(self, state: PipelineState) -> None:
        """Save current pipeline state and transcript to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self.get_cache_key(state.stream_info.url)
        cache_path = self.get_cache_path(state.stream_info.url)
        data = {
            "stream_info": state.stream_info.to_dict(),
            "status": int(state.status),
            "audio_path": str(state.audio_path) if state.audio_path else None,
            "has_transcript": state.transcript is not None,
            "analysis_complete": bool(state.analysis_complete),
            "loudness_spikes": [s.to_dict() for s in state.loudness_spikes],
            "chat_spikes": [s.to_dict() for s in state.chat_spikes],
            "highlights": [h.to_dict() for h in state.highlights],
            "selected_highlights": [h.to_dict() for h in state.selected_highlights],
            "clips": [c.to_dict() for c in state.clips],
            "updated_at": datetime.now().isoformat(),
        }
        if state.transcript is not None:
            transcript_path = self.cache_dir / f"{key}_transcript.json"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(state.transcript.to_json())

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_state(self, url: str) -> PipelineState | None:
        """Load cached pipeline state (including transcript and spikes) if it exists."""
        key = self.get_cache_key(url)
        cache_path = self.get_cache_path(url)
        if not cache_path.exists():
            return None

        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

        state = PipelineState(
            stream_info=StreamInfo.from_dict(data["stream_info"]),
            status=PipelineStatus(int(data["status"])),
            loudness_spikes=[
                TimeRange.from_dict(s) for s in (data.get("loudness_spikes") or [])
            ],
            chat_spikes=[
                TimeRange.from_dict(s) for s in (data.get("chat_spikes") or [])
            ],
            highlights=[
                Highlight.from_dict(h) for h in (data.get("highlights") or [])
            ],
            selected_highlights=[
                Highlight.from_dict(h) for h in (data.get("selected_highlights") or [])
            ],
            clips=[ClipResult.from_dict(c) for c in (data.get("clips") or [])],
            analysis_complete=bool(data.get("analysis_complete", False)),
        )
        if data.get("audio_path"):
            state.audio_path = Path(data["audio_path"])

        transcript_path = self.cache_dir / f"{key}_transcript.json"
        if data.get("has_transcript") and transcript_path.exists():
            with open(transcript_path, encoding="utf-8") as tf:
                state.transcript = TranscriptionResult.from_dict(json.load(tf))

        return state

    def clear(self, url: str) -> None:
        """Remove cached state and transcript for a URL."""
        key = self.get_cache_key(url)
        for path in [
            self.get_cache_path(url),
            self.cache_dir / f"{key}_transcript.json",
        ]:
            if path.exists():
                path.unlink()

    def list_all(self) -> list[PipelineState]:
        """List all cached pipeline states."""
        states: list[PipelineState] = []
        for cache_file in sorted(self.cache_dir.glob("*.json")):
            if cache_file.stem.endswith("_transcript"):
                continue
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                states.append(
                    PipelineState(
                        stream_info=StreamInfo.from_dict(data["stream_info"]),
                        status=PipelineStatus(int(data["status"])),
                        audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
                        highlights=[
                            Highlight.from_dict(h) for h in (data.get("highlights") or [])
                        ],
                        clips=[ClipResult.from_dict(c) for c in (data.get("clips") or [])],
                        analysis_complete=bool(data.get("analysis_complete", False)),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
        return states

    def find_by_audio_path(self, audio_path: Path | str) -> PipelineState | None:
        """Resolve a full cached PipelineState (with transcript & spikes) by its audio_path."""
        target = Path(audio_path)
        try:
            target_norm = os.path.normcase(str(target.resolve()))
        except OSError:
            target_norm = os.path.normcase(str(target))
        for state in self.list_all():
            if state.audio_path is None:
                continue
            try:
                candidate_norm = os.path.normcase(str(state.audio_path.resolve()))
            except OSError:
                candidate_norm = os.path.normcase(str(state.audio_path))
            if state.audio_path == target or candidate_norm == target_norm:
                return self.load_state(state.stream_info.url)
        return None
