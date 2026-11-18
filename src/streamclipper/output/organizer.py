"""Output directory management, filename sanitization, and companion metadata persistence."""
from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
import re
import subprocess

from ..config import Config
from ..models import ClipMetadata, ClipResult

logger = logging.getLogger(__name__)

_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_MULTISPACE_RE = re.compile(r"\s+")


class FileOrganizer:
    """Organizes exported clips into sanitized folder structures with companion metadata."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.output_dir = Path(config.get("general.output_dir", "./output"))
        self.structure_template = str(
            config.get("output.structure", "{game}/{streamer}/{date}")
            or "{game}/{streamer}/{date}"
        )
        self.filename_template = str(
            config.get("output.filename_template", "clip_{index:02d}")
            or "clip_{index:02d}"
        )
        self.save_metadata_json = bool(config.get("output.save_metadata_json", True))
        self.save_transcript_txt = bool(config.get("output.save_transcript_txt", True))
        raw_fmt = str(config.get("video.format", "mp4") or "mp4").strip().lstrip(".")
        self.video_format = self._sanitize_name(raw_fmt, fallback="mp4")

    @staticmethod
    def _sanitize_name(name: str, fallback: str = "unknown") -> str:
        """Strip Windows/POSIX invalid path characters (<>:\"/\\|?*) and leading/trailing dots/spaces."""
        cleaned = _INVALID_PATH_CHARS_RE.sub("", str(name or ""))
        cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip(" .")
        return cleaned or fallback

    def get_clip_path(
        self,
        game: str,
        streamer: str,
        index: int,
        date_str: str | None = None,
    ) -> Path:
        """Resolve sanitized output path honoring output.structure and output.filename_template."""
        safe_game = self._sanitize_name(game, fallback="UnknownGame")
        safe_streamer = self._sanitize_name(streamer, fallback="UnknownStreamer")
        today_str = date.today().isoformat()
        safe_date = (
            self._sanitize_name(date_str, fallback=today_str)
            if date_str is not None
            else today_str
        )

        try:
            rel_structure = self.structure_template.format(
                game=safe_game,
                streamer=safe_streamer,
                date=safe_date,
            )
        except (KeyError, ValueError, IndexError):
            logger.warning(
                "Malformed output.structure template %r; falling back to '{game}/{streamer}/{date}'.",
                self.structure_template,
            )
            rel_structure = f"{safe_game}/{safe_streamer}/{safe_date}"

        safe_parts: list[str] = []
        for raw_part in re.split(r"[/\\]+", rel_structure):
            if not raw_part or raw_part in (".", ".."):
                continue
            cleaned_part = self._sanitize_name(raw_part, fallback="")
            if cleaned_part:
                safe_parts.append(cleaned_part)

        clip_dir = (
            self.output_dir.joinpath(*safe_parts) if safe_parts else self.output_dir
        )
        clip_dir.mkdir(parents=True, exist_ok=True)

        default_stem = f"clip_{index:02d}"
        try:
            raw_stem = self.filename_template.format(
                index=index,
                game=safe_game,
                streamer=safe_streamer,
                date=safe_date,
            )
        except (KeyError, ValueError, IndexError):
            logger.warning(
                "Malformed output.filename_template %r; falling back to %r.",
                self.filename_template,
                default_stem,
            )
            raw_stem = default_stem

        stem = self._sanitize_name(raw_stem, fallback=default_stem)
        return clip_dir / f"{stem}.{self.video_format}"

    @staticmethod
    def _probe_duration_seconds(video_path: Path, metadata: ClipMetadata) -> float:
        """Probe actual video duration via ffprobe, falling back to metadata clip bounds."""
        if video_path.is_file() and video_path.stat().st_size > 0:
            try:
                proc = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                probed = float(proc.stdout.strip())
                if probed > 0.0:
                    return round(probed, 3)
            except (
                FileNotFoundError,
                OSError,
                subprocess.SubprocessError,
                ValueError,
            ):
                pass
        return round(max(0.0, float(metadata.clip_end) - float(metadata.clip_start)), 3)

    def save_clip(
        self,
        video_path: Path,
        metadata: ClipMetadata,
        transcript_text: str = "",
    ) -> ClipResult:
        """Persist companion metadata JSON and transcript TXT beside the exported clip."""
        final_video = Path(video_path)
        final_video.parent.mkdir(parents=True, exist_ok=True)

        metadata_path: Path | None = None
        if self.save_metadata_json:
            metadata_path = final_video.with_suffix(".json")
            metadata_path.write_text(metadata.to_json(), encoding="utf-8")

        if self.save_transcript_txt and transcript_text and str(transcript_text).strip():
            transcript_path = final_video.with_name(f"{final_video.stem}_transcript.txt")
            transcript_path.write_text(str(transcript_text), encoding="utf-8")

        duration = self._probe_duration_seconds(final_video, metadata)
        return ClipResult(
            video_path=final_video,
            metadata_path=metadata_path,
            metadata=metadata,
            duration_seconds=duration,
        )
