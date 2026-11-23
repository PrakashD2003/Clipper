"""Clip exporter coordinating MoviePy rendering, narrator ducking, and ASS burn-in."""
from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from moviepy import CompositeVideoClip

from ..assembly.captions import AssCaptionBurner
from ..assembly.narrator import TransformativeNarrator
from ..config import Config
from ..models import WordTimestamp

logger = logging.getLogger(__name__)


def _log_ffmpeg_stderr(exc: subprocess.CalledProcessError, label: str) -> None:
    """Log the last 500 chars of a failed FFmpeg run's stderr at warning level."""
    stderr = exc.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    logger.warning("%s failed (exit %s): %s", label, exc.returncode, str(stderr or "")[-500:])


def _close_clip_handles(clip: Any) -> None:
    """Close a MoviePy clip and all attached _source_clip / _source_clips handles safely.

    CRITICAL (Windows): Must run inside a try/finally block before deleting or overwriting
    intermediate temp files so ffmpeg/MoviePy readers do not trigger PermissionError.
    """
    if clip is None:
        return

    seen: set[int] = set()
    stack: list[Any] = [clip]

    while stack:
        current = stack.pop()
        if current is None:
            continue
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        source_clip = getattr(current, "_source_clip", None)
        if source_clip is not None and id(source_clip) not in seen:
            stack.append(source_clip)

        source_clips = getattr(current, "_source_clips", None)
        if source_clips:
            try:
                for item in source_clips:
                    if item is not None and id(item) not in seen:
                        stack.append(item)
            except TypeError:
                pass

        close_fn = getattr(current, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.debug("Ignoring error while closing MoviePy clip handle: %s", exc)


class Exporter:
    """Renders composed 9:16 clips, mixes YPP narration, and burns ASS captions + chat ticker."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.codec = str(config.get("video.codec", "libx264") or "libx264").strip().lower()
        self.audio_codec = str(
            config.get("video.audio_codec", "aac") or "aac"
        ).strip().lower()
        raw_fps = config.get("video.fps", 30)
        self.fps = int(raw_fps) if raw_fps is not None else 30
        raw_crf = config.get("video.crf", 19)
        self.crf = int(raw_crf) if raw_crf is not None else 19
        self.preset = str(config.get("video.preset", "p4") or "p4").strip()
        self.libx264_preset = str(config.get("video.libx264_preset", "medium") or "medium").strip()
        self.bitrate = str(config.get("video.bitrate", "12M") or "12M").strip()
        self.maxrate = str(config.get("video.maxrate", "16M") or "16M").strip()
        self.bufsize = str(config.get("video.bufsize", "24M") or "24M").strip()
        self.pix_fmt = str(config.get("video.pix_fmt", "yuv420p") or "yuv420p").strip()
        self.temp_dir = Path(config.get("general.temp_dir", "./temp"))
        self.caption_burner = AssCaptionBurner(config)
        self.narrator = TransformativeNarrator(config)
        # What the last export() actually did (mix ran, probed narration seconds, ...).
        self.last_export_info: dict[str, Any] = self._empty_export_info()

    @staticmethod
    def _empty_export_info() -> dict[str, Any]:
        return {
            "mix_ran": False,
            "narration_expected": False,
            "intro_expected": False,
            "breakdown_expected": False,
            "intro_seconds": 0.0,
            "outro_seconds": 0.0,
            "callout_seconds": 0.0,
            "breakdown_seconds": 0.0,
            "spoken_seconds": 0.0,
            "final_duration": 0.0,
        }

    @staticmethod
    def _notify_progress(progress_callback: Callable | None, value: float) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(value)
        except Exception as exc:
            logger.debug("Progress callback raised an exception: %s", exc)

    def _resolve_effective_mode(self, mode: str | None) -> str:
        """Resolve per-export transformation mode honoring explicit overrides."""
        if mode is not None and str(mode).strip():
            return str(mode).strip().lower()
        if not self.narrator.enabled:
            return "raw"
        return str(self.narrator.mode or "auto").strip().lower()

    def _build_finalize_cmd(
        self,
        input_video: Path,
        output_video: Path,
        codec: str,
    ) -> list[str]:
        """Build single-pass FFmpeg finalization command when ASS burn-in is not needed."""
        normalized_codec = str(codec or "libx264").strip().lower()
        if normalized_codec == "h264_nvenc":
            eff_p = self.preset or "p4"
            v_args = ["-c:v", "h264_nvenc", "-preset", eff_p, "-cq", str(self.crf)]
            if self.bitrate:
                v_args.extend(["-rc:v", "vbr", "-b:v", str(self.bitrate)])
                if self.maxrate:
                    v_args.extend(["-maxrate", str(self.maxrate)])
                if self.bufsize:
                    v_args.extend(["-bufsize", str(self.bufsize)])
                v_args.extend(["-spatial-aq", "1", "-temporal-aq", "1"])
            if self.pix_fmt:
                v_args.extend(["-pix_fmt", str(self.pix_fmt)])
        else:
            v_args = ["-c:v", "libx264", "-crf", str(self.crf)]
            eff_p = self.preset
            if eff_p in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
                eff_p = getattr(self, "libx264_preset", "medium") or "medium"
            if eff_p:
                v_args.extend(["-preset", str(eff_p)])
            if self.pix_fmt:
                v_args.extend(["-pix_fmt", str(self.pix_fmt)])
        return [
            "ffmpeg",
            "-y",
            "-i",
            str(Path(input_video).resolve()),
            *v_args,
            "-c:a",
            "copy",
            str(Path(output_video).resolve()),
        ]

    def _finalize_without_captions(
        self,
        input_video: Path,
        output_video: Path,
    ) -> Path:
        """Transcode intermediate ultrafast CRF 18 MP4 to final codec/CRF (with NVENC fallback)."""
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        selected_codec = str(self.codec or "libx264").strip().lower()
        cmd = self._build_finalize_cmd(input_video, out_path, selected_codec)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            _log_ffmpeg_stderr(exc, f"FFmpeg {selected_codec} finalization")
            if selected_codec != "h264_nvenc":
                raise
            logger.warning(
                "FFmpeg h264_nvenc finalization failed; falling back to libx264."
            )
            fallback_cmd = self._build_finalize_cmd(input_video, out_path, "libx264")
            try:
                subprocess.run(fallback_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc2:
                _log_ffmpeg_stderr(exc2, "FFmpeg libx264 finalization")
                raise
        return out_path

    def export(
        self,
        clip: CompositeVideoClip,
        words: list[WordTimestamp],
        clip_start_offset: float,
        output_path: Path,
        progress_callback: Callable | None = None,
        chat_burst: list[dict[str, Any]] | None = None,
        narration_intro: str = "",
        narration_outro: str = "",
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        mode: str | None = None,
        breakdown_commentary: str = "",
        voice_breakdown_path: Path | None = None,
        replay_start: float | None = None,
        breakdown_callout: str = "",
        layout: Any = None,
        callout: str = "",
        callout_start: float | None = None,
        duck_full: bool = False,
        language: str | None = None,
        style_preset: dict[str, Any] | None = None,
        caption_style: str | None = None,
        moment_type: str | None = None,
    ) -> Path:
        """Render composed clip, mix narrator ducking, burn ASS captions+chat ticker, and release handles."""
        self.last_export_info = self._empty_export_info()
        if language:
            self.narrator.language = str(language).strip().lower()
        final_output = Path(output_path)
        final_output.parent.mkdir(parents=True, exist_ok=True)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(
            tempfile.mkdtemp(prefix="streamclipper_export_", dir=str(self.temp_dir))
        )

        handles_closed = False

        def _release_handles_once() -> None:
            nonlocal handles_closed
            if not handles_closed:
                handles_closed = True
                _close_clip_handles(clip)

        try:
            self._notify_progress(progress_callback, 0.0)
            clip_duration = float(getattr(clip, "duration", 0.0) or 0.0)
            has_audio = getattr(clip, "audio", None) is not None

            raw_mp4 = work_dir / "intermediate_raw.mp4"
            write_kwargs: dict[str, Any] = {
                "fps": self.fps,
                "codec": "libx264",
                "preset": "ultrafast",
                "ffmpeg_params": ["-crf", "18"],
                "audio": has_audio,
                "logger": None,
            }
            if has_audio:
                write_kwargs["audio_codec"] = self.audio_codec
                write_kwargs["temp_audiofile"] = str(work_dir / "temp_audio.m4a")
                write_kwargs["remove_temp"] = True

            # Step 1 & 2: Render high-quality intermediate MP4 and IMMEDIATELY release all MoviePy handles
            try:
                clip.write_videofile(str(raw_mp4), **write_kwargs)
            finally:
                _release_handles_once()

            self._notify_progress(progress_callback, 0.5)

            # Step 3: Resolve effective mode and apply narrator ducking (-c:v copy)
            current_video = raw_mp4
            effective_mode = self._resolve_effective_mode(mode)
            has_narration_input = bool(
                (narration_intro and str(narration_intro).strip())
                or (narration_outro and str(narration_outro).strip())
                or voice_intro_path is not None
                or voice_outro_path is not None
                or (breakdown_commentary and str(breakdown_commentary).strip())
                or voice_breakdown_path is not None
                or (callout and str(callout).strip())
            )
            intro_dur: float | None = None
            outro_dur: float | None = None
            eff_intro: float | None = None
            eff_outro: float | None = None
            breakdown_window: tuple[float | None, float | None] | None = None
            callout_window: tuple[float, float] | None = None
            mix_ran = False
            breakdown_dur: float | None = None
            callout_dur: float | None = None
            self.last_export_info["narration_expected"] = bool(
                effective_mode != "raw" and has_narration_input
            )
            self.last_export_info["intro_expected"] = bool(
                effective_mode != "raw"
                and (
                    (narration_intro and str(narration_intro).strip())
                    or voice_intro_path is not None
                )
            )
            self.last_export_info["breakdown_expected"] = bool(
                effective_mode == "breakdown"
                and (
                    (breakdown_commentary and str(breakdown_commentary).strip())
                    or voice_breakdown_path is not None
                )
            )

            if effective_mode != "raw" and has_narration_input:
                orig_enabled = self.narrator.enabled
                orig_mode = self.narrator.mode
                override_needed = (not orig_enabled) or (
                    str(orig_mode).strip().lower() == "raw"
                )
                try:
                    if override_needed:
                        self.narrator.enabled = True
                        self.narrator.mode = effective_mode

                    prep_kwargs: dict[str, Any] = {
                        "narration_intro": narration_intro,
                        "narration_outro": narration_outro,
                        "voice_intro_path": voice_intro_path,
                        "voice_outro_path": voice_outro_path,
                        "temp_dir": work_dir,
                        "clip_duration": clip_duration or None,
                    }
                    if language is not None:
                        prep_kwargs["language"] = language
                    try:
                        intro_wav, outro_wav = self.narrator.prepare_narration_tracks(**prep_kwargs)
                    except TypeError:
                        prep_kwargs.pop("language", None)
                        intro_wav, outro_wav = self.narrator.prepare_narration_tracks(**prep_kwargs)

                    breakdown_wav = None
                    if effective_mode == "breakdown" and (
                        (breakdown_commentary and str(breakdown_commentary).strip())
                        or voice_breakdown_path is not None
                    ):
                        bd_kwargs: dict[str, Any] = {
                            "breakdown_commentary": breakdown_commentary,
                            "voice_breakdown_path": voice_breakdown_path,
                            "temp_dir": work_dir,
                        }
                        if language is not None:
                            bd_kwargs["language"] = language
                        try:
                            breakdown_wav = self.narrator.prepare_breakdown_track(**bd_kwargs)
                        except TypeError:
                            bd_kwargs.pop("language", None)
                            breakdown_wav = self.narrator.prepare_breakdown_track(**bd_kwargs)

                    callout_wav = None
                    if callout and str(callout).strip():
                        prep_callout = getattr(self.narrator, "prepare_callout_track", None)
                        if callable(prep_callout):
                            co_kwargs: dict[str, Any] = {"temp_dir": work_dir}
                            if language is not None:
                                co_kwargs["language"] = language
                            try:
                                callout_wav = prep_callout(callout, **co_kwargs)
                            except TypeError:
                                co_kwargs.pop("language", None)
                                callout_wav = prep_callout(callout, **co_kwargs)

                    if (
                        intro_wav is not None
                        or outro_wav is not None
                        or breakdown_wav is not None
                        or callout_wav is not None
                    ):
                        intro_dur = self.narrator._probe_audio_duration(intro_wav)
                        outro_dur = self.narrator._probe_audio_duration(outro_wav)
                        breakdown_dur = self.narrator._probe_audio_duration(breakdown_wav)
                        if breakdown_dur is not None and breakdown_dur > 0.01:
                            # Same placement the breakdown ducking filtergraph uses, so
                            # caption cards/suppression match what viewers hear.
                            windows = self.narrator.compute_breakdown_windows(
                                intro_dur,
                                outro_dur,
                                breakdown_dur,
                                clip_duration,
                                replay_start,
                            )
                            eff_intro = windows["eff_intro"]
                            eff_outro = windows["eff_outro"]
                            breakdown_window = (
                                windows["breakdown_start"],
                                windows["breakdown_end"],
                            )
                        else:
                            eff_intro, eff_outro = (
                                self.narrator.get_effective_narration_durations(
                                    intro_dur, outro_dur, clip_duration
                                )
                            )
                        mixed_mp4 = work_dir / "intermediate_mixed.mp4"
                        breakdown_kwargs: dict[str, Any] = (
                            {"breakdown_wav": breakdown_wav, "replay_start_time": replay_start}
                            if breakdown_wav is not None
                            else {}
                        )
                        if callout_wav is not None:
                            c_start = (
                                float(callout_start)
                                if callout_start is not None
                                else 0.4 * clip_duration
                            )
                            breakdown_kwargs["callout_wav"] = callout_wav
                            breakdown_kwargs["callout_start"] = c_start
                            callout_dur = self.narrator._probe_audio_duration(callout_wav)
                            c_dur = callout_dur or 2.0
                            callout_window = (c_start, min(clip_duration, c_start + float(c_dur)))
                        if duck_full:
                            breakdown_kwargs["duck_full"] = True
                        mixed_path = None
                        try:
                            mixed_path = self.narrator.mix_narration_with_ducking(
                                input_video=raw_mp4,
                                output_video=mixed_mp4,
                                intro_wav=intro_wav,
                                outro_wav=outro_wav,
                                clip_duration=clip_duration,
                                **breakdown_kwargs,
                            )
                        except Exception as exc:
                            if isinstance(exc, subprocess.CalledProcessError):
                                _log_ffmpeg_stderr(exc, "Narration mix")
                            logger.warning(
                                "Narration mix failed (%s); keeping un-narrated video.",
                                exc,
                            )
                        if mixed_path:
                            current_video = Path(mixed_path)
                            mix_ran = True
                            parts = {
                                "intro_seconds": intro_dur,
                                "outro_seconds": outro_dur,
                                "callout_seconds": callout_dur,
                                "breakdown_seconds": breakdown_dur,
                            }
                            for key, val in parts.items():
                                self.last_export_info[key] = round(float(val or 0.0), 3)
                        else:
                            if not mixed_path:
                                logger.warning(
                                    "Narration mix produced no output; keeping un-narrated video."
                                )
                            intro_dur = outro_dur = None
                            callout_dur = breakdown_dur = None
                            eff_intro = eff_outro = None
                            breakdown_window = None
                            callout_window = None
                finally:
                    if override_needed:
                        self.narrator.enabled = orig_enabled
                        self.narrator.mode = orig_mode

            if not mix_ran:
                normalize = getattr(self.narrator, "normalize_audio", None)
                if callable(normalize) and has_audio:
                    try:
                        normalized = normalize(current_video, work_dir / "intermediate_norm.mp4")
                        if isinstance(normalized, (str, Path)) and Path(normalized).exists():
                            current_video = Path(normalized)
                    except Exception as exc:
                        logger.warning("Audio normalization failed (%s); continuing.", exc)

            self._notify_progress(progress_callback, 0.75)

            # Step 4: Single-pass ASS burn-in (Hormozi + ChatTicker) or direct codec finalization
            effective_chat_burst = None if effective_mode == "raw" else chat_burst
            has_words = bool(words)
            has_chat = bool(
                self.caption_burner.show_chat_ticker and effective_chat_burst
            )

            is_raw_mode = effective_mode == "raw"
            has_card = (not is_raw_mode) and bool(
                (
                    eff_intro is not None
                    and narration_intro
                    and str(narration_intro).strip()
                )
                or (
                    eff_outro is not None
                    and narration_outro
                    and str(narration_outro).strip()
                )
                or (
                    breakdown_window is not None
                    and breakdown_callout
                    and str(breakdown_callout).strip()
                )
            )

            has_callout_card = (not is_raw_mode) and callout_window is not None
            if has_words or has_chat or has_card or has_callout_card:
                ass_path = work_dir / "captions.ass"
                is_raw = effective_mode == "raw"
                layout_kwargs: dict[str, Any] = {}
                if layout is not None:
                    layout_kwargs["layout"] = layout
                if has_callout_card:
                    layout_kwargs["callout"] = callout
                    layout_kwargs["callout_window"] = callout_window
                burner_style_kwargs: dict[str, Any] = {}
                if style_preset:
                    burner_style_kwargs["style_preset"] = style_preset
                if caption_style:
                    burner_style_kwargs["caption_style"] = caption_style
                if moment_type:
                    burner_style_kwargs["moment_type"] = moment_type

                self.caption_burner.generate_ass(
                    words=words or [],
                    clip_start_offset=clip_start_offset,
                    ass_path=ass_path,
                    chat_burst=effective_chat_burst,
                    narration_intro="" if is_raw else narration_intro,
                    narration_outro="" if is_raw else narration_outro,
                    intro_dur=None if is_raw else eff_intro,
                    outro_dur=None if is_raw else eff_outro,
                    clip_duration=clip_duration,
                    breakdown_callout="" if is_raw else breakdown_callout,
                    breakdown_window=None if is_raw else breakdown_window,
                    **layout_kwargs,
                    **burner_style_kwargs,
                )
                try:
                    result_path = self.caption_burner.burn_captions(
                        input_video=current_video,
                        ass_path=ass_path,
                        output_video=final_output,
                        crf=self.crf,
                        codec=self.codec,
                    )
                except subprocess.CalledProcessError as exc:
                    _log_ffmpeg_stderr(exc, "FFmpeg caption burn-in")
                    raise
            else:
                result_path = self._finalize_without_captions(
                    current_video, final_output
                )

            info = self.last_export_info
            info["mix_ran"] = bool(mix_ran)
            info["spoken_seconds"] = round(
                sum(
                    float(info[k])
                    for k in (
                        "intro_seconds",
                        "outro_seconds",
                        "callout_seconds",
                        "breakdown_seconds",
                    )
                ),
                3,
            )
            try:
                info["final_duration"] = round(
                    float(self.narrator._probe_audio_duration(Path(result_path)) or 0.0), 3
                )
            except Exception as exc:
                logger.debug("Could not probe final export duration: %s", exc)
            self._notify_progress(progress_callback, 1.0)
            return Path(result_path)
        finally:
            # Step 5: Guarantee handles are closed BEFORE unlinking intermediate temp directory
            _release_handles_once()
            shutil.rmtree(work_dir, ignore_errors=True)
