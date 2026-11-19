"""Top-3 countdown compilation: one Short built from a stream's three best moments."""
from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from ..config import Config

logger = logging.getLogger(__name__)

MAX_COMPILATION_SECONDS = 57.0
_BED_GAIN = 0.18


def plan_item_windows(
    peak_offsets: list[float | None],
    source_durations: list[float],
    item_seconds: float = 14.0,
    max_total: float = MAX_COMPILATION_SECONDS,
) -> list[tuple[float, float]]:
    """Per-item (start, end) windows inside each source segment, centred ~40% before the peak.

    Each window is at most `item_seconds`, shrunk equally so the total stays <= `max_total`.
    """
    n = len(source_durations)
    if n == 0:
        return []
    per_item = max(1.0, min(float(item_seconds), float(max_total) / n))
    windows: list[tuple[float, float]] = []
    for peak, dur in zip(peak_offsets, source_durations):
        dur = max(0.0, float(dur))
        length = min(per_item, dur) if dur > 0 else per_item
        anchor = float(peak) if peak is not None else dur / 2.0
        start = anchor - 0.6 * length
        start = max(0.0, min(start, max(0.0, dur - length)))
        windows.append((round(start, 3), round(start + length, 3)))
    return windows


def build_countdown_filtergraph(
    narration_windows: list[tuple[float, float]],
    total: float,
    target_lufs: float | None = None,
) -> str:
    """FFmpeg filtergraph: stream bed (input 0) ducked under each narration (inputs 1..N).

    Narration inputs are normalised to 48 kHz stereo and delayed to their window start.
    When `target_lufs` is given the mixed track gets the same final loudnorm as single clips.
    """
    duck = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in narration_windows) or "0"
    parts = [
        "[0:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume='if({duck},{_BED_GAIN},1)':eval=frame[bed]"
    ]
    labels = ["[bed]"]
    for i, (start, _end) in enumerate(narration_windows, start=1):
        ms = int(round(start * 1000))
        parts.append(
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[n{i}]"
        )
        labels.append(f"[n{i}]")
    norm = f",loudnorm=I={target_lufs:g}:TP=-1.5:LRA=11,aresample=48000" if target_lufs is not None else ""
    parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,"
        f"atrim=0:{total:.3f}{norm}[aout]"
    )
    return ";".join(parts)


def _ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def build_countdown_ass(
    lines: list[str], windows: list[tuple[float, float]], caption_band_top: int = 1488
) -> str:
    """Minimal ASS script: one card per countdown line, shown in the caption band."""
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Countdown,Arial,76,&H00FFFFFF,&H000000FF,&H99000000,&H00000000,-1,0,0,0,"
        "100,100,0,0,1,6,0,8,60,60,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for line, (start, end) in zip(lines, windows):
        safe = (
            str(line).replace("\r", " ").replace("\n", " ").replace("{", "(").replace("}", ")")
        )
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Countdown,,0,0,"
            f"{caption_band_top + 40},,{safe}"
        )
    return header + "\n".join(events) + "\n"


def build_top3_compilation(
    item_segments: list[Path],
    peak_offsets: list[float | None],
    facecams: list[Any],
    countdown_lines: list[str],
    output_path: Path,
    config: Config,
    *,
    item_seconds: float | None = None,
    narrator: Any = None,
    info: dict[str, Any] | None = None,
) -> Path:
    """Render a countdown Short (#3 -> #1) from three downloaded segments.

    `item_segments`, `peak_offsets`, `facecams` and `countdown_lines` are given in play order
    (#3 first). Total length is capped at 57 s. When `info` is given it is filled with what
    actually happened: `mix_ran`, `narration_expected`, `spoken_seconds`, `final_duration`.
    """
    from moviepy import VideoFileClip, concatenate_videoclips

    from .captions import AssCaptionBurner
    from .composer import VideoComposer

    n = len(item_segments)
    if n < 2 or len(peak_offsets) != n or len(facecams) != n:
        raise ValueError("build_top3_compilation needs matching per-item inputs (>=2 items)")
    item_len = float(
        item_seconds
        if item_seconds is not None
        else config.get("transformation.top3_item_seconds", 14)
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(config.get("general.temp_dir", "./temp"))
    temp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="streamclipper_top3_", dir=str(temp_root)))

    composer = VideoComposer(config)
    composed: list[Any] = []
    final: Any = None
    try:
        durations = []
        for seg in item_segments:
            probe = VideoFileClip(str(seg))
            try:
                durations.append(float(probe.duration or 0.0))
            finally:
                probe.close()
        windows = plan_item_windows(peak_offsets, durations, item_len)
        for seg, cam, peak, win in zip(item_segments, facecams, peak_offsets, windows):
            composed.append(
                composer.compose(seg, cam, peak_offset=peak, trim_range=win)
            )
        final = concatenate_videoclips(composed, method="compose")
        total = float(final.duration or 0.0)
        raw_mp4 = work / "top3_raw.mp4"
        has_audio = getattr(final, "audio", None) is not None
        kwargs: dict[str, Any] = {
            "fps": int(config.get("video.fps", 30) or 30),
            "codec": "libx264",
            "preset": "ultrafast",
            "ffmpeg_params": ["-crf", "18"],
            "audio": has_audio,
            "logger": None,
        }
        if has_audio:
            kwargs["audio_codec"] = str(config.get("video.audio_codec", "aac") or "aac")
        try:
            final.write_videofile(str(raw_mp4), **kwargs)
        finally:
            for handle in [final, *composed]:
                for h in (handle, getattr(handle, "_source_clip", None)):
                    try:
                        if h is not None:
                            h.close()
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.debug("close failed: %s", exc)

        # Item start times on the compiled timeline (each composed clip's own duration)
        starts: list[float] = []
        cursor = 0.0
        for clip in composed:
            starts.append(cursor)
            cursor += float(getattr(clip, "duration", 0.0) or 0.0)

        current = raw_mp4
        spoken_seconds = 0.0
        mix_ran = False
        card_windows: list[tuple[float, float]] = []
        wavs: list[Path] = []
        if narrator is not None:
            for i, (line, start) in enumerate(zip(countdown_lines, starts)):
                wav = narrator.synthesize_speech(
                    line, work / f"countdown_{i}.wav", max_seconds=3.5
                )
                if wav is None:
                    continue
                dur = narrator._probe_audio_duration(wav) or 2.5
                wavs.append(Path(wav))
                spoken_seconds += float(dur)
                card_windows.append((start + 0.2, min(total, start + 0.2 + float(dur))))
        if wavs and has_audio:
            mixed = work / "top3_mixed.mp4"
            cmd = ["ffmpeg", "-y", "-i", str(raw_mp4)]
            for w in wavs:
                cmd += ["-i", str(w)]
            cmd += [
                "-filter_complex",
                build_countdown_filtergraph(
                    card_windows,
                    total,
                    target_lufs=float(config.get("transformation.target_lufs", -14) or -14),
                ),
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", str(config.get("video.audio_codec", "aac") or "aac"),
                str(mixed),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                current = mixed
                mix_ran = True
            except subprocess.CalledProcessError as exc:
                logger.warning("Countdown mix failed (%s); continuing without voice.", exc)
        if not mix_ran and has_audio and narrator is not None:
            normalize = getattr(narrator, "normalize_audio", None)
            if callable(normalize):
                try:
                    normed = normalize(current, work / "top3_norm.mp4")
                    if Path(normed).is_file():
                        current = Path(normed)
                except Exception as exc:
                    logger.warning("Top-3 loudness normalization failed (%s); continuing.", exc)

        codec = str(config.get("video.codec", "libx264") or "libx264")
        crf = int(config.get("video.crf", 23) or 23)
        cards = [
            (line, w) for line, w in zip(countdown_lines, [(s, min(total, s + 3.0)) for s in starts])
        ]
        if cards:
            ass_path = work / "countdown.ass"
            ass_path.write_text(
                build_countdown_ass([c[0] for c in cards], [c[1] for c in cards]),
                encoding="utf-8",
            )
            AssCaptionBurner(config).burn_captions(
                input_video=current, ass_path=ass_path, output_video=out, crf=crf, codec=codec
            )
        else:
            shutil.copyfile(current, out)
        if info is not None:
            info["mix_ran"] = mix_ran
            info["narration_expected"] = bool(
                narrator is not None and any(str(l).strip() for l in countdown_lines)
            )
            info["spoken_seconds"] = round(spoken_seconds, 3) if mix_ran else 0.0
            info["final_duration"] = round(total, 3)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
