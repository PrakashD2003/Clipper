"""Topic Shorts 9:16 vertical video assembly: teaser and standalone shorts with persistent hook,
credit tags, narration ducking, and single lossy encode.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips

from streamclipper.assembly.captions import (
    AssCaptionBurner,
    _escape_ass_text,
    _escape_ffmpeg_ass_filename,
    _sec_to_ass_time,
    resolve_font,
)
from streamclipper.assembly.composer import VideoComposer
from streamclipper.assembly.longform import _probe_duration
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config
from streamclipper.download.video import download_segment
from streamclipper.output.exporter import _close_clip_handles, _log_ffmpeg_stderr
from streamclipper.output.topic_post import write_short_post
from streamclipper.topics.models import EvidenceClip, ShortPlan, TopicPlan
from streamclipper.topics.planner import plan_dir, save_plan

logger = logging.getLogger(__name__)


def _sanitize_path_segment(value: str, default: str = "unknown") -> str:
    """Sanitize path component for safe filesystem usage."""
    cleaned = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(value or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned if cleaned else default


def _normalize_slot(slot: str) -> str:
    """Normalize slot name or alias into standard ('teaser', 'short_a', 'short_b')."""
    s = str(slot or "").strip().lower()
    if s in ("teaser", "short_teaser"):
        return "teaser"
    if s in ("short_a", "a"):
        return "short_a"
    if s in ("short_b", "b"):
        return "short_b"
    return s


def _calculate_clip_trims(
    clips_with_durations: list[tuple[EvidenceClip, Path, float]],
    target_min: float = 35.0,
    target_max: float = 60.0,
    min_clip_dur: float = 4.0,
    peak_offsets: list[float | None] | None = None,
) -> list[tuple[EvidenceClip, Path, float, float]]:
    """Trim each evidence clip around its middle (or peak_offset) so the Short lands in 35–60 s.

    Guarantees that each clip keeps >= 4 s (or its full duration if shorter than 4 s).
    """
    if not clips_with_durations:
        return []

    num_clips = len(clips_with_durations)
    total_raw = sum(dur for _, _, dur in clips_with_durations)

    # Determine target duration within [target_min, target_max]
    if total_raw <= target_max:
        target_total = total_raw
    else:
        # Clamped target duration in the 45-50s sweet spot
        target_total = min(target_max, max(target_min, 45.0))

    # Ensure target_total is at least num_clips * min_clip_dur if raw footage allows
    min_possible = sum(min(dur, min_clip_dur) for _, _, dur in clips_with_durations)
    target_total = max(target_total, min_possible)

    # Allocate durations per clip proportionally
    if total_raw > target_total and total_raw > 0:
        ratio = target_total / total_raw
        allocated = []
        for _, _, raw_dur in clips_with_durations:
            alloc = max(min(raw_dur, min_clip_dur), min(raw_dur, raw_dur * ratio))
            allocated.append(alloc)
    else:
        allocated = [dur for _, _, dur in clips_with_durations]

    # Trim each clip around peak_offset or its middle
    results: list[tuple[EvidenceClip, Path, float, float]] = []
    for idx, ((clip, path, raw_dur), alloc_dur) in enumerate(zip(clips_with_durations, allocated)):
        peak_offset = (
            peak_offsets[idx]
            if peak_offsets is not None and idx < len(peak_offsets)
            else None
        )
        if alloc_dur < raw_dur:
            if (
                peak_offset is not None
                and not isinstance(peak_offset, bool)
                and isinstance(peak_offset, (int, float))
                and math.isfinite(peak_offset)
                and 0 < peak_offset < raw_dur
            ):
                t_start = peak_offset - 0.60 * alloc_dur
                if t_start < 0.0:
                    t_start = 0.0
                    t_end = min(raw_dur, t_start + alloc_dur)
                elif t_start + alloc_dur > raw_dur:
                    t_end = raw_dur
                    t_start = max(0.0, t_end - alloc_dur)
                else:
                    t_end = t_start + alloc_dur
            else:
                mid = raw_dur / 2.0
                t_start = max(0.0, mid - alloc_dur / 2.0)
                t_end = min(raw_dur, t_start + alloc_dur)
                t_start = max(0.0, t_end - alloc_dur)
        else:
            t_start = 0.0
            t_end = raw_dur
        results.append((clip, path, t_start, t_end))

    return results


def _build_shorts_ass(
    ass_path: Path,
    hook_title: str,
    credit_windows: list[tuple[float, float, str]],
    narration_windows: list[tuple[float, float, str]],
    total_duration: float,
    font_name: str | None = None,
    config: Config | None = None,
) -> Path:
    """Generate 9:16 vertical ASS file with persistent hook title, credit tags,
    and active-word highlighted Hormozi captions positioned in the safe zone (MarginV >= 460).
    """
    ass_path = Path(ass_path)
    ass_path.parent.mkdir(parents=True, exist_ok=True)

    burner = AssCaptionBurner(config) if config is not None else AssCaptionBurner(Config())
    if font_name is not None and str(font_name).strip():
        resolved_header_font = resolve_font(str(font_name).strip())
        burner.font = resolved_header_font
    else:
        resolved_header_font = burner.font

    # Style colors & borders for Hormozi captions
    bgr_raw = burner.highlight_bgr.strip().replace("&", "")
    if bgr_raw.startswith("H") or bgr_raw.startswith("h"):
        bgr_raw = bgr_raw[1:]
    if len(bgr_raw) == 8:
        secondary_style_color = f"&H{bgr_raw.upper()}"
    elif len(bgr_raw) == 6:
        secondary_style_color = f"&H00{bgr_raw.upper()}"
    else:
        secondary_style_color = "&H0000E6FF"

    # Strictly position above YouTube Shorts bottom ~420px title/channel UI zone (MarginV >= 460)
    hormozi_margin_v = max(460, burner.margin_v)

    if burner.box_style:
        hormozi_border_style = 3
        hormozi_outline = burner.box_padding
        hormozi_shadow = 0
        hormozi_outline_col = burner.box_color
        hormozi_back_col = burner.box_color
    else:
        hormozi_border_style = 1
        hormozi_outline = burner.stroke_width
        hormozi_shadow = burner.shadow_offset
        hormozi_outline_col = burner.stroke_color
        hormozi_back_col = burner.shadow_color

    if burner.style == "karaoke":
        style_primary_col = secondary_style_color
        style_secondary_col = burner.primary_color
    else:
        style_primary_col = burner.primary_color
        style_secondary_col = secondary_style_color

    header = f"""[Script Info]
Title: StreamClipper Topic Short
ScriptType: v4.00+
PlayResX: {burner.width}
PlayResY: {burner.height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookTitle,{resolved_header_font},52,&H0000FFFF,&H000000FF,&H99000000,&H99000000,1,0,0,0,100,100,0,0,3,3,0,8,60,60,220,1
Style: CreditTag,{resolved_header_font},28,&H00FFFFFF,&H000000FF,&H99000000,&H99000000,0,0,0,0,100,100,0,0,3,2,0,1,40,40,280,1
Style: Hormozi,{burner.font},{burner.font_size},{style_primary_col},{style_secondary_col},{hormozi_outline_col},{hormozi_back_col},-1,0,0,0,100,100,0,0,{hormozi_border_style},{hormozi_outline},{hormozi_shadow},2,60,60,{hormozi_margin_v},1
Style: NarrationCard,{burner.font},{burner.font_size},{style_primary_col},{style_secondary_col},{hormozi_outline_col},{hormozi_back_col},-1,0,0,0,100,100,0,0,{hormozi_border_style},{hormozi_outline},{hormozi_shadow},2,60,60,{hormozi_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    # 1. Persistent Hook Title (shown across the entire Short duration)
    clean_hook = (hook_title or "").strip()
    if clean_hook and total_duration > 0:
        escaped_hook = _escape_ass_text(clean_hook)
        events.append(
            f"Dialogue: 0,0:00:00.00,{_sec_to_ass_time(total_duration)},HookTitle,,0,0,0,,{escaped_hook}"
        )

    # 2. Credit tags for each clip during its playback window
    for st, et, credit_text in credit_windows:
        st = max(0.0, float(st))
        et = min(float(total_duration), float(et)) if total_duration > 0 else float(et)
        if et <= st or not credit_text:
            continue
        escaped_credit = _escape_ass_text(credit_text)
        events.append(
            f"Dialogue: 0,{_sec_to_ass_time(st)},{_sec_to_ass_time(et)},CreditTag,,0,0,0,,{escaped_credit}"
        )

    # 3. Active-word highlighted Hormozi narration events (chunked 2-3 words with pop/bounce)
    primary_inline_bgr = "&H" + burner.primary_color[4:].rstrip("&") + "&"
    for st, et, narr_text in narration_windows:
        st = max(0.0, float(st))
        et = min(float(total_duration), float(et)) if total_duration > 0 else float(et)
        if et <= st or not narr_text:
            continue
        chunk_events = burner._generate_narration_chunk_events(
            narr_text, st, et, primary_inline_bgr
        )
        events.extend(chunk_events)

    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return ass_path


def render_topic_shorts(
    config: Config,
    plan: TopicPlan,
    *,
    slots: tuple[str, ...] | list[str] | str = ("teaser", "short_a", "short_b"),
    progress: Any = None,
    draft: bool = False,
) -> dict[str, Any]:
    """Render 9:16 vertical topic Shorts from an approved TopicPlan.

    Performs:
    1. Status checks: refuses unapproved plans.
    2. Gathers evidence clips from each short's point_indexes (reuses T5 segments).
    3. Trims clips around their middle so total duration lands in 35–60 s (>= 4 s per clip).
    4. Composes each clip into 9:16 with blur_fill and concatenates.
    5. Voices narration (intro at start, callout before strongest clip, outro at end) with plan.language.
    6. Mixes narration with ducking.
    7. Applies the narration gate: quarantines the Short if expected intro voice did not mix.
    8. Generates 9:16 ASS overlay (persistent hook title, credit tags, narration cards).
    9. Single final lossy encode with loudnorm (-14 LUFS) and ASS burn-in.
    10. Writes post files and updates plan.outputs.
    """
    if plan.status not in ("approved", "rendered"):
        raise ValueError(
            f"Topic plan '{plan.plan_id}' must be 'approved' to render (current: '{plan.status}')."
        )

    render_config = config
    if draft:
        render_config = copy.copy(config)
        render_config._data = copy.deepcopy(config._data)
        render_config._data.setdefault("transformation", {})["tts_engine"] = "edge-tts"
        render_config._data.setdefault("transformation", {})["hinglish_tts_engines"] = ["edge-tts"]
    config = render_config

    if isinstance(slots, str):
        raw_slots = [_normalize_slot(s) for s in slots.split(",") if s.strip()]
        target_slots = tuple(raw_slots) if raw_slots else ("teaser", "short_a", "short_b")
    else:
        target_slots = tuple(_normalize_slot(s) for s in slots if str(s).strip())

    out_dir = plan_dir(config, plan.game, plan.topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    qname = _sanitize_path_segment(
        str(config.get("output.quarantine_dir", "_quarantine") or "_quarantine"),
        "_quarantine",
    )
    quarantine_base = (
        Path(config.get("general.output_dir", "./output"))
        / qname
        / _sanitize_path_segment(plan.game, "UnknownGame")
        / "topics"
        / _sanitize_path_segment(plan.topic_id, "unknown")
    )

    temp_base = Path(config.get("general.temp_dir", "./temp"))
    topic_temp_dir = temp_base / "topics" / plan.plan_id
    topic_temp_dir.mkdir(parents=True, exist_ok=True)

    matching_shorts = [s for s in plan.shorts if _normalize_slot(s.slot) in target_slots]
    if not matching_shorts:
        logger.warning(
            "No shorts found in plan '%s' matching slots %s; available: %s",
            plan.plan_id,
            target_slots,
            [s.slot for s in plan.shorts],
        )

    composer = VideoComposer(config)
    narrator = TransformativeNarrator(config)
    if plan.language:
        narrator.language = str(plan.language).strip().lower()

    codec = str(config.get("video.codec", "libx264") or "libx264").strip().lower()
    crf = int(config.get("video.crf", 23) or 23)
    font_name = str(config.get("captions.font", "Arial") or "Arial")

    rendered_shorts: list[dict[str, Any]] = []
    quarantined_shorts: list[dict[str, Any]] = []

    total_shorts = len(matching_shorts)
    for idx, short in enumerate(matching_shorts, start=1):
        if progress and getattr(progress, "cancelled", False):
            logger.info("Topic Shorts render cancelled by user.")
            break

        slot_name = _normalize_slot(short.slot)
        base_stem = slot_name if slot_name.startswith("short_") else f"short_{slot_name}"
        filename = f"{base_stem}_draft.mp4" if draft else f"{base_stem}.mp4"
        final_short_path = out_dir / filename

        if progress and hasattr(progress, "start_clip"):
            progress.start_clip(idx, total_shorts)
        if progress and hasattr(progress, "stage"):
            progress.stage(f"rendering short {slot_name}")

        temp_short_dir = topic_temp_dir / f"short_{slot_name}"
        temp_short_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gather evidence clips from short.point_indexes
        points_map = {p.index: p for p in plan.points}
        short_clips: list[EvidenceClip] = []
        for p_idx in short.point_indexes:
            pt = points_map.get(p_idx)
            if pt and pt.evidence:
                short_clips.extend(pt.evidence)

        if not short_clips:
            # Fallback to first available point evidence in plan
            for pt in plan.points:
                if pt.evidence:
                    short_clips.extend(pt.evidence)
                    break

        if not short_clips:
            logger.error("No evidence clips found for short %s in plan %s", slot_name, plan.plan_id)
            continue

        # 2. Locate or download segments in topic_temp_dir and probe durations
        clips_with_durations: list[tuple[EvidenceClip, Path, float]] = []
        for clip in short_clips:
            c_start = max(0.0, float(clip.start))
            c_end = max(c_start + 0.1, float(clip.end))
            max_s = 10.0 if clip.kind == "official" else 20.0
            if (c_end - c_start) > max_s:
                c_end = c_start + max_s

            seg_path = download_segment(
                clip.url,
                c_start,
                c_end,
                topic_temp_dir,
                buffer_seconds=0.0,
                video_id=clip.source_id or None,
            )
            dur = _probe_duration(seg_path)
            if dur <= 0:
                dur = max(4.0, c_end - c_start)
            clips_with_durations.append((clip, seg_path, dur))

        # 3. Calculate trims around peak_offsets / middle so total duration lands in 35–60 s
        peak_offsets: list[float | None] = []
        for c, _, _ in clips_with_durations:
            if isinstance(c, dict):
                po = c.get("peak_offset")
                pt = c.get("peak_time")
                c_start = float(c.get("start", 0.0))
                c_end = float(c.get("end", 0.0))
            else:
                po = getattr(c, "peak_offset", None)
                pt = getattr(c, "peak_time", None)
                c_start = float(getattr(c, "start", 0.0))
                c_end = float(getattr(c, "end", 0.0))

            if po is None and pt is not None:
                try:
                    pt_val = float(pt)
                    if c_start < pt_val < c_end:
                        po = pt_val - c_start
                except (TypeError, ValueError):
                    pass
            peak_offsets.append(po)
        trimmed_clips = _calculate_clip_trims(
            clips_with_durations,
            target_min=35.0,
            target_max=60.0,
            min_clip_dur=4.0,
            peak_offsets=peak_offsets,
        )

        # 4. Compose each clip into 9:16 layout (blur_fill) and concatenate
        composed_beat_clips: list[CompositeVideoClip] = []
        all_opened_sources: list[Any] = []
        clip_timeline_spans: list[tuple[float, float, EvidenceClip]] = []
        t_curr = 0.0

        for clip, seg_path, t_start, t_end in trimmed_clips:
            beat_dur = max(0.1, t_end - t_start)
            comp = composer.compose(seg_path, None, trim_range=(t_start, t_end), layout_kind="blur_fill")
            if getattr(comp, "audio", None) is None:
                from moviepy import AudioClip
                comp = comp.with_audio(AudioClip(lambda t: [0.0, 0.0], duration=comp.duration))
            all_opened_sources.extend(getattr(comp, "_source_clips", [comp]))
            composed_beat_clips.append(comp)
            clip_timeline_spans.append((t_curr, t_curr + beat_dur, clip))
            t_curr += beat_dur

        concat_clip = concatenate_videoclips(composed_beat_clips, method="compose")
        concat_clip._source_clips = all_opened_sources
        total_duration = t_curr

        raw_intermediate_mp4 = temp_short_dir / "intermediate_raw.mp4"
        try:
            concat_clip.write_videofile(
                str(raw_intermediate_mp4),
                fps=30,
                codec="libx264",
                preset="ultrafast",
                ffmpeg_params=["-crf", "18"],
                audio=True,
                audio_codec="aac",
                logger=None,
            )
        finally:
            _close_clip_handles(concat_clip)
            for c in composed_beat_clips:
                _close_clip_handles(c)
            for s in all_opened_sources:
                _close_clip_handles(s)

        # 5. Voicing: intro, callout before strongest clip, outro
        intro_wav, outro_wav = narrator.prepare_narration_tracks(
            narration_intro=short.narration_intro,
            narration_outro=short.narration_outro,
            temp_dir=temp_short_dir,
            clip_duration=total_duration,
        )

        callout_wav = None
        if short.callout and str(short.callout).strip():
            callout_wav = narrator.prepare_callout_track(short.callout, temp_dir=temp_short_dir)

        # Position callout before strongest clip (second clip if >= 2 clips, else at 35% of clip)
        strongest_idx = 1 if len(trimmed_clips) > 1 else 0
        callout_start = clip_timeline_spans[strongest_idx][0] if strongest_idx > 0 else 0.35 * total_duration

        # 6. Audio mix with ducking
        mixed_intermediate_mp4 = temp_short_dir / "intermediate_mixed.mp4"
        mix_ran = False
        mixed_path = None
        try:
            mixed_path = narrator.mix_narration_with_ducking(
                input_video=raw_intermediate_mp4,
                output_video=mixed_intermediate_mp4,
                intro_wav=intro_wav,
                outro_wav=outro_wav,
                callout_wav=callout_wav,
                callout_start=callout_start,
                clip_duration=total_duration,
            )
            if mixed_path and Path(mixed_path).is_file():
                mix_ran = True
        except Exception as exc:
            logger.warning("Narration mix failed on short %s: %s", slot_name, exc)

        # 7. Apply Narration Gate
        intro_expected = bool(short.narration_intro and str(short.narration_intro).strip())
        intro_dur = narrator._probe_audio_duration(intro_wav) if intro_wav else 0.0
        gate_failed = False
        quarantine_reason = ""

        if bool(config.get("output.require_narration", True)):
            if intro_expected and not mix_ran:
                gate_failed = True
                quarantine_reason = "narration mix did not run (TTS failed or produced no audio); clip would ship without voiceover"
            elif intro_expected and mix_ran and not (intro_dur and intro_dur > 0.0):
                gate_failed = True
                quarantine_reason = "intro voiceover was expected but not mixed; clip would open without narration"

        if gate_failed:
            plan.outputs.pop(slot_name, None)
            plan.outputs.pop(f"{slot_name}_post", None)
            quarantine_base.mkdir(parents=True, exist_ok=True)
            quarantined_video = quarantine_base / filename
            # Finalize the un-narrated video into quarantine with loudnorm so review/inspection is possible
            norm_cmd = [
                "ffmpeg", "-y",
                "-i", str(raw_intermediate_mp4.resolve()),
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
                "-c:v", "libx264", "-crf", str(crf),
                "-c:a", "aac", "-b:a", "192k",
                str(quarantined_video.resolve()),
            ]
            try:
                subprocess.run(norm_cmd, check=True, capture_output=True)
            except Exception:
                shutil.copy2(str(raw_intermediate_mp4), str(quarantined_video))

            quarantined_item = {
                "slot": slot_name,
                "video_path": str(quarantined_video.resolve()),
                "quarantine_reason": quarantine_reason,
            }
            quarantined_shorts.append(quarantined_item)
            if progress and hasattr(progress, "clip_done"):
                progress.clip_done(quarantined_item)
            logger.warning("Quarantined short '%s' (%s): %s", slot_name, quarantined_video, quarantine_reason)
            continue

        # 8. Build ASS overlay: persistent hook title, credit tags, narration cards
        credit_windows: list[tuple[float, float, str]] = []
        for st, et, c in clip_timeline_spans:
            if c.kind == "official":
                raw_ch = (c.channel or "").strip().lstrip("@")
                tag = raw_ch if raw_ch else f"Official {plan.game} footage"
            else:
                raw_ch = (c.channel or "").strip().lstrip("@")
                tag = f"@{raw_ch}" if raw_ch else "@Unknown"
            # Show credit tag during the clip's playback (capped at first 4.0s or full clip)
            credit_et = min(et, st + 4.0)
            credit_windows.append((st, credit_et, tag))

        narration_windows: list[tuple[float, float, str]] = []
        eff_intro = None
        eff_outro = None
        if hasattr(narrator, "get_effective_narration_durations"):
            eff_intro, eff_outro = narrator.get_effective_narration_durations(
                intro_dur,
                narrator._probe_audio_duration(outro_wav) if outro_wav else None,
                total_duration,
            )
        if eff_intro is None:
            eff_intro = intro_dur or 3.0
        if short.narration_intro:
            narration_windows.append((0.0, min(total_duration, eff_intro), short.narration_intro))

        if short.callout:
            callout_dur = narrator._probe_audio_duration(callout_wav) if callout_wav else 2.5
            c_end = min(total_duration, callout_start + (callout_dur or 2.5))
            narration_windows.append((callout_start, c_end, short.callout))

        if short.narration_outro:
            if eff_outro is None:
                eff_outro = narrator._probe_audio_duration(outro_wav) if outro_wav else 3.0
            o_start = max(0.0, total_duration - (eff_outro or 3.0))
            narration_windows.append((o_start, total_duration, short.narration_outro))

        ass_path = temp_short_dir / f"{filename}.ass"
        _build_shorts_ass(
            ass_path=ass_path,
            hook_title=short.hook_title,
            credit_windows=credit_windows,
            narration_windows=narration_windows,
            total_duration=total_duration,
            font_name=font_name,
            config=config,
        )

        # 9. Single final lossy encode with loudnorm (-14 LUFS) and ASS burn-in
        encode_input = mixed_intermediate_mp4 if mix_ran else raw_intermediate_mp4
        escaped_ass = _escape_ffmpeg_ass_filename(ass_path.name)

        def build_encode_cmd(selected_codec: str) -> list[str]:
            v_args = (
                ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
                if selected_codec == "h264_nvenc"
                else ["-c:v", "libx264", "-crf", str(crf)]
            )
            return [
                "ffmpeg", "-y",
                "-i", str(encode_input.resolve()),
                "-vf", f"ass={escaped_ass}",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
                *v_args,
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_short_path.resolve()),
            ]

        try:
            subprocess.run(
                build_encode_cmd(codec),
                check=True,
                capture_output=True,
                cwd=str(ass_path.resolve().parent),
            )
        except subprocess.CalledProcessError as exc:
            _log_ffmpeg_stderr(exc, f"FFmpeg {codec} final encode")
            if codec == "h264_nvenc":
                logger.warning("h264_nvenc failed on short %s; falling back to libx264", slot_name)
                subprocess.run(
                    build_encode_cmd("libx264"),
                    check=True,
                    capture_output=True,
                    cwd=str(ass_path.resolve().parent),
                )
            else:
                raise

        # 10. Generate post file
        post_path = write_short_post(config, plan, short, final_short_path)

        # 11. Update plan outputs
        out_slot = f"{slot_name}_draft" if draft else slot_name
        plan.outputs[out_slot] = str(final_short_path.resolve())
        plan.outputs[f"{out_slot}_post"] = str(post_path.resolve())

        rendered_item = {
            "slot": slot_name,
            "video_path": str(final_short_path.resolve()),
            "duration": _probe_duration(final_short_path) or total_duration,
            "post_path": str(post_path.resolve()),
            "draft": draft,
        }
        rendered_shorts.append(rendered_item)
        if progress and hasattr(progress, "clip_done"):
            progress.clip_done(rendered_item)

    # Save updated outputs to plan.json
    save_plan(config, plan)

    return {
        "status": "complete" if rendered_shorts else ("quarantined" if quarantined_shorts else "empty"),
        "plan_id": plan.plan_id,
        "rendered": rendered_shorts,
        "quarantined": quarantined_shorts,
        "outputs": plan.outputs,
    }
