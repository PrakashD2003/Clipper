"""YPP transformative commentary narrator and dynamic FFmpeg audio ducking."""

from __future__ import annotations

import gc
import logging
import time
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from ..config import Config
from .tts_engines import (
    EdgeTTS,
    ElevenLabsTTS,
    ExternalProcessTTS,
    KokoroTTS,
    _is_espeak_available,
)

logger = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.7
TTS_RETRY_SLEEP_SECONDS = 0.5  # pause before retrying a fully failed engine chain (patchable in tests)


class TransformativeNarrator:
    """Synthesizes adaptive intro/outro voiceovers and dynamically ducks stream audio via FFmpeg."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.enabled: bool = bool(config.get("transformation.enabled", True))
        self.mode: str = str(
            config.get("transformation.mode", "auto") or "auto"
        ).strip().lower()
        self.engine: str = str(
            config.get("transformation.tts_engine", "auto") or "auto"
        )
        self.voice: str = str(
            config.get("transformation.tts_voice", "am_puck") or "am_puck"
        )
        self.tts_speed: float = float(config.get("transformation.tts_speed", 1.1) or 1.1)
        chain = config.get(
            "transformation.tts_engines", ["chatterbox", "kokoro", "edge-tts"]
        )
        self.engine_chain: list[str] = (
            [chain] if isinstance(chain, str) else list(chain or [])
        )
        self.tts_external_python: str = str(
            config.get("transformation.tts_external_python", ".venv-tts/Scripts/python.exe")
            or ""
        )
        self.tts_reference_wav: str = str(
            config.get("transformation.tts_reference_wav", "") or ""
        )
        self.edge_tts_rate: str = str(config.get("transformation.edge_tts_rate", "+10%") or "")
        self.target_lufs: float = float(config.get("transformation.target_lufs", -14) or -14)
        raw_gain = config.get("transformation.narration_gain_db", 3)
        self.narration_gain_db: float = float(3 if raw_gain is None else raw_gain)
        raw_xf = config.get("transformation.crossfade_seconds", 0.25)
        self.crossfade_seconds: float = float(0.25 if raw_xf is None else raw_xf)
        self.callout_enabled: bool = bool(config.get("transformation.callout_enabled", True))
        self.edge_tts_voice: str = str(
            config.get("transformation.edge_tts_voice", "en-US-AndrewMultilingualNeural")
            or "en-US-AndrewMultilingualNeural"
        )
        self.elevenlabs_voice_id: str = str(
            config.get("transformation.elevenlabs_voice_id", "") or ""
        )
        # Separate Hindi/Hinglish voice; falls back to the main voice when unset.
        self.elevenlabs_hinglish_voice_id: str = str(
            config.get("transformation.elevenlabs_hinglish_voice_id", "") or ""
        )
        self.elevenlabs_model_id: str = str(
            config.get("transformation.elevenlabs_model_id", "eleven_multilingual_v2")
            or "eleven_multilingual_v2"
        )
        raw_duck = config.get("transformation.duck_stream_volume", 0.18)
        self.duck_volume: float = float(0.18 if raw_duck is None else raw_duck)
        hch = config.get("transformation.hinglish_tts_engines", ["elevenlabs", "edge-tts"])
        self.hinglish_engine_chain: list[str] = (
            [hch] if isinstance(hch, str) else list(hch or [])
        )
        self.hinglish_edge_tts_voice: str = str(
            config.get("transformation.hinglish_edge_tts_voice", "en-IN-PrabhatNeural")
            or "en-IN-PrabhatNeural"
        )
        self.language: str = "english"
        self._kokoro = KokoroTTS(self.voice, self.tts_speed)

    @property
    def _kpipelines(self) -> dict[str, Any]:
        """Cached Kokoro pipelines (owned by the Kokoro engine)."""
        return self._kokoro._pipelines

    def release(self) -> None:
        """Drop cached Kokoro pipelines and free GPU memory when available."""
        self._kokoro.clear()
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("torch.cuda.empty_cache unavailable: %s", exc)

    def _elevenlabs_voice_for(self, language: str | None = None) -> str:
        lang = str(language or self.language or "english").strip().lower()
        if lang == "hinglish" and self.elevenlabs_hinglish_voice_id.strip():
            return self.elevenlabs_hinglish_voice_id
        return self.elevenlabs_voice_id

    def _elevenlabs_available(self, language: str | None = None) -> bool:
        return ElevenLabsTTS(
            self._elevenlabs_voice_for(language), self.elevenlabs_model_id
        ).available()

    def _hinglish_chain(self) -> list[str]:
        """Hinglish chain: only multilingual-capable engines (never English-only ones)."""
        if (self.engine or "auto").lower().strip() == "none":
            return []
        chain: list[str] = []
        for name in self.hinglish_engine_chain:
            n = str(name).lower().strip().replace("_", "-")
            if n not in ("elevenlabs", "edge-tts") or n in chain:
                continue
            if n == "elevenlabs" and not self._elevenlabs_available("hinglish"):
                continue
            chain.append(n)
        return chain or ["edge-tts"]

    def _engine_chain(self, language: str | None = None) -> list[str]:
        """Resolve the ordered TTS fallback chain (legacy single ``tts_engine`` wins when set)."""
        lang = str(language or self.language or "english").strip().lower()
        if lang == "hinglish":
            return self._hinglish_chain()
        chain = self._english_engine_chain()
        if (
            (self.engine or "auto").lower().strip() == "auto"
            and "elevenlabs" not in chain
            and self._elevenlabs_available("english")
        ):
            chain.insert(0, "elevenlabs")
        return chain

    def _english_engine_chain(self) -> list[str]:
        legacy = (self.engine or "auto").lower().strip()
        known = ("chatterbox", "kokoro", "edge-tts", "elevenlabs")
        if legacy == "none":
            return []
        if legacy == "elevenlabs":
            return ["elevenlabs", "kokoro", "edge-tts"]
        if legacy == "kokoro":
            return ["kokoro", "edge-tts"]
        if legacy in ("edge-tts", "edge_tts"):
            return ["edge-tts"]
        if legacy == "chatterbox":
            return ["chatterbox", "kokoro", "edge-tts"]
        if legacy != "auto":
            logger.warning("Unknown TTS engine %r; using configured engine chain", legacy)
        chain: list[str] = []
        for name in self.engine_chain:
            n = str(name).lower().strip().replace("_", "-")
            if n in known and n not in chain:
                chain.append(n)
        return chain or ["kokoro", "edge-tts"]

    def _run_engine(
        self, name: str, text: str, output_wav: Path, language: str | None = None
    ) -> Path | None:
        if name == "chatterbox":
            return self._synthesize_chatterbox(text, output_wav)
        if name == "kokoro":
            return self._synthesize_kokoro(text, output_wav)
        if name == "edge-tts":
            if str(language or self.language).strip().lower() == "hinglish":
                return self._synthesize_edge_tts(text, output_wav, language="hinglish")
            return self._synthesize_edge_tts(text, output_wav)
        if name == "elevenlabs":
            if str(language or self.language).strip().lower() == "hinglish":
                return self._synthesize_elevenlabs(text, output_wav, language="hinglish")
            return self._synthesize_elevenlabs(text, output_wav)
        return None

    def synthesize_speech(
        self,
        text: str,
        output_wav: Path,
        max_seconds: float | None = None,
        language: str | None = None,
    ) -> Path | None:
        """Synthesize text to WAV through the TTS engine chain (first engine returning audio wins)."""
        if not text or not text.strip() or not self.enabled:
            return None

        clean_text = self._fit_text_to_budget(text.strip(), max_seconds)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        lang = str(language or self.language or "english").strip().lower()
        try:
            retries = max(0, int(self.config.get("transformation.tts_retries", 1) or 0))
        except (TypeError, ValueError):
            retries = 1
        for attempt in range(retries + 1):
            if attempt > 0:
                logger.warning(
                    "All TTS engines failed; retrying chain (%d/%d).", attempt, retries
                )
                time.sleep(TTS_RETRY_SLEEP_SECONDS)
            for name in self._engine_chain(lang):
                res = self._run_engine(name, clean_text, output_wav, language=lang)
                if res is not None and res.is_file() and res.stat().st_size > 0:
                    return res
                if name == "kokoro":
                    logger.warning("Kokoro produced no audio (espeak-ng missing?); falling back to edge-tts")
                else:
                    logger.info("TTS engine %s produced no audio; trying next", name)
        logger.debug("No TTS engine succeeded")
        return None

    @staticmethod
    def _fit_text_to_budget(text: str, max_seconds: float | None) -> str:
        """Truncate text so its estimated spoken duration (2.7 words/sec) fits max_seconds."""
        if max_seconds is None or max_seconds <= 0:
            return text
        max_words = max(1, int(max_seconds * WORDS_PER_SECOND))
        words = text.split()
        if len(words) <= max_words:
            return text
        head = " ".join(words[:max_words])
        cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        if head.endswith((".", "!", "?")):
            cut = len(head) - 1
        truncated = head[: cut + 1] if cut > 0 else head
        logger.info(
            "Narration estimated at %.1fs exceeds %.1fs budget; truncated %d -> %d words",
            len(words) / WORDS_PER_SECOND,
            max_seconds,
            len(words),
            len(truncated.split()),
        )
        return truncated.strip()

    def _synthesize_kokoro(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize locally with Kokoro-82M if installed and espeak-ng is available."""
        if os.name == "nt" and not _is_espeak_available():
            logger.debug("espeak-ng not found on Windows; skipping Kokoro synthesis")
            return None
        self._kokoro.espeak_check = lambda: _is_espeak_available()
        self._kokoro.voice = self.voice
        self._kokoro.speed = self.tts_speed
        return self._kokoro.synthesize(text, output_wav)

    def _synthesize_edge_tts(
        self, text: str, output_wav: Path, language: str | None = None
    ) -> Path | None:
        """Synthesize with edge-tts using self.edge_tts_voice / edge_tts_rate."""
        lang = str(language or self.language or "english").strip().lower()
        voice = (
            self.hinglish_edge_tts_voice
            if lang == "hinglish"
            else self.edge_tts_voice or "en-US-AndrewMultilingualNeural"
        )
        eng = EdgeTTS(voice, self.edge_tts_rate)
        return eng.synthesize(text, output_wav)

    def _synthesize_elevenlabs(
        self, text: str, output_wav: Path, language: str | None = None
    ) -> Path | None:
        """Synthesize with ElevenLabs when ELEVENLABS_API_KEY & voice_id are set."""
        eng = ElevenLabsTTS(self._elevenlabs_voice_for(language), self.elevenlabs_model_id)
        return eng.synthesize(text, output_wav)

    def _synthesize_chatterbox(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize via the external-process Chatterbox worker (separate venv)."""
        eng = ExternalProcessTTS(self.tts_external_python, self.tts_reference_wav)
        return eng.synthesize(text, output_wav)

    def resolve_narration_track(
        self,
        custom_path: Path | None,
        text: str,
        output_wav: Path,
        max_seconds: float | None = None,
    ) -> Path | None:
        """Resolve narration track with priority: custom mic file > synthesized TTS."""
        if custom_path is not None:
            p = Path(custom_path)
            if p.is_file() and p.stat().st_size > 0:
                return p
        if text and text.strip():
            return self.synthesize_speech(text, output_wav, max_seconds=max_seconds)
        return None

    def prepare_narration_tracks(
        self,
        narration_intro: str = "",
        narration_outro: str = "",
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        temp_dir: Path | None = None,
        clip_duration: float | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Prepare intro and outro narration WAV paths respecting custom mic priority."""
        if not self.enabled or str(self.mode or "").strip().lower() == "raw":
            return None, None

        temp = temp_dir or Path(tempfile.gettempdir())
        temp.mkdir(parents=True, exist_ok=True)
        budget = clip_duration * 0.30 if clip_duration and clip_duration > 0 else None

        try:
            intro_wav = self.resolve_narration_track(
                voice_intro_path,
                narration_intro,
                temp / "narrator_intro.wav",
                max_seconds=budget,
            )
            outro_wav = self.resolve_narration_track(
                voice_outro_path,
                narration_outro,
                temp / "narrator_outro.wav",
                max_seconds=budget,
            )
        finally:
            self.release()
        return intro_wav, outro_wav

    def prepare_breakdown_track(
        self,
        breakdown_commentary: str = "",
        voice_breakdown_path: Path | None = None,
        temp_dir: Path | None = None,
    ) -> Path | None:
        """Resolve breakdown commentary track with priority: custom mic file > synthesized TTS."""
        if not self.enabled or str(self.mode or "").strip().lower() == "raw":
            return None
        temp = temp_dir or Path(tempfile.gettempdir())
        temp.mkdir(parents=True, exist_ok=True)
        return self.resolve_narration_track(
            voice_breakdown_path,
            breakdown_commentary,
            temp / "narrator_breakdown.wav",
        )

    def get_effective_narration_durations(
        self,
        intro_dur: float | None,
        outro_dur: float | None,
        clip_duration: float,
    ) -> tuple[float | None, float | None]:
        """Compute effective clamped durations for intro and outro used in the ducking filtergraph.

        Guarantees that narration windows do not exceed the short-clip overlap budget
        (clamped to combined 75% budget on short clips) and prevents climax audio bleed.
        """
        has_intro = intro_dur is not None and float(intro_dur) > 0.01
        has_outro = outro_dur is not None and float(outro_dur) > 0.01
        clip_dur = max(float(clip_duration), 0.1)

        if not has_intro and not has_outro:
            return None, None

        if has_intro and has_outro:
            intro_d = float(intro_dur)  # type: ignore[arg-type]
            outro_d = float(outro_dur)  # type: ignore[arg-type]
            max_combined = clip_dur * 0.75
            if (intro_d + outro_d) > max_combined:
                scale = max_combined / (intro_d + outro_d)
                eff_intro = intro_d * scale
                eff_outro = outro_d * scale
            else:
                eff_intro = intro_d
                eff_outro = outro_d
            return eff_intro, eff_outro

        if has_intro:
            intro_d = float(intro_dur)  # type: ignore[arg-type]
            return min(intro_d, clip_dur * 0.75), None

        outro_d = float(outro_dur)  # type: ignore[arg-type]
        return None, min(outro_d, clip_dur * 0.75)

    @staticmethod
    def compute_breakdown_windows(
        intro_dur: float | None,
        outro_dur: float | None,
        breakdown_dur: float | None,
        clip_duration: float,
        replay_start_time: float | None = None,
    ) -> dict[str, float | None]:
        """Single source of truth for breakdown-mode cue placement (audio AND captions).

        Returns ``eff_intro``, ``breakdown_start``, ``breakdown_end``, ``eff_outro`` and
        ``outro_start`` in clip-relative seconds (None when that cue is absent).
        """
        clip_dur = max(float(clip_duration), 0.1)
        has_intro = intro_dur is not None and float(intro_dur) > 0.01
        has_outro = outro_dur is not None and float(outro_dur) > 0.01
        has_breakdown = breakdown_dur is not None and float(breakdown_dur) > 0.01

        eff_intro: float | None = None
        if has_intro:
            limit = (
                float(replay_start_time) if replay_start_time is not None else clip_dur * 0.5
            )
            eff_intro = min(float(intro_dur), max(0.01, limit))  # type: ignore[arg-type]

        breakdown_start: float | None = None
        breakdown_end: float | None = None
        if has_breakdown:
            rep_start = (
                float(replay_start_time)
                if replay_start_time is not None
                else (eff_intro if eff_intro is not None else 0.0)
            )
            breakdown_start = max(0.0, min(clip_dur, rep_start))
            avail = max(0.01, clip_dur - breakdown_start)
            breakdown_end = min(
                clip_dur, breakdown_start + min(float(breakdown_dur), avail)  # type: ignore[arg-type]
            )

        eff_outro: float | None = None
        outro_start: float | None = None
        if has_outro:
            floor = breakdown_end if breakdown_end is not None else (eff_intro or 0.0)
            eff_outro = min(float(outro_dur), max(0.01, clip_dur - floor))  # type: ignore[arg-type]
            outro_start = max(floor, clip_dur - eff_outro)

        return {
            "eff_intro": eff_intro,
            "breakdown_start": breakdown_start,
            "breakdown_end": breakdown_end,
            "eff_outro": eff_outro,
            "outro_start": outro_start,
        }

    def _ramped_duck_expr(
        self, windows: list[tuple[float, float]], duck_vol: float, ramp: float
    ) -> str:
        """Duck envelope with linear ``ramp``-second fades into/out of every window."""
        r = max(float(ramp), 0.01)
        depth = 1.0 - duck_vol
        terms = [
            f"clip(min((t-({a - r:.2f}))/{r:.2f},({b + r:.2f}-t)/{r:.2f}),0,1)"
            for a, b in windows
        ]
        level = terms[0] if len(terms) == 1 else "max(" + ",".join(terms) + ")"
        return f"1-{depth:.2f}*{level}"

    def _build_ducking_filtergraph(
        self,
        intro_dur: float | None,
        outro_dur: float | None,
        clip_duration: float,
        has_video_audio: bool = True,
        *,
        breakdown_dur: float | None = None,
        replay_start_time: float | None = None,
        callout_dur: float | None = None,
        callout_start: float | None = None,
        duck_full: bool = False,
        finalize: bool = False,
    ) -> tuple[str, int]:
        """Build dynamic FFmpeg filter_complex string and input count.

        Input order: 0 = stream, then intro, breakdown, outro, callout (only those present).
        - Stream audio is ducked (``volume=...:eval=frame`` - mandatory, the default
          ``eval=once`` would freeze the level at t=0) under intro/outro/breakdown/callout
          windows only, or for the whole clip when ``duck_full``.
        - Narrator WAVs are normalized to 48 kHz stereo before ``amix`` (stereo ``adelay``).
        - Short clips: intro+outro are clamped to 75% of the clip and ``atrim``-ed.
        - ``finalize=True`` (used by the real mix): smooth ``crossfade_seconds`` duck ramps,
          per-track loudnorm + ``narration_gain_db``, and a final ``loudnorm`` to
          ``target_lufs``. ``finalize=False`` keeps the plain step-ducked graph.
        """
        has_intro = intro_dur is not None and float(intro_dur) > 0.01
        has_outro = outro_dur is not None and float(outro_dur) > 0.01
        has_breakdown = breakdown_dur is not None and float(breakdown_dur) > 0.01
        has_callout = callout_dur is not None and float(callout_dur) > 0.01
        clip_dur = max(float(clip_duration), 0.1)
        norm = "aresample=48000,aformat=channel_layouts=stereo"
        any_track = has_intro or has_outro or has_breakdown or has_callout

        if not any_track:
            if not has_video_audio:
                return (
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=0:{clip_dur:.2f},asetpts=PTS-STARTPTS[aout]",
                    1,
                )
            return f"[0:a]{norm}[aout]", 1

        duck_vol_str = f"{self.duck_volume:.2f}"
        silent_base = (
            f"anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=0:{clip_dur:.2f},asetpts=PTS-STARTPTS[a0];"
        )

        # tracks: (trim_filter, delay_ms | None, duck window | None)
        tracks: list[tuple[str, int | None, tuple[float, float] | None]] = []

        def _trim(eff: float, orig: float) -> str:
            if eff < orig - 1e-6:
                return f",atrim=0:{max(0.01, eff):.2f},asetpts=PTS-STARTPTS"
            return ""

        if has_breakdown:
            win = self.compute_breakdown_windows(
                intro_dur, outro_dur, breakdown_dur, clip_dur, replay_start_time
            )
            eff_intro = win["eff_intro"]
            rep_start = win["breakdown_start"]
            breakdown_end = win["breakdown_end"]
            eff_outro = win["eff_outro"]
            if eff_intro is not None:
                tracks.append(
                    (_trim(eff_intro, float(intro_dur)), None, (0.0, eff_intro))  # type: ignore[arg-type]
                )
            rep_delay = int(round(rep_start * 1000))
            tracks.append(
                (
                    _trim(breakdown_end - rep_start, float(breakdown_dur)),  # type: ignore[arg-type]
                    rep_delay,
                    (rep_start, breakdown_end),
                )
            )
            if eff_outro is not None:
                o_start = win["outro_start"]
                tracks.append(
                    (
                        _trim(eff_outro, float(outro_dur)),  # type: ignore[arg-type]
                        int(round(o_start * 1000)),
                        (o_start, clip_dur),
                    )
                )
        else:
            eff_intro, eff_outro = self.get_effective_narration_durations(
                intro_dur, outro_dur, clip_dur
            )
            if eff_intro is not None:
                tracks.append(
                    (_trim(eff_intro, float(intro_dur)), None, (0.0, eff_intro))  # type: ignore[arg-type]
                )
            if eff_outro is not None:
                o_start = max(eff_intro or 0.0, clip_dur - eff_outro)
                tracks.append(
                    (
                        _trim(eff_outro, float(outro_dur)),  # type: ignore[arg-type]
                        int(round(o_start * 1000)),
                        (o_start, clip_dur),
                    )
                )

        if has_callout:
            c_start = float(callout_start) if callout_start is not None else clip_dur * 0.35
            c_dur = float(callout_dur)  # type: ignore[arg-type]
            c_start = max(0.0, min(c_start, clip_dur - 0.2))
            c_eff = min(c_dur, max(0.01, clip_dur - c_start))
            tracks.append(
                (
                    _trim(c_eff, c_dur),
                    int(round(c_start * 1000)),
                    (c_start, c_start + c_eff),
                )
            )

        windows = [t[2] for t in tracks if t[2] is not None]
        if duck_full:
            vol_expr = duck_vol_str
        elif finalize:
            vol_expr = self._ramped_duck_expr(windows, self.duck_volume, self.crossfade_seconds)
        else:
            vol_expr = (
                "if("
                + "+".join(
                    f"between(t,{0 if a == 0 else format(a, '.2f')},{b:.2f})" for a, b in windows
                )
                + f",{duck_vol_str},1.0)"
            )

        base_branch = (
            f"[0:a]volume='{vol_expr}':eval=frame,{norm}[a0];"
            if has_video_audio
            else silent_base
        )

        narr_tail = ""
        if finalize:
            narr_tail = (
                f",loudnorm=I=-16:TP=-1.5:LRA=11,{norm},"
                f"volume={self.narration_gain_db:.1f}dB"
            )
        parts: list[str] = []
        for idx, (trim, delay_ms, _w) in enumerate(tracks, start=1):
            delay = f",adelay={delay_ms}|{delay_ms}" if delay_ms is not None else ""
            parts.append(f"[{idx}:a]{norm}{trim}{narr_tail}{delay}[a{idx}];")

        total_inputs = 1 + len(tracks)
        amix_in = "".join(f"[a{i}]" for i in range(total_inputs))
        out_label = "[amixed]" if finalize else "[aout]"
        mix_part = f"{amix_in}amix=inputs={total_inputs}:duration=first:normalize=0{out_label}"
        graph = base_branch + "".join(parts) + mix_part
        if finalize:
            graph += f";[amixed]loudnorm=I={self.target_lufs:g}:TP=-1.5:LRA=11,{norm}[aout]"
        return graph, total_inputs

    def _probe_audio_duration(self, audio_path: Path | None) -> float | None:
        """Probe narration track duration via soundfile with ffprobe fallback (supports .wav/.m4a/.mp3/.aac)."""
        if audio_path is None:
            return None
        p = Path(audio_path)
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            info = sf.info(str(p))
            if info.duration > 0.05:
                return float(info.duration)
            return None
        except Exception as exc:
            logger.debug("soundfile could not probe %s (%s); trying ffprobe", p, exc)

        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(p),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
            val = float(res.stdout.strip())
            if val > 0.05:
                return val
        except Exception as exc:
            logger.debug("ffprobe failed reading audio duration for %s: %s", p, exc)
        return None

    def _has_audio_stream(self, video_path: Path) -> bool:
        """Check whether input_video has at least one audio stream (defaults True if probe unavailable)."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
            return bool(res.stdout.strip())
        except Exception as exc:
            logger.debug("Audio stream probe failed (%s); assuming audio present", exc)
            return True

    def mix_narration_with_ducking(
        self,
        input_video: Path,
        output_video: Path,
        intro_wav: Path | None,
        outro_wav: Path | None,
        clip_duration: float,
        *,
        breakdown_wav: Path | None = None,
        replay_start_time: float | None = None,
        replay_start: float | None = None,
        callout_wav: Path | None = None,
        callout_start: float | None = None,
        duck_full: bool = False,
    ) -> Path:
        """Mix intro/outro/breakdown/callout voiceovers over ducked stream audio (-c:v copy).

        The callout is voiced over the moment (stream ducked under it only); ``duck_full``
        keeps the stream ducked for the whole clip. The mix ends in loudnorm to ``target_lufs``.
        """
        input_video = Path(input_video)
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        effective_replay_start = (
            replay_start_time if replay_start_time is not None else replay_start
        )

        # Bypass mixing if disabled, raw mode, or no voiceover exists
        if not self.enabled or str(self.mode or "").strip().lower() == "raw":
            return self._bypass_copy(input_video, output_video)

        intro_dur = self._probe_audio_duration(intro_wav)
        outro_dur = self._probe_audio_duration(outro_wav)
        breakdown_dur = self._probe_audio_duration(breakdown_wav)

        callout_dur = self._probe_audio_duration(callout_wav)

        if (
            intro_dur is None
            and outro_dur is None
            and breakdown_dur is None
            and callout_dur is None
        ):
            return self._bypass_copy(input_video, output_video)

        # Ensure clip duration is positive
        eff_clip_duration = clip_duration
        if eff_clip_duration <= 0.0:
            eff_clip_duration = self._probe_clip_duration(input_video)

        has_video_audio = self._has_audio_stream(input_video)
        filtergraph, input_count = self._build_ducking_filtergraph(
            intro_dur=intro_dur,
            outro_dur=outro_dur,
            clip_duration=eff_clip_duration,
            has_video_audio=has_video_audio,
            breakdown_dur=breakdown_dur,
            replay_start_time=effective_replay_start,
            callout_dur=callout_dur,
            callout_start=callout_start,
            duck_full=duck_full,
            finalize=True,
        )

        same_file = False
        try:
            same_file = input_video.resolve() == output_video.resolve()
        except OSError:
            same_file = str(input_video) == str(output_video)

        target_out = (
            output_video.with_name(f"{output_video.stem}_mix_tmp{output_video.suffix}")
            if same_file
            else output_video
        )

        cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_video)]
        if intro_dur is not None:
            cmd.extend(["-i", str(intro_wav)])
        if breakdown_dur is not None:
            cmd.extend(["-i", str(breakdown_wav)])
        if outro_dur is not None:
            cmd.extend(["-i", str(outro_wav)])
        if callout_dur is not None:
            cmd.extend(["-i", str(callout_wav)])

        cmd.extend(
            [
                "-filter_complex",
                filtergraph,
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(target_out),
            ]
        )

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if same_file and target_out.is_file():
                shutil.move(str(target_out), str(output_video))
            return output_video
        except subprocess.CalledProcessError as exc:
            if target_out.exists() and target_out != output_video:
                try:
                    target_out.unlink()
                except OSError:
                    pass
            stderr_tail = (exc.stderr or "")[-500:]
            raise RuntimeError(
                f"FFmpeg ducking mix failed (code {exc.returncode}): {stderr_tail}"
            ) from exc
        except Exception as exc:
            logger.warning(
                "Unexpected error during narration mixing: %s. Falling back to original video.",
                exc,
            )
            if target_out.exists() and target_out != output_video:
                try:
                    target_out.unlink()
                except OSError:
                    pass
            return self._bypass_copy(input_video, output_video)

    def prepare_callout_track(
        self, callout: str, temp_dir: Path | None = None
    ) -> Path | None:
        """Synthesize the short build-up callout (``transformation.callout_enabled``)."""
        if (
            not callout
            or not callout.strip()
            or not self.enabled
            or not self.callout_enabled
            or str(self.mode or "").strip().lower() == "raw"
        ):
            return None
        temp = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        temp.mkdir(parents=True, exist_ok=True)
        try:
            return self.synthesize_speech(callout, temp / "narrator_callout.wav", max_seconds=3.0)
        finally:
            self.release()

    def normalize_audio(self, input_video: Path, output_video: Path) -> Path:
        """Loudnorm the clip audio to ``target_lufs`` with ``-c:v copy`` (no narration mix ran)."""
        input_video = Path(input_video)
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        if not self._has_audio_stream(input_video):
            return self._bypass_copy(input_video, output_video)

        try:
            same_file = input_video.resolve() == output_video.resolve()
        except OSError:
            same_file = str(input_video) == str(output_video)
        target_out = (
            output_video.with_name(f"{output_video.stem}_norm_tmp{output_video.suffix}")
            if same_file
            else output_video
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(input_video),
            "-map", "0:v", "-map", "0:a:0",
            "-af", f"loudnorm=I={self.target_lufs:g}:TP=-1.5:LRA=11,aresample=48000",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(target_out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if same_file and target_out.is_file():
                shutil.move(str(target_out), str(output_video))
            return output_video
        except Exception as exc:
            logger.warning("Audio normalization failed (%s); keeping original audio", exc)
            if target_out != output_video and target_out.exists():
                target_out.unlink(missing_ok=True)
            return self._bypass_copy(input_video, output_video)

    def _bypass_copy(self, input_video: Path, output_video: Path) -> Path:
        """Safely copy or preserve video without mixing."""
        try:
            if input_video.resolve() != output_video.resolve():
                shutil.copyfile(str(input_video), str(output_video))
        except OSError:
            if str(input_video) != str(output_video):
                shutil.copyfile(str(input_video), str(output_video))
        return output_video

    def _probe_clip_duration(self, video_path: Path) -> float:
        """Probe video duration via ffprobe or soundfile fallback."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception as exc:
            logger.debug("Clip duration probe failed (%s); defaulting to 30s", exc)
        return 30.0
