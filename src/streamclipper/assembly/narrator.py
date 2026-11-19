"""YPP transformative commentary narrator and dynamic FFmpeg audio ducking."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import soundfile as sf

from ..config import Config

logger = logging.getLogger(__name__)


def _is_espeak_available() -> bool:
    """Check if espeak-ng / espeak binary or library is available on the system."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return True
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True
    if os.name == "nt":
        prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        std_candidates = [
            Path(prog_files) / "eSpeak NG" / "libespeak-ng.dll",
            Path(prog_files_x86) / "eSpeak NG" / "libespeak-ng.dll",
        ]
        for p in std_candidates:
            if p.exists():
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(p)
                return True
    return False


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
            config.get("transformation.tts_voice", "af_heart") or "af_heart"
        )
        self.edge_tts_voice: str = str(
            config.get("transformation.edge_tts_voice", "en-US-GuyNeural")
            or "en-US-GuyNeural"
        )
        self.elevenlabs_voice_id: str = str(
            config.get("transformation.elevenlabs_voice_id", "") or ""
        )
        self.elevenlabs_model_id: str = str(
            config.get("transformation.elevenlabs_model_id", "eleven_multilingual_v2")
            or "eleven_multilingual_v2"
        )
        raw_duck = config.get("transformation.duck_stream_volume", 0.18)
        self.duck_volume: float = float(0.18 if raw_duck is None else raw_duck)

    def synthesize_speech(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize text to WAV via configured provider (kokoro | edge-tts | elevenlabs | auto).

        Note: When engine is 'edge-tts' (or 'auto' falls back from kokoro to edge-tts),
        self.edge_tts_voice ('en-US-GuyNeural') is used rather than a Kokoro voice ID like 'af_heart'.
        """
        if not text or not text.strip() or not self.enabled:
            return None

        clean_text = text.strip()
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        engine = (self.engine or "auto").lower().strip()
        if engine == "none":
            return None

        if engine == "elevenlabs":
            res = self._synthesize_elevenlabs(clean_text, output_wav)
            if res is not None and res.is_file() and res.stat().st_size > 0:
                return res
            logger.info("ElevenLabs unconfigured or failed; falling back to auto TTS")
            engine = "auto"

        if engine == "kokoro":
            return self._synthesize_kokoro(clean_text, output_wav)

        if engine in ("edge-tts", "edge_tts"):
            return self._synthesize_edge_tts(clean_text, output_wav)

        if engine == "auto":
            # 1. Primary: local free Kokoro
            res = self._synthesize_kokoro(clean_text, output_wav)
            if res is not None and res.is_file() and res.stat().st_size > 0:
                return res
            # 2. Fallback: free cloud edge-tts
            res = self._synthesize_edge_tts(clean_text, output_wav)
            if res is not None and res.is_file() and res.stat().st_size > 0:
                return res
            logger.debug("No TTS engine succeeded in auto mode")
            return None

        # Unknown engine -> fallback to auto
        logger.warning("Unknown TTS engine %r; attempting auto fallback", engine)
        res = self._synthesize_kokoro(clean_text, output_wav)
        if res is not None and res.is_file() and res.stat().st_size > 0:
            return res
        return self._synthesize_edge_tts(clean_text, output_wav)

    def _synthesize_kokoro(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize locally with Kokoro-82M if installed and espeak-ng is available."""
        if os.name == "nt" and not _is_espeak_available():
            logger.debug(
                "espeak-ng not found on Windows (PHONEMIZER_ESPEAK_LIBRARY not set); "
                "skipping Kokoro synthesis and falling back to edge-tts"
            )
            return None

        try:
            from kokoro import KPipeline  # type: ignore
        except (ImportError, ModuleNotFoundError):
            logger.debug("Kokoro not installed; skipping Kokoro synthesis")
            return None

        try:
            lang_code = "b" if self.voice.startswith("b") else "a"
            pipeline = KPipeline(lang_code=lang_code)
            generator = pipeline(text, voice=self.voice, speed=1, split_pattern=r"\n+")
            audios: list[Any] = []
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audios.append(audio)
            if not audios:
                return None

            import numpy as np

            combined = np.concatenate(audios) if len(audios) > 1 else audios[0]
            sf.write(str(output_wav), combined, 24000)
            if output_wav.is_file() and output_wav.stat().st_size > 0:
                return output_wav
            return None
        except Exception as exc:
            logger.warning("Kokoro synthesis failed: %s", exc)
            return None

    def _synthesize_edge_tts(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize with edge-tts using self.edge_tts_voice (converting mp3 output to WAV via ffmpeg)."""
        try:
            import edge_tts  # type: ignore
        except (ImportError, ModuleNotFoundError):
            logger.debug("edge-tts not installed; skipping Edge-TTS synthesis")
            return None

        temp_mp3 = output_wav.with_name(f"{output_wav.stem}_edgetts.mp3")
        try:
            voice = self.edge_tts_voice or "en-US-GuyNeural"

            async def _generate() -> None:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(temp_mp3))

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(_generate())).result()
            else:
                asyncio.run(_generate())

            if not temp_mp3.is_file() or temp_mp3.stat().st_size == 0:
                return None

            # Convert mp3 to WAV via ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_mp3),
                "-ar",
                "48000",
                "-ac",
                "2",
                str(output_wav),
            ]
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if output_wav.is_file() and output_wav.stat().st_size > 0:
                return output_wav
            return None
        except Exception as exc:
            logger.warning("Edge-TTS synthesis failed: %s", exc)
            return None
        finally:
            if temp_mp3.exists():
                try:
                    temp_mp3.unlink()
                except OSError:
                    pass

    def _synthesize_elevenlabs(self, text: str, output_wav: Path) -> Path | None:
        """Synthesize with ElevenLabs API (incl. personal voice clone) when ELEVENLABS_API_KEY & voice_id are set."""
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        voice_id = self.elevenlabs_voice_id.strip()
        if not api_key or not voice_id:
            logger.debug("ElevenLabs unconfigured; missing API key or voice_id")
            return None

        temp_mp3 = output_wav.with_name(f"{output_wav.stem}_elevenlabs.mp3")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.elevenlabs_model_id or "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                status_code = getattr(resp, "status", None) or getattr(resp, "code", None) or 200
                if isinstance(status_code, int) and status_code != 200:
                    logger.warning("ElevenLabs API returned HTTP %s", status_code)
                    return None
                audio_bytes = resp.read()

            if not audio_bytes:
                return None

            temp_mp3.write_bytes(audio_bytes)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_mp3),
                "-ar",
                "48000",
                "-ac",
                "2",
                str(output_wav),
            ]
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if output_wav.is_file() and output_wav.stat().st_size > 0:
                return output_wav
            return None
        except Exception as exc:
            logger.warning("ElevenLabs synthesis failed: %s", exc)
            return None
        finally:
            if temp_mp3.exists():
                try:
                    temp_mp3.unlink()
                except OSError:
                    pass

    def resolve_narration_track(
        self,
        custom_path: Path | None,
        text: str,
        output_wav: Path,
    ) -> Path | None:
        """Resolve narration track with priority: custom mic file > synthesized TTS."""
        if custom_path is not None:
            p = Path(custom_path)
            if p.is_file() and p.stat().st_size > 0:
                return p
        if text and text.strip():
            return self.synthesize_speech(text, output_wav)
        return None

    def prepare_narration_tracks(
        self,
        narration_intro: str = "",
        narration_outro: str = "",
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        temp_dir: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Prepare intro and outro narration WAV paths respecting custom mic priority."""
        if not self.enabled or str(self.mode or "").strip().lower() == "raw":
            return None, None

        temp = temp_dir or Path(tempfile.gettempdir())
        temp.mkdir(parents=True, exist_ok=True)

        intro_wav = self.resolve_narration_track(
            voice_intro_path,
            narration_intro,
            temp / "narrator_intro.wav",
        )
        outro_wav = self.resolve_narration_track(
            voice_outro_path,
            narration_outro,
            temp / "narrator_outro.wav",
        )
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

    def _build_ducking_filtergraph(
        self,
        intro_dur: float | None,
        outro_dur: float | None,
        clip_duration: float,
        has_video_audio: bool = True,
        *,
        breakdown_dur: float | None = None,
        replay_start_time: float | None = None,
    ) -> tuple[str, int]:
        """Build dynamic FFmpeg filter_complex string and input count.

        - Handles intro-only (2 inputs), outro-only (2 inputs), intro+outro (3 inputs),
          or multi-cue breakdown ducking (intro + breakdown, breakdown only, etc.).
        - Stage 1: Optional intro voiceover ([0, intro_dur], ducks stream audio to 18%).
        - Stage 2: Climax stream audio plays at 100% full volume.
        - Stage 3: Instant Replay section ([replay_start, replay_start + breakdown_dur])
          ducks slowed stream audio to 18% under host breakdown commentary.
        - When ``has_video_audio`` is False (silent clip), synthesizes a 48kHz stereo
          silent base track of ``clip_duration`` via ``anullsrc`` so intro/outro voiceovers
          still attach cleanly instead of failing on a missing ``[0:a]`` stream.
        - Normalizes stream audio and all narrator WAV inputs to 48000 Hz stereo
          (`aresample=48000,aformat=channel_layouts=stereo`) before `amix` so 24kHz mono
          Kokoro/Edge-TTS/ElevenLabs tracks mix cleanly with 48kHz stereo stream audio,
          using stereo `adelay={delay_ms}|{delay_ms}` for delayed narrator tracks.
        - Clamps narration windows on short clips so intro and outro never overlap
          (if intro_dur + outro_dur > clip_duration * 0.75, trims outro or scales windows safely).
        - CRITICAL: Ducks [0:a] using `volume='if(between(t,...),0.18,1.0)':eval=frame`
          (MUST include `:eval=frame` because FFmpeg's `volume` defaults to `eval=once`
          at t=0, which would leave the entire clip stuck at 18% volume!).
        - Uses `amix=inputs={n}:duration=first:normalize=0`.
        """
        has_intro = intro_dur is not None and float(intro_dur) > 0.01
        has_outro = outro_dur is not None and float(outro_dur) > 0.01
        has_breakdown = breakdown_dur is not None and float(breakdown_dur) > 0.01
        clip_dur = max(float(clip_duration), 0.1)

        if not has_intro and not has_outro and not has_breakdown:
            if not has_video_audio:
                return (
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=0:{clip_dur:.2f},asetpts=PTS-STARTPTS[aout]",
                    1,
                )
            return "[0:a]aresample=48000,aformat=channel_layouts=stereo[aout]", 1

        duck_vol_str = f"{self.duck_volume:.2f}"
        silent_base = (
            f"anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=0:{clip_dur:.2f},asetpts=PTS-STARTPTS[a0];"
        )

        # Multi-cue breakdown mode ducking branch
        if has_breakdown:
            win = self.compute_breakdown_windows(
                intro_dur, outro_dur, breakdown_dur, clip_dur, replay_start_time
            )
            eff_intro = win["eff_intro"]
            rep_start = win["breakdown_start"]
            breakdown_end = win["breakdown_end"]
            eff_outro = win["eff_outro"]

            intro_trim = ""
            if eff_intro is not None and eff_intro < float(intro_dur) - 1e-6:  # type: ignore[arg-type]
                intro_trim = f",atrim=0:{max(0.01, eff_intro):.2f},asetpts=PTS-STARTPTS"

            breakdown_d = float(breakdown_dur)  # type: ignore[arg-type]
            eff_breakdown = breakdown_end - rep_start
            breakdown_delay_ms = int(round(rep_start * 1000))
            breakdown_trim = ""
            if eff_breakdown < breakdown_d - 1e-6:
                breakdown_trim = f",atrim=0:{max(0.01, eff_breakdown):.2f},asetpts=PTS-STARTPTS"

            outro_trim = ""
            if eff_outro is not None:
                outro_start = win["outro_start"]
                outro_end = clip_dur
                outro_delay_ms = int(round(outro_start * 1000))
                if eff_outro < float(outro_dur) - 1e-6:  # type: ignore[arg-type]
                    outro_trim = f",atrim=0:{max(0.01, eff_outro):.2f},asetpts=PTS-STARTPTS"

            conditions: list[str] = []
            if eff_intro is not None:
                conditions.append(f"between(t,0,{eff_intro:.2f})")
            conditions.append(f"between(t,{rep_start:.2f},{breakdown_end:.2f})")
            if eff_outro is not None:
                conditions.append(f"between(t,{outro_start:.2f},{outro_end:.2f})")

            vol_expr = f"if({'+'.join(conditions)},{duck_vol_str},1.0)"
            base_branch = (
                f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
                if has_video_audio
                else silent_base
            )

            tracks_parts: list[str] = []
            current_input = 1
            if eff_intro is not None:
                tracks_parts.append(
                    f"[{current_input}:a]aresample=48000,aformat=channel_layouts=stereo{intro_trim}[a{current_input}];"
                )
                current_input += 1

            tracks_parts.append(
                f"[{current_input}:a]aresample=48000,aformat=channel_layouts=stereo{breakdown_trim},adelay={breakdown_delay_ms}|{breakdown_delay_ms}[a{current_input}];"
            )
            current_input += 1

            if eff_outro is not None:
                tracks_parts.append(
                    f"[{current_input}:a]aresample=48000,aformat=channel_layouts=stereo{outro_trim},adelay={outro_delay_ms}|{outro_delay_ms}[a{current_input}];"
                )
                current_input += 1

            total_inputs = current_input
            amix_in = "".join(f"[a{i}]" for i in range(total_inputs))
            mix_part = f"{amix_in}amix=inputs={total_inputs}:duration=first:normalize=0[aout]"
            filter_complex = base_branch + "".join(tracks_parts) + mix_part
            return filter_complex, total_inputs

        # Standard 2-cue (intro / outro) branch (preserves exact existing formatting)
        eff_intro, eff_outro = self.get_effective_narration_durations(
            intro_dur, outro_dur, clip_dur
        )

        if eff_intro is not None and eff_outro is not None:
            intro_d = float(intro_dur)  # type: ignore[arg-type]
            outro_d = float(outro_dur)  # type: ignore[arg-type]
            outro_start = max(eff_intro, clip_dur - eff_outro)
            outro_end = clip_dur
            intro_end = eff_intro
            delay_ms = int(round(outro_start * 1000))

            intro_trim = (
                f",atrim=0:{max(0.01, eff_intro):.2f},asetpts=PTS-STARTPTS"
                if eff_intro < intro_d - 1e-6
                else ""
            )
            outro_trim = (
                f",atrim=0:{max(0.01, eff_outro):.2f},asetpts=PTS-STARTPTS"
                if eff_outro < outro_d - 1e-6
                else ""
            )

            vol_expr = (
                f"if(between(t,0,{intro_end:.2f})+"
                f"between(t,{outro_start:.2f},{outro_end:.2f}),"
                f"{duck_vol_str},1.0)"
            )
            base_branch = (
                f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
                if has_video_audio
                else silent_base
            )
            filter_complex = (
                f"{base_branch}"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo{intro_trim}[a1];"
                f"[2:a]aresample=48000,aformat=channel_layouts=stereo{outro_trim},adelay={delay_ms}|{delay_ms}[a2];"
                f"[a0][a1][a2]amix=inputs=3:duration=first:normalize=0[aout]"
            )
            return filter_complex, 3

        if eff_intro is not None and eff_outro is None:
            intro_d = float(intro_dur)  # type: ignore[arg-type]
            intro_end = eff_intro
            intro_trim = (
                f",atrim=0:{max(0.01, eff_intro):.2f},asetpts=PTS-STARTPTS"
                if eff_intro < intro_d - 1e-6
                else ""
            )

            vol_expr = f"if(between(t,0,{intro_end:.2f}),{duck_vol_str},1.0)"
            base_branch = (
                f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
                if has_video_audio
                else silent_base
            )
            filter_complex = (
                f"{base_branch}"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo{intro_trim}[a1];"
                f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            return filter_complex, 2

        # eff_outro is not None and eff_intro is None
        outro_d = float(outro_dur)  # type: ignore[arg-type]
        outro_start = max(0.0, clip_dur - eff_outro)  # type: ignore[operator]
        outro_end = clip_dur
        delay_ms = int(round(outro_start * 1000))
        outro_trim = (
            f",atrim=0:{max(0.01, eff_outro):.2f},asetpts=PTS-STARTPTS"  # type: ignore[operator]
            if eff_outro < outro_d - 1e-6  # type: ignore[operator]
            else ""
        )

        vol_expr = (
            f"if(between(t,{outro_start:.2f},{outro_end:.2f}),{duck_vol_str},1.0)"
        )
        base_branch = (
            f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
            if has_video_audio
            else silent_base
        )
        filter_complex = (
            f"{base_branch}"
            f"[1:a]aresample=48000,aformat=channel_layouts=stereo{outro_trim},adelay={delay_ms}|{delay_ms}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        return filter_complex, 2

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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return bool(res.stdout.strip())
        except Exception:
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
    ) -> Path:
        """Mix intro/outro/breakdown voiceovers over ducked stream audio while copying video stream (-c:v copy)."""
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

        if intro_dur is None and outro_dur is None and breakdown_dur is None:
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
                text=True,
            )
            if same_file and target_out.is_file():
                shutil.move(str(target_out), str(output_video))
            return output_video
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "FFmpeg ducking mix failed (code %d): %s\nFalling back to original video.",
                exc.returncode,
                exc.stderr[:500] if exc.stderr else "",
            )
            if target_out.exists() and target_out != output_video:
                try:
                    target_out.unlink()
                except OSError:
                    pass
            return self._bypass_copy(input_video, output_video)
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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception:
            pass
        return 30.0
