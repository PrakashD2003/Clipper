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


class TransformativeNarrator:
    """Synthesizes adaptive intro/outro voiceovers and dynamically ducks stream audio via FFmpeg."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.enabled: bool = bool(config.get("transformation.enabled", True))
        self.mode: str = str(config.get("transformation.mode", "auto"))
        self.engine: str = str(config.get("transformation.tts_engine", "auto"))
        self.voice: str = str(config.get("transformation.tts_voice", "af_heart"))
        self.edge_tts_voice: str = str(
            config.get("transformation.edge_tts_voice", "en-US-GuyNeural")
        )
        self.elevenlabs_voice_id: str = str(
            config.get("transformation.elevenlabs_voice_id", "")
        )
        self.elevenlabs_model_id: str = str(
            config.get("transformation.elevenlabs_model_id", "eleven_multilingual_v2")
        )
        self.duck_volume: float = float(
            config.get("transformation.duck_stream_volume", 0.18)
        )

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
        """Synthesize locally with Kokoro-82M if installed."""
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
        if not self.enabled or self.mode == "raw":
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

    def _build_ducking_filtergraph(
        self,
        intro_dur: float | None,
        outro_dur: float | None,
        clip_duration: float,
    ) -> tuple[str, int]:
        """Build dynamic FFmpeg filter_complex string and input count.

        - Handles intro-only (2 inputs), outro-only (2 inputs), or intro+outro (3 inputs).
        - Normalizes stream audio and all narrator WAV inputs to 48000 Hz stereo
          (`aresample=48000,aformat=channel_layouts=stereo`) before `amix` so 24kHz mono
          Kokoro/Edge-TTS/ElevenLabs tracks mix cleanly with 48kHz stereo stream audio,
          using stereo `adelay={delay_ms}|{delay_ms}` for the outro track.
        - Clamps narration windows on short clips so intro and outro never overlap
          (if intro_dur + outro_dur > clip_duration * 0.75, trims outro or scales windows safely).
        - CRITICAL: Ducks [0:a] using `volume='if(between(t,...),0.18,1.0)':eval=frame`
          (MUST include `:eval=frame` because FFmpeg's `volume` defaults to `eval=once`
          at t=0, which would leave the entire clip stuck at 18% volume!).
        - Uses `amix=inputs={n}:duration=first:normalize=0`.
        """
        has_intro = intro_dur is not None and intro_dur > 0.01
        has_outro = outro_dur is not None and outro_dur > 0.01
        clip_dur = max(float(clip_duration), 0.1)

        if not has_intro and not has_outro:
            return "[0:a]aresample=48000,aformat=channel_layouts=stereo[aout]", 1

        duck_vol_str = f"{self.duck_volume:.2f}"

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

            outro_start = max(eff_intro, clip_dur - eff_outro)
            outro_end = clip_dur
            intro_end = eff_intro
            delay_ms = int(round(outro_start * 1000))

            vol_expr = (
                f"if(between(t,0,{intro_end:.2f})+"
                f"between(t,{outro_start:.2f},{outro_end:.2f}),"
                f"{duck_vol_str},1.0)"
            )
            filter_complex = (
                f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
                f"[2:a]aresample=48000,aformat=channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[a2];"
                f"[a0][a1][a2]amix=inputs=3:duration=first:normalize=0[aout]"
            )
            return filter_complex, 3

        if has_intro and not has_outro:
            intro_d = float(intro_dur)  # type: ignore[arg-type]
            eff_intro = min(intro_d, clip_dur * 0.75)
            intro_end = eff_intro

            vol_expr = f"if(between(t,0,{intro_end:.2f}),{duck_vol_str},1.0)"
            filter_complex = (
                f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
                f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            return filter_complex, 2

        # has_outro and not has_intro
        outro_d = float(outro_dur)  # type: ignore[arg-type]
        eff_outro = min(outro_d, clip_dur * 0.75)
        outro_start = max(0.0, clip_dur - eff_outro)
        outro_end = clip_dur
        delay_ms = int(round(outro_start * 1000))

        vol_expr = (
            f"if(between(t,{outro_start:.2f},{outro_end:.2f}),{duck_vol_str},1.0)"
        )
        filter_complex = (
            f"[0:a]volume='{vol_expr}':eval=frame,aresample=48000,aformat=channel_layouts=stereo[a0];"
            f"[1:a]aresample=48000,aformat=channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        return filter_complex, 2

    def mix_narration_with_ducking(
        self,
        input_video: Path,
        output_video: Path,
        intro_wav: Path | None,
        outro_wav: Path | None,
        clip_duration: float,
    ) -> Path:
        """Mix intro/outro voiceovers over ducked stream audio while copying video stream (-c:v copy)."""
        input_video = Path(input_video)
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Bypass mixing if disabled, raw mode, or neither voiceover exists
        if not self.enabled or self.mode == "raw":
            return self._bypass_copy(input_video, output_video)

        intro_dur: float | None = None
        outro_dur: float | None = None

        if intro_wav is not None:
            p_intro = Path(intro_wav)
            if p_intro.is_file() and p_intro.stat().st_size > 0:
                try:
                    info = sf.info(str(p_intro))
                    if info.duration > 0.05:
                        intro_dur = float(info.duration)
                except Exception as exc:
                    logger.debug("Failed reading intro_wav info: %s", exc)

        if outro_wav is not None:
            p_outro = Path(outro_wav)
            if p_outro.is_file() and p_outro.stat().st_size > 0:
                try:
                    info = sf.info(str(p_outro))
                    if info.duration > 0.05:
                        outro_dur = float(info.duration)
                except Exception as exc:
                    logger.debug("Failed reading outro_wav info: %s", exc)

        if intro_dur is None and outro_dur is None:
            return self._bypass_copy(input_video, output_video)

        # Ensure clip duration is positive
        eff_clip_duration = clip_duration
        if eff_clip_duration <= 0.0:
            eff_clip_duration = self._probe_clip_duration(input_video)

        filtergraph, input_count = self._build_ducking_filtergraph(
            intro_dur=intro_dur,
            outro_dur=outro_dur,
            clip_duration=eff_clip_duration,
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
        if intro_dur is not None and outro_dur is not None:
            cmd.extend(["-i", str(intro_wav), "-i", str(outro_wav)])
        elif intro_dur is not None:
            cmd.extend(["-i", str(intro_wav)])
        elif outro_dur is not None:
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
