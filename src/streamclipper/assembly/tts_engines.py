"""Pluggable TTS engines used by the narrator (first engine returning audio wins)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import soundfile as sf

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_espeak_available() -> bool:
    """Check if espeak-ng / espeak binary or library is available on the system."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return True
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True
    if os.name == "nt":
        prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        for p in (
            Path(prog_files) / "eSpeak NG" / "libespeak-ng.dll",
            Path(prog_files_x86) / "eSpeak NG" / "libespeak-ng.dll",
        ):
            if p.exists():
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(p)
                return True
    return False


def _nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def _mp3_to_wav(src: Path, dst: Path) -> bool:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "48000", "-ac", "2", str(dst)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _nonempty(dst)


@runtime_checkable
class TTSEngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def synthesize(self, text: str, out_wav: Path) -> Path | None: ...


class KokoroTTS:
    """Local Kokoro-82M (needs espeak-ng on Windows)."""

    name = "kokoro"

    def __init__(self, voice: str = "am_puck", speed: float = 1.1) -> None:
        self.voice = voice
        self.speed = float(speed)
        self._pipelines: dict[str, Any] = {}
        self.espeak_check: Any = None  # optional override callable() -> bool

    def _espeak_ok(self) -> bool:
        return bool((self.espeak_check or _is_espeak_available)())

    def available(self) -> bool:
        if os.name == "nt" and not self._espeak_ok():
            return False
        try:
            import importlib.util

            return importlib.util.find_spec("kokoro") is not None
        except (ImportError, ValueError):
            return False

    def clear(self) -> None:
        self._pipelines.clear()

    def synthesize(self, text: str, out_wav: Path) -> Path | None:
        if os.name == "nt" and not self._espeak_ok():
            logger.debug("espeak-ng not found on Windows; skipping Kokoro")
            return None
        try:
            from kokoro import KPipeline  # type: ignore
        except (ImportError, ModuleNotFoundError):
            logger.debug("Kokoro not installed; skipping")
            return None
        try:
            lang_code = "b" if self.voice.startswith("b") else "a"
            pipeline = self._pipelines.get(lang_code)
            if pipeline is None:
                pipeline = KPipeline(lang_code=lang_code)
                self._pipelines[lang_code] = pipeline
            audios: list[Any] = []
            for _, _, audio in pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
            ):
                if audio is not None and len(audio) > 0:
                    audios.append(audio)
            if not audios:
                return None
            import numpy as np

            combined = np.concatenate(audios) if len(audios) > 1 else audios[0]
            sf.write(str(out_wav), combined, 24000)
            return out_wav if _nonempty(out_wav) else None
        except Exception as exc:
            logger.warning("Kokoro synthesis failed: %s", exc)
            return None


class EdgeTTS:
    """Free cloud edge-tts."""

    name = "edge-tts"

    def __init__(self, voice: str = "en-US-AndrewMultilingualNeural", rate: str = "+10%") -> None:
        self.voice = voice
        self.rate = rate

    def available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("edge_tts") is not None
        except (ImportError, ValueError):
            return False

    def synthesize(self, text: str, out_wav: Path) -> Path | None:
        try:
            import edge_tts  # type: ignore
        except (ImportError, ModuleNotFoundError):
            logger.debug("edge-tts not installed; skipping")
            return None

        temp_mp3 = out_wav.with_name(f"{out_wav.stem}_edgetts.mp3")
        try:
            kwargs: dict[str, Any] = {}
            if self.rate:
                kwargs["rate"] = self.rate

            async def _generate() -> None:
                communicate = edge_tts.Communicate(text, self.voice, **kwargs)
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

            if not _nonempty(temp_mp3):
                return None
            return out_wav if _mp3_to_wav(temp_mp3, out_wav) else None
        except Exception as exc:
            logger.warning("Edge-TTS synthesis failed: %s", exc)
            return None
        finally:
            temp_mp3.unlink(missing_ok=True)


class ElevenLabsTTS:
    """ElevenLabs API (incl. personal voice clone) via ELEVENLABS_API_KEY + voice_id."""

    name = "elevenlabs"

    def __init__(self, voice_id: str = "", model_id: str = "eleven_multilingual_v2") -> None:
        self.voice_id = (voice_id or "").strip()
        self.model_id = model_id or "eleven_multilingual_v2"

    def available(self) -> bool:
        return bool(os.environ.get("ELEVENLABS_API_KEY", "").strip() and self.voice_id)

    def synthesize(self, text: str, out_wav: Path) -> Path | None:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key or not self.voice_id:
            logger.debug("ElevenLabs unconfigured; missing API key or voice_id")
            return None

        temp_mp3 = out_wav.with_name(f"{out_wav.stem}_elevenlabs.mp3")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
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
            return out_wav if _mp3_to_wav(temp_mp3, out_wav) else None
        except Exception as exc:
            logger.warning("ElevenLabs synthesis failed: %s", exc)
            return None
        finally:
            temp_mp3.unlink(missing_ok=True)


class ExternalProcessTTS:
    """Runs a TTS worker in a separate Python env (e.g. Chatterbox in ``.venv-tts``)."""

    TIMEOUT_SECONDS = 120

    def __init__(
        self,
        python_path: str = "",
        ref_wav: str = "",
        module: str = "streamclipper.tts_worker.chatterbox_worker",
        name: str = "chatterbox",
    ) -> None:
        self.python_path = (python_path or "").strip()
        self.ref_wav = (ref_wav or "").strip()
        self.module = module
        self.name = name

    def _python(self) -> Path | None:
        if not self.python_path:
            return None
        p = Path(self.python_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    def available(self) -> bool:
        p = self._python()
        return p is not None and p.is_file()

    def synthesize(self, text: str, out_wav: Path) -> Path | None:
        py = self._python()
        if py is None or not py.is_file():
            logger.debug("External TTS python not found (%s); skipping %s", py, self.name)
            return None
        cmd = [str(py), "-m", self.module, "--text", text, "--out", str(out_wav)]
        if self.ref_wav:
            cmd += ["--ref", self.ref_wav]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        try:
            out_wav.unlink(missing_ok=True)
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("%s TTS worker timed out after %ss", self.name, self.TIMEOUT_SECONDS)
            return None
        except Exception as exc:
            logger.warning("%s TTS worker failed: %s", self.name, exc)
            return None
        return out_wav if _nonempty(out_wav) else None
