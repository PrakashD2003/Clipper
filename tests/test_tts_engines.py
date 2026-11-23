"""Tests for pluggable TTS engines (Batch C: C1, C2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from streamclipper.assembly import tts_engines
from streamclipper.assembly.tts_engines import (
    EdgeTTS,
    ElevenLabsTTS,
    ExternalProcessTTS,
    KokoroTTS,
    TTSEngine,
)


def _wav(path: Path) -> Path:
    sf.write(str(path), (0.1 * np.ones(2400)).astype(np.float32), 24000)
    return path


def test_C1_engines_satisfy_protocol() -> None:
    for eng in (KokoroTTS(), EdgeTTS(), ElevenLabsTTS(), ExternalProcessTTS()):
        assert isinstance(eng, TTSEngine)
    assert ExternalProcessTTS().name == "chatterbox"


def test_C1_external_process_builds_command_and_returns_wav(tmp_path: Path) -> None:
    fake_py = tmp_path / "py.exe"
    fake_py.write_text("")
    ref = tmp_path / "ref.wav"
    out = tmp_path / "out.wav"
    eng = ExternalProcessTTS(str(fake_py), str(ref))
    assert eng.available()
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        _wav(Path(cmd[cmd.index("--out") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("subprocess.run", side_effect=fake_run):
        assert eng.synthesize("hello there", out) == out
    cmd = seen["cmd"]
    assert cmd[:3] == [str(fake_py), "-m", "streamclipper.tts_worker.chatterbox_worker"]
    assert cmd[cmd.index("--text") + 1] == "hello there"
    assert cmd[cmd.index("--ref") + 1] == str(ref)
    assert seen["kw"]["timeout"] == 120


def test_C1_external_process_failures_return_none(tmp_path: Path) -> None:
    fake_py = tmp_path / "py.exe"
    fake_py.write_text("")
    out = tmp_path / "out.wav"
    eng = ExternalProcessTTS(str(fake_py))
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("x", 120)):
        assert eng.synthesize("t", out) is None
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "x")):
        assert eng.synthesize("t", out) is None
    # success exit code but empty/missing output
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
        assert eng.synthesize("t", out) is None
    # missing interpreter
    assert ExternalProcessTTS(str(tmp_path / "nope.exe")).synthesize("t", out) is None
    assert not ExternalProcessTTS("").available()


def test_C1_external_process_relative_python_resolves_from_repo_root() -> None:
    eng = ExternalProcessTTS(".venv-tts/Scripts/python.exe")
    assert eng._python() == tts_engines.REPO_ROOT / ".venv-tts/Scripts/python.exe"


def test_C1_kokoro_unavailable_without_espeak_on_windows() -> None:
    with (
        patch.object(tts_engines.os, "name", "nt"),
        patch.object(tts_engines, "_is_espeak_available", return_value=False),
    ):
        eng = KokoroTTS()
        assert not eng.available()
        assert eng.synthesize("hi", Path("x.wav")) is None


def test_C1_elevenlabs_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    eng = ElevenLabsTTS("voice")
    assert not eng.available()
    assert eng.synthesize("hi", Path("x.wav")) is None


def test_C2_worker_is_standalone_cli() -> None:
    src = Path(tts_engines.__file__).resolve().parents[1] / "tts_worker" / "chatterbox_worker.py"
    text = src.read_text(encoding="utf-8")
    assert "streamclipper" not in text.replace("streamclipper.tts_worker", "")
    assert "ChatterboxTurboTTS" in text and "audio_prompt_path" in text
    # CLI parses args without importing heavy deps (help exits before model imports).
    res = subprocess.run(
        [sys.executable, str(src), "--help"], capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0 and "--ref" in res.stdout


def test_elevenlabs_voice_settings_override_is_optional() -> None:
    from streamclipper.assembly.tts_engines import ElevenLabsTTS

    default = ElevenLabsTTS("v")
    assert default.model_id == "eleven_v4" and default.voice_settings is None
    custom = ElevenLabsTTS("v", "eleven_v4_turbo", voice_settings={"stability": 0.4}, output_format="")
    assert custom.model_id == "eleven_v4_turbo"
    assert custom.voice_settings == {"stability": 0.4} and custom.output_format == ""


def test_elevenlabs_synthesize_populates_voice_settings(monkeypatch, tmp_path: Path) -> None:
    import io
    import json
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-123")
    eng = ElevenLabsTTS("voice_abc")

    captured_req: list = []

    def mock_urlopen(req, timeout=30.0):
        captured_req.append(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.code = 200
        mock_resp.read.return_value = b"ID3fake-mp3-bytes"
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = None
        return mock_ctx

    out_wav = tmp_path / "test.wav"

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        patch("streamclipper.assembly.tts_engines._mp3_to_wav", side_effect=lambda src, dst: dst.write_bytes(b"RIFF") and True),
    ):
        res = eng.synthesize("Hello world", out_wav)
        assert res == out_wav
        assert len(captured_req) == 1
        payload = json.loads(captured_req[0].data.decode("utf-8"))
        assert payload["voice_settings"] == {"stability": 0.40, "similarity_boost": 0.85}
        assert payload["model_id"] == "eleven_v4"

        # Explicit custom voice settings take precedence
        eng_custom = ElevenLabsTTS("voice_abc", voice_settings={"stability": 0.75, "similarity_boost": 0.95})
        res_custom = eng_custom.synthesize("Custom voice settings test", out_wav)
        assert res_custom == out_wav
        payload_custom = json.loads(captured_req[-1].data.decode("utf-8"))
        assert payload_custom["voice_settings"] == {"stability": 0.75, "similarity_boost": 0.95}

