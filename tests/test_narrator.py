"""Unit tests for YPP Transformative Narrator and FFmpeg dynamic audio ducking."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure worktree's src and subpackages are prioritized over editable root install
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamclipper
_SC_DIR = str(_REPO_ROOT / "src" / "streamclipper")
if _SC_DIR not in streamclipper.__path__:
    streamclipper.__path__.insert(0, _SC_DIR)

try:
    import streamclipper.assembly
    _ASSEMBLY_DIR = str(_REPO_ROOT / "src" / "streamclipper" / "assembly")
    if _ASSEMBLY_DIR not in streamclipper.assembly.__path__:
        streamclipper.assembly.__path__.insert(0, _ASSEMBLY_DIR)
except (ImportError, AttributeError):
    pass

import numpy as np
import pytest
import soundfile as sf

from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_external_chatterbox():
    """Never launch the external Chatterbox worker from unit tests."""
    with patch.object(TransformativeNarrator, "_synthesize_chatterbox", return_value=None):
        yield


@pytest.fixture
def base_config() -> Config:
    """Fixture providing a default Config instance."""
    return Config()


@pytest.fixture
def custom_config(tmp_path: Path) -> Config:
    """Fixture providing a Config instance with custom transformation settings."""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(
        """
transformation:
  enabled: true
  mode: "narrative"
  tts_engine: "auto"
  tts_voice: "af_bella"
  edge_tts_voice: "en-US-AriaNeural"
  elevenlabs_voice_id: "el_voice_123"
  elevenlabs_model_id: "eleven_turbo_v2"
  duck_stream_volume: 0.15
""",
        encoding="utf-8",
    )
    return Config(cfg_file)


def create_synthetic_wav(path: Path, duration_sec: float = 1.0, sr: int = 24000, freq: float = 440.0) -> Path:
    """Create a real synthetic sine wave WAV file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), sine, sr)
    return path


def create_synthetic_mp4(path: Path, duration_sec: float = 3.0) -> Path:
    """Create a real synthetic MP4 with video and audio streams using ffmpeg lavfi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size=160x120:rate=20",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestTransformativeNarratorConfig:
    """Test configuration initialization and parameter propagation."""

    def test_init_defaults(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        assert narrator.enabled is True
        assert narrator.mode == "auto"
        assert narrator.engine == "auto"
        assert narrator.voice == "am_puck"
        assert narrator.edge_tts_voice == "en-US-AndrewMultilingualNeural"
        assert narrator.elevenlabs_voice_id == ""
        assert narrator.elevenlabs_model_id == "eleven_v4"
        assert pytest.approx(narrator.duck_volume, 0.01) == 0.18

    def test_init_custom_overrides(self, custom_config: Config) -> None:
        narrator = TransformativeNarrator(custom_config)
        assert narrator.enabled is True
        assert narrator.mode == "narrative"
        assert narrator.voice == "af_bella"
        assert narrator.edge_tts_voice == "en-US-AriaNeural"
        assert narrator.elevenlabs_voice_id == "el_voice_123"
        assert narrator.elevenlabs_model_id == "eleven_turbo_v2"
        assert pytest.approx(narrator.duck_volume, 0.01) == 0.15


class TestSynthesizeSpeechRoutingAndFallbacks:
    """Test speech synthesis routing, provider fallbacks, and error handling."""

    def test_empty_or_whitespace_text_returns_none(self, base_config: Config, tmp_path: Path) -> None:
        narrator = TransformativeNarrator(base_config)
        out_wav = tmp_path / "test.wav"
        assert narrator.synthesize_speech("", out_wav) is None
        assert narrator.synthesize_speech("   \n\t  ", out_wav) is None

    def test_disabled_transformation_returns_none(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "disabled.yaml"
        cfg_file.write_text("transformation:\n  enabled: false\n", encoding="utf-8")
        narrator = TransformativeNarrator(Config(cfg_file))
        assert narrator.synthesize_speech("Hello world", tmp_path / "out.wav") is None

    def test_engine_none_returns_none(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "none_engine.yaml"
        cfg_file.write_text("transformation:\n  tts_engine: 'none'\n", encoding="utf-8")
        narrator = TransformativeNarrator(Config(cfg_file))
        assert narrator.synthesize_speech("Hello world", tmp_path / "out.wav") is None

    def test_provider_fallback_kokoro_import_failure_to_edge_tts(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """In auto mode, if Kokoro fails (e.g. not installed), gracefully fall back to edge-tts."""
        narrator = TransformativeNarrator(base_config)
        out_wav = tmp_path / "fallback.wav"

        with (
            patch.object(narrator, "_synthesize_kokoro", return_value=None) as mock_kokoro,
            patch.object(
                narrator, "_synthesize_edge_tts", return_value=create_synthetic_wav(out_wav)
            ) as mock_edge,
        ):
            res = narrator.synthesize_speech("Fall back to Edge-TTS", out_wav)
            mock_kokoro.assert_called_once()
            mock_edge.assert_called_once_with("Fall back to Edge-TTS", out_wav)
            assert res == out_wav
            assert res.is_file()

    def test_auto_kokoro_success_bypasses_edge_tts(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """In auto mode, if Kokoro succeeds, edge-tts is never invoked."""
        narrator = TransformativeNarrator(base_config)
        out_wav = tmp_path / "kokoro_ok.wav"

        with (
            patch.object(
                narrator, "_synthesize_kokoro", return_value=create_synthetic_wav(out_wav)
            ) as mock_kokoro,
            patch.object(narrator, "_synthesize_edge_tts") as mock_edge,
        ):
            res = narrator.synthesize_speech("Local Kokoro success", out_wav)
            mock_kokoro.assert_called_once()
            mock_edge.assert_not_called()
            assert res == out_wav

    def test_explicit_edge_tts_uses_edge_tts_voice_never_kokoro_voice(
        self, tmp_path: Path
    ) -> None:
        """Edge-TTS must use self.edge_tts_voice and NEVER the Kokoro voice ID."""
        cfg_file = tmp_path / "edge_cfg.yaml"
        cfg_file.write_text(
            """
transformation:
  tts_engine: "edge-tts"
  tts_voice: "af_heart"
  edge_tts_voice: "en-US-GuyNeural"
""",
            encoding="utf-8",
        )
        narrator = TransformativeNarrator(Config(cfg_file))
        out_wav = tmp_path / "edge.wav"

        with patch.object(
            narrator, "_synthesize_edge_tts", return_value=create_synthetic_wav(out_wav)
        ) as mock_edge:
            narrator.synthesize_speech("Test edge voice", out_wav)
            mock_edge.assert_called_once_with("Test edge voice", out_wav)
            # Verify the configured edge-tts voice is distinct from kokoro voice
            assert narrator.edge_tts_voice == "en-US-GuyNeural"

    def test_kokoro_skips_and_falls_back_when_espeak_ng_missing_on_windows(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """On Windows, if espeak-ng is not installed, Kokoro is skipped and edge-tts is used."""
        narrator = TransformativeNarrator(base_config)
        out_wav = tmp_path / "fallback.wav"

        with (
            patch("os.name", "nt"),
            patch("streamclipper.assembly.narrator._is_espeak_available", return_value=False),
            patch.object(
                narrator, "_synthesize_edge_tts", return_value=create_synthetic_wav(out_wav)
            ) as mock_edge,
        ):
            # _synthesize_kokoro returns None directly when espeak-ng is missing on Windows
            assert narrator._synthesize_kokoro("test text", out_wav) is None
            # And synthesize_speech falls back to edge-tts in auto mode
            res = narrator.synthesize_speech("test text", out_wav)
            assert res == out_wav
            mock_edge.assert_called_once()
            assert narrator.voice == "am_puck"
            assert narrator.edge_tts_voice != narrator.voice

    def test_espeak_available_sets_phonemizer_env_when_dll_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When libespeak-ng.dll candidate exists on Windows, it sets PHONEMIZER_ESPEAK_LIBRARY."""
        fake_dll = tmp_path / "libespeak-ng.dll"
        fake_dll.write_text("fake binary", encoding="utf-8")
        monkeypatch.delenv("PHONEMIZER_ESPEAK_LIBRARY", raising=False)

        from streamclipper.assembly.narrator import _is_espeak_available

        with (
            patch("os.name", "nt"),
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=True),
        ):
            avail = _is_espeak_available()
            assert avail is True
            assert os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") is not None

    def test_get_effective_narration_durations(self, base_config: Config) -> None:
        """get_effective_narration_durations cleanly scales durations to prevent overlap and climax bleed."""
        narrator = TransformativeNarrator(base_config)
        # None inputs
        assert narrator.get_effective_narration_durations(None, None, 10.0) == (None, None)
        # Within budget (3s + 3s = 6s < 15s budget on 20s clip)
        assert narrator.get_effective_narration_durations(3.0, 3.0, 20.0) == (3.0, 3.0)
        # Exceeds budget (4s + 4s = 8s > 6.0s on 8s clip -> 3.0s, 3.0s)
        eff_i, eff_o = narrator.get_effective_narration_durations(4.0, 4.0, 8.0)
        assert eff_i == pytest.approx(3.0)
        assert eff_o == pytest.approx(3.0)
        # Single intro clamped (9s on 10s clip -> 7.5s)
        eff_i, eff_o = narrator.get_effective_narration_durations(9.0, None, 10.0)
        assert eff_i == pytest.approx(7.5)
        assert eff_o is None

    def test_elevenlabs_unconfigured_falls_back_to_auto(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When elevenlabs is configured but ELEVENLABS_API_KEY is missing, fall back to auto."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        cfg_file = tmp_path / "el_cfg.yaml"
        cfg_file.write_text(
            """
transformation:
  tts_engine: "elevenlabs"
  elevenlabs_voice_id: "test_voice_id"
""",
            encoding="utf-8",
        )
        narrator = TransformativeNarrator(Config(cfg_file))
        out_wav = tmp_path / "el_fallback.wav"

        with (
            patch.object(narrator, "_synthesize_kokoro", return_value=None) as mock_kokoro,
            patch.object(
                narrator, "_synthesize_edge_tts", return_value=create_synthetic_wav(out_wav)
            ) as mock_edge,
        ):
            res = narrator.synthesize_speech("Unconfigured ElevenLabs test", out_wav)
            mock_kokoro.assert_called_once()
            mock_edge.assert_called_once()
            assert res == out_wav

    def test_mocked_elevenlabs_api_call_with_api_key_and_voice_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ELEVENLABS_API_KEY and voice_id set, ElevenLabs API is called with correct headers & body."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "mock_key_xyz123")
        cfg_file = tmp_path / "el_valid.yaml"
        cfg_file.write_text(
            """
transformation:
  tts_engine: "elevenlabs"
  elevenlabs_voice_id: "custom_clone_999"
  elevenlabs_model_id: "eleven_multilingual_v2"
""",
            encoding="utf-8",
        )
        narrator = TransformativeNarrator(Config(cfg_file))
        out_wav = tmp_path / "elevenlabs_out.wav"

        recorded_req = []

        def mock_urlopen(req, timeout=30.0):
            recorded_req.append(req)
            # Create a mock response returning 1-second synthetic MP3/WAV bytes
            raw_audio = io.BytesIO()
            t = np.linspace(0, 0.5, 8000, endpoint=False)
            sf.write(raw_audio, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="WAV")

            mock_enter = MagicMock()
            mock_enter.status = 200
            mock_enter.code = 200
            mock_enter.read.return_value = raw_audio.getvalue()

            mock_ctx = MagicMock()
            mock_ctx.__enter__.return_value = mock_enter
            mock_ctx.__exit__.return_value = None
            return mock_ctx

        with (
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
            patch("subprocess.run") as mock_subproc,
        ):
            # Simulate ffmpeg converting mp3 to wav
            def fake_ffmpeg(cmd, **kwargs):
                create_synthetic_wav(Path(cmd[-1]), duration_sec=0.5)
                return MagicMock(returncode=0)

            mock_subproc.side_effect = fake_ffmpeg

            res = narrator.synthesize_speech("ElevenLabs voiceover test", out_wav)
            assert res == out_wav
            assert len(recorded_req) == 1
            req = recorded_req[0]
            assert "custom_clone_999" in req.full_url
            assert req.get_header("Xi-api-key") == "mock_key_xyz123"
            assert req.get_header("Content-type") == "application/json"

            body = json.loads(req.data.decode("utf-8"))
            assert body["text"] == "ElevenLabs voiceover test"
            assert body["model_id"] == "eleven_multilingual_v2"
            # No override: ElevenLabs uses the settings saved on the voice.
            assert "voice_settings" not in body
            assert "output_format=mp3_44100_128" in req.full_url

    def test_unknown_engine_falls_back_to_auto(self, base_config: Config, tmp_path: Path) -> None:
        narrator = TransformativeNarrator(base_config)
        narrator.engine = "some_unknown_engine"
        out_wav = tmp_path / "unknown_engine.wav"

        with (
            patch.object(narrator, "_synthesize_kokoro", return_value=None),
            patch.object(
                narrator, "_synthesize_edge_tts", return_value=create_synthetic_wav(out_wav)
            ) as mock_edge,
        ):
            res = narrator.synthesize_speech("Unknown engine fallback", out_wav)
            mock_edge.assert_called_once()
            assert res == out_wav


class TestCustomVoiceoverPriority:
    """Test custom human mic voiceover priority over TTS synthesis."""

    def test_custom_voice_intro_takes_priority_over_tts(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        narrator = TransformativeNarrator(base_config)
        custom_mic_intro = create_synthetic_wav(tmp_path / "my_human_intro.wav", duration_sec=2.0)
        out_tts_wav = tmp_path / "should_not_be_created.wav"

        with patch.object(narrator, "synthesize_speech") as mock_synth:
            resolved = narrator.resolve_narration_track(
                custom_path=custom_mic_intro,
                text="AI generated intro script",
                output_wav=out_tts_wav,
            )
            mock_synth.assert_not_called()
            assert resolved == custom_mic_intro
            assert resolved.is_file()

    def test_custom_voice_outro_takes_priority_over_tts(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        narrator = TransformativeNarrator(base_config)
        custom_mic_outro = create_synthetic_wav(tmp_path / "my_human_outro.wav", duration_sec=1.5)
        out_tts_wav = tmp_path / "tts_outro.wav"

        with patch.object(narrator, "synthesize_speech") as mock_synth:
            resolved = narrator.resolve_narration_track(
                custom_path=custom_mic_outro,
                text="AI generated outro script",
                output_wav=out_tts_wav,
            )
            mock_synth.assert_not_called()
            assert resolved == custom_mic_outro

    def test_missing_custom_voice_falls_back_to_tts(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        narrator = TransformativeNarrator(base_config)
        non_existent_path = tmp_path / "ghost_file.wav"
        out_tts_wav = tmp_path / "generated_tts.wav"

        with patch.object(
            narrator, "synthesize_speech", return_value=create_synthetic_wav(out_tts_wav)
        ) as mock_synth:
            resolved = narrator.resolve_narration_track(
                custom_path=non_existent_path,
                text="Generate this speech",
                output_wav=out_tts_wav,
            )
            mock_synth.assert_called_once_with(
                "Generate this speech", out_tts_wav, max_seconds=None
            )
            assert resolved == out_tts_wav

    def test_prepare_narration_tracks_respects_both_priorities(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        narrator = TransformativeNarrator(base_config)
        custom_intro = create_synthetic_wav(tmp_path / "mic_intro.wav", duration_sec=1.0)
        work_dir = tmp_path / "work"

        with patch.object(
            narrator,
            "synthesize_speech",
            side_effect=lambda text, path, **kw: create_synthetic_wav(path, 1.0),
        ) as mock_synth:
            intro_track, outro_track = narrator.prepare_narration_tracks(
                narration_intro="Intro text",
                narration_outro="Outro text",
                voice_intro_path=custom_intro,
                voice_outro_path=None,
                temp_dir=work_dir,
            )
            assert intro_track == custom_intro
            assert outro_track is not None
            assert outro_track.name == "narrator_outro.wav"
            mock_synth.assert_called_once()

    def test_prepare_narration_tracks_disabled_or_raw(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "raw.yaml"
        cfg_file.write_text("transformation:\n  mode: 'raw'\n", encoding="utf-8")
        narrator = TransformativeNarrator(Config(cfg_file))
        intro, outro = narrator.prepare_narration_tracks("intro", "outro")
        assert intro is None
        assert outro is None


class TestDuckingFiltergraph:
    """Test dynamic FFmpeg ducking filtergraph generation, input counts, and audio normalization."""

    def test_dynamic_filtergraph_intro_only(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=3.5, outro_dur=None, clip_duration=12.0
        )
        assert input_count == 2
        assert "[0:a]volume='if(between(t,0,3.50),0.18,1.0)':eval=frame" in filtergraph
        assert "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1]" in filtergraph
        assert "[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]" in filtergraph
        assert "adelay=" not in filtergraph

    def test_dynamic_filtergraph_outro_only(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=None, outro_dur=3.0, clip_duration=12.0
        )
        assert input_count == 2
        # outro starts at 12.0 - 3.0 = 9.0s -> delay 9000ms
        assert "[0:a]volume='if(between(t,9.00,12.00),0.18,1.0)':eval=frame" in filtergraph
        assert "adelay=9000|9000" in filtergraph
        assert "[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]" in filtergraph

    def test_dynamic_filtergraph_both_intro_and_outro(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=3.0, outro_dur=3.0, clip_duration=15.0
        )
        assert input_count == 3
        # Combined ducking expression: intro [0, 3.0] + outro [12.0, 15.0]
        assert "[0:a]volume='if(between(t,0,3.00)+between(t,12.00,15.00),0.18,1.0)':eval=frame" in filtergraph
        assert "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1]" in filtergraph
        assert "[2:a]aresample=48000,aformat=channel_layouts=stereo,adelay=12000|12000[a2]" in filtergraph
        assert "[a0][a1][a2]amix=inputs=3:duration=first:normalize=0[aout]" in filtergraph

    def test_eval_frame_guardrail_present_in_all_filtergraphs(self, base_config: Config) -> None:
        """CRITICAL: FFmpeg volume filter MUST include :eval=frame to prevent t=0 freeze."""
        narrator = TransformativeNarrator(base_config)
        fg_intro, _ = narrator._build_ducking_filtergraph(2.0, None, 10.0)
        fg_outro, _ = narrator._build_ducking_filtergraph(None, 2.0, 10.0)
        fg_both, _ = narrator._build_ducking_filtergraph(2.0, 2.0, 10.0)

        assert ":eval=frame" in fg_intro
        assert ":eval=frame" in fg_outro
        assert ":eval=frame" in fg_both

    def test_stereo_sample_rate_normalization_present(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        fg, _ = narrator._build_ducking_filtergraph(2.0, 2.0, 10.0)
        # All 3 audio tracks must pass through 48kHz stereo normalization
        norm_str = "aresample=48000,aformat=channel_layouts=stereo"
        assert fg.count(norm_str) == 3


class TestShortClipOverlapClamping:
    """Test safe scaling and clamping on short clips to guarantee climax reaction audio preservation."""

    def test_both_windows_scaled_when_exceeding_75_percent_limit(
        self, base_config: Config
    ) -> None:
        """When intro_dur + outro_dur > clip_duration * 0.75, both windows scale down proportionally."""
        narrator = TransformativeNarrator(base_config)
        # Clip duration 8.0s -> max combined narration = 8.0 * 0.75 = 6.0s
        # intro 4.0s + outro 4.0s = 8.0s > 6.0s
        # scale = 6.0 / 8.0 = 0.75 -> effective intro = 3.0s, effective outro = 3.0s
        # outro starts at 8.0 - 3.0 = 5.0s -> climax gap = [3.0s, 5.0s] (2.0s untouched!)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=4.0, outro_dur=4.0, clip_duration=8.0
        )
        assert input_count == 3
        assert "between(t,0,3.00)+between(t,5.00,8.00)" in filtergraph
        assert "adelay=5000|5000" in filtergraph

    def test_intro_only_clamping(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        # Clip 10.0s -> max intro = 7.5s. Intro 9.0s -> clamped to 7.50s
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=9.0, outro_dur=None, clip_duration=10.0
        )
        assert input_count == 2
        assert "between(t,0,7.50)" in filtergraph

    def test_outro_only_clamping(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        # Clip 10.0s -> max outro = 7.5s. Outro 9.0s -> clamped to 7.50s
        # outro starts at 10.0 - 7.5 = 2.50s -> delay 2500ms
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=None, outro_dur=9.0, clip_duration=10.0
        )
        assert input_count == 2
        assert "between(t,2.50,10.00)" in filtergraph
        assert "adelay=2500|2500" in filtergraph

    def test_unclamped_when_within_budget(self, base_config: Config) -> None:
        narrator = TransformativeNarrator(base_config)
        # Clip 20.0s -> budget 15.0s. Intro 3.0s + Outro 3.0s = 6.0s (within budget)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=3.0, outro_dur=3.0, clip_duration=20.0
        )
        assert input_count == 3
        assert "between(t,0,3.00)+between(t,17.00,20.00)" in filtergraph
        assert "adelay=17000|17000" in filtergraph
        assert "atrim=" not in filtergraph

    def test_clamped_narrator_tracks_are_atrimmed_to_prevent_climax_bleed_and_overlap(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """Short-clip clamping must atrim [1:a]/[2:a] so intro never bleeds into climax or outro."""
        narrator = TransformativeNarrator(base_config)
        # Clip 6.0s -> max combined = 4.5s; intro=4.0s, outro=4.0s -> eff_intro=2.25s, eff_outro=2.25s
        fg, input_count = narrator._build_ducking_filtergraph(
            intro_dur=4.0, outro_dur=4.0, clip_duration=6.0
        )
        assert input_count == 3
        assert "[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:2.25,asetpts=PTS-STARTPTS[a1]" in fg
        assert "[2:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:2.25,asetpts=PTS-STARTPTS,adelay=3750|3750[a2]" in fg

        # Verify FFmpeg executes the clamped filtergraph cleanly on real media
        in_video = create_synthetic_mp4(tmp_path / "short_clip.mp4", duration_sec=3.0)
        intro_wav = create_synthetic_wav(tmp_path / "long_intro.wav", duration_sec=2.0, freq=440.0)
        outro_wav = create_synthetic_wav(tmp_path / "long_outro.wav", duration_sec=2.0, freq=880.0)
        out_video = tmp_path / "short_clamped_out.mp4"

        res = narrator.mix_narration_with_ducking(
            input_video=in_video,
            output_video=out_video,
            intro_wav=intro_wav,
            outro_wav=outro_wav,
            clip_duration=3.0,
        )
        assert res == out_video
        assert out_video.is_file() and out_video.stat().st_size > 0


class TestMixNarrationWithDucking:
    """Test mixing voiceovers over stream video with ducking and fallback safety."""

    def test_bypass_when_disabled(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "disabled.yaml"
        cfg_file.write_text("transformation:\n  enabled: false\n", encoding="utf-8")
        narrator = TransformativeNarrator(Config(cfg_file))

        in_vid = tmp_path / "in.mp4"
        in_vid.write_text("mock video", encoding="utf-8")
        out_vid = tmp_path / "out.mp4"

        res = narrator.mix_narration_with_ducking(in_vid, out_vid, tmp_path / "i.wav", None, 10.0)
        assert res == out_vid
        assert out_vid.read_text(encoding="utf-8") == "mock video"

    def test_bypass_when_raw_mode(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "raw_mode.yaml"
        cfg_file.write_text("transformation:\n  mode: 'raw'\n", encoding="utf-8")
        narrator = TransformativeNarrator(Config(cfg_file))

        in_vid = tmp_path / "in.mp4"
        in_vid.write_text("mock raw video", encoding="utf-8")
        out_vid = tmp_path / "out.mp4"

        res = narrator.mix_narration_with_ducking(in_vid, out_vid, tmp_path / "i.wav", None, 10.0)
        assert res == out_vid
        assert out_vid.read_text(encoding="utf-8") == "mock raw video"

    def test_bypass_when_no_audio_tracks_provided(self, base_config: Config, tmp_path: Path) -> None:
        narrator = TransformativeNarrator(base_config)
        in_vid = tmp_path / "in.mp4"
        in_vid.write_text("video content", encoding="utf-8")
        out_vid = tmp_path / "out.mp4"

        res = narrator.mix_narration_with_ducking(in_vid, out_vid, None, None, 10.0)
        assert res == out_vid
        assert out_vid.read_text(encoding="utf-8") == "video content"

    def test_real_e2e_ducking_intro_and_outro(self, base_config: Config, tmp_path: Path) -> None:
        """End-to-end integration test with real FFmpeg, synthetic MP4, and synthetic WAVs."""
        narrator = TransformativeNarrator(base_config)

        in_video = create_synthetic_mp4(tmp_path / "stream_clip.mp4", duration_sec=4.0)
        intro_wav = create_synthetic_wav(tmp_path / "intro.wav", duration_sec=1.0, freq=440.0)
        outro_wav = create_synthetic_wav(tmp_path / "outro.wav", duration_sec=1.0, freq=880.0)
        out_video = tmp_path / "final_ducked_clip.mp4"

        result = narrator.mix_narration_with_ducking(
            input_video=in_video,
            output_video=out_video,
            intro_wav=intro_wav,
            outro_wav=outro_wav,
            clip_duration=4.0,
        )

        assert result == out_video
        assert out_video.is_file()
        assert out_video.stat().st_size > 0

        # Verify output duration via probe
        probe_dur = narrator._probe_clip_duration(out_video)
        assert 3.5 <= probe_dur <= 4.5

    def test_real_e2e_ducking_intro_only(self, base_config: Config, tmp_path: Path) -> None:
        """End-to-end integration test for intro-only (2-input) ducking."""
        narrator = TransformativeNarrator(base_config)

        in_video = create_synthetic_mp4(tmp_path / "stream_clip2.mp4", duration_sec=3.0)
        intro_wav = create_synthetic_wav(tmp_path / "intro2.wav", duration_sec=1.0)
        out_video = tmp_path / "final_intro_only.mp4"

        result = narrator.mix_narration_with_ducking(
            input_video=in_video,
            output_video=out_video,
            intro_wav=intro_wav,
            outro_wav=None,
            clip_duration=3.0,
        )

        assert result == out_video
        assert out_video.is_file()
        assert out_video.stat().st_size > 0

    def test_real_e2e_ducking_outro_only(self, base_config: Config, tmp_path: Path) -> None:
        """End-to-end integration test for outro-only (2-input) ducking."""
        narrator = TransformativeNarrator(base_config)

        in_video = create_synthetic_mp4(tmp_path / "stream_clip3.mp4", duration_sec=3.0)
        outro_wav = create_synthetic_wav(tmp_path / "outro3.wav", duration_sec=1.0)
        out_video = tmp_path / "final_outro_only.mp4"

        result = narrator.mix_narration_with_ducking(
            input_video=in_video,
            output_video=out_video,
            intro_wav=None,
            outro_wav=outro_wav,
            clip_duration=3.0,
        )

        assert result == out_video
        assert out_video.is_file()
        assert out_video.stat().st_size > 0

    def test_in_place_mixing_safe_replacement(self, base_config: Config, tmp_path: Path) -> None:
        """When input_video and output_video point to the same file, in-place replace is handled cleanly."""
        narrator = TransformativeNarrator(base_config)

        video_path = create_synthetic_mp4(tmp_path / "inplace.mp4", duration_sec=3.0)
        intro_wav = create_synthetic_wav(tmp_path / "intro_in.wav", duration_sec=1.0)

        result = narrator.mix_narration_with_ducking(
            input_video=video_path,
            output_video=video_path,
            intro_wav=intro_wav,
            outro_wav=None,
            clip_duration=3.0,
        )

        assert result == video_path
        assert video_path.is_file()
        assert video_path.stat().st_size > 0

    def test_ffmpeg_failure_raises_runtime_error(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """When FFmpeg fails with CalledProcessError, a RuntimeError carrying stderr is raised."""
        narrator = TransformativeNarrator(base_config)

        in_video = tmp_path / "source.mp4"
        in_video.write_text("valid input video data", encoding="utf-8")
        out_video = tmp_path / "target.mp4"
        intro_wav = create_synthetic_wav(tmp_path / "intro.wav", duration_sec=1.0)

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Mock error")):
            with pytest.raises(RuntimeError, match="Mock error"):
                narrator.mix_narration_with_ducking(
                    input_video=in_video,
                    output_video=out_video,
                    intro_wav=intro_wav,
                    outro_wav=None,
                    clip_duration=5.0,
                )

    def test_silent_video_still_receives_narrator_audio_track(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """Silent video clips (no audio stream) must receive narrator intro/outro instead of bypassing."""
        narrator = TransformativeNarrator(base_config)

        silent_video = tmp_path / "silent_input.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x120:d=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(silent_video),
            ],
            check=True,
            capture_output=True,
        )
        intro_wav = create_synthetic_wav(tmp_path / "intro_s.wav", duration_sec=0.8, freq=440.0)
        outro_wav = create_synthetic_wav(tmp_path / "outro_s.wav", duration_sec=0.8, freq=880.0)
        out_video = tmp_path / "silent_with_voiceover.mp4"

        res = narrator.mix_narration_with_ducking(
            input_video=silent_video,
            output_video=out_video,
            intro_wav=intro_wav,
            outro_wav=outro_wav,
            clip_duration=3.0,
        )
        assert res == out_video
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(out_video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "aac"

    def test_non_wav_m4a_custom_voiceover_is_probed_and_mixed(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """Non-WAV custom voiceover containers (e.g., .m4a) must be probed via ffprobe fallback and mixed."""
        narrator = TransformativeNarrator(base_config)
        in_video = create_synthetic_mp4(tmp_path / "stream_m4a.mp4", duration_sec=3.0)
        intro_m4a = tmp_path / "custom_intro.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=660:duration=0.9",
                "-c:a",
                "aac",
                str(intro_m4a),
            ],
            check=True,
            capture_output=True,
        )
        out_video = tmp_path / "mixed_m4a.mp4"

        narrator.mix_narration_with_ducking(
            input_video=in_video,
            output_video=out_video,
            intro_wav=intro_m4a,
            outro_wav=None,
            clip_duration=3.0,
        )
        assert out_video.is_file() and out_video.stat().st_size > 0
        # Byte comparison proves FFmpeg actually mixed the .m4a track rather than bypassing via copy
        assert out_video.read_bytes() != in_video.read_bytes()

    def test_case_insensitive_raw_mode_and_tiny_atrim_floor(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """Uppercase/whitespace 'RAW' config must bypass mixing, and tiny clips must not emit atrim=0:0.00."""
        cfg_file = tmp_path / "raw_upper.yaml"
        cfg_file.write_text("transformation:\n  mode: ' RAW '\n", encoding="utf-8")
        raw_narrator = TransformativeNarrator(Config(cfg_file))
        wav = create_synthetic_wav(tmp_path / "intro_raw.wav", duration_sec=0.5)
        assert raw_narrator.prepare_narration_tracks(
            voice_intro_path=wav, temp_dir=tmp_path
        ) == (None, None)

        narrator = TransformativeNarrator(base_config)
        fg, count = narrator._build_ducking_filtergraph(
            intro_dur=10.0, outro_dur=0.05, clip_duration=0.1
        )
        assert count == 3
        assert "atrim=0:0.00" not in fg


# ---------------------------------------------------------------------------
# Batch C: engine chain, callout, loudness (C3-C5)
# ---------------------------------------------------------------------------

def _chain_narrator(tmp_path: Path, yaml_body: str) -> TransformativeNarrator:
    cfg = tmp_path / "chain.yaml"
    cfg.write_text(yaml_body, encoding="utf-8")
    return TransformativeNarrator(Config(cfg))


def test_C3_chain_first_engine_returning_audio_wins(tmp_path: Path) -> None:
    n = _chain_narrator(tmp_path, "transformation:\n  tts_engines: [chatterbox, kokoro, edge-tts]\n")
    out = tmp_path / "c3.wav"
    with (
        patch.object(n, "_synthesize_chatterbox", return_value=create_synthetic_wav(out)) as cb,
        patch.object(n, "_synthesize_kokoro") as ko,
        patch.object(n, "_synthesize_edge_tts") as ed,
    ):
        assert n.synthesize_speech("hello world", out) == out
    cb.assert_called_once()
    ko.assert_not_called()
    ed.assert_not_called()


def test_C3_chain_falls_through_failed_engines(tmp_path: Path) -> None:
    n = _chain_narrator(tmp_path, "transformation:\n  tts_engines: [chatterbox, kokoro, edge-tts]\n")
    out = tmp_path / "c3b.wav"
    with (
        patch.object(n, "_synthesize_chatterbox", return_value=None),
        patch.object(n, "_synthesize_kokoro", return_value=None),
        patch.object(n, "_synthesize_edge_tts", return_value=create_synthetic_wav(out)) as ed,
    ):
        assert n.synthesize_speech("hello world", out) == out
    ed.assert_called_once()


def test_C3_legacy_engine_overrides_chain(tmp_path: Path) -> None:
    n = _chain_narrator(
        tmp_path, 'transformation:\n  tts_engine: "edge-tts"\n  tts_engines: [chatterbox, kokoro]\n'
    )
    out = tmp_path / "c3c.wav"
    with (
        patch.object(n, "_synthesize_chatterbox") as cb,
        patch.object(n, "_synthesize_edge_tts", return_value=create_synthetic_wav(out)),
    ):
        assert n.synthesize_speech("hello", out) == out
    cb.assert_not_called()


def test_C3_kokoro_and_edge_use_configured_voice_speed_rate(tmp_path: Path) -> None:
    n = _chain_narrator(
        tmp_path,
        "transformation:\n  tts_voice: am_fenrir\n  tts_speed: 1.25\n"
        '  edge_tts_voice: en-US-AriaNeural\n  edge_tts_rate: "+20%"\n',
    )
    assert n.voice == "am_fenrir" and n.tts_speed == 1.25
    assert n.edge_tts_rate == "+20%"
    with patch("streamclipper.assembly.narrator.EdgeTTS") as edge_cls:
        edge_cls.return_value.synthesize.return_value = None
        n._synthesize_edge_tts("x", tmp_path / "e.wav")
    edge_cls.assert_called_once_with("en-US-AriaNeural", "+20%")


def test_C4_finalize_graph_has_callout_gain_ramps_and_loudnorm(tmp_path: Path) -> None:
    n = _chain_narrator(
        tmp_path,
        "transformation:\n  target_lufs: -14\n  narration_gain_db: 3\n  crossfade_seconds: 0.25\n",
    )
    fg, count = n._build_ducking_filtergraph(
        3.0, 3.0, 20.0, callout_dur=1.5, callout_start=8.0, finalize=True
    )
    assert count == 4
    assert ":eval=frame" in fg
    assert "clip(min(" in fg and "max(" in fg  # smooth ramps across three windows
    assert fg.count("loudnorm=I=-16:TP=-1.5:LRA=11") == 3
    assert fg.count("volume=3.0dB") == 3
    assert "[3:a]" in fg and "adelay=8000|8000" in fg
    assert "amix=inputs=4:duration=first:normalize=0[amixed]" in fg
    assert fg.endswith("[amixed]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo[aout]")


def test_C4_duck_full_keeps_stream_ducked_whole_clip(base_config: Config) -> None:
    n = TransformativeNarrator(base_config)
    fg, _ = n._build_ducking_filtergraph(
        None, None, 10.0, callout_dur=1.0, callout_start=4.0, duck_full=True, finalize=True
    )
    assert "[0:a]volume='0.18':eval=frame" in fg


def test_C4_callout_only_window_ducks_only_under_callout(base_config: Config) -> None:
    n = TransformativeNarrator(base_config)
    fg, count = n._build_ducking_filtergraph(
        None, None, 10.0, callout_dur=1.0, callout_start=4.0
    )
    assert count == 2
    assert "volume='if(between(t,4.00,5.00),0.18,1.0)':eval=frame" in fg


def test_C4_real_mix_with_callout_and_loudnorm(base_config: Config, tmp_path: Path) -> None:
    n = TransformativeNarrator(base_config)
    vid = create_synthetic_mp4(tmp_path / "in.mp4", duration_sec=6.0)
    intro = create_synthetic_wav(tmp_path / "i.wav", 1.0)
    callout = create_synthetic_wav(tmp_path / "c.wav", 1.0, freq=660.0)
    out = tmp_path / "mixed.mp4"
    res = n.mix_narration_with_ducking(
        vid, out, intro, None, 6.0, callout_wav=callout, callout_start=2.5
    )
    assert res == out and out.stat().st_size > 0
    # callout alone (no intro/outro) still mixes
    out2 = tmp_path / "mixed2.mp4"
    n.mix_narration_with_ducking(vid, out2, None, None, 6.0, callout_wav=callout, duck_full=True)
    assert out2.stat().st_size > 0


def test_C4_prepare_callout_track(base_config: Config, tmp_path: Path) -> None:
    n = TransformativeNarrator(base_config)
    wav = tmp_path / "narrator_callout.wav"
    with patch.object(n, "synthesize_speech", return_value=wav) as syn:
        assert n.prepare_callout_track("Wait for it", tmp_path) == wav
    assert syn.call_args.args[0] == "Wait for it"
    assert n.prepare_callout_track("", tmp_path) is None
    n.callout_enabled = False
    assert n.prepare_callout_track("Wait for it", tmp_path) is None


def test_C5_normalize_audio_copies_video_and_loudnorms(base_config: Config, tmp_path: Path) -> None:
    n = TransformativeNarrator(base_config)
    vid = create_synthetic_mp4(tmp_path / "n_in.mp4", duration_sec=3.0)
    out = tmp_path / "n_out.mp4"
    with patch("subprocess.run", wraps=subprocess.run) as spy:
        assert n.normalize_audio(vid, out) == out
    ff = [c.args[0] for c in spy.call_args_list if c.args[0][0] == "ffmpeg"][0]
    assert ff[ff.index("-c:v") + 1] == "copy"
    assert any("loudnorm=I=" in a for a in ff)
    assert out.stat().st_size > 0
    # in-place
    n.normalize_audio(vid, vid)
    assert vid.stat().st_size > 0
