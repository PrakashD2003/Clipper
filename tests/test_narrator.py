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
        assert narrator.voice == "af_heart"
        assert narrator.edge_tts_voice == "en-US-GuyNeural"
        assert narrator.elevenlabs_voice_id == ""
        assert narrator.elevenlabs_model_id == "eleven_multilingual_v2"
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
            assert narrator.voice == "af_heart"
            assert narrator.edge_tts_voice != narrator.voice

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
            mock_synth.assert_called_once_with("Generate this speech", out_tts_wav)
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
            side_effect=lambda text, path: create_synthetic_wav(path, 1.0),
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

    def test_ffmpeg_failure_falls_back_cleanly_to_copy(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """When FFmpeg fails with CalledProcessError, fallback safely copies original video."""
        narrator = TransformativeNarrator(base_config)

        in_video = tmp_path / "source.mp4"
        in_video.write_text("valid input video data", encoding="utf-8")
        out_video = tmp_path / "target.mp4"
        intro_wav = create_synthetic_wav(tmp_path / "intro.wav", duration_sec=1.0)

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Mock error")):
            result = narrator.mix_narration_with_ducking(
                input_video=in_video,
                output_video=out_video,
                intro_wav=intro_wav,
                outro_wav=None,
                clip_duration=5.0,
            )
            assert result == out_video
            assert out_video.is_file()
            assert out_video.read_text(encoding="utf-8") == "valid input video data"
