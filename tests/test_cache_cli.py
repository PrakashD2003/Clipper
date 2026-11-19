"""Unit tests for StepCache and the StreamClipper Click CLI."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

import streamclipper.cli as cli_mod
import streamclipper.config as config_mod
from streamclipper.cache import StepCache
from streamclipper.cli import cli
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import (
    ClipMetadata,
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    SignalType,
    StreamInfo,
    TimeRange,
    TranscriptSegment,
    TranscriptionResult,
    WordTimestamp,
)


def _make_config(
    tmp_path: Path,
    *,
    overrides: dict | None = None,
    profile: HardwareProfile | None = None,
) -> Config:
    """Create a Config isolated to tmp_path cache dir with a deterministic HardwareProfile."""
    cfg_data: dict = {
        "general": {
            "cache_dir": str(tmp_path / ".cache"),
            "output_dir": str(tmp_path / "output"),
            "temp_dir": str(tmp_path / "temp"),
        }
    }
    if overrides:
        for section, values in overrides.items():
            if section in cfg_data and isinstance(values, dict):
                cfg_data[section].update(values)
            else:
                cfg_data[section] = values

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    hw = profile or HardwareProfile(
        tier="cuda_mid_tensor",
        gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        vram_gb=6.44,
        compute_capability="8.9",
        has_tensor_cores=True,
        has_nvenc=True,
        device="cuda",
        model_size="large-v2",
        compute_type="int8_float16",
        batch_size=8,
        llm_model="qwen3:8b",
        video_codec="h264_nvenc",
    )
    return Config(user_config_path=cfg_path, hardware_profile=hw)


def _make_full_pipeline_state(
    url: str = "https://youtube.com/watch?v=gta6_test",
    title: str = "GTA 6 Launch Day Stream!",
    status: PipelineStatus = PipelineStatus.ASSEMBLING,
    include_transcript: bool = True,
) -> PipelineState:
    """Build a populated PipelineState with nested transcript, spikes, highlights, and clips."""
    w1 = WordTimestamp(text="NO", start=121.0, end=121.3, confidence=0.98)
    w2 = WordTimestamp(text="WAY!", start=121.35, end=121.8, confidence=0.95)
    transcript = (
        TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="NO WAY!",
                    start=121.0,
                    end=121.8,
                    words=[w1, w2],
                )
            ],
            full_text="NO WAY!",
            language="en",
        )
        if include_transcript
        else None
    )

    highlight = Highlight(
        start=115.0,
        end=165.0,
        score=9.4,
        title="INSANE POLICE CHASE ESCAPE",
        reason="Peak tension getaway with huge chat reaction",
        best_quote="NO WAY!",
        loudness_score=0.88,
        chat_score=0.91,
        final_score=0.90,
        transcript_chunk="NO WAY!",
        word_timestamps=[w1, w2],
    )

    clip_meta = ClipMetadata(
        hook_title="INSANE POLICE CHASE ESCAPE",
        caption="Wait for the jump!",
        hashtags=["#GTA6", "#gaming"],
        streamer="TestStreamer",
        game="GTA 6",
        source_url=url,
        clip_start=115.0,
        clip_end=165.0,
        highlight_score=9.4,
    )
    clip_result = ClipResult(
        video_path=Path("output/GTA_6/TestStreamer/2026-11-18/clip_01.mp4"),
        metadata_path=Path("output/GTA_6/TestStreamer/2026-11-18/clip_01.json"),
        metadata=clip_meta,
        duration_seconds=50.0,
    )

    return PipelineState(
        stream_info=StreamInfo(
            url=url,
            title=title,
            platform=Platform.YOUTUBE,
            streamer="TestStreamer",
            game="GTA 6",
            duration_seconds=14400,
            view_count=42000,
            video_id="gta6_test",
        ),
        status=status,
        audio_path=Path("temp/gta6_test.wav"),
        transcript=transcript,
        loudness_spikes=[
            TimeRange(
                start=120.0,
                end=132.0,
                score=3.8,
                signal_type=SignalType.LOUDNESS,
                metadata={"rms_db": -4.2},
            )
        ],
        chat_spikes=[
            TimeRange(
                start=122.0,
                end=138.0,
                score=4.5,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata={"peak_mps": 35},
            )
        ],
        highlights=[highlight],
        selected_highlights=[highlight],
        clips=[clip_result],
    )


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    name: str = "NVIDIA GeForce RTX 4070",
    total_memory: int = 12_000_000_000,
    major: int = 8,
    minor: int = 9,
) -> None:
    """Inject a mock torch module exposing total_memory (and NOT total_mem)."""
    fake_props = SimpleNamespace(
        name=name,
        total_memory=total_memory,
        major=major,
        minor=minor,
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_properties=lambda idx=0: fake_props,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


# ---------------------------------------------------------------------------
# StepCache Tests
# ---------------------------------------------------------------------------


class TestStepCache:
    def test_cache_key_and_path_are_deterministic(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        url_a = "https://youtube.com/watch?v=alpha"
        url_b = "https://youtube.com/watch?v=beta"

        key_a1 = cache.get_cache_key(url_a)
        key_a2 = cache.get_cache_key(url_a)
        key_b = cache.get_cache_key(url_b)

        assert key_a1 == key_a2
        assert len(key_a1) == 16
        assert key_a1 != key_b
        assert cache.get_cache_path(url_a) == cache.cache_dir / f"{key_a1}.json"

    def test_load_state_returns_none_when_missing(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        assert cache.load_state("https://youtube.com/watch?v=does_not_exist") is None

    def test_save_and_load_full_state_with_nested_transcript(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        state = _make_full_pipeline_state(status=PipelineStatus.ASSEMBLING)

        cache.save_state(state)

        key = cache.get_cache_key(state.stream_info.url)
        state_file = cache.get_cache_path(state.stream_info.url)
        transcript_file = cache.cache_dir / f"{key}_transcript.json"

        assert state_file.exists()
        assert transcript_file.exists()

        # Verify persisted JSON schema uses int(status)
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        assert raw["status"] == int(PipelineStatus.ASSEMBLING)
        assert isinstance(raw["status"], int)
        assert raw["has_transcript"] is True
        assert "updated_at" in raw

        # Load and verify full restoration
        loaded = cache.load_state(state.stream_info.url)
        assert loaded is not None
        assert loaded.stream_info == state.stream_info
        assert loaded.status is PipelineStatus.ASSEMBLING
        assert loaded.status < PipelineStatus.COMPLETE
        assert loaded.audio_path == Path("temp/gta6_test.wav")
        assert loaded.loudness_spikes == state.loudness_spikes
        assert loaded.chat_spikes == state.chat_spikes
        assert loaded.highlights == state.highlights
        assert loaded.selected_highlights == state.selected_highlights
        assert loaded.clips == state.clips

        assert loaded.transcript == state.transcript
        assert isinstance(loaded.transcript, TranscriptionResult)
        assert isinstance(loaded.transcript.segments[0], TranscriptSegment)
        assert isinstance(loaded.transcript.segments[0].words[0], WordTimestamp)
        assert loaded.transcript.segments[0].words[1].text == "WAY!"

    def test_save_and_load_state_without_transcript(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        state = _make_full_pipeline_state(
            status=PipelineStatus.DOWNLOADING,
            include_transcript=False,
        )
        cache.save_state(state)

        key = cache.get_cache_key(state.stream_info.url)
        assert not (cache.cache_dir / f"{key}_transcript.json").exists()

        loaded = cache.load_state(state.stream_info.url)
        assert loaded is not None
        assert loaded.status is PipelineStatus.DOWNLOADING
        assert loaded.transcript is None

    def test_list_all_ignores_transcript_and_corrupt_files(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        state_1 = _make_full_pipeline_state(
            url="https://youtube.com/watch?v=stream1",
            title="Stream One",
            status=PipelineStatus.ANALYZING,
        )
        state_2 = _make_full_pipeline_state(
            url="https://youtube.com/watch?v=stream2",
            title="Stream Two",
            status=PipelineStatus.COMPLETE,
        )
        cache.save_state(state_1)
        cache.save_state(state_2)

        # Drop malformed JSON files to confirm resilience
        (cache.cache_dir / "corrupt.json").write_text("{not-valid-json", encoding="utf-8")
        (cache.cache_dir / "missing_keys.json").write_text(
            json.dumps({"hello": "world"}), encoding="utf-8"
        )

        listed = cache.list_all()
        assert len(listed) == 2
        by_url = {s.stream_info.url: s for s in listed}
        assert by_url["https://youtube.com/watch?v=stream1"].status is PipelineStatus.ANALYZING
        assert by_url["https://youtube.com/watch?v=stream2"].status is PipelineStatus.COMPLETE
        assert len(by_url["https://youtube.com/watch?v=stream1"].highlights) == 1
        assert len(by_url["https://youtube.com/watch?v=stream2"].clips) == 1

    def test_clear_removes_state_and_transcript_cleanly(self, tmp_path: Path) -> None:
        cache = StepCache(_make_config(tmp_path))
        url_1 = "https://youtube.com/watch?v=remove_me"
        url_2 = "https://youtube.com/watch?v=keep_me"

        cache.save_state(_make_full_pipeline_state(url=url_1, title="Remove Me"))
        cache.save_state(_make_full_pipeline_state(url=url_2, title="Keep Me"))

        key_1 = cache.get_cache_key(url_1)
        assert cache.get_cache_path(url_1).exists()
        assert (cache.cache_dir / f"{key_1}_transcript.json").exists()

        cache.clear(url_1)
        assert not cache.get_cache_path(url_1).exists()
        assert not (cache.cache_dir / f"{key_1}_transcript.json").exists()
        assert cache.load_state(url_1) is None

        # Calling clear again on an already-cleared URL should not raise
        cache.clear(url_1)

        # Other cached URL must remain intact
        remaining = cache.list_all()
        assert len(remaining) == 1
        assert remaining[0].stream_info.url == url_2

    def test_find_by_audio_path_resolves_full_state_and_analysis_complete(
        self, tmp_path: Path
    ) -> None:
        cache = StepCache(_make_config(tmp_path))
        wav_path = tmp_path / "temp" / "gta6_test.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFF")

        state = _make_full_pipeline_state(status=PipelineStatus.SCORING)
        state.audio_path = wav_path
        state.analysis_complete = True
        cache.save_state(state)

        found = cache.find_by_audio_path(wav_path)
        assert found is not None
        assert found.stream_info.url == state.stream_info.url
        assert found.transcript is not None
        assert found.analysis_complete is True
        assert cache.find_by_audio_path(tmp_path / "missing.wav") is None


# ---------------------------------------------------------------------------
# Click CLI Tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_lists_all_commands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("clip", "auto", "setup", "status"):
            assert cmd in result.output
        assert "--config" in result.output
        assert "--verbose" in result.output
        assert "--version" in result.output

    def test_version_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_clip_command_accepts_all_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"general": {"cache_dir": str(tmp_path / ".cache")}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

        captured_calls: list[dict] = []

        class _StubPipeline:
            def __init__(self, config: Config) -> None:
                self.config = config

            async def process_stream(self, **kwargs) -> list[ClipResult]:
                captured_calls.append(kwargs)
                return []

        import streamclipper.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "Pipeline", _StubPipeline)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "--verbose",
                "clip",
                "https://youtube.com/watch?v=demo123",
                "--max-clips",
                "5",
                "--game",
                "GTA 6",
                "--streamer",
                "Kai",
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0
        assert "https://youtube.com/watch?v=demo123" in result.output
        assert len(captured_calls) == 1
        assert captured_calls[0]["url"] == "https://youtube.com/watch?v=demo123"
        assert captured_calls[0]["max_clips"] == 5
        assert captured_calls[0]["game"] == "GTA 6"
        assert captured_calls[0]["streamer"] == "Kai"
        assert captured_calls[0]["interactive"] is False

    def test_auto_command_uses_config_and_game_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"discovery": {"default_game": "GTA 6"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

        discovered_games: list[str | None] = []

        class _StubDiscovery:
            def __init__(self, config: Config) -> None:
                self.config = config

            def discover(self, game: str | None = None) -> list[StreamInfo]:
                discovered_games.append(game)
                return []

        import streamclipper.discovery.youtube as yt_mod

        monkeypatch.setattr(yt_mod, "YouTubeDiscovery", _StubDiscovery)

        runner = CliRunner()
        res_default = runner.invoke(cli, ["--config", str(cfg_file), "auto"])
        assert res_default.exit_code == 0
        assert "GTA 6" in res_default.output

        res_override = runner.invoke(
            cli,
            ["--config", str(cfg_file), "auto", "--game", "Elden Ring", "--max-streams", "3"],
        )
        assert res_override.exit_code == 0
        assert "Elden Ring" in res_override.output
        assert discovered_games == ["GTA 6", "Elden Ring"]

    def test_setup_renders_dependencies_gpu_and_hardware_profile_with_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce RTX 4070",
            total_memory=12_000_000_000,
            major=8,
            minor=9,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: True)
        monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "transcription": {"model_size": "medium"},
                    "video": {"codec": "libx264"},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg_file), "setup"])
        assert result.exit_code == 0

        # All 7 dependency checks appear
        for dep in (
            "yt-dlp",
            "FFmpeg",
            "WhisperX",
            "SoundFile",
            "OpenCV",
            "MoviePy v2",
            "Ollama SDK",
        ):
            assert dep in result.output

        # GPU line uses total_memory / 1e9 and major.minor
        assert "NVIDIA GeForce RTX 4070" in result.output
        assert "12.0GB VRAM" in result.output
        assert "CC 8.9" in result.output

        # HardwareProfile table shows tier, resolved settings, and provenance
        assert "Resolved Hardware Profile (cuda_high_vram)" in result.output
        assert "transcription.device" in result.output
        assert "transcription.model_size" in result.output
        assert "transcription.compute_type" in result.output
        assert "transcription.batch_size" in result.output
        assert "llm.model" in result.output
        assert "video.codec" in result.output
        assert "auto" in result.output
        assert "user override" in result.output

        # Verify overridden values vs auto-resolved values appear in the table rows
        lines = result.output.splitlines()
        model_size_line = next(ln for ln in lines if "transcription.model_size" in ln)
        assert "medium" in model_size_line
        assert "user override" in model_size_line

        codec_line = next(ln for ln in lines if "video.codec" in ln)
        assert "libx264" in codec_line
        assert "user override" in codec_line

        device_line = next(ln for ln in lines if "transcription.device" in ln)
        assert "cuda" in device_line
        assert "auto" in device_line

        llm_line = next(ln for ln in lines if "llm.model" in ln)
        assert "qwen3:8b" in llm_line
        assert "auto" in llm_line

    def test_setup_without_cuda_reports_cpu_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_torch(monkeypatch, cuda_available=False)
        monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump({}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg_file), "setup"])
        assert result.exit_code == 0
        assert "No CUDA GPU detected" in result.output
        assert "Resolved Hardware Profile (cpu_only)" in result.output

    def test_check_dependency_handles_missing_module_and_missing_cmd(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli_mod, "console", Console(width=120, force_terminal=False))
        cli_mod._check_dependency("MissingPkg", "definitely_nonexistent_pkg_xyz")
        cli_mod._check_dependency("MissingBin", None, cmd="definitely_nonexistent_bin_xyz --version")
        captured = capsys.readouterr().out
        assert "MissingPkg — not installed" in captured
        assert "MissingBin — not installed" in captured

    def test_status_empty_and_populated_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))
        cache_dir = tmp_path / ".cache"
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"general": {"cache_dir": str(cache_dir)}}),
            encoding="utf-8",
        )

        runner = CliRunner()
        empty_res = runner.invoke(cli, ["--config", str(cfg_file), "status"])
        assert empty_res.exit_code == 0
        assert "No cached pipelines found." in empty_res.output

        # Populate cache with two pipelines (one with title, one falling back to URL)
        cache = StepCache(Config(user_config_path=cfg_file))
        cache.save_state(
            _make_full_pipeline_state(
                url="https://youtube.com/watch?v=vid1",
                title="GTA 6 Crazy Heist",
                status=PipelineStatus.ANALYZING,
            )
        )
        cache.save_state(
            _make_full_pipeline_state(
                url="https://youtube.com/watch?v=vid2",
                title="",
                status=PipelineStatus.COMPLETE,
            )
        )

        populated_res = runner.invoke(cli, ["--config", str(cfg_file), "status"])
        assert populated_res.exit_code == 0
        assert "Pipeline Status" in populated_res.output
        assert "GTA 6 Crazy Heist" in populated_res.output
        assert "analyzing" in populated_res.output
        assert "https://youtube.com/watch?v=vid2" in populated_res.output
        assert "complete" in populated_res.output

    def test_missing_config_path_fails_cleanly_and_verbose_sets_debug_logging(
        self, tmp_path: Path
    ) -> None:
        import logging

        runner = CliRunner()
        bad_res = runner.invoke(
            cli, ["--config", str(tmp_path / "does_not_exist.yaml"), "status"]
        )
        assert bad_res.exit_code != 0

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"general": {"cache_dir": str(tmp_path / ".cache")}}),
            encoding="utf-8",
        )
        ok_res = runner.invoke(cli, ["--config", str(cfg_file), "--verbose", "status"])
        assert ok_res.exit_code == 0
        assert logging.getLogger("streamclipper").level == logging.DEBUG


# ---------------------------------------------------------------------------
# Review regression tests (A5, M5, L14)
# ---------------------------------------------------------------------------


def test_A5_cache_key_canonicalizes_youtube_urls_and_loads_legacy(tmp_path: Path) -> None:
    import hashlib

    cache = StepCache(_make_config(tmp_path))
    vid = "dQw4w9WgXcQ"
    spellings = [
        f"https://youtu.be/{vid}",
        f"https://www.youtube.com/watch?v={vid}",
        f"https://www.youtube.com/watch?v={vid}&t=42s",
        f"https://www.youtube.com/shorts/{vid}",
        f"https://www.youtube.com/live/{vid}",
    ]
    keys = {cache.get_cache_key(u) for u in spellings}
    assert keys == {hashlib.sha256(f"yt:{vid}".encode()).hexdigest()[:16]}
    assert cache.get_cache_key(" https://example.com/x ") == hashlib.sha256(
        b"https://example.com/x"
    ).hexdigest()[:16]

    # Legacy file keyed by sha256(raw url) still loads.
    url = spellings[1]
    state = _make_full_pipeline_state(url=url, include_transcript=False)
    cache.save_state(state)
    canonical = cache.get_cache_path(url)
    legacy = cache.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"
    canonical.rename(legacy)
    loaded = cache.load_state(url)
    assert loaded is not None
    assert loaded.stream_info.url == url
    cache.save_state(loaded)
    assert canonical.exists()


def test_M5_truncated_state_returns_none_and_quarantines(tmp_path: Path) -> None:
    cache = StepCache(_make_config(tmp_path))
    url = "https://youtube.com/watch?v=gta6_test"
    cache.save_state(_make_full_pipeline_state(url=url, include_transcript=False))
    path = cache.get_cache_path(url)
    path.write_text(path.read_text(encoding="utf-8")[:30], encoding="utf-8")

    assert cache.load_state(url) is None
    assert not path.exists()
    assert path.with_name(path.name + ".corrupt").exists()
    assert not list(cache.cache_dir.glob("*.tmp"))


def test_L14_auto_exits_nonzero_on_failure_and_clip_wraps_valueerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"general": {"cache_dir": str(tmp_path / ".cache")}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))
    _install_fake_torch(monkeypatch, cuda_available=False)

    class _StubDiscovery:
        def __init__(self, config: Config) -> None:
            pass

        def discover(self, game: str | None = None) -> list[StreamInfo]:
            return [
                StreamInfo(
                    url="https://youtube.com/watch?v=aaaaaaaaaaa",
                    title="A",
                    platform=Platform.YOUTUBE,
                    streamer="S",
                    game="GTA 6",
                    duration_seconds=100,
                    video_id="aaaaaaaaaaa",
                )
            ]

    class _StubPipeline:
        def __init__(self, config: Config) -> None:
            pass

        async def process_streams_parallel(self, urls, **kwargs):
            return [RuntimeError("boom") for _ in urls]

        async def process_stream(self, **kwargs):
            raise ValueError("bad mode")

    import streamclipper.discovery.youtube as yt_mod

    monkeypatch.setattr(yt_mod, "YouTubeDiscovery", _StubDiscovery)
    # Stub the module so the test never imports torch/pandas via the real pipeline.
    monkeypatch.setitem(
        sys.modules, "streamclipper.pipeline", SimpleNamespace(Pipeline=_StubPipeline)
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["--config", str(cfg_file), "auto"])
    assert res.exit_code == 1
    assert "Finished auto-run" in res.output

    res = runner.invoke(
        cli, ["--config", str(cfg_file), "clip", "https://youtube.com/watch?v=aaaaaaaaaaa"]
    )
    assert res.exit_code == 1
    assert "Error: bad mode" in res.output
    assert "Traceback" not in res.output


def test_M6_requested_max_clips_round_trips_through_cache(tmp_path: Path) -> None:
    """M6 integration: StepCache persists requested_max_clips so resume compares real values."""
    cache = StepCache(_make_config(tmp_path))
    url = "https://youtube.com/watch?v=gta6_test"
    state = _make_full_pipeline_state(url=url, include_transcript=False)
    state.requested_max_clips = 3
    cache.save_state(state)
    reloaded = cache.load_state(url)
    assert reloaded is not None
    assert reloaded.requested_max_clips == 3


def test_G3_scoring_checkpoint_files_are_not_listed_as_states(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Scorer checkpoints share cache_dir; list_all must skip them without warnings."""
    cache = StepCache(_make_config(tmp_path))
    cache.save_state(_make_full_pipeline_state(include_transcript=False))
    (cache.cache_dir / "scoring_ckpt_0123456789abcdef.json").write_text(
        json.dumps({"chunks": {}}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        states = cache.list_all()
    assert len(states) == 1
    assert "scoring_ckpt" not in caplog.text
