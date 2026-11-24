"""Unit tests for data models, config loading/validation, and HardwareProfile auto-tuning."""
import json
import subprocess
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import streamclipper.config as config_mod
from streamclipper.config import (
    Config,
    HardwareProfile,
    _detect_nvenc_support,
    detect_hardware_profile,
    resolve_hardware_config,
)
from streamclipper.models import (
    ClipMetadata,
    ClipResult,
    FacecamRegion,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    SerializableModel,
    SignalType,
    StreamInfo,
    TimeRange,
    TranscriptSegment,
    TranscriptionResult,
    WordTimestamp,
)


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    name: str = "",
    total_memory: int = 0,
    major: int = 0,
    minor: int = 0,
) -> None:
    """Inject a mock `torch` module into sys.modules for deterministic hardware tests."""
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
# HardwareProfile Detection & NVENC Tests
# ---------------------------------------------------------------------------


class TestHardwareDetection:
    @pytest.mark.parametrize(
        "has_nvenc,expected_codec",
        [(True, "h264_nvenc"), (False, "libx264")],
    )
    def test_tier_high_vram_rtx_4070_12gb(
        self, monkeypatch: pytest.MonkeyPatch, has_nvenc: bool, expected_codec: str
    ) -> None:
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce RTX 4070",
            total_memory=12_000_000_000,
            major=8,
            minor=9,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: has_nvenc)

        hw = detect_hardware_profile()
        assert hw.tier == "cuda_high_vram"
        assert hw.gpu_name == "NVIDIA GeForce RTX 4070"
        assert hw.vram_gb == 12.0
        assert hw.compute_capability == "8.9"
        assert hw.has_tensor_cores is True
        assert hw.has_nvenc is has_nvenc
        assert hw.device == "cuda"
        assert hw.model_size == "large-v2"
        assert hw.compute_type == "float16"
        assert hw.batch_size == 16
        assert hw.llm_model == "qwen3:8b"
        assert hw.video_codec == expected_codec

    @pytest.mark.parametrize(
        "has_nvenc,expected_codec",
        [(True, "h264_nvenc"), (False, "libx264")],
    )
    def test_tier_mid_vram_tensor_rtx_4050_6gb(
        self, monkeypatch: pytest.MonkeyPatch, has_nvenc: bool, expected_codec: str
    ) -> None:
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce RTX 4050 Laptop GPU",
            total_memory=6_442_450_944,  # ~6.44 GB
            major=8,
            minor=9,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: has_nvenc)

        hw = detect_hardware_profile()
        assert hw.tier == "cuda_mid_tensor"
        assert hw.compute_capability == "8.9"
        assert hw.has_tensor_cores is True
        assert hw.has_nvenc is has_nvenc
        assert hw.device == "cuda"
        assert hw.model_size == "large-v2"
        assert hw.compute_type == "int8_float16"
        assert hw.batch_size == 8
        assert hw.llm_model == "qwen3:8b"
        assert hw.video_codec == expected_codec

    def test_tier_mid_vram_rtx_turing_cc75_detected_as_tensor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RTX 2060 (CC 7.5, 6GB) still matches Tensor Core tier via 'RTX' in name."""
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce RTX 2060",
            total_memory=6_000_000_000,
            major=7,
            minor=5,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: True)

        hw = detect_hardware_profile()
        assert hw.tier == "cuda_mid_tensor"
        assert hw.has_tensor_cores is True
        assert hw.model_size == "large-v2"
        assert hw.compute_type == "int8_float16"
        assert hw.batch_size == 8
        assert hw.llm_model == "qwen3:8b"

    @pytest.mark.parametrize(
        "has_nvenc,expected_codec",
        [(True, "h264_nvenc"), (False, "libx264")],
    )
    def test_tier_mid_vram_no_tensor_gtx_1660_super_6gb(
        self, monkeypatch: pytest.MonkeyPatch, has_nvenc: bool, expected_codec: str
    ) -> None:
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce GTX 1660 SUPER",
            total_memory=6_442_450_944,
            major=7,
            minor=5,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: has_nvenc)

        hw = detect_hardware_profile()
        assert hw.tier == "cuda_mid_no_tensor"
        assert hw.compute_capability == "7.5"
        assert hw.has_tensor_cores is False
        assert hw.has_nvenc is has_nvenc
        assert hw.device == "cuda"
        assert hw.model_size == "medium"
        assert hw.compute_type == "int8"
        assert hw.batch_size == 4
        assert hw.llm_model == "qwen3:4b"
        assert hw.video_codec == expected_codec

    def test_tier_low_vram_4gb_fallback_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_torch(
            monkeypatch,
            cuda_available=True,
            name="NVIDIA GeForce GTX 1650",
            total_memory=4_294_967_296,  # ~4.29 GB (< 5.5 GB)
            major=7,
            minor=5,
        )
        monkeypatch.setattr(config_mod, "_detect_nvenc_support", lambda: True)

        hw = detect_hardware_profile()
        assert hw.tier == "cpu_fallback"
        assert hw.device == "cpu"
        assert hw.model_size == "base"
        assert hw.compute_type == "int8"
        assert hw.batch_size == 4
        assert hw.llm_model == "qwen3:4b"
        assert hw.video_codec == "libx264"

    def test_tier_cpu_only_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_torch(monkeypatch, cuda_available=False)

        hw = detect_hardware_profile()
        assert hw.tier == "cpu_only"
        assert hw.gpu_name == "CPU"
        assert hw.vram_gb == 0.0
        assert hw.compute_capability == "N/A"
        assert hw.has_tensor_cores is False
        assert hw.has_nvenc is False
        assert hw.device == "cpu"
        assert hw.model_size == "base"
        assert hw.compute_type == "int8"
        assert hw.batch_size == 4
        assert hw.llm_model == "qwen3:4b"
        assert hw.video_codec == "libx264"

    def test_detect_nvenc_support_present_absent_and_missing_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=" V..... h264_nvenc NVIDIA NVENC H.264 encoder\n"
            ),
        )
        assert _detect_nvenc_support() is True

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=" V..... libx264 libx264 H.264\n"
            ),
        )
        assert _detect_nvenc_support() is False

        def _raise_not_found(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not found")

        monkeypatch.setattr(subprocess, "run", _raise_not_found)
        assert _detect_nvenc_support() is False


# ---------------------------------------------------------------------------
# Config Loading, Merging, Override Precedence & Validation Tests
# ---------------------------------------------------------------------------


class TestConfigSystem:
    def _gtx_1660_profile(self, has_nvenc: bool = True) -> HardwareProfile:
        return HardwareProfile(
            tier="cuda_mid_no_tensor",
            gpu_name="NVIDIA GeForce GTX 1660 SUPER",
            vram_gb=6.44,
            compute_capability="7.5",
            has_tensor_cores=False,
            has_nvenc=has_nvenc,
            device="cuda",
            model_size="medium",
            compute_type="int8",
            batch_size=4,
            llm_model="qwen3:4b",
            video_codec="h264_nvenc" if has_nvenc else "libx264",
        )

    def test_defaults_and_dot_notation_lookup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        profile = self._gtx_1660_profile(has_nvenc=True)

        cfg = Config(hardware_profile=profile)

        assert cfg.get("analysis.weights.loudness") == 0.20
        assert cfg.get("analysis.weights.chat_velocity") == 0.20
        assert cfg.get("analysis.weights.heatmap") == 0.20
        assert cfg.get("analysis.weights.llm_score") == 0.40
        assert cfg.get("video.resolution") == [1080, 1920]
        assert cfg.get("captions.highlight_color") == "#FFE600"
        assert cfg.get("nonexistent.nested.key", "fallback") == "fallback"
        assert cfg.hardware_profile == profile

        # All 6 auto keys resolved from profile
        assert cfg.get("transcription.device") == "cuda"
        assert cfg.get("transcription.model_size") == "medium"
        assert cfg.get("transcription.compute_type") == "int8"
        assert cfg.get("transcription.batch_size") == 4
        assert cfg.get("llm.model") == "qwen3:4b"
        assert cfg.get("video.codec") == "h264_nvenc"
        for key in (
            "transcription.device",
            "transcription.model_size",
            "transcription.compute_type",
            "transcription.batch_size",
            "llm.model",
            "video.codec",
        ):
            assert cfg.is_auto_resolved(key) is True

    def test_user_overrides_preserved_over_hardware_auto_config(
        self, tmp_path: Path
    ) -> None:
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text(
            yaml.safe_dump(
                {
                    "discovery": {"default_game": "Valorant"},
                    "transcription": {
                        "model_size": "large-v2",
                        "batch_size": 12,
                    },
                    "llm": {"model": "qwen3:14b"},
                    "video": {"codec": "libx264"},
                }
            ),
            encoding="utf-8",
        )

        # Even on GTX 1660 SUPER, explicit overrides must NOT be overwritten
        cfg = Config(
            user_config_path=user_cfg,
            hardware_profile=self._gtx_1660_profile(has_nvenc=True),
        )

        assert cfg.get("discovery.default_game") == "Valorant"
        # Deep merge preserved untouched sibling keys
        assert cfg.get("discovery.platform") == "youtube"

        # Overridden hardware keys kept verbatim
        assert cfg.get("transcription.model_size") == "large-v2"
        assert cfg.get("transcription.batch_size") == 12
        assert cfg.get("llm.model") == "qwen3:14b"
        assert cfg.get("video.codec") == "libx264"
        assert cfg.is_auto_resolved("transcription.model_size") is False
        assert cfg.is_auto_resolved("transcription.batch_size") is False
        assert cfg.is_auto_resolved("llm.model") is False
        assert cfg.is_auto_resolved("video.codec") is False

        # Untouched 'auto' keys still resolved from hardware profile
        assert cfg.get("transcription.device") == "cuda"
        assert cfg.get("transcription.compute_type") == "int8"
        assert cfg.is_auto_resolved("transcription.device") is True
        assert cfg.is_auto_resolved("transcription.compute_type") is True

    def test_resolve_hardware_config_cpu_float16_safety_guard(self) -> None:
        cpu_profile = HardwareProfile(
            tier="cpu_only",
            gpu_name="CPU",
            vram_gb=0.0,
            compute_capability="N/A",
            has_tensor_cores=False,
            has_nvenc=False,
            device="cpu",
            model_size="base",
            compute_type="int8",
            batch_size=4,
            llm_model="qwen3:4b",
            video_codec="libx264",
        )
        data = {
            "transcription": {
                "device": "cpu",
                "model_size": "small",
                "compute_type": "float16",
                "batch_size": 4,
            },
            "llm": {"model": "auto"},
            "video": {"codec": "auto"},
        }
        resolved, hw = resolve_hardware_config(data, profile=cpu_profile)
        assert hw == cpu_profile
        assert resolved["transcription"]["compute_type"] == "int8"
        assert resolved["transcription"]["model_size"] == "small"
        assert resolved["llm"]["model"] == "qwen3:4b"
        assert resolved["video"]["codec"] == "libx264"

    def test_validation_rejects_invalid_weight_sum(self, tmp_path: Path) -> None:
        bad_cfg = tmp_path / "bad_weights.yaml"
        bad_cfg.write_text(
            yaml.safe_dump(
                {
                    "analysis": {
                        "weights": {
                            "loudness": 0.5,
                            "chat_velocity": 0.5,
                            "llm_score": 0.5,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Signal weights must sum to 1.0"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    def test_validation_rejects_invalid_resolution(self, tmp_path: Path) -> None:
        bad_cfg = tmp_path / "bad_res.yaml"
        bad_cfg.write_text(
            yaml.safe_dump({"video": {"resolution": [1080]}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="video.resolution"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    @pytest.mark.parametrize("bad_batch_size", [0, -2, "fast", True])
    def test_validation_rejects_invalid_batch_size(
        self, tmp_path: Path, bad_batch_size: object
    ) -> None:
        bad_cfg = tmp_path / "bad_batch.yaml"
        bad_cfg.write_text(
            yaml.safe_dump({"transcription": {"batch_size": bad_batch_size}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="transcription.batch_size"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    def test_packaged_default_config_matches_repo_root(self) -> None:
        repo_cfg = Path(__file__).parent.parent / "config.default.yaml"
        pkg_cfg = Path(__file__).parent.parent / "src" / "streamclipper" / "config.default.yaml"
        assert pkg_cfg.is_file()
        assert pkg_cfg.read_text(encoding="utf-8") == repo_cfg.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "bad_weights",
        [
            None,
            [],
            {"loudness": -0.5, "chat_velocity": 0.5, "llm_score": 1.0},
            {"loudness": 0.25, "chat_velocity": 0.25, "llm_score": 0.25, "bogus": 0.25},
            {"loudness": True, "chat_velocity": 0.0, "llm_score": 0.0},
        ],
    )
    def test_validation_rejects_malformed_or_negative_weights(
        self, tmp_path: Path, bad_weights: object
    ) -> None:
        bad_cfg = tmp_path / "bad_weights_edge.yaml"
        bad_cfg.write_text(
            yaml.safe_dump({"analysis": {"weights": bad_weights}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    @pytest.mark.parametrize("bad_res", [[0, 1920], [1080, -1], [True, 1920], ["1080", "1920"]])
    def test_validation_rejects_non_positive_or_non_int_resolution(
        self, tmp_path: Path, bad_res: object
    ) -> None:
        bad_cfg = tmp_path / "bad_res_edge.yaml"
        bad_cfg.write_text(
            yaml.safe_dump({"video": {"resolution": bad_res}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="video.resolution"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    @pytest.mark.parametrize("valid_mode", ["auto", "narrative", "story", "raw"])
    def test_validation_accepts_all_four_transformation_modes(
        self, tmp_path: Path, valid_mode: str
    ) -> None:
        cfg_file = tmp_path / f"mode_{valid_mode}.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"transformation": {"mode": valid_mode}}),
            encoding="utf-8",
        )
        cfg = Config(user_config_path=cfg_file, hardware_profile=self._gtx_1660_profile())
        assert cfg.get("transformation.mode") == valid_mode

    def test_validation_rejects_invalid_transformation_mode(self, tmp_path: Path) -> None:
        bad_cfg = tmp_path / "bad_mode.yaml"
        bad_cfg.write_text(
            yaml.safe_dump({"transformation": {"mode": "turbo"}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="transformation.mode"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())

    def test_validation_rejects_non_mapping_user_yaml(self, tmp_path: Path) -> None:
        bad_cfg = tmp_path / "list_root.yaml"
        bad_cfg.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            Config(user_config_path=bad_cfg, hardware_profile=self._gtx_1660_profile())


# ---------------------------------------------------------------------------
# Data Models, PipelineStatus IntEnum & Full JSON Round-Trip Tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_wildcard_import_exports_all_models(self) -> None:
        namespace: dict = {}
        exec("from streamclipper.models import *", namespace)
        for name in (
            "Platform",
            "SignalType",
            "PipelineStatus",
            "SerializableModel",
            "StreamInfo",
            "TimeRange",
            "WordTimestamp",
            "TranscriptSegment",
            "TranscriptionResult",
            "Highlight",
            "FacecamRegion",
            "ClipMetadata",
            "ClipResult",
            "PipelineState",
        ):
            assert name in namespace

    def test_pipeline_status_is_int_enum_with_numeric_ordering(self) -> None:
        assert issubclass(PipelineStatus, IntEnum)
        assert PipelineStatus.PENDING == 1
        assert PipelineStatus.COMPLETE == 7
        assert PipelineStatus.FAILED == 99

        # Key regression check: alphabetically "PENDING" > "ANALYZING",
        # so IntEnum is essential for numeric stage comparison.
        assert PipelineStatus.PENDING < PipelineStatus.ANALYZING
        assert (
            PipelineStatus.PENDING
            < PipelineStatus.DOWNLOADING
            < PipelineStatus.TRANSCRIBING
            < PipelineStatus.ANALYZING
            < PipelineStatus.SCORING
            < PipelineStatus.ASSEMBLING
            < PipelineStatus.COMPLETE
            < PipelineStatus.FAILED
        )

    def test_all_models_json_round_trip(self) -> None:
        stream_info = StreamInfo(
            url="https://youtube.com/watch?v=abc123",
            title="GTA 6 Launch Day Stream!",
            platform=Platform.YOUTUBE,
            streamer="TestStreamer",
            game="GTA 6",
            duration_seconds=14400,
            view_count=50000,
            video_id="abc123",
        )
        assert StreamInfo.from_dict(json.loads(stream_info.to_json())) == stream_info

        time_range = TimeRange(
            start=120.5,
            end=135.25,
            score=4.2,
            signal_type=SignalType.CHAT_VELOCITY,
            metadata={"peak_mps": 42, "tags": ["hype", "clutch"]},
        )
        assert TimeRange.from_dict(json.loads(time_range.to_json())) == time_range

        w1 = WordTimestamp(text="NO", start=121.0, end=121.3, confidence=0.98)
        w2 = WordTimestamp(text="WAY!", start=121.35, end=121.8, confidence=0.95)
        assert WordTimestamp.from_dict(json.loads(w1.to_json())) == w1

        seg = TranscriptSegment(text="NO WAY!", start=121.0, end=121.8, words=[w1, w2])
        assert TranscriptSegment.from_dict(json.loads(seg.to_json())) == seg

        transcript = TranscriptionResult(
            segments=[seg],
            full_text="NO WAY!",
            language="en",
        )
        transcript_roundtrip = TranscriptionResult.from_dict(
            json.loads(transcript.to_json())
        )
        assert transcript_roundtrip == transcript
        assert isinstance(transcript_roundtrip.segments[0], TranscriptSegment)
        assert isinstance(transcript_roundtrip.segments[0].words[0], WordTimestamp)

        highlight = Highlight(
            start=115.0,
            end=165.0,
            score=9.2,
            title="INSANE POLICE CHASE ESCAPE",
            reason="High tension escape with huge chat reaction",
            best_quote="NO WAY!",
            loudness_score=0.85,
            chat_score=0.92,
            final_score=0.89,
            transcript_chunk="NO WAY!",
            word_timestamps=[w1, w2],
        )
        assert Highlight.from_dict(json.loads(highlight.to_json())) == highlight

        facecam = FacecamRegion(x=1500, y=750, w=380, h=280, confidence=0.91)
        assert FacecamRegion.from_dict(json.loads(facecam.to_json())) == facecam

        clip_meta = ClipMetadata(
            hook_title="INSANE POLICE CHASE ESCAPE",
            caption="Wait until the ending!",
            hashtags=["#GTA6", "#gaming", "#clips"],
            streamer="TestStreamer",
            game="GTA 6",
            source_url="https://youtube.com/watch?v=abc123",
            clip_start=115.0,
            clip_end=165.0,
            highlight_score=9.2,
        )
        assert ClipMetadata.from_dict(json.loads(clip_meta.to_json())) == clip_meta

        clip_result = ClipResult(
            video_path=Path("output/GTA_6/TestStreamer/2026-11-18/clip_01.mp4"),
            metadata_path=Path("output/GTA_6/TestStreamer/2026-11-18/clip_01.json"),
            metadata=clip_meta,
            duration_seconds=50.0,
        )
        clip_roundtrip = ClipResult.from_dict(json.loads(clip_result.to_json()))
        assert clip_roundtrip == clip_result
        assert isinstance(clip_roundtrip.video_path, Path)
        assert isinstance(clip_roundtrip.metadata_path, Path)

        state = PipelineState(
            stream_info=stream_info,
            status=PipelineStatus.ASSEMBLING,
            audio_path=Path("temp/abc123.wav"),
            transcript=transcript,
            loudness_spikes=[
                TimeRange(start=120.0, end=130.0, score=3.1, signal_type=SignalType.LOUDNESS)
            ],
            chat_spikes=[time_range],
            highlights=[highlight],
            selected_highlights=[highlight],
            clips=[clip_result],
            cache_dir=Path(".cache"),
        )
        state_json = state.to_json()
        state_dict = json.loads(state_json)
        assert state_dict["status"] == 6
        assert state_dict["stream_info"]["platform"] == "youtube"
        assert isinstance(state_dict["audio_path"], str)

        state_roundtrip = PipelineState.from_dict(state_dict)
        assert state_roundtrip == state
        assert state_roundtrip.status is PipelineStatus.ASSEMBLING
        assert isinstance(state_roundtrip.audio_path, Path)
        assert isinstance(state_roundtrip.cache_dir, Path)

    def test_from_dict_tolerates_explicit_json_null_fields(self) -> None:
        """Cached JSON with explicit null list/dict/string fields must not raise TypeError."""
        tr = TimeRange.from_dict({"start": 1.0, "end": 5.0, "score": None, "metadata": None})
        assert tr.score == 0.0
        assert tr.metadata == {}

        seg = TranscriptSegment.from_dict({"text": "hi", "start": 0.0, "end": 1.0, "words": None})
        assert seg.words == []

        res = TranscriptionResult.from_dict({"segments": None, "full_text": None, "language": None})
        assert res.segments == []
        assert res.full_text == ""
        assert res.language == "en"

        hl = Highlight.from_dict(
            {
                "start": 0.0,
                "end": 40.0,
                "score": 8.0,
                "title": None,
                "reason": None,
                "word_timestamps": None,
                "chat_burst": None,
                "sub_clips": None,
                "peak_time": None,
            }
        )
        assert hl.word_timestamps == []
        assert hl.chat_burst == []
        assert hl.sub_clips == []
        assert hl.peak_time == 0.0

        meta = ClipMetadata.from_dict(
            {
                "hook_title": "T",
                "caption": "C",
                "hashtags": None,
                "streamer": "S",
                "game": "G",
                "source_url": "U",
                "clip_start": 0.0,
                "clip_end": 30.0,
                "highlight_score": 8.0,
            }
        )
        assert meta.hashtags == []

        cr = ClipResult.from_dict(
            {
                "video_path": "clip.mp4",
                "metadata_path": None,
                "metadata": meta.to_dict(),
                "duration_seconds": 30.0,
            }
        )
        assert cr.metadata_path is None

        ps = PipelineState.from_dict(
            {
                "stream_info": {
                    "url": "https://youtu.be/x",
                    "title": None,
                    "platform": "youtube",
                    "streamer": None,
                    "game": None,
                    "duration_seconds": None,
                    "view_count": None,
                    "video_id": None,
                },
                "status": None,
                "loudness_spikes": None,
                "chat_spikes": None,
                "highlights": None,
                "selected_highlights": None,
                "clips": None,
            }
        )
        assert ps.status is PipelineStatus.PENDING
        assert ps.loudness_spikes == []
        assert ps.clips == []


def test_L15_detect_hardware_profile_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cuda_available=False)
    first = detect_hardware_profile()
    # Changing the environment has no effect until the cache is cleared.
    _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        name="NVIDIA GeForce RTX 4070",
        total_memory=12_000_000_000,
        major=8,
        minor=9,
    )
    assert detect_hardware_profile() is first
    detect_hardware_profile.cache_clear()
    assert detect_hardware_profile().tier != "cpu_only"


def test_L16_hardware_detection_failure_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(idx: int = 0):
        raise RuntimeError("cuda exploded")

    fake_cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=_boom)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with caplog.at_level("WARNING", logger="streamclipper.config"):
        hw = detect_hardware_profile()
    assert hw.tier == "cpu_only"
    assert "cuda exploded" in caplog.text


def test_G12_default_config_copies_are_identical() -> None:
    repo_root = Path(config_mod.__file__).resolve().parents[2]
    root_copy = repo_root / "config.default.yaml"
    pkg_copy = Path(config_mod.__file__).resolve().parent / "config.default.yaml"

    def _norm(p: Path) -> str:
        return p.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    assert _norm(root_copy) == _norm(pkg_copy)


def test_F1_default_config_has_v3_keys_and_hardware_stays_auto() -> None:
    cfg = Config()
    assert cfg.get("video.game_fit") == "full_bleed"
    assert cfg.get("facecam.min_detection_ratio") == 0.6
    assert cfg.get("facecam.max_center_jitter") == 0.05
    assert cfg.get("facecam.overrides") == {}
    assert cfg.get("analysis.max_clip_seconds") == 45
    assert cfg.get("analysis.skip_intro_minutes") == 5
    assert cfg.get("content_gate.enabled") is True
    assert cfg.get("output.max_final_seconds") == 55
    assert cfg.get("output.queued_dir") == "queued"
    assert cfg.get("transformation.tts_engines") == ["chatterbox", "kokoro", "edge-tts"]
    assert cfg.get("transformation.target_lufs") == -14
    assert cfg.get("transformation.format_rotation") == ["narrative", "story", "breakdown"]
    assert cfg.get("transformation.bgm_path") == ""
    assert cfg.get("transformation.bgm_volume_db") == -26.0
    assert cfg.get("longform.card_seconds") == 1.2
    assert cfg.get("longform.transition_sfx_path") == ""
    assert cfg.get("discovery.blocked_streamers") == []
    raw = yaml.safe_load(
        (Path(config_mod.__file__).resolve().parent / "config.default.yaml").read_text("utf-8")
    )
    assert raw["video"]["codec"] == "auto"
    assert raw["transcription"]["model_size"] == "auto"
    assert raw["llm"]["model"] == "auto"


def test_F2_game_fit_and_top3_mode_validation(tmp_path: Path) -> None:
    for bad in ("stretch", "", 3):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"video": {"game_fit": bad}}), encoding="utf-8")
        with pytest.raises(ValueError, match="game_fit"):
            Config(user_config_path=p)
    ok = tmp_path / "ok.yaml"
    ok.write_text(
        yaml.safe_dump({"video": {"game_fit": "balanced"}, "transformation": {"mode": "top3"}}),
        encoding="utf-8",
    )
    cfg = Config(user_config_path=ok)
    assert cfg.get("video.game_fit") == "balanced"
    assert cfg.get("transformation.mode") == "top3"

    ok_fb = tmp_path / "ok_fb.yaml"
    ok_fb.write_text(yaml.safe_dump({"video": {"game_fit": "full_bleed"}}), encoding="utf-8")
    cfg_fb = Config(user_config_path=ok_fb)
    assert cfg_fb.get("video.game_fit") == "full_bleed"


def test_repo_user_config_fallback_when_running_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When working directory is outside repo and has no config.yaml, fall back to repo-root config.yaml."""
    outside_dir = tmp_path / "outside_cwd"
    outside_dir.mkdir()
    repo_cfg = tmp_path / "repo_config.yaml"
    repo_cfg.write_text(
        "transformation:\n  elevenlabs_voice_id: 'fallback_voice_id'\n", encoding="utf-8"
    )
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", Path("config.yaml"))
    monkeypatch.setattr(config_mod, "_REPO_USER_CONFIG_PATH", repo_cfg)
    monkeypatch.chdir(outside_dir)

    cfg = Config()
    assert cfg.get("transformation.elevenlabs_voice_id") == "fallback_voice_id"


def test_T6_channels_block_defaults_and_topics_ledger_name() -> None:
    cfg = Config()
    assert cfg.get("channels.en.output_dir") == "./output/en"
    assert cfg.get("channels.en.narration_language") == "english"
    assert cfg.get("channels.en.ledger_name") == "topics_ledger_en.json"
    assert cfg.get("channels.en.discovery_language") == "en-US"
    assert cfg.get("channels.en.discovery_region") == "US"

    assert cfg.get("channels.hi.output_dir") == "./output/hi"
    assert cfg.get("channels.hi.narration_language") == "hinglish"
    assert cfg.get("channels.hi.ledger_name") == "topics_ledger_hi.json"
    assert cfg.get("channels.hi.discovery_language") == "hi"
    assert cfg.get("channels.hi.discovery_region") == "IN"

    assert cfg.get("topics.ledger_name") == "topics_ledger.json"


def test_T6_apply_channel_profile_isolation_and_shared_resources() -> None:
    cfg_base = Config()
    default_out = cfg_base.get("general.output_dir")
    default_lang = cfg_base.get("transformation.narration_language")
    default_ledger = cfg_base.get("topics.ledger_name")
    shared_cache = cfg_base.get("general.cache_dir")
    shared_db = cfg_base.get("library.db_path")

    # None preserves exact existing defaults
    cfg_none = Config()
    res = cfg_none.apply_channel_profile(None)
    assert res is cfg_none
    assert cfg_none.get("general.output_dir") == default_out
    assert cfg_none.get("transformation.narration_language") == default_lang
    assert cfg_none.get("topics.ledger_name") == default_ledger

    # English channel profile
    cfg_en = Config()
    cfg_en.apply_channel_profile("en")
    assert cfg_en.get("general.output_dir") == "./output/en"
    assert cfg_en.get("transformation.narration_language") == "english"
    assert cfg_en.get("topics.ledger_name") == "topics_ledger_en.json"
    assert cfg_en.get("discovery.language") == "en-US"
    assert cfg_en.get("discovery.region") == "US"
    assert cfg_en.get("general.cache_dir") == shared_cache
    assert cfg_en.get("library.db_path") == shared_db

    # Hinglish channel profile
    cfg_hi = Config()
    cfg_hi.apply_channel_profile("hi")
    assert cfg_hi.get("general.output_dir") == "./output/hi"
    assert cfg_hi.get("transformation.narration_language") == "hinglish"
    assert cfg_hi.get("topics.ledger_name") == "topics_ledger_hi.json"
    assert cfg_hi.get("discovery.language") == "hi"
    assert cfg_hi.get("discovery.region") == "IN"
    assert cfg_hi.get("general.cache_dir") == shared_cache
    assert cfg_hi.get("library.db_path") == shared_db

    # Case insensitive
    cfg_upper = Config()
    cfg_upper.apply_channel_profile("EN")
    assert cfg_upper.get("general.output_dir") == "./output/en"

    # Invalid channel raises ValueError
    with pytest.raises(ValueError, match="Unknown channel profile"):
        Config().apply_channel_profile("nonexistent_channel")


def test_T6_topic_ledger_channel_isolation(tmp_path: Path) -> None:
    from streamclipper.topics.ledger import TopicLedger

    cache = tmp_path / "shared_cache"
    cache.mkdir()

    cfg_en = Config()
    cfg_en._data["general"]["cache_dir"] = str(cache)
    cfg_en.apply_channel_profile("en")

    cfg_hi = Config()
    cfg_hi._data["general"]["cache_dir"] = str(cache)
    cfg_hi.apply_channel_profile("hi")

    ledger_en = TopicLedger(cfg_en)
    ledger_hi = TopicLedger(cfg_hi)

    assert ledger_en.path == cache / "topics_ledger_en.json"
    assert ledger_hi.path == cache / "topics_ledger_hi.json"

    # Record in English ledger
    ledger_en.record(
        topic_id="en-topic-1",
        title="Top 5 English Moments",
        game="GTA 6",
        shape="numbered",
    )

    # Record in Hinglish ledger
    ledger_hi.record(
        topic_id="hi-topic-1",
        title="Top 5 Hinglish Moments",
        game="GTA 6",
        shape="countdown",
    )

    # Verify separate files exist in the same shared cache dir
    assert (cache / "topics_ledger_en.json").exists()
    assert (cache / "topics_ledger_hi.json").exists()

    en_entries = ledger_en._load()
    hi_entries = ledger_hi._load()

    assert len(en_entries) == 1
    assert en_entries[0]["topic_id"] == "en-topic-1"
    assert len(hi_entries) == 1
    assert hi_entries[0]["topic_id"] == "hi-topic-1"

    # Default ledger path
    cfg_default = Config()
    cfg_default._data["general"]["cache_dir"] = str(cache)
    ledger_default = TopicLedger(cfg_default)
    assert ledger_default.path == cache / "topics_ledger.json"


def test_T6_cli_channel_option_applied() -> None:
    from unittest.mock import patch, MagicMock
    from click.testing import CliRunner
    from streamclipper.cli import cli

    runner = CliRunner()

    # 1. clip with --channel hi
    with patch("streamclipper.pipeline.Pipeline") as mock_pipeline:
        mock_instance = MagicMock()
        mock_instance.process_stream = MagicMock()
        import asyncio
        async def mock_process(*args: object, **kwargs: object) -> list:
            return []
        mock_instance.process_stream.side_effect = mock_process
        mock_pipeline.return_value = mock_instance

        res = runner.invoke(cli, ["clip", "https://youtube.com/watch?v=123", "--channel", "hi", "--no-interactive"])
        assert res.exit_code == 0, f"res.output: {res.output}"
        cfg_passed = mock_pipeline.call_args[0][0]
        assert cfg_passed.get("general.output_dir") == "./output/hi"
        assert cfg_passed.get("transformation.narration_language") == "hinglish"
        assert cfg_passed.get("topics.ledger_name") == "topics_ledger_hi.json"

    # 2. auto with --channel en
    with patch("streamclipper.discovery.youtube.YouTubeDiscovery") as mock_disc:
        mock_disc_inst = MagicMock()
        mock_disc_inst.discover.return_value = []
        mock_disc.return_value = mock_disc_inst

        res = runner.invoke(cli, ["auto", "--channel", "en"])
        assert res.exit_code == 0, f"res.output: {res.output}"
        cfg_passed = mock_disc.call_args[0][0]
        assert cfg_passed.get("general.output_dir") == "./output/en"
        assert cfg_passed.get("transformation.narration_language") == "english"
        assert cfg_passed.get("topics.ledger_name") == "topics_ledger_en.json"

    # 3. topics plan with --channel hi
    with patch("streamclipper.topics.planner.plan_topic_local") as mock_plan:
        mock_plan_inst = MagicMock()
        mock_plan_inst.problems = []
        mock_plan_inst.plan_id = "test-plan"
        mock_plan.return_value = mock_plan_inst
        with patch("streamclipper.topics.planner.format_plan_outline", return_value="Outline"):
            res = runner.invoke(cli, ["topics", "plan", "GTA 6", "Secret Vault", "--channel", "hi"])
            assert res.exit_code == 0, f"res.output: {res.output}"
            cfg_passed = mock_plan.call_args[0][0]
            assert cfg_passed.get("general.output_dir") == "./output/hi"
            assert cfg_passed.get("transformation.narration_language") == "hinglish"
            assert cfg_passed.get("topics.ledger_name") == "topics_ledger_hi.json"

    # 4. topics render with --channel en
    with patch("streamclipper.topics.planner.load_plan") as mock_load, \
         patch("streamclipper.assembly.longform.render_long_video", return_value={}) as mock_render:
        mock_p = MagicMock()
        mock_p.status = "approved"
        mock_load.return_value = mock_p
        res = runner.invoke(cli, ["topics", "render", "test-plan", "--channel", "en"])
        assert res.exit_code == 0, f"res.output: {res.output}"
        cfg_passed = mock_load.call_args[0][0]
        assert cfg_passed.get("general.output_dir") == "./output/en"
        assert cfg_passed.get("transformation.narration_language") == "english"

    # 5. topics shorts with --channel hi
    with patch("streamclipper.topics.planner.load_plan") as mock_load, \
         patch("streamclipper.assembly.topic_shorts.render_topic_shorts", return_value={}) as mock_shorts:
        mock_p = MagicMock()
        mock_p.status = "approved"
        mock_load.return_value = mock_p
        res = runner.invoke(cli, ["topics", "shorts", "test-plan", "--channel", "hi"])
        assert res.exit_code == 0, f"res.output: {res.output}"
        cfg_passed = mock_load.call_args[0][0]
        assert cfg_passed.get("general.output_dir") == "./output/hi"
        assert cfg_passed.get("transformation.narration_language") == "hinglish"


