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

        assert cfg.get("analysis.weights.loudness") == 0.25
        assert cfg.get("analysis.weights.chat_velocity") == 0.25
        assert cfg.get("analysis.weights.llm_score") == 0.50
        assert cfg.get("video.resolution") == [1080, 1920]
        assert cfg.get("captions.highlight_color") == "#FFD700"
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
