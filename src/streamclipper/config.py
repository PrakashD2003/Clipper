"""Configuration loading with defaults, validation, merging, and dynamic hardware auto-tuning."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import subprocess
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.default.yaml"
USER_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class HardwareProfile:
    """Resolved hardware capabilities and recommended runtime parameters."""
    tier: str
    gpu_name: str
    vram_gb: float
    compute_capability: str
    has_tensor_cores: bool
    has_nvenc: bool
    device: str
    model_size: str
    compute_type: str
    batch_size: int
    llm_model: str
    video_codec: str


def _detect_nvenc_support() -> bool:
    """Return True if FFmpeg is installed and exposes the h264_nvenc encoder."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0 and "h264_nvenc" in proc.stdout
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def detect_hardware_profile(check_nvenc: bool = True) -> HardwareProfile:
    """Inspect CUDA GPU properties and FFmpeg encoders to build a HardwareProfile."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            vram_gb = float(props.total_memory) / 1e9
            major = int(props.major)
            minor = int(props.minor)
            cc = f"{major}.{minor}"
            has_tensor_cores = (major >= 8) or ("RTX" in gpu_name.upper())
            has_nvenc = _detect_nvenc_support() if check_nvenc else False
            video_codec = "h264_nvenc" if has_nvenc else "libx264"

            if vram_gb >= 10.0:
                return HardwareProfile(
                    tier="cuda_high_vram",
                    gpu_name=gpu_name,
                    vram_gb=round(vram_gb, 2),
                    compute_capability=cc,
                    has_tensor_cores=has_tensor_cores,
                    has_nvenc=has_nvenc,
                    device="cuda",
                    model_size="large-v2",
                    compute_type="float16",
                    batch_size=16,
                    llm_model="qwen3:8b",
                    video_codec=video_codec,
                )

            if vram_gb >= 5.5 and has_tensor_cores:
                return HardwareProfile(
                    tier="cuda_mid_tensor",
                    gpu_name=gpu_name,
                    vram_gb=round(vram_gb, 2),
                    compute_capability=cc,
                    has_tensor_cores=True,
                    has_nvenc=has_nvenc,
                    device="cuda",
                    model_size="large-v2",
                    compute_type="int8_float16",
                    batch_size=8,
                    llm_model="qwen3:8b",
                    video_codec=video_codec,
                )

            if vram_gb >= 5.5:
                return HardwareProfile(
                    tier="cuda_mid_no_tensor",
                    gpu_name=gpu_name,
                    vram_gb=round(vram_gb, 2),
                    compute_capability=cc,
                    has_tensor_cores=False,
                    has_nvenc=has_nvenc,
                    device="cuda",
                    model_size="medium",
                    compute_type="int8",
                    batch_size=4,
                    llm_model="qwen3:4b",
                    video_codec=video_codec,
                )

            # CUDA GPU present but < 5.5 GB VRAM — fall back safely to CPU for ASR/LLM
            return HardwareProfile(
                tier="cpu_fallback",
                gpu_name=gpu_name,
                vram_gb=round(vram_gb, 2),
                compute_capability=cc,
                has_tensor_cores=has_tensor_cores,
                has_nvenc=has_nvenc,
                device="cpu",
                model_size="base",
                compute_type="int8",
                batch_size=4,
                llm_model="qwen3:4b",
                video_codec="libx264",
            )
    except Exception:
        pass

    return HardwareProfile(
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


def resolve_hardware_config(
    data: dict[str, Any], profile: HardwareProfile | None = None
) -> tuple[dict[str, Any], HardwareProfile]:
    """Replace 'auto' hardware keys in data with values from HardwareProfile.

    Explicit user overrides (any value != 'auto') are preserved untouched.
    """
    hw = profile or detect_hardware_profile()

    transcription = data.setdefault("transcription", {})
    if transcription.get("device") == "auto":
        transcription["device"] = hw.device
    if transcription.get("model_size") == "auto":
        transcription["model_size"] = hw.model_size
    if transcription.get("compute_type") == "auto":
        transcription["compute_type"] = hw.compute_type
    if transcription.get("batch_size") == "auto":
        transcription["batch_size"] = hw.batch_size

    # Safety guard: float16 / int8_float16 are invalid on CPU
    if transcription.get("device") == "cpu" and transcription.get("compute_type") in (
        "float16",
        "int8_float16",
    ):
        transcription["compute_type"] = "int8"

    llm = data.setdefault("llm", {})
    if llm.get("model") == "auto":
        llm["model"] = hw.llm_model

    video = data.setdefault("video", {})
    if video.get("codec") == "auto":
        video["codec"] = hw.video_codec

    return data, hw


class Config:
    """Hierarchical config: user config overrides defaults, then resolves 'auto' hardware settings."""

    def __init__(
        self,
        user_config_path: Path | None = None,
        hardware_profile: HardwareProfile | None = None,
    ):
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        config_path = Path(user_config_path) if user_config_path is not None else USER_CONFIG_PATH
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                user_data = yaml.safe_load(f) or {}
            self._data = self._deep_merge(self._data, user_data)

        self._raw_auto_keys = {
            key: self.get(key) == "auto"
            for key in (
                "transcription.device",
                "transcription.model_size",
                "transcription.compute_type",
                "transcription.batch_size",
                "llm.model",
                "video.codec",
            )
        }
        self._data, self.hardware_profile = resolve_hardware_config(
            self._data, profile=hardware_profile
        )
        self._validate()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Access nested config: config.get('analysis.weights.loudness')."""
        keys = dotted_key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def is_auto_resolved(self, dotted_key: str) -> bool:
        """Return True if this setting came from dynamic 'auto' hardware resolution."""
        return bool(self._raw_auto_keys.get(dotted_key, False))

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    def _validate(self) -> None:
        """Validate critical config values."""
        weights = self.get("analysis.weights", {})
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Signal weights must sum to 1.0, got {total}")

        resolution = self.get("video.resolution", [1080, 1920])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("video.resolution must be [width, height]")

        batch_size = self.get("transcription.batch_size", 4)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"transcription.batch_size must be a positive int, got {batch_size!r}")

    @property
    def data(self) -> dict:
        return self._data
