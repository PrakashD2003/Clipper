"""Shared pytest fixtures."""
import os

import pytest

# Never let a developer's real .env (API keys) leak into tests.
os.environ.setdefault("STREAMCLIPPER_NO_DOTENV", "1")

from streamclipper.config import detect_hardware_profile


@pytest.fixture(autouse=True)
def _no_tts_retry_sleep(monkeypatch):
    """TTS chain retries must not slow the suite down."""
    import streamclipper.assembly.narrator as narrator_mod

    monkeypatch.setattr(narrator_mod, "TTS_RETRY_SLEEP_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _clear_hardware_profile_cache():
    """detect_hardware_profile is memoized; reset so monkeypatched torch/subprocess apply."""
    detect_hardware_profile.cache_clear()
    yield
    detect_hardware_profile.cache_clear()
