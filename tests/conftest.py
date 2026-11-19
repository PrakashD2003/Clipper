"""Shared pytest fixtures."""
import pytest

from streamclipper.config import detect_hardware_profile


@pytest.fixture(autouse=True)
def _clear_hardware_profile_cache():
    """detect_hardware_profile is memoized; reset so monkeypatched torch/subprocess apply."""
    detect_hardware_profile.cache_clear()
    yield
    detect_hardware_profile.cache_clear()
