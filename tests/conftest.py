"""Pytest configuration and fixtures for video offset finder tests."""

from pathlib import Path

import pytest

# Base fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_DIR = FIXTURES_DIR / "synthetic"
BBB_DIR = FIXTURES_DIR / "bbb"


@pytest.fixture
def synthetic_reference() -> Path:
    """Path to synthetic testsrc reference video (10s, 25fps, 160x90)."""
    return SYNTHETIC_DIR / "reference.mp4"


@pytest.fixture
def synthetic_offset_2s() -> Path:
    """Synthetic video starting at 2s into reference."""
    return SYNTHETIC_DIR / "offset_2s.mp4"


@pytest.fixture
def synthetic_offset_3p5s() -> Path:
    """Synthetic video starting at 3.5s into reference."""
    return SYNTHETIC_DIR / "offset_3p5s.mp4"


@pytest.fixture
def synthetic_offset_5s() -> Path:
    """Synthetic video starting at 5s into reference."""
    return SYNTHETIC_DIR / "offset_5s.mp4"


@pytest.fixture
def bbb_reference() -> Path:
    """Path to Big Buck Bunny reference video (10s, 60fps, 160x90)."""
    return BBB_DIR / "reference.mp4"


@pytest.fixture
def bbb_offset_2s() -> Path:
    """Big Buck Bunny video starting at 2s into reference."""
    return BBB_DIR / "offset_2s.mp4"


@pytest.fixture
def bbb_offset_3p5s() -> Path:
    """Big Buck Bunny video starting at 3.5s into reference."""
    return BBB_DIR / "offset_3p5s.mp4"


@pytest.fixture
def bbb_offset_5s() -> Path:
    """Big Buck Bunny video starting at 5s into reference."""
    return BBB_DIR / "offset_5s.mp4"
