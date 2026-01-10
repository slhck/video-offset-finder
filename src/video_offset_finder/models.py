"""Data models and enums for video offset finding."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CompareType(str, Enum):
    """Supported comparison algorithms."""

    PHASH = "phash"  # DCT-based perceptual hash, good general choice
    DHASH = "dhash"  # Difference hash, fast, good for video
    AHASH = "ahash"  # Average hash, fastest but less robust
    WHASH = "whash"  # Wavelet hash, most robust but slowest
    SAD = "sad"  # Sum of Absolute Differences, direct pixel comparison


# Backward compatibility alias
HashType = CompareType


@dataclass
class VideoInfo:
    """Basic video metadata."""

    path: Path
    fps: float
    duration: float
    frame_count: int
    width: int
    height: int


@dataclass
class OffsetResult:
    """Result of offset detection."""

    offset_frames: int
    offset_seconds: float
    confidence: float  # Lower is better (Hamming distance)
    fps_used: float
    method: str
