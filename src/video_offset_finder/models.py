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
    offset_timestamp: str  # HH:MM:SS.ms format
    confidence: float  # Lower is better (Hamming distance)
    fps_used: float
    method: str
    second_best_confidence: float | None = None
    overlap_frames: int = 0


@dataclass(frozen=True)
class CorrelationResult:
    """Detailed result for signature cross-correlation."""

    offset_frames: int
    distance: float
    second_best_distance: float | None
    overlap_frames: int
