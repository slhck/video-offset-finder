"""
Video Offset Finder - Find temporal offset between videos using perceptual hashing or SAD.

This library provides tools for finding the temporal offset between two videos,
which is useful for video synchronization, A/V sync analysis, and quality comparison.

Example usage:
    from pathlib import Path
    from video_offset_finder import find_offset, CompareType

    result = find_offset(
        ref_path=Path("reference.mp4"),
        dist_path=Path("distorted.mp4"),
        compare_type=CompareType.PHASH,
    )
    print(f"Offset: {result.offset_seconds}s")
"""

from importlib.metadata import version

from .finder import find_offset
from .hashing import (
    compute_hash,
    compute_sad_signature,
    compute_video_hashes,
    compute_video_signatures,
    cross_correlate_hashes,
    cross_correlate_signatures,
)
from .models import CompareType, HashType, OffsetResult, VideoInfo
from .video import extract_frames, get_video_info

__version__ = version("video-offset-finder")

__all__ = [
    # Main function
    "find_offset",
    # Models
    "CompareType",
    "HashType",  # Backward compatibility alias
    "OffsetResult",
    "VideoInfo",
    # Video utilities
    "get_video_info",
    "extract_frames",
    # Comparison utilities
    "compute_hash",
    "compute_sad_signature",
    "compute_video_hashes",
    "compute_video_signatures",
    "cross_correlate_hashes",
    "cross_correlate_signatures",
    # Version
    "__version__",
]
