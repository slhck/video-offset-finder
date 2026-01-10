"""Perceptual hashing and comparison functions for video frames."""

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from .models import CompareType
from .video import extract_frames, get_video_info

# Type alias for frame signatures (either hash or pixel array)
FrameSignature = Union[imagehash.ImageHash, np.ndarray]

# Default resize dimensions for SAD comparison
SAD_RESIZE_WIDTH = 64
SAD_RESIZE_HEIGHT = 64


def compute_hash(
    image: Image.Image,
    compare_type: CompareType = CompareType.PHASH,
    hash_size: int = 16,
) -> imagehash.ImageHash:
    """Compute perceptual hash of an image."""
    if compare_type == CompareType.PHASH:
        return imagehash.phash(image, hash_size=hash_size)
    elif compare_type == CompareType.DHASH:
        return imagehash.dhash(image, hash_size=hash_size)
    elif compare_type == CompareType.AHASH:
        return imagehash.average_hash(image, hash_size=hash_size)
    elif compare_type == CompareType.WHASH:
        return imagehash.whash(image, hash_size=hash_size)
    else:
        raise ValueError(f"Unknown hash type: {compare_type}")


def compute_sad_signature(
    image: Image.Image,
    width: int = SAD_RESIZE_WIDTH,
    height: int = SAD_RESIZE_HEIGHT,
) -> np.ndarray:
    """
    Compute SAD signature (resized grayscale pixel array) of an image.

    Args:
        image: PIL Image to process
        width: Target width for resizing
        height: Target height for resizing

    Returns:
        Flattened numpy array of grayscale pixel values (0-255)
    """
    # Convert to grayscale and resize
    gray = image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    return np.array(gray, dtype=np.float32).flatten()


def compute_video_signatures(
    path: Path,
    fps: float,
    compare_type: CompareType = CompareType.PHASH,
    hash_size: int = 16,
    start_time: float = 0,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    desc: str = "Computing signatures",
) -> list[tuple[float, FrameSignature]]:
    """
    Compute frame signatures (hashes or SAD arrays) for video frames.

    Args:
        path: Video file path
        fps: Target frames per second for extraction
        compare_type: Comparison algorithm to use
        hash_size: Hash size (only used for hash-based methods)
        start_time: Start time in seconds
        max_duration: Maximum duration to process
        max_frames: Maximum number of frames to process
        desc: Description for progress bar

    Returns:
        List of (timestamp, signature) tuples
    """
    signatures: list[tuple[float, FrameSignature]] = []
    frames = extract_frames(
        path,
        fps,
        start_time=start_time,
        max_duration=max_duration,
        max_frames=max_frames,
    )

    # Estimate total frames for progress bar
    # Use ceiling to avoid underestimating (which causes tqdm to drop the progress bar)
    video_info = get_video_info(path)
    duration = max_duration or (video_info.duration - start_time)
    estimated_frames = min(
        math.ceil(duration * fps) + 1 if duration > 0 else video_info.frame_count,
        max_frames or float("inf"),
    )

    for timestamp, image in tqdm(frames, total=estimated_frames, desc=desc):
        sig: FrameSignature
        if compare_type == CompareType.SAD:
            sig = compute_sad_signature(image)
        else:
            sig = compute_hash(image, compare_type, hash_size)
        signatures.append((timestamp, sig))

    return signatures


# Backward compatibility alias
def compute_video_hashes(
    path: Path,
    fps: float,
    compare_type: Optional[CompareType] = None,
    hash_size: int = 16,
    start_time: float = 0,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    desc: str = "Computing hashes",
    # Backward compatibility parameter
    hash_type: Optional[CompareType] = None,
) -> list[tuple[float, FrameSignature]]:
    """Backward compatible alias for compute_video_signatures."""
    # Support old hash_type parameter for backward compatibility
    actual_type = compare_type or hash_type or CompareType.PHASH
    return compute_video_signatures(
        path, fps, actual_type, hash_size, start_time, max_duration, max_frames, desc
    )


def hash_to_array(h: imagehash.ImageHash) -> np.ndarray:
    """Convert ImageHash to numpy array of bits."""
    return np.array(h.hash.flatten(), dtype=np.int8)


def cross_correlate_signatures(
    ref_sigs: list[tuple[float, FrameSignature]],
    dist_sigs: list[tuple[float, FrameSignature]],
    compare_type: CompareType = CompareType.PHASH,
) -> tuple[int, float]:
    """
    Find optimal alignment using cross-correlation of frame signatures.

    For hash-based methods, computes Hamming distance.
    For SAD, computes Sum of Absolute Differences.

    Args:
        ref_sigs: Reference video signatures (timestamp, signature)
        dist_sigs: Distorted video signatures (timestamp, signature)
        compare_type: Comparison algorithm being used

    Returns:
        Tuple of (best_offset_in_dist_frames, min_avg_distance)
    """
    if compare_type == CompareType.SAD:
        return _cross_correlate_sad(ref_sigs, dist_sigs)
    else:
        return _cross_correlate_hashes(ref_sigs, dist_sigs)


def _cross_correlate_hashes(
    ref_hashes: Sequence[tuple[float, FrameSignature]],
    dist_hashes: Sequence[tuple[float, FrameSignature]],
) -> tuple[int, float]:
    """
    Find optimal alignment using cross-correlation of hash distances.

    This finds the global optimum by computing the total Hamming distance
    at each possible offset.

    Returns:
        Tuple of (best_offset_in_dist_frames, min_avg_distance)
    """
    ref_arrays = np.array([hash_to_array(h) for _, h in ref_hashes])  # type: ignore
    dist_arrays = np.array([hash_to_array(h) for _, h in dist_hashes])  # type: ignore

    n_ref = len(ref_arrays)
    n_dist = len(dist_arrays)

    # We slide dist over ref, looking for where dist starts relative to ref
    # Positive offset means dist is delayed (starts later than ref)
    # Negative offset means dist is ahead (starts before ref)

    best_offset = 0
    min_avg_distance = float("inf")

    # Search range: from dist being fully ahead to fully behind
    for offset in range(-n_dist + 1, n_ref):
        # Determine overlap region
        if offset >= 0:
            ref_start = offset
            dist_start = 0
        else:
            ref_start = 0
            dist_start = -offset

        ref_end = min(n_ref, offset + n_dist)
        dist_end = dist_start + (ref_end - ref_start)

        if ref_end <= ref_start:
            continue

        # Compute Hamming distances for overlapping frames
        ref_slice = ref_arrays[ref_start:ref_end]
        dist_slice = dist_arrays[dist_start:dist_end]

        # Hamming distance = sum of XOR bits
        distances = np.sum(ref_slice != dist_slice, axis=1)
        avg_distance = np.mean(distances)

        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            best_offset = offset

    return best_offset, min_avg_distance


def _cross_correlate_sad(
    ref_sigs: list[tuple[float, FrameSignature]],
    dist_sigs: list[tuple[float, FrameSignature]],
) -> tuple[int, float]:
    """
    Find optimal alignment using cross-correlation of SAD (Sum of Absolute Differences).

    Returns:
        Tuple of (best_offset_in_dist_frames, min_avg_sad)
    """
    ref_arrays = np.array([sig for _, sig in ref_sigs])
    dist_arrays = np.array([sig for _, sig in dist_sigs])

    n_ref = len(ref_arrays)
    n_dist = len(dist_arrays)

    best_offset = 0
    min_avg_sad = float("inf")

    # Search range: from dist being fully ahead to fully behind
    for offset in range(-n_dist + 1, n_ref):
        # Determine overlap region
        if offset >= 0:
            ref_start = offset
            dist_start = 0
        else:
            ref_start = 0
            dist_start = -offset

        ref_end = min(n_ref, offset + n_dist)
        dist_end = dist_start + (ref_end - ref_start)

        if ref_end <= ref_start:
            continue

        # Compute SAD for overlapping frames
        ref_slice = ref_arrays[ref_start:ref_end]
        dist_slice = dist_arrays[dist_start:dist_end]

        # Sum of Absolute Differences per frame, then average
        sad_per_frame = np.sum(np.abs(ref_slice - dist_slice), axis=1)
        avg_sad = np.mean(sad_per_frame)

        if avg_sad < min_avg_sad:
            min_avg_sad = avg_sad
            best_offset = offset

    return best_offset, min_avg_sad


# Backward compatibility alias
def cross_correlate_hashes(
    ref_hashes: list[tuple[float, imagehash.ImageHash]],
    dist_hashes: list[tuple[float, imagehash.ImageHash]],
) -> tuple[int, float]:
    """Backward compatible alias for hash-based cross correlation."""
    return _cross_correlate_hashes(ref_hashes, dist_hashes)
