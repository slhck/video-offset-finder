"""Perceptual hashing and comparison functions for video frames."""

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from .models import CompareType, CorrelationResult, VideoInfo
from .video import extract_frames, get_video_info

# Type alias for frame signatures (either hash or pixel array)
FrameSignature = Union[imagehash.ImageHash, np.ndarray]

# Default resize dimensions for SAD comparison
SAD_RESIZE_WIDTH = 64
SAD_RESIZE_HEIGHT = 64
_BYTE_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def signature_image_size(
    compare_type: CompareType, hash_size: int
) -> Optional[tuple[int, int]]:
    """Return the smallest safe decoder output for a comparison algorithm."""
    if compare_type == CompareType.PHASH:
        return (hash_size * 4, hash_size * 4)
    if compare_type == CompareType.DHASH:
        return (hash_size + 1, hash_size)
    if compare_type == CompareType.AHASH:
        return (hash_size, hash_size)
    if compare_type == CompareType.SAD:
        return (SAD_RESIZE_WIDTH, SAD_RESIZE_HEIGHT)
    # ImageHash derives the wavelet image scale from the input dimensions.
    return None


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
    return np.array(gray, dtype=np.uint8).flatten()


def compute_video_signatures(
    path: Path,
    fps: float,
    compare_type: CompareType = CompareType.PHASH,
    hash_size: int = 16,
    start_time: float = 0,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    desc: str = "Computing signatures",
    quiet: bool = False,
    video_info: Optional[VideoInfo] = None,
    packed_hashes: bool = False,
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
        quiet: If True, suppress progress bar
        video_info: Previously-read metadata, used to avoid reopening the video
        packed_hashes: Store hash bits compactly for internal reusable caches

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
        image_size=signature_image_size(compare_type, hash_size),
    )

    # Estimate total frames for progress bar
    # Use ceiling to avoid underestimating (which causes tqdm to drop the progress bar)
    video_info = video_info or get_video_info(path)
    duration = max_duration or (video_info.duration - start_time)
    estimated_frames = min(
        math.ceil(duration * fps) + 1 if duration > 0 else video_info.frame_count,
        max_frames or float("inf"),
    )

    for timestamp, image in tqdm(
        frames, total=estimated_frames, desc=desc, disable=quiet
    ):
        sig: FrameSignature
        if compare_type == CompareType.SAD:
            sig = compute_sad_signature(image)
        else:
            frame_hash = compute_hash(image, compare_type, hash_size)
            sig = hash_to_packed(frame_hash) if packed_hashes else frame_hash
        signatures.append((timestamp, sig))

    return signatures


def hash_to_array(h: imagehash.ImageHash) -> np.ndarray:
    """Convert ImageHash to numpy array of bits."""
    return np.array(h.hash.flatten(), dtype=np.int8)


def hash_to_packed(h: FrameSignature) -> np.ndarray:
    """Convert an ImageHash to packed bytes, accepting an existing packed hash."""
    if isinstance(h, imagehash.ImageHash):
        return np.packbits(hash_to_array(h))
    return h


def cross_correlate_signatures(
    ref_sigs: list[tuple[float, FrameSignature]],
    dist_sigs: list[tuple[float, FrameSignature]],
    compare_type: CompareType = CompareType.PHASH,
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: Optional[int] = None,
    max_offset_frames: Optional[int] = None,
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
    result = cross_correlate_signatures_detailed(
        ref_sigs,
        dist_sigs,
        compare_type,
        min_overlap_fraction=min_overlap_fraction,
        min_overlap_frames=min_overlap_frames,
        min_offset_frames=min_offset_frames,
        max_offset_frames=max_offset_frames,
    )
    return result.offset_frames, result.distance


def cross_correlate_signatures_detailed(
    ref_sigs: list[tuple[float, FrameSignature]],
    dist_sigs: list[tuple[float, FrameSignature]],
    compare_type: CompareType = CompareType.PHASH,
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: Optional[int] = None,
    max_offset_frames: Optional[int] = None,
) -> CorrelationResult:
    """Find the best alignment and report ambiguity and overlap diagnostics."""
    if not 0 < min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must be in the interval (0, 1]")
    if min_overlap_frames < 1:
        raise ValueError("min_overlap_frames must be at least one")

    kwargs = {
        "min_overlap_fraction": min_overlap_fraction,
        "min_overlap_frames": min_overlap_frames,
        "min_offset_frames": min_offset_frames,
        "max_offset_frames": max_offset_frames,
    }
    if compare_type == CompareType.SAD:
        return _cross_correlate_sad(ref_sigs, dist_sigs, **kwargs)
    return _cross_correlate_hashes(ref_sigs, dist_sigs, **kwargs)


def _candidate_offsets(
    n_ref: int,
    n_dist: int,
    min_overlap_fraction: float,
    min_overlap_frames: int,
    min_offset_frames: Optional[int],
    max_offset_frames: Optional[int],
) -> tuple[range, int]:
    if n_ref == 0 or n_dist == 0:
        raise ValueError("Cannot correlate empty signature sequences")
    shortest = min(n_ref, n_dist)
    required_overlap = min(
        shortest,
        max(min_overlap_frames, math.ceil(shortest * min_overlap_fraction)),
    )
    lower_bound = -n_dist + 1 if min_offset_frames is None else min_offset_frames
    upper_bound = n_ref - 1 if max_offset_frames is None else max_offset_frames
    lower = max(-n_dist + required_overlap, lower_bound)
    upper = min(n_ref - required_overlap, upper_bound)
    if lower > upper:
        raise ValueError("No offsets satisfy the overlap and offset bounds")
    return range(lower, upper + 1), required_overlap


def _result_from_candidates(
    candidates: list[tuple[float, int, int]],
) -> CorrelationResult:
    candidates.sort(key=lambda item: (item[0], -item[2], abs(item[1])))
    best_distance, best_offset, best_overlap = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else None
    return CorrelationResult(best_offset, best_distance, second, best_overlap)


def _cross_correlate_hashes(
    ref_hashes: Sequence[tuple[float, FrameSignature]],
    dist_hashes: Sequence[tuple[float, FrameSignature]],
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: Optional[int] = None,
    max_offset_frames: Optional[int] = None,
) -> CorrelationResult:
    """
    Find optimal alignment using cross-correlation of hash distances.

    This finds the global optimum by computing the total Hamming distance
    at each possible offset.

    Returns:
        Tuple of (best_offset_in_dist_frames, min_avg_distance)
    """
    if not ref_hashes or not dist_hashes:
        raise ValueError("Cannot correlate empty signature sequences")
    ref_arrays = np.array([hash_to_packed(h) for _, h in ref_hashes])
    dist_arrays = np.array([hash_to_packed(h) for _, h in dist_hashes])

    n_ref = len(ref_arrays)
    n_dist = len(dist_arrays)

    # We slide dist over ref, looking for where dist starts relative to ref
    # Positive offset means dist is delayed (starts later than ref)
    # Negative offset means dist is ahead (starts before ref)

    offsets, _ = _candidate_offsets(
        n_ref,
        n_dist,
        min_overlap_fraction,
        min_overlap_frames,
        min_offset_frames,
        max_offset_frames,
    )
    candidates: list[tuple[float, int, int]] = []
    for offset in offsets:
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

        # Packed XOR plus a lookup-table population count supports NumPy 1.24.
        distances = _BYTE_POPCOUNT[np.bitwise_xor(ref_slice, dist_slice)].sum(axis=1)
        candidates.append((float(np.mean(distances)), offset, len(ref_slice)))

    return _result_from_candidates(candidates)


def _cross_correlate_sad(
    ref_sigs: list[tuple[float, FrameSignature]],
    dist_sigs: list[tuple[float, FrameSignature]],
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: Optional[int] = None,
    max_offset_frames: Optional[int] = None,
) -> CorrelationResult:
    """
    Find optimal alignment using cross-correlation of SAD (Sum of Absolute Differences).

    Returns:
        Tuple of (best_offset_in_dist_frames, min_avg_sad)
    """
    ref_arrays = np.array([sig for _, sig in ref_sigs])
    dist_arrays = np.array([sig for _, sig in dist_sigs])

    n_ref = len(ref_arrays)
    n_dist = len(dist_arrays)

    offsets, _ = _candidate_offsets(
        n_ref,
        n_dist,
        min_overlap_fraction,
        min_overlap_frames,
        min_offset_frames,
        max_offset_frames,
    )
    candidates: list[tuple[float, int, int]] = []
    for offset in offsets:
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
        sad_per_frame = np.sum(
            np.abs(ref_slice.astype(np.int16) - dist_slice.astype(np.int16)), axis=1
        )
        candidates.append((float(np.mean(sad_per_frame)), offset, len(ref_slice)))

    return _result_from_candidates(candidates)
