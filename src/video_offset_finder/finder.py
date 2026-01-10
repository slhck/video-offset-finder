"""Main offset finding algorithm using hierarchical search."""

import logging
from pathlib import Path
from typing import Optional

from .hashing import compute_video_signatures, cross_correlate_signatures
from .models import CompareType, OffsetResult
from .video import get_video_info


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    millis = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02}.{millis:03}"


def find_offset(
    ref_path: Path,
    dist_path: Path,
    compare_type: Optional[CompareType] = None,
    hash_size: int = 16,
    coarse_fps: float = 1.0,
    fine_fps: float = 10.0,
    start_offset: float = 0,
    max_search_offset: Optional[float] = None,
    max_duration: Optional[float] = None,
    refine_window: float = 2.0,
    frame_accurate: bool = True,
    quiet: bool = False,
) -> OffsetResult:
    """
    Find video offset using hierarchical coarse-to-fine search.

    This is the main entry point for finding temporal offset between two videos.

    Args:
        ref_path: Reference video path
        dist_path: Distorted/delayed video path
        compare_type: Comparison algorithm to use (hash-based or SAD)
        hash_size: Hash size (larger = more precise but slower, ignored for SAD)
        coarse_fps: FPS for initial coarse search
        fine_fps: FPS for intermediate refinement
        start_offset: Known minimum offset (seconds) to start search
        max_search_offset: Maximum offset to search (seconds)
        max_duration: Maximum video duration to analyze (seconds)
        refine_window: Window size (seconds) around coarse result for refinement
        frame_accurate: If True, do final pass at native FPS for exact frame matching
        quiet: If True, suppress progress bars

    Returns:
        OffsetResult with detected offset
    """
    compare_type = compare_type or CompareType.PHASH

    ref_info = get_video_info(ref_path)
    dist_info = get_video_info(dist_path)

    logging.debug(
        f"Reference: {ref_info.width}x{ref_info.height} @ {ref_info.fps:.2f} fps, "
        f"{ref_info.duration:.2f}s"
    )
    logging.debug(
        f"Distorted: {dist_info.width}x{dist_info.height} @ {dist_info.fps:.2f} fps, "
        f"{dist_info.duration:.2f}s"
    )

    # Use the higher of the two frame rates for frame-accurate matching
    native_fps = max(ref_info.fps, dist_info.fps)

    # Phase 1: Coarse search at low FPS
    logging.debug(f"Phase 1: Coarse search at {coarse_fps} fps")

    # Compute reference hashes
    ref_max_duration = max_duration
    if max_search_offset:
        # Need to cover search range + some overlap
        ref_max_duration = min(
            max_search_offset + (max_duration or dist_info.duration),
            ref_info.duration,
        )

    ref_sigs = compute_video_signatures(
        ref_path,
        coarse_fps,
        compare_type,
        hash_size,
        start_time=start_offset,
        max_duration=ref_max_duration,
        desc="Reference (coarse)",
        quiet=quiet,
    )

    # Compute distorted video signatures
    dist_search_duration = max_search_offset if max_search_offset else None
    dist_sigs = compute_video_signatures(
        dist_path,
        coarse_fps,
        compare_type,
        hash_size,
        max_duration=dist_search_duration,
        desc="Distorted (coarse)",
        quiet=quiet,
    )

    # Find best offset via cross-correlation
    coarse_offset_frames, coarse_distance = cross_correlate_signatures(
        ref_sigs, dist_sigs, compare_type
    )
    coarse_offset_seconds = coarse_offset_frames / coarse_fps + start_offset

    logging.debug(
        f"Coarse result: offset = {coarse_offset_seconds:.2f}s "
        f"({coarse_offset_frames} frames at {coarse_fps} fps), "
        f"distance = {coarse_distance:.2f}"
    )

    current_offset = coarse_offset_seconds
    current_distance = coarse_distance
    current_fps = coarse_fps

    # Phase 2: Intermediate refinement (if fine_fps specified and different from coarse)
    if fine_fps > coarse_fps:
        logging.debug(f"Phase 2: Fine search at {fine_fps} fps")

        fine_start = max(0, current_offset - refine_window)
        fine_duration = min(refine_window * 2, max_duration or dist_info.duration)

        ref_sigs_fine = compute_video_signatures(
            ref_path,
            fine_fps,
            compare_type,
            hash_size,
            start_time=fine_start,
            max_duration=fine_duration + refine_window,
            desc="Reference (fine)",
            quiet=quiet,
        )

        dist_sigs_fine = compute_video_signatures(
            dist_path,
            fine_fps,
            compare_type,
            hash_size,
            max_duration=fine_duration,
            desc="Distorted (fine)",
            quiet=quiet,
        )

        fine_offset_frames, fine_distance = cross_correlate_signatures(
            ref_sigs_fine, dist_sigs_fine, compare_type
        )
        current_offset = fine_offset_frames / fine_fps + fine_start
        current_distance = fine_distance
        current_fps = fine_fps

        logging.debug(
            f"Fine result: offset = {current_offset:.2f}s, distance = {current_distance:.2f}"
        )

    # Phase 3: Frame-accurate search at native FPS
    if frame_accurate and native_fps > current_fps:
        logging.debug(
            f"Phase 3: Frame-accurate search at {native_fps:.2f} fps (native)"
        )

        # Narrow window for final refinement (0.5s should be plenty after fine search)
        frame_window = 0.5
        frame_start = max(0, current_offset - frame_window)
        # Only need to analyze a few seconds of the distorted video
        frame_duration = min(frame_window * 2, dist_info.duration)

        ref_sigs_native = compute_video_signatures(
            ref_path,
            native_fps,
            compare_type,
            hash_size,
            start_time=frame_start,
            max_duration=frame_duration + frame_window,
            desc="Reference (native)",
            quiet=quiet,
        )

        dist_sigs_native = compute_video_signatures(
            dist_path,
            native_fps,
            compare_type,
            hash_size,
            max_duration=frame_duration,
            desc="Distorted (native)",
            quiet=quiet,
        )

        native_offset_frames, native_distance = cross_correlate_signatures(
            ref_sigs_native, dist_sigs_native, compare_type
        )
        native_offset_seconds = native_offset_frames / native_fps + frame_start

        logging.debug(
            f"Frame-accurate result: offset = {native_offset_seconds:.4f}s "
            f"({int(native_offset_seconds * native_fps)} frames), "
            f"distance = {native_distance:.2f}"
        )

        return OffsetResult(
            offset_frames=int(round(native_offset_seconds * native_fps)),
            offset_seconds=native_offset_seconds,
            offset_timestamp=format_timestamp(native_offset_seconds),
            confidence=native_distance,
            fps_used=native_fps,
            method=f"frame_accurate_{compare_type.value}",
        )

    return OffsetResult(
        offset_frames=int(round(current_offset * ref_info.fps)),
        offset_seconds=current_offset,
        offset_timestamp=format_timestamp(current_offset),
        confidence=current_distance,
        fps_used=current_fps,
        method=f"hierarchical_{compare_type.value}",
    )
