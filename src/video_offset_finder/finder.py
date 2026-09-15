"""Main offset finding algorithm using hierarchical search."""

import logging
import math
from pathlib import Path
from typing import Optional, Sequence

from .hashing import (
    FrameSignature,
    compute_video_signatures,
    cross_correlate_signatures_detailed,
)
from .models import CompareType, CorrelationResult, OffsetResult, VideoInfo
from .video import get_video_info

MAX_CACHED_HASH_FRAMES = 60_000
MAX_CACHED_SAD_FRAMES = 10_000


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    sign = "-" if seconds < 0 else ""
    total_millis = int(abs(seconds) * 1000)
    total_seconds, millis = divmod(total_millis, 1000)
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{sign}{hrs:02}:{mins:02}:{secs:02}.{millis:03}"


def _resample_cached_signatures(
    signatures: Sequence[tuple[float, FrameSignature]],
    fps: float,
    start_time: float = 0,
    max_duration: Optional[float] = None,
) -> list[tuple[float, FrameSignature]]:
    """Select CFR slots from a higher-rate signature cache using hold-previous."""
    if not signatures:
        return []
    interval = 1.0 / fps
    slot = max(0.0, start_time)
    end = signatures[-1][0]
    if max_duration is not None:
        end = min(end, start_time + max_duration)

    selected: list[tuple[float, FrameSignature]] = []
    index = 0
    while index + 1 < len(signatures) and signatures[index + 1][0] <= slot:
        index += 1
    while slot <= end + 1e-9:
        while index + 1 < len(signatures) and signatures[index + 1][0] <= slot + 1e-9:
            index += 1
        if signatures[index][0] <= slot + 1e-9:
            selected.append((slot, signatures[index][1]))
        slot += interval
    return selected


def _can_cache(info: VideoInfo, cache_fps: float, compare_type: CompareType) -> bool:
    limit = (
        MAX_CACHED_SAD_FRAMES
        if compare_type == CompareType.SAD
        else MAX_CACHED_HASH_FRAMES
    )
    return math.ceil(info.duration * cache_fps) <= limit


def find_offset(
    ref_path: Path,
    dist_path: Path,
    compare_type: Optional[CompareType] = None,
    hash_size: int = 16,
    coarse_fps: float = 1.0,
    fine_fps: float = 10.0,
    start_offset: Optional[float] = None,
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
        start_offset: Optional minimum offset (seconds) to search
        max_search_offset: Maximum offset to search (seconds)
        max_duration: Maximum video duration to analyze (seconds)
        refine_window: Window size (seconds) around coarse result for refinement
        frame_accurate: If True, do final pass at native FPS for exact frame matching
        quiet: If True, suppress progress bars

    Returns:
        OffsetResult with detected offset
    """
    compare_type = compare_type or CompareType.PHASH

    if coarse_fps <= 0 or fine_fps <= 0:
        raise ValueError("coarse_fps and fine_fps must be greater than zero")
    if (
        start_offset is not None
        and max_search_offset is not None
        and max_search_offset < start_offset
    ):
        raise ValueError("max_search_offset must not be less than start_offset")
    search_start = start_offset or 0.0

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

    # For ordinary clips, compute native-cadence signatures once and derive all
    # three search phases from their timestamps. Long inputs fall back to the
    # phase-specific path to avoid excessive hashing and, for SAD, memory use.
    cache_fps = max(native_fps, fine_fps, coarse_fps)
    use_cache = _can_cache(ref_info, cache_fps, compare_type) and _can_cache(
        dist_info, cache_fps, compare_type
    )
    ref_cache: Optional[list[tuple[float, FrameSignature]]] = None
    dist_cache: Optional[list[tuple[float, FrameSignature]]] = None
    if use_cache:
        logging.debug(f"Decoding each input once at {cache_fps:.2f} fps")
        ref_cache = compute_video_signatures(
            ref_path,
            cache_fps,
            compare_type,
            hash_size,
            desc="Reference",
            quiet=quiet,
            video_info=ref_info,
            packed_hashes=compare_type != CompareType.SAD,
        )
        dist_cache = compute_video_signatures(
            dist_path,
            cache_fps,
            compare_type,
            hash_size,
            desc="Distorted",
            quiet=quiet,
            video_info=dist_info,
            packed_hashes=compare_type != CompareType.SAD,
        )

    def signatures_for(
        path: Path,
        info: VideoInfo,
        cache: Optional[list[tuple[float, FrameSignature]]],
        fps: float,
        start_time: float,
        duration: Optional[float],
        desc: str,
    ) -> list[tuple[float, FrameSignature]]:
        if cache is not None:
            return _resample_cached_signatures(cache, fps, start_time, duration)
        return compute_video_signatures(
            path,
            fps,
            compare_type,
            hash_size,
            start_time=start_time,
            max_duration=duration,
            desc=desc,
            quiet=quiet,
            video_info=info,
        )

    def correlate(
        ref_sigs: list[tuple[float, FrameSignature]],
        dist_sigs: list[tuple[float, FrameSignature]],
        fps: float,
        ref_start: float,
        dist_start: float,
    ) -> CorrelationResult:
        min_frames = None
        max_frames = None
        local_base = ref_start - dist_start
        if start_offset is not None:
            min_frames = math.ceil((start_offset - local_base) * fps - 1e-9)
        if max_search_offset is not None:
            max_frames = math.floor((max_search_offset - local_base) * fps + 1e-9)
        return cross_correlate_signatures_detailed(
            ref_sigs,
            dist_sigs,
            compare_type,
            min_offset_frames=min_frames,
            max_offset_frames=max_frames,
        )

    # Phase 1: Coarse search at low FPS
    logging.debug(f"Phase 1: Coarse search at {coarse_fps} fps")

    # Compute reference hashes
    ref_max_duration = max_duration
    if max_search_offset is not None:
        # Need to cover search range + some overlap
        ref_max_duration = min(
            (max_search_offset - search_start) + (max_duration or dist_info.duration),
            max(0.0, ref_info.duration - search_start),
        )

    ref_sigs = signatures_for(
        ref_path,
        ref_info,
        ref_cache,
        coarse_fps,
        search_start,
        ref_max_duration,
        "Reference (coarse)",
    )

    # Compute distorted video signatures
    dist_search_duration = max_duration
    dist_sigs = signatures_for(
        dist_path,
        dist_info,
        dist_cache,
        coarse_fps,
        0,
        dist_search_duration,
        "Distorted (coarse)",
    )

    # Find best offset via cross-correlation
    coarse_result = correlate(ref_sigs, dist_sigs, coarse_fps, search_start, 0)
    coarse_offset_seconds = coarse_result.offset_frames / coarse_fps + search_start

    logging.debug(
        f"Coarse result: offset = {coarse_offset_seconds:.2f}s "
        f"({coarse_result.offset_frames} frames at {coarse_fps} fps), "
        f"distance = {coarse_result.distance:.2f}"
    )

    current_offset = coarse_offset_seconds
    current_result = coarse_result
    current_fps = coarse_fps

    # Phase 2: Intermediate refinement (if fine_fps specified and different from coarse)
    if fine_fps > coarse_fps:
        logging.debug(f"Phase 2: Fine search at {fine_fps} fps")

        # Calculate search windows for both videos based on offset sign
        # For positive offset: match is at ref[offset], dist[0]
        # For negative offset: match is at ref[0], dist[-offset]
        if current_offset >= 0:
            ref_fine_start = max(0, current_offset - refine_window)
            dist_fine_start = 0.0
        else:
            ref_fine_start = 0.0
            dist_fine_start = max(0, -current_offset - refine_window)

        fine_duration = min(refine_window * 2, max_duration or dist_info.duration)

        ref_sigs_fine = signatures_for(
            ref_path,
            ref_info,
            ref_cache,
            fine_fps,
            ref_fine_start,
            fine_duration + refine_window,
            "Reference (fine)",
        )

        dist_sigs_fine = signatures_for(
            dist_path,
            dist_info,
            dist_cache,
            fine_fps,
            dist_fine_start,
            fine_duration,
            "Distorted (fine)",
        )

        fine_result = correlate(
            ref_sigs_fine,
            dist_sigs_fine,
            fine_fps,
            ref_fine_start,
            dist_fine_start,
        )
        # Account for both start positions when calculating final offset
        current_offset = (
            fine_result.offset_frames / fine_fps + ref_fine_start - dist_fine_start
        )
        current_result = fine_result
        current_fps = fine_fps

        logging.debug(
            f"Fine result: offset = {current_offset:.2f}s, "
            f"distance = {current_result.distance:.2f}"
        )

    # Phase 3: Frame-accurate search at native FPS
    if frame_accurate and native_fps > current_fps:
        logging.debug(
            f"Phase 3: Frame-accurate search at {native_fps:.2f} fps (native)"
        )

        # Narrow window for final refinement (0.5s should be plenty after fine search)
        frame_window = 0.5
        # Only need to analyze a few seconds of each video
        frame_duration = min(frame_window * 2, dist_info.duration)

        # Calculate search windows for both videos based on offset sign
        if current_offset >= 0:
            ref_frame_start = max(0, current_offset - frame_window)
            dist_frame_start = 0.0
        else:
            ref_frame_start = 0.0
            dist_frame_start = max(0, -current_offset - frame_window)

        ref_sigs_native = signatures_for(
            ref_path,
            ref_info,
            ref_cache,
            native_fps,
            ref_frame_start,
            frame_duration + frame_window,
            "Reference (native)",
        )

        dist_sigs_native = signatures_for(
            dist_path,
            dist_info,
            dist_cache,
            native_fps,
            dist_frame_start,
            frame_duration,
            "Distorted (native)",
        )

        native_result = correlate(
            ref_sigs_native,
            dist_sigs_native,
            native_fps,
            ref_frame_start,
            dist_frame_start,
        )
        # Account for both start positions when calculating final offset
        native_offset_seconds = (
            native_result.offset_frames / native_fps
            + ref_frame_start
            - dist_frame_start
        )

        logging.debug(
            f"Frame-accurate result: offset = {native_offset_seconds:.4f}s "
            f"({int(native_offset_seconds * native_fps)} frames), "
            f"distance = {native_result.distance:.2f}"
        )

        return OffsetResult(
            offset_frames=int(round(native_offset_seconds * native_fps)),
            offset_seconds=native_offset_seconds,
            offset_timestamp=format_timestamp(native_offset_seconds),
            confidence=native_result.distance,
            fps_used=native_fps,
            method=f"frame_accurate_{compare_type.value}",
            second_best_confidence=native_result.second_best_distance,
            overlap_frames=native_result.overlap_frames,
        )

    return OffsetResult(
        offset_frames=int(round(current_offset * ref_info.fps)),
        offset_seconds=current_offset,
        offset_timestamp=format_timestamp(current_offset),
        confidence=current_result.distance,
        fps_used=current_fps,
        method=f"hierarchical_{compare_type.value}",
        second_best_confidence=current_result.second_best_distance,
        overlap_frames=current_result.overlap_frames,
    )
