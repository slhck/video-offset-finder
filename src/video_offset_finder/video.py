"""Video processing utilities for frame extraction and metadata."""

from pathlib import Path
from typing import Iterator, Optional

import av
from PIL import Image

from .models import VideoInfo


def get_video_info(path: Path) -> VideoInfo:
    """Extract video metadata using PyAV."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 25)
        time_base = stream.time_base or 1
        duration = float(stream.duration * time_base) if stream.duration else 0.0
        frame_count = stream.frames or int(duration * fps)

    return VideoInfo(
        path=path,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        width=stream.width,
        height=stream.height,
    )


def extract_frames(
    path: Path,
    target_fps: float,
    start_time: float = 0,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[float, Image.Image]]:
    """
    Extract frames from video at specified FPS.

    Args:
        path: Video file path
        target_fps: Desired output frame rate
        start_time: Start time in seconds (relative to video start, not PTS)
        max_duration: Maximum duration to extract (seconds)
        max_frames: Maximum number of frames to extract

    Yields:
        Tuple of (relative_timestamp_seconds, PIL.Image)
        The timestamp is relative to the video start (0-based), not absolute PTS.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        source_fps = float(stream.average_rate or stream.base_rate or 25)
        time_base = float(stream.time_base) if stream.time_base else 1.0
        frame_interval = source_fps / target_fps

        first_pts_time: Optional[float] = None

        # Seek to start_time if specified (with some margin before)
        # This avoids decoding all frames from the beginning
        if start_time > 0.5:
            # First, decode one frame to get the baseline first_pts_time
            # This is needed for proper timestamp normalization after seeking
            for first_frame in container.decode(video=0):
                if first_frame.pts is not None:
                    first_pts_time = float(first_frame.pts * time_base)
                break

            # Now seek to target position (0.5s before start_time to account for keyframes)
            seek_time = max(0, start_time - 0.5)
            seek_pts = int(seek_time / time_base)
            container.seek(seek_pts, stream=stream)

        frames_in_range = 0  # Count frames within the extraction range
        next_sample_idx: float = 0  # Next frame index (within range) to sample
        frames_yielded = 0

        for frame in container.decode(video=0):
            # Get absolute PTS timestamp
            if frame.pts is not None:
                abs_timestamp = float(frame.pts * time_base)
            else:
                # Fallback for frames without PTS
                abs_timestamp = 0  # Will be corrected by first_pts_time logic

            # Normalize to video-relative time (first frame = 0)
            if first_pts_time is None:
                first_pts_time = abs_timestamp

            relative_time = abs_timestamp - first_pts_time

            # Skip frames before start_time
            if relative_time < start_time:
                continue

            # Check duration limit (relative to start_time)
            if max_duration and (relative_time - start_time) > max_duration:
                break

            # Check frame limit
            if max_frames and frames_yielded >= max_frames:
                break

            # Sample at target FPS (using frame count within extraction range)
            if frames_in_range >= next_sample_idx:
                yield relative_time, frame.to_image()
                next_sample_idx += frame_interval
                frames_yielded += 1

            frames_in_range += 1
