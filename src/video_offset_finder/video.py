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
    image_size: Optional[tuple[int, int]] = None,
) -> Iterator[tuple[float, Image.Image]]:
    """
    Extract frames from video at specified FPS.

    Args:
        path: Video file path
        target_fps: Desired output frame rate
        start_time: Start time in seconds (relative to video start, not PTS)
        max_duration: Maximum duration to extract (seconds)
        max_frames: Maximum number of frames to extract
        image_size: Optional decoder-side output size (width, height)

    Yields:
        Tuple of (sampling_timestamp_seconds, PIL.Image). Sampling uses a
        constant-rate timeline and holds the most recent decoded frame for a
        slot. This matches conventional CFR conversion when source timestamps
        are irregular or the target rate is higher than the source rate.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")

        source_fps = float(stream.average_rate or stream.base_rate or 25)
        time_base = float(stream.time_base) if stream.time_base else 1.0
        sample_interval = 1.0 / target_fps

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

        next_sample_time = max(0.0, start_time)
        frames_yielded = 0
        decoded_index = 0
        previous_frame: Optional[av.VideoFrame] = None

        def to_image(frame: av.VideoFrame) -> Image.Image:
            if image_size is None:
                return frame.to_image()
            return frame.to_image(width=image_size[0], height=image_size[1])

        for frame in container.decode(video=0):
            # Get absolute PTS timestamp
            if frame.pts is not None:
                abs_timestamp = float(frame.pts * time_base)
            else:
                # Preserve a useful monotonic timeline for uncommon streams
                # whose decoded frames do not carry timestamps.
                abs_timestamp = decoded_index / source_fps
            decoded_index += 1

            # Normalize to video-relative time (first frame = 0)
            if first_pts_time is None:
                first_pts_time = abs_timestamp

            relative_time = abs_timestamp - first_pts_time

            # Emit every CFR slot preceding this decoded frame using the most
            # recent frame. In particular, a missing source timestamp becomes
            # a repeated frame rather than shifting the entire sequence.
            while previous_frame is not None and next_sample_time < relative_time:
                if (
                    max_duration is not None
                    and next_sample_time > start_time + max_duration
                ):
                    return
                if max_frames is not None and frames_yielded >= max_frames:
                    return
                yield next_sample_time, to_image(previous_frame)
                next_sample_time += sample_interval
                frames_yielded += 1

            previous_frame = frame

        # Emit the final exact slot, but do not extend the video beyond the
        # timestamp of its last decoded frame.
        if previous_frame is not None and next_sample_time <= relative_time:
            if max_duration is None or next_sample_time <= start_time + max_duration:
                if max_frames is None or frames_yielded < max_frames:
                    yield next_sample_time, to_image(previous_frame)
