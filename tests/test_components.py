"""Tests for individual components: video utilities and hashing."""

from fractions import Fraction
from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image

from video_offset_finder import (
    CompareType,
    compute_hash,
    compute_sad_signature,
    compute_video_signatures,
    cross_correlate_signatures,
    cross_correlate_signatures_detailed,
    extract_frames,
    find_offset,
    get_video_info,
)
from video_offset_finder.hashing import FrameSignature

# Hash types (excluding SAD which is not a hash algorithm)
HASH_TYPES = [
    CompareType.PHASH,
    CompareType.DHASH,
    CompareType.AHASH,
    CompareType.WHASH,
]


class TestVideoInfo:
    """Tests for get_video_info function."""

    def test_get_video_info_synthetic(self, synthetic_reference: Path) -> None:
        """Test video info extraction for synthetic video."""
        info = get_video_info(synthetic_reference)

        assert info.path == synthetic_reference
        assert info.width == 160
        assert info.height == 90
        assert 24 <= info.fps <= 26  # Should be ~25 fps
        assert 9 <= info.duration <= 11  # Should be ~10 seconds

    def test_get_video_info_bbb(self, bbb_reference: Path) -> None:
        """Test video info extraction for Big Buck Bunny."""
        info = get_video_info(bbb_reference)

        assert info.path == bbb_reference
        assert info.width == 160
        assert info.height == 90
        assert 59 <= info.fps <= 61  # Should be ~60 fps
        assert 9 <= info.duration <= 11  # Should be ~10 seconds


class TestFrameExtraction:
    """Tests for extract_frames function."""

    def test_extract_frames_count(self, synthetic_reference: Path) -> None:
        """Test that frame extraction yields expected number of frames."""
        frames = list(
            extract_frames(synthetic_reference, target_fps=5.0, max_frames=10)
        )

        assert len(frames) == 10

    def test_extract_frames_yields_images(self, synthetic_reference: Path) -> None:
        """Test that extracted frames are PIL Images."""
        frames = list(extract_frames(synthetic_reference, target_fps=1.0, max_frames=3))

        for timestamp, image in frames:
            assert isinstance(timestamp, float)
            assert isinstance(image, Image.Image)

    def test_extract_frames_timestamps_increase(self, bbb_reference: Path) -> None:
        """Test that timestamps increase monotonically."""
        frames = list(extract_frames(bbb_reference, target_fps=5.0, max_frames=5))

        timestamps = [t for t, _ in frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_extract_frames_with_start_time(self, synthetic_reference: Path) -> None:
        """Test frame extraction starting from a specific time."""
        frames = list(
            extract_frames(
                synthetic_reference, target_fps=5.0, start_time=2.0, max_frames=5
            )
        )

        # First frame should be at or after 2 seconds
        assert frames[0][0] >= 2.0

    def test_extract_frames_with_max_duration(self, synthetic_reference: Path) -> None:
        """Test frame extraction with max duration limit."""
        frames = list(
            extract_frames(synthetic_reference, target_fps=10.0, max_duration=1.0)
        )

        # Should get roughly 10 frames in 1 second at 10 fps
        assert 8 <= len(frames) <= 12

    def test_extract_frames_holds_previous_frame_for_missing_pts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Irregular PTS are resampled on time, including CFR duplicates."""

        class FakeFrame:
            def __init__(self, pts: int) -> None:
                self.pts = pts

            def to_image(self, **_kwargs: int) -> Image.Image:
                return Image.new("L", (1, 1), self.pts)

        class FakeStream:
            average_rate = Fraction(4, 1)
            base_rate = Fraction(4, 1)
            time_base = Fraction(1, 4)
            thread_type = ""

        class FakeStreams:
            video = [FakeStream()]

        class FakeContainer:
            streams = FakeStreams()

            def __enter__(self) -> "FakeContainer":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def decode(self, video: int = 0) -> list[FakeFrame]:
                assert video == 0
                return [FakeFrame(0), FakeFrame(2), FakeFrame(3)]

        monkeypatch.setattr(
            "video_offset_finder.video.av.open", lambda _path: FakeContainer()
        )
        frames = list(extract_frames(Path("irregular.mp4"), target_fps=4.0))

        assert [timestamp for timestamp, _ in frames] == [0.0, 0.25, 0.5, 0.75]
        assert [image.getpixel((0, 0)) for _, image in frames] == [0, 0, 2, 3]


class TestHashing:
    """Tests for hashing functions."""

    def test_compute_hash_returns_hash(self, synthetic_reference: Path) -> None:
        """Test that compute_hash returns an ImageHash."""
        frames = list(extract_frames(synthetic_reference, target_fps=1.0, max_frames=1))
        _, image = frames[0]

        hash_result = compute_hash(image, CompareType.PHASH, hash_size=16)

        assert hash_result is not None
        assert len(str(hash_result)) > 0

    @pytest.mark.parametrize("hash_type", HASH_TYPES)
    def test_all_hash_types_work(
        self, synthetic_reference: Path, hash_type: CompareType
    ) -> None:
        """Test all hash types produce valid hashes."""
        frames = list(extract_frames(synthetic_reference, target_fps=1.0, max_frames=1))
        _, image = frames[0]

        hash_result = compute_hash(image, hash_type, hash_size=8)

        assert hash_result is not None

    def test_sad_signature_works(self, synthetic_reference: Path) -> None:
        """Test SAD signature computation."""
        frames = list(extract_frames(synthetic_reference, target_fps=1.0, max_frames=1))
        _, image = frames[0]

        sig = compute_sad_signature(image)

        assert sig is not None
        assert sig.shape == (64 * 64,)  # Default 64x64 grayscale

    def test_same_image_same_hash(self, synthetic_reference: Path) -> None:
        """Test that the same image produces the same hash."""
        frames = list(extract_frames(synthetic_reference, target_fps=1.0, max_frames=1))
        _, image = frames[0]

        hash1 = compute_hash(image, CompareType.PHASH, hash_size=16)
        hash2 = compute_hash(image, CompareType.PHASH, hash_size=16)

        assert hash1 == hash2

    def test_compute_video_signatures(self, synthetic_reference: Path) -> None:
        """Test compute_video_signatures returns list of tuples."""
        hashes = compute_video_signatures(
            synthetic_reference,
            fps=2.0,
            compare_type=CompareType.PHASH,
            hash_size=8,
            max_frames=5,
        )

        assert len(hashes) == 5
        for timestamp, hash_val in hashes:
            assert isinstance(timestamp, float)
            assert hash_val is not None


class TestHashSimilarity:
    """Tests for hash similarity between related frames."""

    def test_similar_frames_have_low_distance(self, bbb_reference: Path) -> None:
        """Adjacent frames should have similar hashes (after initial fade-in)."""
        hashes = compute_video_signatures(
            bbb_reference,
            fps=30.0,  # High fps = consecutive frames are similar
            compare_type=CompareType.PHASH,
            hash_size=16,
            start_time=5.0,  # Skip title screen fade-in
            max_frames=10,
        )

        # Check distance between adjacent frames
        distances = []
        for i in range(len(hashes) - 1):
            _, hash1 = hashes[i]
            _, hash2 = hashes[i + 1]
            distances.append(hash1 - hash2)

        # Average distance should be low for adjacent frames
        avg_distance = sum(distances) / len(distances)
        assert avg_distance < 50, (
            f"Average adjacent frame distance {avg_distance} too high"
        )


class TestCorrelation:
    """Tests for overlap-safe correlation and diagnostics."""

    def test_rejects_perfect_single_frame_edge_match(self) -> None:
        ref: list[tuple[float, FrameSignature]] = [
            (float(i), np.array([10], dtype=np.uint8)) for i in range(3)
        ]
        dist: list[tuple[float, FrameSignature]] = [
            (0.0, np.array([0], dtype=np.uint8)),
            (1.0, np.array([0], dtype=np.uint8)),
            (2.0, np.array([10], dtype=np.uint8)),
        ]

        result = cross_correlate_signatures_detailed(
            ref, dist, CompareType.SAD, min_overlap_fraction=0.5
        )

        assert result.offset_frames != 2
        assert result.overlap_frames >= 2
        assert result.second_best_distance is not None

    def test_packed_hash_correlation_matches_identical_sequence(self) -> None:
        hashes: list[tuple[float, FrameSignature]] = [
            (
                float(value),
                imagehash.ImageHash(np.unpackbits(np.array([value], dtype=np.uint8))),
            )
            for value in (3, 17, 99, 201)
        ]

        offset, distance = cross_correlate_signatures(hashes, hashes)

        assert offset == 0
        assert distance == 0


class TestSignatureCaching:
    """Tests for reuse of decoded signatures between search phases."""

    def test_short_inputs_are_decoded_once_each(
        self,
        synthetic_reference: Path,
        synthetic_offset_2s: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def counted_compute(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return compute_video_signatures(*args, **kwargs)  # type: ignore

        monkeypatch.setattr(
            "video_offset_finder.finder.compute_video_signatures", counted_compute
        )

        find_offset(
            synthetic_reference,
            synthetic_offset_2s,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=True,
            quiet=True,
        )

        assert calls == 2
