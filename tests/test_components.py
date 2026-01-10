"""Tests for individual components: video utilities and hashing."""

from pathlib import Path

import pytest
from PIL import Image

from video_offset_finder import (
    CompareType,
    compute_hash,
    compute_sad_signature,
    compute_video_signatures,
    extract_frames,
    get_video_info,
)

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
