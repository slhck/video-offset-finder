"""Tests for video offset finding using real video content (Big Buck Bunny)."""

from pathlib import Path

import pytest

from video_offset_finder import CompareType, find_offset


class TestBBBOffsetDetection:
    """Test offset detection with Big Buck Bunny video clips."""

    @pytest.mark.parametrize(
        "offset_fixture,expected_offset",
        [
            ("bbb_offset_2s", 2.0),
            ("bbb_offset_3p5s", 3.5),
            ("bbb_offset_5s", 5.0),
        ],
    )
    def test_coarse_search(
        self,
        bbb_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test coarse-only search finds offset within 1 second for real content."""
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=bbb_reference,
            dist_path=dist_path,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=1.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds - expected_offset) < 1.0, (
            f"Expected offset ~{expected_offset}s, got {result.offset_seconds}s"
        )

    @pytest.mark.parametrize(
        "offset_fixture,expected_offset",
        [
            ("bbb_offset_2s", 2.0),
            ("bbb_offset_3p5s", 3.5),
            ("bbb_offset_5s", 5.0),
        ],
    )
    def test_hierarchical_search(
        self,
        bbb_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test hierarchical search (coarse + fine) finds offset within 0.2 seconds."""
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=bbb_reference,
            dist_path=dist_path,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=10.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds - expected_offset) < 0.2, (
            f"Expected offset ~{expected_offset}s, got {result.offset_seconds}s"
        )

    @pytest.mark.parametrize(
        "offset_fixture,expected_offset",
        [
            ("bbb_offset_2s", 2.0),
            ("bbb_offset_5s", 5.0),
        ],
    )
    def test_frame_accurate_search(
        self,
        bbb_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test frame-accurate search finds offset within 1 second for real content.

        Note: BBB content with title fade-in can be challenging for exact matching.
        The tolerance is looser than synthetic content tests.
        """
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=bbb_reference,
            dist_path=dist_path,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=10.0,
            frame_accurate=True,
        )

        assert abs(result.offset_seconds - expected_offset) < 1.0, (
            f"Expected offset ~{expected_offset}s, got {result.offset_seconds}s"
        )


class TestBBBCompareTypes:
    """Test different comparison algorithms with Big Buck Bunny content."""

    @pytest.mark.parametrize("compare_type", list(CompareType))
    def test_all_compare_types_find_offset(
        self,
        bbb_reference: Path,
        bbb_offset_2s: Path,
        compare_type: CompareType,
    ) -> None:
        """All compare types should detect 2s offset within tolerance for real content."""
        result = find_offset(
            ref_path=bbb_reference,
            dist_path=bbb_offset_2s,
            compare_type=compare_type,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Compare type {compare_type.value} failed: expected ~2s, got {result.offset_seconds}s"
        )


class TestBBBEdgeCases:
    """Test edge cases with Big Buck Bunny content."""

    def test_identical_videos(self, bbb_reference: Path) -> None:
        """Identical real videos should have zero offset."""
        result = find_offset(
            ref_path=bbb_reference,
            dist_path=bbb_reference,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds) < 0.5, (
            f"Expected zero offset, got {result.offset_seconds}s"
        )

    def test_real_content_confidence(
        self, bbb_reference: Path, bbb_offset_2s: Path
    ) -> None:
        """Real content should have low confidence (good match)."""
        result = find_offset(
            ref_path=bbb_reference,
            dist_path=bbb_offset_2s,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        # Real content from the same source should match well
        assert result.confidence < 30, (
            f"Confidence too high for identical source content: {result.confidence}"
        )
