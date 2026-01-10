"""Tests for video offset finding using synthetic testsrc videos."""

from pathlib import Path

import pytest

from video_offset_finder import CompareType, find_offset


class TestSyntheticOffsetDetection:
    """Test offset detection with synthetic testsrc videos."""

    @pytest.mark.parametrize(
        "offset_fixture,expected_offset",
        [
            ("synthetic_offset_2s", 2.0),
            ("synthetic_offset_3p5s", 3.5),
            ("synthetic_offset_5s", 5.0),
        ],
    )
    def test_coarse_search(
        self,
        synthetic_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test coarse-only search finds offset within 1 second."""
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=synthetic_reference,
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
            ("synthetic_offset_2s", 2.0),
            ("synthetic_offset_3p5s", 3.5),
            ("synthetic_offset_5s", 5.0),
        ],
    )
    def test_hierarchical_search(
        self,
        synthetic_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test hierarchical search (coarse + fine) finds offset within 0.2 seconds."""
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=synthetic_reference,
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
            ("synthetic_offset_2s", 2.0),
            ("synthetic_offset_5s", 5.0),
        ],
    )
    def test_frame_accurate_search(
        self,
        synthetic_reference: Path,
        offset_fixture: str,
        expected_offset: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test frame-accurate search finds offset within 0.1 seconds."""
        dist_path = request.getfixturevalue(offset_fixture)

        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=dist_path,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=10.0,
            frame_accurate=True,
        )

        assert abs(result.offset_seconds - expected_offset) < 0.1, (
            f"Expected offset ~{expected_offset}s, got {result.offset_seconds}s"
        )


class TestSyntheticCompareTypes:
    """Test different comparison algorithms with synthetic videos."""

    @pytest.mark.parametrize(
        "compare_type",
        [CompareType.PHASH, CompareType.DHASH, CompareType.WHASH, CompareType.SAD],
    )
    def test_robust_compare_types_find_offset(
        self,
        synthetic_reference: Path,
        synthetic_offset_2s: Path,
        compare_type: CompareType,
    ) -> None:
        """Robust comparison types should detect 2s offset within tolerance."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=compare_type,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Compare type {compare_type.value} failed: expected ~2s, got {result.offset_seconds}s"
        )

    @pytest.mark.xfail(reason="ahash struggles with synthetic testsrc patterns")
    def test_ahash_finds_offset(
        self,
        synthetic_reference: Path,
        synthetic_offset_2s: Path,
    ) -> None:
        """Average hash may struggle with synthetic content patterns."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.AHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Compare type ahash failed: expected ~2s, got {result.offset_seconds}s"
        )


class TestSyntheticEdgeCases:
    """Test edge cases with synthetic videos."""

    def test_identical_videos(self, synthetic_reference: Path) -> None:
        """Identical videos should have zero offset."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_reference,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        assert abs(result.offset_seconds) < 0.5, (
            f"Expected zero offset, got {result.offset_seconds}s"
        )

    def test_confidence_value(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """Result should include reasonable confidence value."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
        )

        # Confidence is distance metric, should be reasonably low for matching content
        assert result.confidence < 50, f"Confidence too high: {result.confidence}"
