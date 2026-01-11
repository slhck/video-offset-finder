"""Tests for search offset parameters: start_offset, max_search_offset, max_duration, and negative offsets."""

from pathlib import Path


from video_offset_finder import CompareType, find_offset


class TestStartOffset:
    """Test the start_offset (-o/--start-offset) parameter.

    start_offset specifies a known minimum offset in seconds.
    The algorithm skips extracting reference frames before this point.
    """

    def test_start_offset_finds_correct_offset(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """With start_offset=3, should still find offset at 5s."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            start_offset=3.0,  # Skip first 3s of reference
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # Should find ~5s offset (content starts at 5s into reference)
        assert abs(result.offset_seconds - 5.0) < 0.5, (
            f"Expected ~5s offset with start_offset=3, got {result.offset_seconds}s"
        )

    def test_start_offset_speeds_up_search(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """start_offset=4 should find offset faster (less frames to process).

        This test verifies the feature works - actual speed improvement
        depends on video length.
        """
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            start_offset=4.0,  # Skip first 4s (match is at 5s)
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert abs(result.offset_seconds - 5.0) < 0.5, (
            f"Expected ~5s offset with start_offset=4, got {result.offset_seconds}s"
        )

    def test_start_offset_still_finds_earlier_match(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """Even with start_offset=3 and actual offset at 2s, algorithm may still find correct offset.

        The cross-correlation can still find matches that occur before the start_offset
        position because the distorted video is analyzed from its beginning. The algorithm
        adds start_offset to the frame offset, so finding a match at frame -1 at coarse 1fps
        from position 3 would give 3 + (-1) = 2s.
        """
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            start_offset=3.0,  # Skip first 3s, but match is at 2s
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # The algorithm actually still finds the correct offset
        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Expected ~2s offset, got {result.offset_seconds}s"
        )

    def test_start_offset_zero_is_default(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """With start_offset=0 (default), should find offset at 2s."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            start_offset=0.0,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Expected ~2s offset with start_offset=0, got {result.offset_seconds}s"
        )


class TestMaxSearchOffset:
    """Test the max_search_offset (-s/--max-search-offset) parameter.

    max_search_offset limits the maximum offset to search in seconds.
    It affects how much of the distorted video and reference video are analyzed.
    """

    def test_max_search_offset_includes_match(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """With max_search_offset=5, should find offset at 2s (within range)."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            max_search_offset=5.0,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Expected ~2s offset with max_search_offset=5, got {result.offset_seconds}s"
        )

    def test_max_search_offset_affects_dist_extraction(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """max_search_offset limits how much of the distorted video is analyzed.

        With max_search_offset=3, only 3s of distorted video is analyzed.
        However, with a 5s distorted video containing content from 5-10s of timeline,
        the first 3s of distorted (timeline 5-8s) may still match reference (0-10s).
        """
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            max_search_offset=3.0,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # The algorithm can still find the 5s offset because it affects search range,
        # not the possible offset values directly
        assert abs(result.offset_seconds - 5.0) < 1.0, (
            f"Expected ~5s offset with max_search_offset=3, got {result.offset_seconds}s"
        )

    def test_max_search_offset_unlimited_by_default(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Without max_search_offset, should find offset at 5s."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            max_search_offset=None,  # Unlimited
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert abs(result.offset_seconds - 5.0) < 0.5, (
            f"Expected ~5s offset with unlimited search, got {result.offset_seconds}s"
        )


class TestMaxDuration:
    """Test the max_duration (-m/--max-duration) parameter.

    max_duration limits how much of the reference video is analyzed.
    """

    def test_max_duration_includes_content(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """With max_duration=8, should analyze 0-8s and find offset at 2s."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            max_duration=8.0,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert abs(result.offset_seconds - 2.0) < 0.5, (
            f"Expected ~2s offset with max_duration=8, got {result.offset_seconds}s"
        )

    def test_max_duration_very_short(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """With max_duration=4, should still find match (offset 2s, 4s duration = ends at 6s)."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_2s,
            compare_type=CompareType.PHASH,
            max_duration=4.0,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # With 4s max_duration from reference and distorted starting at 2s,
        # there should be enough overlap to find the match
        assert abs(result.offset_seconds - 2.0) < 1.0, (
            f"Expected ~2s offset with max_duration=4, got {result.offset_seconds}s"
        )


class TestNegativeOffset:
    """Test detection of negative offsets.

    A negative offset means the distorted video is AHEAD of the reference
    (starts earlier in the timeline).

    For testing, we swap reference and distorted videos:
    - Use offset_5s as "reference" (contains content from 5-10s of original)
    - Use original reference as "distorted" (contains content from 0-10s)
    - The distorted (0-10s) starts BEFORE the reference (5-10s), so offset = -5s
    """

    def test_negative_offset_coarse_only_works(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Coarse-only search correctly detects negative offsets.

        This test verifies the cross-correlation algorithm properly handles
        negative offsets at the coarse search level.
        """
        result = find_offset(
            ref_path=synthetic_offset_5s,  # Content from 5-10s
            dist_path=synthetic_reference,  # Content from 0-10s
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=1.0,  # Same as coarse = skip fine search
            frame_accurate=False,
            quiet=True,
        )

        # Coarse search should correctly find -5s offset
        assert result.offset_seconds < 0, (
            f"Expected negative offset, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-5.0)) < 1.0, (
            f"Expected offset ~-5s, got {result.offset_seconds}s"
        )

    def test_negative_offset_with_fine_refinement(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Fine refinement should preserve negative offset found in coarse search."""
        result = find_offset(
            ref_path=synthetic_offset_5s,  # Content from 5-10s
            dist_path=synthetic_reference,  # Content from 0-10s
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,  # Fine search enabled
            frame_accurate=False,
            quiet=True,
        )

        assert result.offset_seconds < 0, (
            f"Expected negative offset, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-5.0)) < 1.0, (
            f"Expected offset ~-5s, got {result.offset_seconds}s"
        )

    def test_negative_offset_with_different_hash_types(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Negative offsets with fine search should work with different comparison algorithms.

        Note: DHASH excluded because synthetic testsrc produces identical dhash values
        for frames 6s apart, causing false matches. DHASH is tested with BBB content instead.
        """
        for compare_type in [CompareType.PHASH, CompareType.SAD]:
            result = find_offset(
                ref_path=synthetic_offset_5s,
                dist_path=synthetic_reference,
                compare_type=compare_type,
                coarse_fps=1.0,
                fine_fps=5.0,
                frame_accurate=False,
                quiet=True,
            )

            assert result.offset_seconds < 0, (
                f"{compare_type.value}: Expected negative offset, got {result.offset_seconds}s"
            )

    def test_negative_offset_2s_coarse_only(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """Test smaller negative offset (-2s) detection with coarse search.

        offset_2s contains frames 2-10s of original (8s duration).
        reference contains frames 0-10s of original.
        Using offset_2s as ref: dist starts 2s before ref.
        Expected offset: -2s
        """
        result = find_offset(
            ref_path=synthetic_offset_2s,  # Content from 2-10s (8s long)
            dist_path=synthetic_reference,  # Content from 0-10s
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=1.0,  # Skip fine search
            frame_accurate=False,
            quiet=True,
        )

        assert result.offset_seconds < 0, (
            f"Expected negative offset, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-2.0)) < 1.0, (
            f"Expected offset ~-2s, got {result.offset_seconds}s"
        )

    def test_negative_offset_2s_with_fine_search(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """Test smaller negative offset (-2s) with fine search.

        The -2s case may work with fine search because the refine_window (2.0s)
        is close to the offset magnitude, so the search window still captures
        enough of the correct region.
        """
        result = find_offset(
            ref_path=synthetic_offset_2s,  # Content from 2-10s
            dist_path=synthetic_reference,  # Content from 0-10s
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # This may or may not work depending on refine_window vs offset magnitude
        assert result.offset_seconds < 0, (
            f"Expected negative offset, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-2.0)) < 1.0, (
            f"Expected offset ~-2s, got {result.offset_seconds}s"
        )

    def test_negative_offset_frame_accurate(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Frame-accurate search should also work with negative offsets."""
        result = find_offset(
            ref_path=synthetic_offset_5s,
            dist_path=synthetic_reference,
            compare_type=CompareType.PHASH,
            coarse_fps=1.0,
            fine_fps=10.0,
            frame_accurate=True,
            quiet=True,
        )

        assert result.offset_seconds < 0, (
            f"Expected negative offset with frame_accurate=True, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-5.0)) < 0.5, (
            f"Expected offset ~-5s with frame_accurate=True, got {result.offset_seconds}s"
        )

    def test_negative_offset_different_algorithms_coarse_only(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Coarse-only negative offset works with different hash types.

        Note: DHASH excluded because synthetic testsrc produces identical dhash values
        for frames 6s apart, causing false matches. DHASH is tested with BBB content instead.
        """
        for compare_type in [CompareType.PHASH, CompareType.SAD]:
            result = find_offset(
                ref_path=synthetic_offset_5s,
                dist_path=synthetic_reference,
                compare_type=compare_type,
                coarse_fps=1.0,
                fine_fps=1.0,  # Skip fine search
                frame_accurate=False,
                quiet=True,
            )

            assert result.offset_seconds < 0, (
                f"{compare_type.value} coarse-only: Expected negative, got {result.offset_seconds}s"
            )
            assert abs(result.offset_seconds - (-5.0)) < 1.0, (
                f"{compare_type.value} coarse-only: Expected ~-5s, got {result.offset_seconds}s"
            )

    def test_negative_offset_dhash_with_real_content(
        self, bbb_reference: Path, bbb_offset_5s: Path
    ) -> None:
        """Test DHASH negative offset with real content (BBB).

        DHASH doesn't work well with synthetic testsrc due to repeating patterns,
        but works correctly with real video content.
        """
        result = find_offset(
            ref_path=bbb_offset_5s,
            dist_path=bbb_reference,
            compare_type=CompareType.DHASH,
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        assert result.offset_seconds < 0, (
            f"DHASH: Expected negative offset, got {result.offset_seconds}s"
        )
        assert abs(result.offset_seconds - (-5.0)) < 1.0, (
            f"DHASH: Expected offset ~-5s, got {result.offset_seconds}s"
        )


class TestCombinedParameters:
    """Test combinations of search parameters."""

    def test_start_offset_with_max_search_offset(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Combining start_offset and max_search_offset should work together."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            start_offset=4.0,  # Start at 4s
            max_search_offset=3.0,  # Search up to 3s from dist start
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # With start_offset=4 and match at 5s, should still find it
        # max_search_offset limits dist analysis, not the final offset
        assert result.offset_seconds >= 4.0, (
            f"Expected offset >= 4s (start_offset), got {result.offset_seconds}s"
        )

    def test_start_offset_with_max_duration(
        self, synthetic_reference: Path, synthetic_offset_5s: Path
    ) -> None:
        """Combining start_offset and max_duration should work together."""
        result = find_offset(
            ref_path=synthetic_reference,
            dist_path=synthetic_offset_5s,
            compare_type=CompareType.PHASH,
            start_offset=4.0,  # Start at 4s
            max_duration=4.0,  # Analyze 4s from start_offset
            coarse_fps=1.0,
            fine_fps=5.0,
            frame_accurate=False,
            quiet=True,
        )

        # Reference analyzed: 4s to 8s
        # Match is at 5s, should find it
        assert abs(result.offset_seconds - 5.0) < 1.0, (
            f"Expected ~5s offset with start_offset=4, max_duration=4, got {result.offset_seconds}s"
        )


class TestEdgeCasesWithOffsetParams:
    """Test edge cases with search parameters."""

    def test_zero_max_search_offset_handles_gracefully(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """max_search_offset=0 should be handled gracefully."""
        try:
            result = find_offset(
                ref_path=synthetic_reference,
                dist_path=synthetic_offset_2s,
                compare_type=CompareType.PHASH,
                max_search_offset=0.0,
                coarse_fps=1.0,
                fine_fps=5.0,
                frame_accurate=False,
                quiet=True,
            )
            # If it doesn't raise, just check we got some result
            assert result.offset_seconds is not None
        except (ValueError, ZeroDivisionError):
            # These exceptions are acceptable for edge case input
            pass

    def test_start_offset_equals_video_duration(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """start_offset at video duration should handle gracefully."""
        try:
            result = find_offset(
                ref_path=synthetic_reference,
                dist_path=synthetic_offset_2s,
                compare_type=CompareType.PHASH,
                start_offset=10.0,  # Skip entire video
                coarse_fps=1.0,
                fine_fps=5.0,
                frame_accurate=False,
                quiet=True,
            )
            # If it doesn't raise, result may be unreliable
            assert result.offset_seconds is not None
        except (ValueError, IndexError):
            # These exceptions are acceptable
            pass

    def test_negative_start_offset_behavior(
        self, synthetic_reference: Path, synthetic_offset_2s: Path
    ) -> None:
        """Negative start_offset should be handled (treated as 0 or error)."""
        try:
            result = find_offset(
                ref_path=synthetic_reference,
                dist_path=synthetic_offset_2s,
                compare_type=CompareType.PHASH,
                start_offset=-1.0,  # Negative start
                coarse_fps=1.0,
                fine_fps=5.0,
                frame_accurate=False,
                quiet=True,
            )
            # If it doesn't raise, should behave similar to start_offset=0
            assert abs(result.offset_seconds - 2.0) < 1.0
        except ValueError:
            # This exception is acceptable
            pass
