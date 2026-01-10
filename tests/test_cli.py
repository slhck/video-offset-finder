"""Tests for CLI output format."""

import json
import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCliOutputFormat:
    """Test CLI JSON output format validation."""

    @pytest.mark.parametrize(
        "fixture_dir,ref_name,dist_name",
        [
            ("synthetic", "reference.mp4", "offset_2s.mp4"),
            ("bbb", "reference.mp4", "offset_2s.mp4"),
        ],
    )
    def test_json_output_format(
        self, fixture_dir: str, ref_name: str, dist_name: str
    ) -> None:
        """Validate JSON output contains all required fields."""
        ref_path = FIXTURES_DIR / fixture_dir / ref_name
        dist_path = FIXTURES_DIR / fixture_dir / dist_name

        result = subprocess.run(
            [
                "video-offset-finder",
                str(ref_path),
                str(dist_path),
                "--coarse-fps",
                "1",
                "--fine-fps",
                "5",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output = json.loads(result.stdout)

        # Validate top-level fields
        assert "date" in output
        assert "reference" in output
        assert "distorted" in output
        assert "offset_frames" in output
        assert "offset_seconds" in output
        assert "offset_timestamp" in output
        assert "confidence" in output
        assert "fps_used" in output
        assert "method" in output
        assert "settings" in output

        # Validate types
        assert isinstance(output["offset_frames"], int)
        assert isinstance(output["offset_seconds"], (int, float))
        assert isinstance(output["offset_timestamp"], str)
        assert isinstance(output["confidence"], (int, float))

        # Validate timestamp format (HH:MM:SS.mmm)
        ts = output["offset_timestamp"]
        assert len(ts) == 12, f"Timestamp wrong length: {ts}"
        assert ts[2] == ":" and ts[5] == ":" and ts[8] == "."
