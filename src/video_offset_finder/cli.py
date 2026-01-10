"""Command-line interface for video offset finder."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .finder import find_offset
from .models import CompareType


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="""\
Find the temporal offset between two videos using perceptual hashing.

This approach is robust to:
- Different resolutions
- Different quality/compression levels
- Color grading differences

The algorithm uses hierarchical search: first at low FPS to find
approximate offset, then refines at higher FPS.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ref", type=Path, help="Reference video")
    parser.add_argument("dist", type=Path, help="Distorted/delayed video")
    parser.add_argument(
        "-t",
        "--compare-type",
        type=CompareType,
        choices=list(CompareType),
        default=CompareType.PHASH,
        metavar="{phash,dhash,ahash,whash,sad}",
        help="Comparison algorithm: phash (default, best quality), dhash (fast), "
        "ahash (fastest), whash (most robust), sad (direct pixel comparison)",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="Hash size in bits (default: 16, larger = more precise)",
    )
    parser.add_argument(
        "--coarse-fps",
        type=float,
        default=1.0,
        help="FPS for coarse search (default: 1.0)",
    )
    parser.add_argument(
        "--fine-fps",
        type=float,
        default=10.0,
        help="FPS for fine search (default: 10.0)",
    )
    parser.add_argument(
        "-o",
        "--start-offset",
        type=float,
        default=0,
        help="Known minimum offset in seconds (default: 0)",
    )
    parser.add_argument(
        "-s",
        "--max-search-offset",
        type=float,
        help="Maximum offset to search in seconds (default: unlimited)",
    )
    parser.add_argument(
        "-m",
        "--max-duration",
        type=float,
        help="Maximum duration to analyze in seconds (default: unlimited)",
    )
    parser.add_argument(
        "--refine-window",
        type=float,
        default=2.0,
        help="Window size around coarse result for refinement (default: 2.0s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.ref.exists():
        logging.error(f"Reference video not found: {args.ref}")
        sys.exit(1)
    if not args.dist.exists():
        logging.error(f"Distorted video not found: {args.dist}")
        sys.exit(1)

    start_time = datetime.now()

    result = find_offset(
        ref_path=args.ref,
        dist_path=args.dist,
        compare_type=args.compare_type,
        hash_size=args.hash_size,
        coarse_fps=args.coarse_fps,
        fine_fps=args.fine_fps,
        start_offset=args.start_offset,
        max_search_offset=args.max_search_offset,
        max_duration=args.max_duration,
        refine_window=args.refine_window,
    )

    end_time = datetime.now()
    compute_time = (end_time - start_time).total_seconds()

    logging.debug(f"Computation finished in {compute_time:.2f} seconds")
    logging.debug(
        f"Found offset: {result.offset_seconds:.3f}s ({result.offset_frames} frames)"
    )

    # Output JSON result
    output = {
        "date": datetime.now().isoformat(),
        "reference": str(args.ref),
        "distorted": str(args.dist),
        "offset_frames": result.offset_frames,
        "offset_seconds": result.offset_seconds,
        "confidence": result.confidence,
        "fps_used": result.fps_used,
        "method": result.method,
        "settings": {
            "compare_type": args.compare_type.value,
            "hash_size": args.hash_size,
            "coarse_fps": args.coarse_fps,
            "fine_fps": args.fine_fps,
            "start_offset": args.start_offset,
            "max_search_offset": args.max_search_offset,
            "max_duration": args.max_duration,
            "refine_window": args.refine_window,
            "compute_time": compute_time,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
