# Video Offset Finder

[![PyPI version](https://img.shields.io/pypi/v/video-offset-finder.svg)](https://pypi.org/project/video-offset-finder)

[![Python package](https://github.com/slhck/video-offset-finder/actions/workflows/python-package.yml/badge.svg)](https://github.com/slhck/video-offset-finder/actions/workflows/python-package.yml)

Find the temporal offset between two videos using perceptual hashing or direct pixel comparison (SAD).

**Contents:**

- [Why Do We Need This?](#why-do-we-need-this)
- [Requirements and Installation](#requirements-and-installation)
- [Usage](#usage)
- [Output Format](#output-format)
- [How Does It Work?](#how-does-it-work)
  - [Hashing/Comparison Algorithms](#hashingcomparison-algorithms)
  - [Overall Flow](#overall-flow)
  - [Search Parameters Visualized](#search-parameters-visualized)
- [API](#api)
  - [Available Functions](#available-functions)
  - [OffsetResult Fields](#offsetresult-fields)
- [License](#license)

## Why Do We Need This?

I've too often encountered slightly offset video files, which are a pain to sync for calculating full-reference video quality metrics (like VMAF). Based on an earlier, PSNR-based Python script, this is now a fully-featured – and much faster! – tool to find the temporal offset between two videos.

This tool is generally useful for:

- Synchronizing videos from different sources
- A/V sync analysis
- Video quality comparison (aligning reference and test videos)
- Finding where a clip appears in a longer video

The default algorithm uses perceptual hashing and therefore is robust to:

- Different resolutions
- Different quality/compression levels
- Color grading differences
- Minor geometric distortions

## Requirements and Installation

Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uvx video-offset-finder
```

Using [pipx](https://pipx.pypa.io/latest/installation/):

```bash
pipx install video-offset-finder
```

Or, using pip:

```bash
pip install video-offset-finder
```

## Usage

Let's say you have two video files, `reference.mp4` and `distorted.mp4`, and you want to find the temporal offset between them. You can use the command-line tool as follows:

```bash
# Find offset between reference and distorted/delayed video
uvx video-offset-finder reference.mp4 distorted.mp4

# With hints about expected offset (faster)
uvx video-offset-finder ref.mp4 dist.mp4 --start-offset 10 --max-search-offset 15

# Verbose output
uvx video-offset-finder ref.mp4 dist.mp4 -v
```

The tool will output JSON with the detected offset and confidence score. For the output format, see [Output Format](#output-format).

Full usage:

```text
usage: video-offset-finder [-h] [-t {phash,dhash,ahash,whash,sad}]
                           [--hash-size HASH_SIZE] [--coarse-fps COARSE_FPS]
                           [--fine-fps FINE_FPS] [-o START_OFFSET]
                           [-s MAX_SEARCH_OFFSET] [-m MAX_DURATION]
                           [--refine-window REFINE_WINDOW] [-v] [-q] [--version]
                           ref dist

positional arguments:
  ref                   Reference video
  dist                  Distorted/delayed video

options:
  -h, --help            show this help message and exit
  -t, --compare-type {phash,dhash,ahash,whash,sad}
                        Comparison algorithm: phash (default, best quality), dhash
                        (fast), ahash (fastest), whash (most robust), sad (direct
                        pixel comparison)
  --hash-size HASH_SIZE
                        Hash size in bits (default: 16, larger = more precise)
  --coarse-fps COARSE_FPS
                        FPS for coarse search (default: 1.0)
  --fine-fps FINE_FPS   FPS for fine search (default: 10.0)
  -o, --start-offset START_OFFSET
                        Known minimum offset in seconds (default: 0)
  -s, --max-search-offset MAX_SEARCH_OFFSET
                        Maximum offset to search in seconds (default: unlimited)
  -m, --max-duration MAX_DURATION
                        Maximum duration to analyze in seconds (default: unlimited)
  --refine-window REFINE_WINDOW
                        Window size around coarse result for refinement (default: 2.0s)
  -v, --verbose         Enable debug logging
  -q, --quiet           Suppress progress bars and logging (only output JSON)
  --version             show program's version number and exit
```

## Output Format

The tool outputs JSON to stdout:

```json
{
  "date": "2025-01-09T20:15:30.123456",
  "reference": "reference.mp4",
  "distorted": "distorted.mp4",
  "offset_frames": 150,
  "offset_seconds": 5.005,
  "offset_timestamp": "00:00:05.005",
  "confidence": 2.34,
  "fps_used": 29.97,
  "method": "frame_accurate_phash",
  "settings": {
    "compare_type": "phash",
    "hash_size": 16,
    "coarse_fps": 1.0,
    "fine_fps": 10.0,
    "start_offset": 0,
    "max_search_offset": null,
    "max_duration": null,
    "refine_window": 2.0,
    "compute_time": 12.45
  }
}
```

The fields are as follows:

| Field              | Description                                                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `offset_frames`    | Offset in frames (at `fps_used` rate)                                                                                   |
| `offset_seconds`   | Offset in seconds                                                                                                       |
| `offset_timestamp` | Offset in `HH:MM:SS.sss` format                                                                                         |
| `confidence`       | Average distance (lower = better match, 0 = identical). Hamming distance for hash algorithms, SAD for pixel comparison. |
| `fps_used`         | Frame rate used for final measurement                                                                                   |
| `method`           | Algorithm used for final result                                                                                         |
| `compute_time`     | Processing time in seconds                                                                                              |

> [!NOTE]
>
> A **positive offset** means the distorted video is delayed relative to the reference (starts later). A **negative offset** means the distorted video is ahead (starts earlier).

## How Does It Work?

This section explains the frame comparison methods, the overall search algorithm, and visualizes how the search parameters affect the process.

### Hashing/Comparison Algorithms

There are different algorithms available for comparing frames, each with their own trade-offs:

| Algorithm | Speed   | Robustness | Best For                         |
| --------- | ------- | ---------- | -------------------------------- |
| `phash`   | Medium  | High       | General use (default)            |
| `dhash`   | Fast    | Medium     | Fast processing                  |
| `ahash`   | Fastest | Lower      | Very fast estimates              |
| `whash`   | Slowest | Highest    | Difficult comparisons            |
| `sad`     | Fast    | Medium     | Identical/similar quality videos |

The first four are "perceptual hash" algorithms from the ImageHash library:

- **phash** (Perceptual Hash): Applies a [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) (DCT) to capture low-frequency components, similar to JPEG compression. Most robust to scaling and minor edits.
- **dhash** (Difference Hash): Compares the brightness of adjacent pixels horizontally. Fast and effective for detecting shifts/translations.
- **ahash** (Average Hash): Compares each pixel to the average brightness of the image. Simplest and fastest, but less robust to changes.
- **whash** (Wavelet Hash): Uses [Haar wavelet](https://en.wikipedia.org/wiki/Haar_wavelet) decomposition for multi-resolution analysis. Most robust to compression artifacts and color changes.

All hash algorithms reduce an image to a compact binary fingerprint. For more details, see the [ImageHash library documentation](https://github.com/JohannesBuchner/imagehash).

The last algorithm is direct pixel comparison:

- **sad** (Sum of Absolute Differences): Directly compares pixel values between frames after resizing to a common resolution (64x64 grayscale). Computes the sum of absolute differences between corresponding pixels. Fast and effective when videos have similar quality/encoding, but less robust to compression artifacts or color grading differences than perceptual hashes. This will not work when the videos have different resolutions.

### Overall Flow

The tool uses a hierarchical coarse-to-fine search, where each pass computes frame signatures and immediately performs cross-correlation, then uses that result to narrow the search window for the next pass:

1. **Coarse pass** (1 fps): Compute signatures for both videos at low frame rate, find approximate offset via cross-correlation
2. **Fine pass** (10 fps): Compute signatures only within a ±2s window around the coarse result, refine the offset
3. **Frame-accurate pass** (native fps): Compute signatures within a ±0.5s window around the fine result for exact frame matching

This speeds up the process significantly while maintaining accuracy.

Cross-correlation finds the global optimum by computing the total distance (Hamming for hashes, SAD for pixel comparison) at each possible offset, avoiding local minima that can trap simple difference-based approaches.

### Search Parameters Visualized

The following diagrams show how the offset detection and search parameters work.

#### Default Case: Cut Video Within Source

The most common scenario: a shorter "distorted" video is a clip extracted from the longer "reference" video:

```text
Reference (source):
|======================================================|
0s                                                    60s

Distorted (cut):
                    |=================|
                   15s               35s

                    ↑
                    └── offset = 15s (positive: distorted
                        starts later in timeline)

Result: offset_seconds = 15.0
```

#### Negative Offset: Distorted Starts Earlier

When the distorted video contains content that appears before the reference:

```text
Reference:
                    |==============================|
                   10s                            50s

Distorted:
|============================================|
0s                                          40s

↑
└── offset = -10s (negative: distorted starts earlier in timeline)

Result: offset_seconds = -10.0
```

#### Using `--start-offset` to Skip Reference Start

If you know the match is not in the first N seconds of the reference, use `-o/--start-offset` to skip extracting those frames:

```text
Reference (60s total):
|======================================================|
0s                                                    60s

With --start-offset 20, frames extracted from reference:
|xxxxxxxxxxxxxxxxxxxx|=================================|
0s   (not extracted) 20s                              60s

Distorted (20s clip that matches at 30s):
                              |=================|
                             30s               50s

Offset found = 30s
```

Skipping the first 20s of the reference speeds up processing. Matches before 20s in the reference cannot be found.

#### Using `--max-search-offset` to Limit Analysis

Use `-s/--max-search-offset` to limit how much of each video is analyzed, reducing processing time:

```text
Reference (60s), Distorted (20s), --max-search-offset 25:

Reference frames extracted (25s + 20s = 45s):
|==========================================|xxxxxxxxxxx|
0s                                        45s         60s
                                           (not extracted)

Distorted frames extracted (up to 25s, but video is only 20s):
|===================|
0s                 20s  (full distorted used)
```

The algorithm analyzes fewer reference frames, speeding up processing. The cross-correlation still searches all possible alignments between the extracted frame sets.

#### Using `--max-duration` to Limit Analysis Length

Use `-m/--max-duration` to analyze only the first N seconds of the reference:

```text
Reference (60s), --max-duration 30:

Reference frames extracted:
|==============================|xxxxxxxxxxxxxxxxxxxxxxxxxxx|
0s                            30s                         60s
                               (not extracted)
```

This is useful for very long videos when you expect the match to be near the beginning.

## API

Use as a library in your Python code:

```python
from pathlib import Path
from video_offset_finder import find_offset, CompareType

# Basic usage
result = find_offset(
    ref_path=Path("reference.mp4"),
    dist_path=Path("distorted.mp4"),
)
print(f"Offset: {result.offset_seconds:.3f}s ({result.offset_frames} frames)")

# With options (using perceptual hash)
result = find_offset(
    ref_path=Path("reference.mp4"),
    dist_path=Path("distorted.mp4"),
    compare_type=CompareType.DHASH,  # Faster hash algorithm
    coarse_fps=2.0,                  # More samples in coarse pass
    fine_fps=15.0,                   # Higher precision in fine pass
    start_offset=5.0,                # Known minimum offset
    max_search_offset=20.0,          # Limit search range
    max_duration=60.0,               # Only analyze first 60s
    frame_accurate=True,             # Final pass at native FPS
    quiet=True,                      # Suppress progress bars
)

# Using SAD (direct pixel comparison)
result = find_offset(
    ref_path=Path("reference.mp4"),
    dist_path=Path("distorted.mp4"),
    compare_type=CompareType.SAD,    # Sum of Absolute Differences
)
```

### Available Functions

```python
from video_offset_finder import (
    # Main function
    find_offset,

    # Models
    CompareType,    # Enum: PHASH, DHASH, AHASH, WHASH, SAD
    VideoInfo,      # Dataclass with video metadata
    OffsetResult,   # Dataclass with detection result

    # Video utilities
    get_video_info,   # Extract video metadata
    extract_frames,   # Generator yielding (timestamp, PIL.Image) tuples

    # Comparison utilities
    compute_hash,               # Compute perceptual hash for a single image
    compute_sad_signature,      # Compute SAD signature for a single image
    compute_video_signatures,   # Compute signatures for all frames in a video
    cross_correlate_signatures, # Find best alignment between signature sequences
)
```

### OffsetResult Fields

```python
@dataclass
class OffsetResult:
    offset_frames: int    # Offset in frames
    offset_seconds: float # Offset in seconds
    confidence: float     # Distance metric (lower = better)
    fps_used: float       # FPS used for measurement
    method: str           # Algorithm identifier
```

## License

MIT License

Copyright (c) 2025 Werner Robitza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
