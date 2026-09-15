# Developer Documentation

## Architecture Overview

Structure of the main source code directory:

```
src/video_offset_finder/
├── __init__.py    # Public API exports
├── cli.py         # Command-line interface
├── finder.py      # Main offset-finding algorithm
├── models.py      # Data models and enums
├── hashing.py     # Perceptual hashing and comparison
└── video.py       # Video processing utilities
```

## Default Flow

The search has three steps. For shorter videos, it reads each video once and reuses the frame signatures in all three steps. For longer videos, it reads only the parts needed for each step to keep memory use under control.

```
┌─────────────────────────────┐
│  Reference + Distorted Video│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Decode + signature cache    │
│  ─────────────────────────  │
│ PTS-based CFR sampling      │
│ Decoder-side scaling        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 1: Coarse (1 fps)    │
│  ─────────────────────────  │
│  Select cached signatures   │
│  Cross-correlate            │
│  → Offset ±1s               │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 2: Fine (10 fps)     │
│  ─────────────────────────  │
│  Window ±2s                 │
│  Select cached signatures   │
│  Cross-correlate            │
│  → Offset ±0.1s             │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 3: Native fps        │
│  ─────────────────────────  │
│  Window ±0.5s               │
│  Select cached signatures   │
│  Cross-correlate            │
│  → Frame-accurate           │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│        OffsetResult         │
└─────────────────────────────┘
```

## Cross-Correlation Algorithm

The core matching algorithm compares signatures across all possible temporal offsets:

```
For each offset allowed by the search bounds and minimum overlap:
    1. Determine overlapping frame regions
    2. Compute distance between aligned frames:
       - Hash-based: packed XOR + byte population count
       - SAD-based: Sum of absolute pixel differences
    3. Average distance across overlapping frames
    4. Track best and second-best distance plus overlap length

The compatibility API returns `(best_offset, minimum_distance)`. The detailed
API returns `CorrelationResult` with the runner-up score and overlap length.
```
