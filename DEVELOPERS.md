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

The algorithm uses a three-phase hierarchical search to efficiently find the temporal offset between two videos:

```
┌─────────────────────────────┐
│  Reference + Distorted Video│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 1: Coarse (1 fps)    │
│  ─────────────────────────  │
│  Extract frames             │
│  Compute signatures         │
│  Cross-correlate            │
│  → Offset ±1s               │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 2: Fine (10 fps)     │
│  ─────────────────────────  │
│  Window ±2s                 │
│  Extract frames             │
│  Compute signatures         │
│  Cross-correlate            │
│  → Offset ±0.1s             │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Phase 3: Native fps        │
│  ─────────────────────────  │
│  Window ±0.5s               │
│  Extract frames             │
│  Compute signatures         │
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
For each offset from (-n_dist+1) to n_ref:
    1. Determine overlapping frame regions
    2. Compute distance between aligned frames:
       - Hash-based: Hamming distance (bit differences)
       - SAD-based: Sum of absolute pixel differences
    3. Average distance across overlapping frames
    4. Track offset with minimum average distance

Return: (best_offset, minimum_distance)
```
