"""Utilities for consistent FPS-aware resampling."""

from __future__ import annotations

import numpy as np


def get_resample_indices(num_frames: int, input_fps: int, output_fps: int) -> np.ndarray:
    """Return nearest-neighbor indices for FPS conversion."""
    if num_frames <= 0:
        return np.array([], dtype=int)
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("input_fps and output_fps must be positive")
    if input_fps == output_fps:
        return np.arange(num_frames, dtype=int)

    duration = num_frames / input_fps
    num_output_frames = max(1, int(round(duration * output_fps)))
    indices = np.linspace(0, num_frames - 1, num_output_frames).round().astype(int)
    return np.clip(indices, 0, num_frames - 1)
