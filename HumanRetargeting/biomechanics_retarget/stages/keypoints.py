"""Keypoint extraction stage helpers."""

from __future__ import annotations

from pathlib import Path

from HumanRetargeting.biomechanics_retarget.extract_keypoints_from_overground import (
    extract_keypoints_for_retargeting,
)


def run_keypoint_extraction(
    *,
    input_file: Path,
    output_file: Path,
    fps: int,
    output_fps: int,
) -> Path:
    """Extract retargeter-compatible keypoints for one overground motion."""
    extract_keypoints_for_retargeting(
        input_file=input_file,
        output_file=output_file,
        fps=fps,
        output_fps=output_fps,
    )
    return output_file
