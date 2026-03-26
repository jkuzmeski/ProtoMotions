"""Overground conversion stage helpers."""

from __future__ import annotations

from pathlib import Path

from HumanRetargeting.biomechanics_retarget.treadmill2overground import (
    process_motion_file,
)


def run_overground_trial(
    *,
    motion_file: Path,
    output_dir: Path,
    fps: int,
    coordinate_transform: str,
    speed_override: float | None,
) -> Path | None:
    """Run the treadmill-to-overground transform for one motion file."""
    success = process_motion_file(
        motion_file=motion_file,
        output_dir=output_dir,
        fps=fps,
        coordinate_transform=coordinate_transform,
        speed_override=speed_override,
    )
    if not success:
        return None
    output_file = output_dir / f"{motion_file.stem}.npy"
    return output_file if output_file.exists() else None
