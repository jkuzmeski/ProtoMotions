"""Motion packaging stage helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from protomotions.components.motion_lib import MotionLib, MotionLibConfig


def create_motion_manifest(
    *,
    motion_files: list[Path],
    output_file: Path,
    fps: int,
) -> Path:
    """Create a reproducible YAML manifest for a packaged motion library."""
    motions = []
    for motion_path in sorted(motion_files):
        motion = torch.load(motion_path, map_location="cpu", weights_only=False)
        num_frames = int(motion["rigid_body_pos"].shape[0])
        duration = (num_frames - 1) / float(fps) if num_frames > 0 else 0.0
        motions.append(
            {
                "file": str(motion_path.resolve().as_posix()),
                "fps": fps,
                "weight": 1.0,
                "sub_motions": [
                    {
                        "idx": 0,
                        "timings": {"start": 0.0, "end": duration},
                        "weight": 1.0,
                    }
                ],
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        yaml.safe_dump({"motions": motions}, sort_keys=False),
        encoding="utf-8",
    )
    return output_file


def package_motion_library(
    *,
    manifest_file: Path,
    output_file: Path,
    device: str = "cpu",
) -> Path:
    """Package motion files into a MotionLib .pt file."""
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(manifest_file)), device=device)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    motion_lib.save_to_file(str(output_file))
    return output_file
