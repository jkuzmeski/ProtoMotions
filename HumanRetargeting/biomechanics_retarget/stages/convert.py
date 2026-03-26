"""Motion conversion stage helpers."""

from __future__ import annotations

from pathlib import Path

from HumanRetargeting.biomechanics_retarget.convert_retargeted_to_motion import (
    convert_npz_to_motion,
)


def run_motion_conversion(
    *,
    npz_file: Path,
    output_file: Path,
    model_xml: Path,
    input_fps: int,
    output_fps: int,
    contact_file: Path | None,
    apply_motion_filter: bool,
) -> Path:
    """Convert one retargeted PyRoki motion to ProtoMotions .motion format."""
    success = convert_npz_to_motion(
        npz_file=npz_file,
        output_file=output_file,
        model_xml=model_xml,
        input_fps=input_fps,
        output_fps=output_fps,
        contact_file=contact_file,
        apply_motion_filter=apply_motion_filter,
        height_offset=0.0,
    )
    if not success:
        raise RuntimeError(f"Conversion rejected by motion filter for {npz_file.name}")
    return output_file
