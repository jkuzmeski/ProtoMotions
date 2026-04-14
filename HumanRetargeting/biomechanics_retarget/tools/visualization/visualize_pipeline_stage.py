#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Standalone matplotlib viewer for stage-by-stage pipeline comparisons."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import typer
import yaml

from HumanRetargeting.biomechanics_retarget.pipeline_visualization import (
    ensure_interactive_matplotlib_backend,
    load_keypoint_positions,
    load_motion_body_positions,
    load_packaged_motion_body_positions,
    load_retargeted_body_positions,
    resample_positions,
    show_stage_comparison,
)


app = typer.Typer(pretty_exceptions_enable=False)


class VisualizationStage(str, Enum):
    """Supported pairwise pipeline comparisons."""

    FULL = "full"
    OVERGROUND_KEYPOINTS = "overground-keypoints"
    KEYPOINTS_RETARGET = "keypoints-retarget"
    RETARGET_MOTION = "retarget-motion"
    MOTION_PACKAGE = "motion-package"


def _load_run_profile_data(processed_dir: Path) -> dict:
    profile_path = processed_dir / "profile.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Expected run profile at {profile_path}. Re-run the pipeline or pass --model-xml."
        )
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Run profile at {profile_path} must be a mapping")
    return data


def _load_run_subject_id(processed_dir: Path) -> str:
    data = _load_run_profile_data(processed_dir)
    subject_id = data.get("subject_id")
    if not subject_id:
        raise ValueError(f"Run profile at {processed_dir / 'profile.yaml'} does not define subject_id")
    return str(subject_id)


def _resolve_model_xml(processed_dir: Path, model_xml: Path | None) -> Path:
    if model_xml is not None:
        return model_xml.resolve()
    subject_id = _load_run_subject_id(processed_dir)
    inferred = (
        REPO_ROOT
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
        / f"smpl_humanoid_lower_body_subject_{subject_id}.xml"
    )
    if not inferred.exists():
        raise FileNotFoundError(
            f"Could not infer model XML for subject {subject_id}: {inferred}"
        )
    return inferred


def _load_run_fps(processed_dir: Path) -> int:
    data = _load_run_profile_data(processed_dir)
    fps = data.get("fps")
    if fps is None:
        raise ValueError(f"Run profile at {processed_dir / 'profile.yaml'} does not define fps")
    return int(fps)


def _load_overground_positions(overground_file: Path, processed_dir: Path) -> tuple[np.ndarray, int]:
    positions = np.asarray(np.load(overground_file, allow_pickle=True), dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"Expected overground positions with shape [T, N, 3], got {positions.shape}"
        )
    return positions, _load_run_fps(processed_dir)


def _resolve_packaged_file(processed_dir: Path) -> Path:
    packaged_files = sorted((processed_dir / "packaged_data").glob("*.pt"))
    if len(packaged_files) != 1:
        raise FileNotFoundError(
            "Expected exactly one packaged .pt file in packaged_data. "
            f"Found {len(packaged_files)} in {processed_dir / 'packaged_data'}."
        )
    return packaged_files[0]


def _find_motion_index(packaged_file: Path, motion_file: Path) -> int:
    package = torch.load(packaged_file, map_location="cpu", weights_only=False)
    target = str(motion_file.resolve())
    motion_files = list(package["motion_files"])
    if target not in motion_files:
        raise ValueError(f"{motion_file.name} is not present in {packaged_file.name}")
    return motion_files.index(target)


def _compare_overground_keypoints(
    *,
    processed_dir: Path,
    trial: str,
    seconds: float,
    start_sec: float,
) -> None:
    overground_file = processed_dir / "overground_data" / f"{trial}.npy"
    keypoint_file = processed_dir / "keypoints" / f"{trial}.npy"
    overground_positions, overground_fps = _load_overground_positions(overground_file, processed_dir)
    keypoint_positions, keypoint_fps = load_keypoint_positions(keypoint_file)
    overground_positions = resample_positions(
        overground_positions,
        source_fps=overground_fps,
        target_fps=keypoint_fps,
    )
    show_stage_comparison(
        before_positions=overground_positions,
        after_positions=keypoint_positions,
        before_label="Overground",
        after_label="Keypoints",
        stage_name="Overground to Keypoints",
        motion_name=trial,
        fps=keypoint_fps,
        seconds=seconds,
        start_sec=start_sec,
    )


def _compare_keypoints_retarget(
    *,
    processed_dir: Path,
    trial: str,
    model_xml: Path,
    seconds: float,
    start_sec: float,
) -> None:
    keypoint_file = processed_dir / "keypoints" / f"{trial}.npy"
    retargeted_file = processed_dir / "retargeted_motions" / f"{trial}_retargeted.npz"
    keypoint_positions, fps = load_keypoint_positions(keypoint_file)
    retargeted_positions, _ = load_retargeted_body_positions(retargeted_file, model_xml)
    show_stage_comparison(
        before_positions=keypoint_positions,
        after_positions=retargeted_positions,
        before_label="Keypoints",
        after_label="Retargeted FK",
        stage_name="Keypoints to Retarget",
        motion_name=trial,
        fps=fps,
        seconds=seconds,
        start_sec=start_sec,
    )


def _compare_retarget_motion(
    *,
    processed_dir: Path,
    trial: str,
    model_xml: Path,
    seconds: float,
    start_sec: float,
) -> None:
    retargeted_file = processed_dir / "retargeted_motions" / f"{trial}_retargeted.npz"
    motion_file = processed_dir / "motion_files" / f"{trial}.motion"
    retargeted_positions, fps = load_retargeted_body_positions(retargeted_file, model_xml)
    motion_positions, _ = load_motion_body_positions(motion_file, model_xml)
    show_stage_comparison(
        before_positions=retargeted_positions,
        after_positions=motion_positions,
        before_label="Retargeted FK",
        after_label="Motion File",
        stage_name="Retarget to Motion",
        motion_name=trial,
        fps=fps,
        seconds=seconds,
        start_sec=start_sec,
    )


def _compare_motion_package(
    *,
    processed_dir: Path,
    trial: str,
    model_xml: Path,
    seconds: float,
    start_sec: float,
) -> None:
    motion_file = processed_dir / "motion_files" / f"{trial}.motion"
    packaged_file = _resolve_packaged_file(processed_dir)
    motion_index = _find_motion_index(packaged_file, motion_file)
    motion_positions, fps = load_motion_body_positions(motion_file, model_xml)
    packaged_positions, _ = load_packaged_motion_body_positions(
        packaged_file,
        model_xml,
        motion_index,
    )
    show_stage_comparison(
        before_positions=motion_positions,
        after_positions=packaged_positions,
        before_label="Motion File",
        after_label="Packaged MotionLib",
        stage_name="Motion to Package",
        motion_name=trial,
        fps=fps,
        seconds=seconds,
        start_sec=start_sec,
    )


@app.command()
def main(
    processed_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    trial: str = typer.Argument(..., help="Trial stem, for example S02_15ms_Long."),
    stage: VisualizationStage = typer.Option(
        VisualizationStage.FULL,
        "--stage",
        help="Show one pairwise comparison or walk through the full pipeline.",
    ),
    model_xml: Path | None = typer.Option(
        None,
        "--model-xml",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional MJCF override. By default this is inferred from processed_data/<subject>/profile.yaml.",
    ),
    seconds: float = typer.Option(2.0, "--seconds", help="Clip duration to visualize."),
    start_sec: float = typer.Option(0.0, "--start-sec", help="Clip start time in seconds."),
) -> None:
    """Show blocking matplotlib comparisons for one processed pipeline motion."""
    processed_dir = processed_dir.resolve()
    model_xml = _resolve_model_xml(processed_dir, model_xml)
    backend = ensure_interactive_matplotlib_backend()
    typer.echo(f"Using matplotlib backend: {backend}")
    typer.echo(f"Model XML: {model_xml}")

    if stage in {VisualizationStage.FULL, VisualizationStage.OVERGROUND_KEYPOINTS}:
        _compare_overground_keypoints(
            processed_dir=processed_dir,
            trial=trial,
            seconds=seconds,
            start_sec=start_sec,
        )
    if stage in {VisualizationStage.FULL, VisualizationStage.KEYPOINTS_RETARGET}:
        _compare_keypoints_retarget(
            processed_dir=processed_dir,
            trial=trial,
            model_xml=model_xml,
            seconds=seconds,
            start_sec=start_sec,
        )
    if stage in {VisualizationStage.FULL, VisualizationStage.RETARGET_MOTION}:
        _compare_retarget_motion(
            processed_dir=processed_dir,
            trial=trial,
            model_xml=model_xml,
            seconds=seconds,
            start_sec=start_sec,
        )
    if stage in {VisualizationStage.FULL, VisualizationStage.MOTION_PACKAGE}:
        _compare_motion_package(
            processed_dir=processed_dir,
            trial=trial,
            model_xml=model_xml,
            seconds=seconds,
            start_sec=start_sec,
        )


if __name__ == "__main__":
    app()
