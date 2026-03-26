#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Reference-based quality checks for lower-body PyRoki retargeting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
)


app = typer.Typer(pretty_exceptions_enable=False)

BODY_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
]
SOURCE_ORIENTATION_TO_BODY = {
    0: "Pelvis",
    3: "L_Ankle",
    4: "L_Toe",
    7: "R_Ankle",
    8: "R_Toe",
}


@dataclass(frozen=True)
class RetargetQCThresholds:
    max_mean_keypoint_error_m: float = 0.06
    max_peak_keypoint_error_m: float = 0.35
    min_heading_alignment_mean: float = 0.9
    min_heading_alignment_min: float = 0.6
    max_orientation_error_deg: float = 35.0
    enforced_orientation_bodies: list[str] | None = None
    min_contact_rate: float = 0.08
    max_contact_rate_delta: float = 0.15
    max_mean_contact_slip_mps: float = 1.0

    def __post_init__(self) -> None:
        if self.enforced_orientation_bodies is None:
            object.__setattr__(self, "enforced_orientation_bodies", ["Pelvis"])


def load_thresholds_from_config(config_file: Path) -> RetargetQCThresholds:
    """Load retarget QC thresholds from a checked-in config file."""
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"QC config at {config_file} must be a mapping")
    return RetargetQCThresholds(**dict(data.get("retarget") or {}))


def _load_keypoint_data(keypoint_file: Path) -> dict[str, np.ndarray]:
    data = np.load(keypoint_file, allow_pickle=True)
    if data.ndim == 0:
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported keypoint payload in {keypoint_file}")
    return data


def _load_retarget_qpos(retargeted_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(retargeted_file, allow_pickle=True)
    return (
        np.asarray(data["base_frame_pos"], dtype=np.float32),
        np.asarray(data["base_frame_wxyz"], dtype=np.float32),
        np.asarray(data["joint_angles"], dtype=np.float32),
    )


def _normalize_xy(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = np.asarray(vectors[..., :2], dtype=np.float32)
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-6
    normalized = np.zeros_like(projected)
    normalized[valid] = projected[valid] / norms[valid]
    return normalized, valid


def _heading_alignment(source_rot: np.ndarray, target_rot: np.ndarray) -> tuple[float, float]:
    source_heading, source_valid = _normalize_xy(source_rot[:, :, 0])
    target_heading, target_valid = _normalize_xy(target_rot[:, :, 0])
    valid = source_valid & target_valid
    if not np.any(valid):
        return 1.0, 1.0

    dots = np.sum(source_heading[valid] * target_heading[valid], axis=-1)
    return float(np.mean(dots)), float(np.min(dots))


def _orientation_error_deg(source_rot: np.ndarray, target_rot: np.ndarray) -> float:
    relative = np.matmul(np.swapaxes(target_rot, -1, -2), source_rot)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos_theta))))


def _contact_rate(contacts: np.ndarray) -> float:
    if contacts.ndim == 1:
        return float(np.mean(contacts))
    return float(np.mean(np.max(contacts, axis=1)))


def _mean_contact_slip(
    body_positions: np.ndarray,
    left_contacts: np.ndarray,
    right_contacts: np.ndarray,
    fps: float,
) -> float:
    if body_positions.shape[0] <= 1:
        return 0.0

    speeds = np.linalg.norm(np.diff(body_positions, axis=0), axis=-1) * fps
    left_weights = np.max(left_contacts[1:], axis=1)
    right_weights = np.max(right_contacts[1:], axis=1)

    left_slip = speeds[:, [3, 4]].mean(axis=1)
    right_slip = speeds[:, [7, 8]].mean(axis=1)

    numerator = float((left_slip * left_weights).sum() + (right_slip * right_weights).sum())
    denominator = float(left_weights.sum() + right_weights.sum())
    if denominator <= 1e-6:
        return float("inf")
    return numerator / denominator


def evaluate_retargeted_motion(
    *,
    keypoint_file: Path,
    retargeted_file: Path,
    model_xml: Path,
    thresholds: RetargetQCThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RetargetQCThresholds()
    keypoint_data = _load_keypoint_data(keypoint_file)
    root_pos, root_rot_wxyz, joint_angles = _load_retarget_qpos(retargeted_file)

    source_positions = np.asarray(keypoint_data["positions"], dtype=np.float32)
    source_orientations = np.asarray(keypoint_data["orientations"], dtype=np.float32)
    left_contacts = np.asarray(keypoint_data["left_foot_contacts"], dtype=np.float32)
    right_contacts = np.asarray(keypoint_data["right_foot_contacts"], dtype=np.float32)

    motion_fps = float(keypoint_data.get("fps", 30))
    motion_len = min(
        len(source_positions),
        len(root_pos),
        len(root_rot_wxyz),
        len(joint_angles),
        len(left_contacts),
        len(right_contacts),
    )
    if motion_len == 0:
        raise ValueError(f"No overlapping frames between {keypoint_file} and {retargeted_file}")

    source_positions = source_positions[:motion_len]
    source_orientations = source_orientations[:motion_len]
    left_contacts = left_contacts[:motion_len]
    right_contacts = right_contacts[:motion_len]
    qpos = torch.cat(
        [
            torch.from_numpy(root_pos[:motion_len]),
            torch.from_numpy(root_rot_wxyz[:motion_len]),
            torch.from_numpy(joint_angles[:motion_len]),
        ],
        dim=-1,
    ).to(dtype=torch.float32)

    kinematic_info = extract_kinematic_info(str(model_xml))
    root_translation, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    world_positions, world_rotations = compute_forward_kinematics_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_translation,
        joint_rot_mats=joint_rot_mats,
    )

    body_indices = [kinematic_info.body_names.index(name) for name in BODY_NAMES]
    world_positions_np = world_positions[:, body_indices].detach().cpu().numpy()
    world_rotations_np = world_rotations[:, body_indices].detach().cpu().numpy()

    position_error = np.linalg.norm(world_positions_np - source_positions, axis=-1)
    per_body_error = {
        name: float(value)
        for name, value in zip(BODY_NAMES, position_error.mean(axis=0), strict=True)
    }

    heading_mean, heading_min = _heading_alignment(
        source_rot=source_orientations[:, 0],
        target_rot=world_rotations_np[:, 0],
    )

    orientation_errors = {}
    for source_idx, body_name in SOURCE_ORIENTATION_TO_BODY.items():
        body_idx = BODY_NAMES.index(body_name)
        orientation_errors[body_name] = _orientation_error_deg(
            source_rot=source_orientations[:, source_idx],
            target_rot=world_rotations_np[:, body_idx],
        )

    left_contact_rate = _contact_rate(left_contacts)
    right_contact_rate = _contact_rate(right_contacts)
    mean_contact_slip = _mean_contact_slip(
        body_positions=world_positions_np,
        left_contacts=left_contacts,
        right_contacts=right_contacts,
        fps=motion_fps,
    )

    failures: list[str] = []
    if float(position_error.mean()) > thresholds.max_mean_keypoint_error_m:
        failures.append("mean_keypoint_error")
    if float(position_error.max()) > thresholds.max_peak_keypoint_error_m:
        failures.append("peak_keypoint_error")
    if heading_mean < thresholds.min_heading_alignment_mean:
        failures.append("heading_alignment_mean")
    if heading_min < thresholds.min_heading_alignment_min:
        failures.append("heading_alignment_min")
    enforced_orientation_errors = {
        name: orientation_errors[name]
        for name in thresholds.enforced_orientation_bodies
        if name in orientation_errors
    }
    if enforced_orientation_errors and (
        max(enforced_orientation_errors.values()) > thresholds.max_orientation_error_deg
    ):
        failures.append("orientation_error")
    if left_contact_rate < thresholds.min_contact_rate:
        failures.append("left_contact_rate")
    if right_contact_rate < thresholds.min_contact_rate:
        failures.append("right_contact_rate")
    if abs(left_contact_rate - right_contact_rate) > thresholds.max_contact_rate_delta:
        failures.append("contact_rate_delta")
    if mean_contact_slip > thresholds.max_mean_contact_slip_mps:
        failures.append("mean_contact_slip")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "keypoint_file": str(keypoint_file),
        "retargeted_file": str(retargeted_file),
        "model_xml": str(model_xml),
        "num_frames": motion_len,
        "fps": motion_fps,
        "metrics": {
            "mean_keypoint_error_m": float(position_error.mean()),
            "peak_keypoint_error_m": float(position_error.max()),
            "per_body_mean_error_m": per_body_error,
            "pelvis_heading_alignment_mean": heading_mean,
            "pelvis_heading_alignment_min": heading_min,
            "orientation_error_deg": orientation_errors,
            "enforced_orientation_error_deg": enforced_orientation_errors,
            "left_contact_rate": left_contact_rate,
            "right_contact_rate": right_contact_rate,
            "contact_rate_delta": abs(left_contact_rate - right_contact_rate),
            "mean_contact_slip_mps": mean_contact_slip,
            "root_height_range_m": float(np.ptp(root_pos[:motion_len, 2])),
        },
        "thresholds": asdict(thresholds),
    }


def save_qc_report(report: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


@app.command()
def main(
    keypoint_file: Path = typer.Argument(..., exists=True, help="Reference keypoint .npy file."),
    retargeted_file: Path = typer.Argument(..., exists=True, help="PyRoki retargeted .npz file."),
    model_xml: Path = typer.Option(..., "--model-xml", exists=True, help="Target MJCF used for FK."),
    qc_config: Path | None = typer.Option(None, "--qc-config", exists=True, help="Optional QC threshold config."),
    report_file: Path | None = typer.Option(None, "--report-file", help="Optional JSON output path."),
    fail_on_error: bool = typer.Option(False, "--fail-on-error", help="Exit non-zero when QC fails."),
) -> None:
    """Evaluate one retargeted motion against its source keypoints."""
    thresholds = load_thresholds_from_config(qc_config) if qc_config is not None else None
    report = evaluate_retargeted_motion(
        keypoint_file=keypoint_file,
        retargeted_file=retargeted_file,
        model_xml=model_xml,
        thresholds=thresholds,
    )
    if report_file is not None:
        save_qc_report(report, report_file)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if fail_on_error and not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
