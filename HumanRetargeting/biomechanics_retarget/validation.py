"""Contract validation utilities for the production biomechanics pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from HumanRetargeting.biomechanics_retarget.retarget_qc import (
    RetargetQCThresholds,
    evaluate_retargeted_motion,
)
from protomotions.components.pose_lib import extract_kinematic_info


REQUIRED_RETARGET_KEYS = (
    "base_frame_pos",
    "base_frame_wxyz",
    "joint_angles",
    "joint_names",
)
REQUIRED_MOTION_KEYS = (
    "fps",
    "state_conversion",
    "rigid_body_pos",
    "rigid_body_rot",
    "rigid_body_vel",
    "rigid_body_ang_vel",
    "dof_pos",
    "dof_vel",
    "rigid_body_contacts",
)
REQUIRED_PACKAGE_KEYS = (
    "gts",
    "grs",
    "gvs",
    "gavs",
    "dps",
    "dvs",
    "contacts",
    "motion_files",
    "motion_num_frames",
    "motion_dt",
    "motion_lengths",
    "length_starts",
    "motion_weights",
)

JOINT_LIMIT_TOLERANCE_RAD = 2e-5


def load_qc_thresholds(config_file: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"QC config at {config_file} must be a mapping")
    return data


def _decode_joint_names(joint_names: list[Any]) -> list[str]:
    if joint_names and isinstance(joint_names[0], bytes):
        return [name.decode("utf-8") for name in joint_names]
    return [str(name) for name in joint_names]


def validate_retargeted_npz(
    *,
    npz_file: Path,
    keypoint_file: Path,
    model_xml: Path,
    qc_config_file: Path,
) -> dict[str, Any]:
    """Validate one retargeted NPZ against contract and quality rules."""
    data = np.load(npz_file, allow_pickle=True)
    failures: list[str] = []

    missing_keys = [key for key in REQUIRED_RETARGET_KEYS if key not in data]
    if missing_keys:
        failures.append("missing_keys")
        return {
            "passed": False,
            "failures": failures,
            "npz_file": str(npz_file),
            "keypoint_file": str(keypoint_file),
            "model_xml": str(model_xml),
            "metrics": {
                "missing_keys": missing_keys,
            },
        }

    root_pos = np.asarray(data["base_frame_pos"], dtype=np.float32)
    root_rot = np.asarray(data["base_frame_wxyz"], dtype=np.float32)
    joint_angles = np.asarray(data["joint_angles"], dtype=np.float32)
    joint_names = _decode_joint_names(data["joint_names"].tolist())

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        failures.append("invalid_root_pos_shape")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        failures.append("invalid_root_rot_shape")
    if joint_angles.ndim != 2:
        failures.append("invalid_joint_angle_shape")

    num_frames = int(root_pos.shape[0]) if root_pos.ndim == 2 else 0
    if root_rot.shape[0] != num_frames or joint_angles.shape[0] != num_frames:
        failures.append("inconsistent_frame_count")

    kinematic_info = extract_kinematic_info(str(model_xml))
    expected_joint_names = list(kinematic_info.dof_names)
    missing_targets = [
        joint_name for joint_name in expected_joint_names if joint_name not in joint_names
    ]
    unexpected_sources = [
        joint_name for joint_name in joint_names if joint_name not in expected_joint_names
    ]
    if missing_targets or unexpected_sources:
        failures.append("joint_name_mismatch")

    ordered_joint_angles = joint_angles
    if not missing_targets and not unexpected_sources and joint_names != expected_joint_names:
        reorder_indices = [joint_names.index(name) for name in expected_joint_names]
        ordered_joint_angles = joint_angles[:, reorder_indices]

    lower = kinematic_info.dof_limits_lower.numpy()
    upper = kinematic_info.dof_limits_upper.numpy()
    below = ordered_joint_angles < lower[None, :] - JOINT_LIMIT_TOLERANCE_RAD
    above = ordered_joint_angles > upper[None, :] + JOINT_LIMIT_TOLERANCE_RAD
    out_of_limit = np.logical_or(below, above)
    violating_frames = int(out_of_limit.any(axis=1).sum())
    if violating_frames > 0:
        failures.append("joint_limit_violation")

    qc_thresholds = load_qc_thresholds(qc_config_file)
    retarget_thresholds = RetargetQCThresholds(**dict(qc_thresholds.get("retarget") or {}))
    quality_report = evaluate_retargeted_motion(
        keypoint_file=keypoint_file,
        retargeted_file=npz_file,
        model_xml=model_xml,
        thresholds=retarget_thresholds,
    )
    if not quality_report["passed"]:
        failures.append("retarget_quality")

    max_abs_violation = 0.0
    if below.any():
        max_abs_violation = max(
            max_abs_violation,
            float(np.max(lower[None, :] - ordered_joint_angles, where=below, initial=0.0)),
        )
    if above.any():
        max_abs_violation = max(
            max_abs_violation,
            float(np.max(ordered_joint_angles - upper[None, :], where=above, initial=0.0)),
        )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "npz_file": str(npz_file),
        "keypoint_file": str(keypoint_file),
        "model_xml": str(model_xml),
        "metrics": {
            "num_frames": num_frames,
            "expected_num_dofs": len(expected_joint_names),
            "actual_num_dofs": int(joint_angles.shape[1]) if joint_angles.ndim == 2 else 0,
            "violating_frames": violating_frames,
            "max_abs_limit_violation_rad": max_abs_violation,
            "joint_order_matches_target": joint_names == expected_joint_names,
            "missing_target_joints": missing_targets,
            "unexpected_source_joints": unexpected_sources,
        },
        "quality_report": quality_report,
    }


def validate_motion_file(
    *,
    motion_file: Path,
    model_xml: Path,
    qc_config_file: Path,
) -> dict[str, Any]:
    """Validate one .motion file against MotionLib-facing invariants."""
    motion = torch.load(motion_file, map_location="cpu", weights_only=False)
    failures: list[str] = []

    missing_keys = [key for key in REQUIRED_MOTION_KEYS if key not in motion]
    if missing_keys:
        failures.append("missing_keys")
        return {
            "passed": False,
            "failures": failures,
            "motion_file": str(motion_file),
            "model_xml": str(model_xml),
            "metrics": {
                "missing_keys": missing_keys,
            },
        }

    kinematic_info = extract_kinematic_info(str(model_xml))
    dof_pos = motion["dof_pos"]
    dof_vel = motion["dof_vel"]
    lower = kinematic_info.dof_limits_lower
    upper = kinematic_info.dof_limits_upper

    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(kinematic_info.dof_names):
        failures.append("invalid_dof_pos_shape")
    if dof_vel.shape != dof_pos.shape:
        failures.append("invalid_dof_vel_shape")

    violating_frames = int(
        (
            (dof_pos < lower - JOINT_LIMIT_TOLERANCE_RAD)
            | (dof_pos > upper + JOINT_LIMIT_TOLERANCE_RAD)
        ).any(dim=1).sum()
    )
    if violating_frames > 0:
        failures.append("joint_limit_violation")

    max_jump = float((dof_pos[1:] - dof_pos[:-1]).abs().max()) if dof_pos.shape[0] > 1 else 0.0
    motion_thresholds = load_qc_thresholds(qc_config_file)
    max_allowed_jump = float((motion_thresholds.get("motion") or {}).get("max_frame_dof_jump_rad", float(torch.pi)))
    if max_jump > max_allowed_jump + 1e-5:
        failures.append("non_interpolation_safe_dof_jump")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "motion_file": str(motion_file),
        "model_xml": str(model_xml),
        "metrics": {
            "num_frames": int(dof_pos.shape[0]),
            "num_dofs": int(dof_pos.shape[1]),
            "violating_frames": violating_frames,
            "max_frame_dof_jump_rad": max_jump,
            "max_allowed_frame_dof_jump_rad": max_allowed_jump,
        },
    }


def validate_packaged_motion_lib(
    *,
    packaged_file: Path,
    expected_motion_files: list[Path],
) -> dict[str, Any]:
    """Validate one packaged MotionLib against exact ordering and slice equality."""
    package = torch.load(packaged_file, map_location="cpu", weights_only=False)
    failures: list[str] = []

    missing_keys = [key for key in REQUIRED_PACKAGE_KEYS if key not in package]
    if missing_keys:
        failures.append("missing_keys")
        return {
            "passed": False,
            "failures": failures,
            "packaged_file": str(packaged_file),
            "metrics": {
                "missing_keys": missing_keys,
            },
        }

    actual_motion_files = [Path(path) for path in package["motion_files"]]
    expected_resolved = [path.resolve() for path in expected_motion_files]

    def _same_motion_path(left: Path, right: Path) -> bool:
        try:
            return left.samefile(right)
        except FileNotFoundError:
            return False
        except OSError:
            return left.resolve() == right.resolve()

    order_matches = len(actual_motion_files) == len(expected_resolved) and all(
        _same_motion_path(actual, expected)
        for actual, expected in zip(actual_motion_files, expected_resolved, strict=True)
    )
    if not order_matches:
        failures.append("motion_file_order_mismatch")

    slice_mismatches: list[str] = []
    for idx, motion_path in enumerate(actual_motion_files):
        motion = torch.load(motion_path, map_location="cpu", weights_only=False)
        start = int(package["length_starts"][idx])
        frames = int(package["motion_num_frames"][idx])
        end = start + frames

        expected_slices = {
            "gts": motion["rigid_body_pos"],
            "grs": motion["rigid_body_rot"],
            "gvs": motion["rigid_body_vel"],
            "gavs": motion["rigid_body_ang_vel"],
            "dps": motion["dof_pos"],
            "dvs": motion["dof_vel"],
            "contacts": motion["rigid_body_contacts"],
        }
        if any(
            not torch.equal(package[key][start:end], expected)
            for key, expected in expected_slices.items()
        ):
            slice_mismatches.append(motion_path.name)
    if slice_mismatches:
        failures.append("packaged_slice_mismatch")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "packaged_file": str(packaged_file),
        "metrics": {
            "num_motions": len(actual_motion_files),
            "motion_files": [path.name for path in actual_motion_files],
            "slice_mismatch_files": slice_mismatches,
        },
    }


def write_validation_report(report: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def ensure_validation_passed(report: dict[str, Any], failure_message: str) -> None:
    if not report["passed"]:
        raise RuntimeError(failure_message)
