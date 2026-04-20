#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Extract keypoints from overground motion data for lower-body retargeting.

This script converts the joint position data from treadmill2overground.py
output into the keypoint format expected by the lower-body retargeting scripts.

The lower-body SMPL humanoid has 9 joints:
    - Pelvis (root)
    - L_Hip, L_Knee, L_Ankle, L_Toe
    - R_Hip, R_Knee, R_Ankle, R_Toe

Output format (compatible with the lower-body retargeter):
    - positions: (T, N_KEYPOINTS, 3) - XYZ coordinates
    - orientations: (T, N_KEYPOINTS, 3, 3) - rotation matrices (estimated)
    - left_foot_contacts: (T, 2) - contact labels for left ankle and toebase
    - right_foot_contacts: (T, 2) - contact labels for right ankle and toebase

Usage:
    python extract_keypoints_from_overground.py input.npy output.npy --fps 200

Author: BioMotions Team
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from scipy.ndimage import binary_closing, binary_opening
try:
    from .fps_utils import get_resample_indices
except ImportError:
    from fps_utils import get_resample_indices

app = typer.Typer(pretty_exceptions_enable=False)

# Joint ordering from treadmill2overground.py
TREADMILL_JOINT_NAMES = [
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

# Mapping to lower-body retargeting keypoint names (9 keypoints for lower body)
PYROKI_KEYPOINT_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
]

# Indices into TREADMILL_JOINT_NAMES for convenience.
_IDX_PELVIS = 0
_IDX_L_HIP = 1
_IDX_L_KNEE = 2
_IDX_L_ANKLE = 3
_IDX_L_TOE = 4
_IDX_R_HIP = 5
_IDX_R_KNEE = 6
_IDX_R_ANKLE = 7
_IDX_R_TOE = 8


def extract_anthropometry_from_keypoints(
    keypoint_files: list[Path],
) -> dict[str, float]:
    """Extract segment lengths from finished keypoint .npy files.

    Averages across all provided trials to get robust estimates of
    pelvis width, thigh / shank / foot lengths per side.  The returned
    dictionary uses the same keys expected by ``SubjectProfile`` so it
    can be fed directly into a profile YAML.
    """
    thighs_l, thighs_r = [], []
    shanks_l, shanks_r = [], []
    feet_l, feet_r = [], []
    pelvis_widths = []

    for kf in keypoint_files:
        data = np.load(kf, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
        positions = np.asarray(data["positions"], dtype=np.float32)

        thighs_l.append(float(np.linalg.norm(positions[:, _IDX_L_KNEE] - positions[:, _IDX_L_HIP], axis=1).mean()))
        thighs_r.append(float(np.linalg.norm(positions[:, _IDX_R_KNEE] - positions[:, _IDX_R_HIP], axis=1).mean()))
        shanks_l.append(float(np.linalg.norm(positions[:, _IDX_L_ANKLE] - positions[:, _IDX_L_KNEE], axis=1).mean()))
        shanks_r.append(float(np.linalg.norm(positions[:, _IDX_R_ANKLE] - positions[:, _IDX_R_KNEE], axis=1).mean()))
        feet_l.append(float(np.linalg.norm(positions[:, _IDX_L_TOE] - positions[:, _IDX_L_ANKLE], axis=1).mean()))
        feet_r.append(float(np.linalg.norm(positions[:, _IDX_R_TOE] - positions[:, _IDX_R_ANKLE], axis=1).mean()))
        pelvis_widths.append(float(np.linalg.norm(positions[:, _IDX_L_HIP] - positions[:, _IDX_R_HIP], axis=1).mean()))

    thigh = float(np.mean(thighs_l + thighs_r))
    shank = float(np.mean(shanks_l + shanks_r))
    # foot_length_m is the forward (X) reach used by the asset scaler.
    # Approximate from the 3D ankle-to-toe distance using the base model's
    # toe Z-drop ratio: z_drop/foot_3d ≈ 0.055/0.135 ≈ 0.407.
    foot_3d = float(np.mean(feet_l + feet_r))
    foot_z_ratio = 0.055 / 0.135  # base model proportions
    foot_length = float(np.sqrt(max(foot_3d**2 - (foot_z_ratio * foot_3d) ** 2, 0.0)))

    foot_3d_l = float(np.mean(feet_l))
    foot_3d_r = float(np.mean(feet_r))
    foot_length_l = float(np.sqrt(max(foot_3d_l**2 - (foot_z_ratio * foot_3d_l) ** 2, 0.0)))
    foot_length_r = float(np.sqrt(max(foot_3d_r**2 - (foot_z_ratio * foot_3d_r) ** 2, 0.0)))

    return {
        "pelvis_width_m": round(float(np.mean(pelvis_widths)), 4),
        "thigh_length_m": round(thigh, 4),
        "shank_length_m": round(shank, 4),
        "foot_length_m": round(foot_length, 4),
        "left_thigh_length_m": round(float(np.mean(thighs_l)), 4),
        "right_thigh_length_m": round(float(np.mean(thighs_r)), 4),
        "left_shank_length_m": round(float(np.mean(shanks_l)), 4),
        "right_shank_length_m": round(float(np.mean(shanks_r)), 4),
        "left_foot_length_m": round(foot_length_l, 4),
        "right_foot_length_m": round(foot_length_r, 4),
    }


def calculate_kinematics(
    positions: np.ndarray, fps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate velocities and accelerations from positions."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    time_delta = 1.0 / fps
    velocities = np.gradient(positions, time_delta, axis=0)
    accelerations = np.gradient(velocities, time_delta, axis=0)
    return velocities, accelerations


def detect_foot_contacts(
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    foot_accelerations: np.ndarray,
) -> np.ndarray:
    """
    Detect support phases using adaptive per-joint clearance and speed thresholds.

    The lower-body treadmill dataset keeps ankle joints several centimeters above the
    floor even during stance, so absolute world-height thresholds are brittle. Detect
    contact from each joint's own clearance envelope plus low horizontal/vertical
    speed windows instead.

    Returns:
        stance_mask: Boolean array of shape (T,) indicating stance phase
    """
    del foot_accelerations

    clearance = foot_positions[:, 2] - np.percentile(foot_positions[:, 2], 1.0)
    horizontal_speed = np.linalg.norm(foot_velocities[:, :2], axis=1)
    vertical_speed = np.abs(foot_velocities[:, 2])

    height_threshold = max(
        0.012,
        min(0.08, float(np.percentile(clearance, 20.0) + 0.008)),
    )
    horizontal_speed_threshold = max(
        0.2,
        min(1.0, float(np.percentile(horizontal_speed, 15.0) + 0.15)),
    )
    vertical_speed_threshold = max(
        0.08,
        min(0.5, float(np.percentile(vertical_speed, 20.0) + 0.05)),
    )

    stance_mask = (
        (clearance <= height_threshold)
        & (horizontal_speed <= horizontal_speed_threshold)
        & (vertical_speed <= vertical_speed_threshold)
    )

    # Prefer broad stance windows to isolated spikes; PyRoki treats support as a
    # trajectory-level constraint rather than an instantaneous event.
    stance_mask = binary_closing(stance_mask, structure=np.ones(7))
    stance_mask = binary_opening(stance_mask, structure=np.ones(3))
    return stance_mask


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return fallback.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def _project_ground(vector: np.ndarray) -> np.ndarray:
    projected = np.asarray(vector, dtype=np.float32).copy()
    projected[2] = 0.0
    return projected


def _orthonormalize_basis(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    x_axis = _normalize(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
    y_axis = _normalize(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize(z_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis), y_axis)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)


def estimate_orientations(
    positions: np.ndarray,
) -> np.ndarray:
    """
    Estimate joint orientations from position data.
    
    For lower-body retargeting, we primarily need:
    - Pelvis orientation (from hip positions)
    - Foot orientations (estimated from ankle-toe vectors)
    
    Returns:
        orientations: (T, N_joints, 3, 3) rotation matrices
    """
    n_frames, n_joints, _ = positions.shape
    orientations = np.zeros((n_frames, n_joints, 3, 3))
    
    # Initialize all as identity
    for t in range(n_frames):
        for j in range(n_joints):
            orientations[t, j] = np.eye(3)
    
    # Canonical source frame:
    # - x: forward
    # - y: left
    # - z: up
    l_hip_idx = TREADMILL_JOINT_NAMES.index("L_Hip")
    r_hip_idx = TREADMILL_JOINT_NAMES.index("R_Hip")
    pelvis_idx = TREADMILL_JOINT_NAMES.index("Pelvis")
    l_ankle_idx = TREADMILL_JOINT_NAMES.index("L_Ankle")
    l_toe_idx = TREADMILL_JOINT_NAMES.index("L_Toe")
    r_ankle_idx = TREADMILL_JOINT_NAMES.index("R_Ankle")
    r_toe_idx = TREADMILL_JOINT_NAMES.index("R_Toe")

    pelvis_positions = positions[:, pelvis_idx]
    pelvis_velocities = np.gradient(pelvis_positions, axis=0)

    prev_pelvis_basis = np.eye(3, dtype=np.float32)
    for t in range(n_frames):
        l_hip = positions[t, l_hip_idx]
        r_hip = positions[t, r_hip_idx]
        lateral_axis = _project_ground(l_hip - r_hip)
        lateral_axis = _normalize(lateral_axis, prev_pelvis_basis[:, 1])

        heading_hint = _project_ground(pelvis_velocities[t])
        left_foot_heading = _project_ground(positions[t, l_toe_idx] - positions[t, l_ankle_idx])
        right_foot_heading = _project_ground(positions[t, r_toe_idx] - positions[t, r_ankle_idx])
        heading_hint = heading_hint + 0.5 * (left_foot_heading + right_foot_heading)

        x_axis = np.cross(lateral_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        x_axis = _normalize(x_axis, prev_pelvis_basis[:, 0])
        if np.dot(x_axis, heading_hint) < 0.0:
            x_axis = -x_axis
            lateral_axis = -lateral_axis

        rot_matrix = _orthonormalize_basis(
            x_axis=x_axis,
            y_axis=lateral_axis,
            z_axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        if np.dot(rot_matrix[:, 0], prev_pelvis_basis[:, 0]) < 0.0:
            rot_matrix[:, :2] *= -1.0

        orientations[t, pelvis_idx] = rot_matrix
        prev_pelvis_basis = rot_matrix

    # Estimate foot orientations from ankle-toe vectors in the same x-forward frame.
    prev_left_basis = prev_pelvis_basis.copy()
    prev_right_basis = prev_pelvis_basis.copy()
    for t in range(n_frames):
        l_ankle = positions[t, l_ankle_idx]
        l_toe = positions[t, l_toe_idx]
        r_ankle = positions[t, r_ankle_idx]
        r_toe = positions[t, r_toe_idx]

        pelvis_basis = orientations[t, pelvis_idx]

        l_forward = _project_ground(l_toe - l_ankle)
        l_forward = _normalize(l_forward, pelvis_basis[:, 0])
        if np.dot(l_forward, pelvis_basis[:, 0]) < 0.0:
            l_forward = -l_forward
        l_basis = _orthonormalize_basis(
            x_axis=l_forward,
            y_axis=np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), l_forward),
            z_axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        if np.dot(l_basis[:, 0], prev_left_basis[:, 0]) < 0.0:
            l_basis[:, :2] *= -1.0
        orientations[t, l_ankle_idx] = l_basis
        orientations[t, l_toe_idx] = l_basis
        prev_left_basis = l_basis

        r_forward = _project_ground(r_toe - r_ankle)
        r_forward = _normalize(r_forward, pelvis_basis[:, 0])
        if np.dot(r_forward, pelvis_basis[:, 0]) < 0.0:
            r_forward = -r_forward
        r_basis = _orthonormalize_basis(
            x_axis=r_forward,
            y_axis=np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), r_forward),
            z_axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        if np.dot(r_basis[:, 0], prev_right_basis[:, 0]) < 0.0:
            r_basis[:, :2] *= -1.0
        orientations[t, r_ankle_idx] = r_basis
        orientations[t, r_toe_idx] = r_basis
        prev_right_basis = r_basis

    return orientations


def extract_foot_contacts(
    positions: np.ndarray,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract foot contact labels from position data.
    
    Returns:
        left_foot_contacts: (T, 2) - contacts for ankle and toebase
        right_foot_contacts: (T, 2) - contacts for ankle and toebase
    """
    n_frames = positions.shape[0]
    
    # Get foot positions
    l_ankle_idx = TREADMILL_JOINT_NAMES.index("L_Ankle")
    l_toe_idx = TREADMILL_JOINT_NAMES.index("L_Toe")
    r_ankle_idx = TREADMILL_JOINT_NAMES.index("R_Ankle")
    r_toe_idx = TREADMILL_JOINT_NAMES.index("R_Toe")
    
    l_ankle_pos = positions[:, l_ankle_idx, :]
    l_toe_pos = positions[:, l_toe_idx, :]
    r_ankle_pos = positions[:, r_ankle_idx, :]
    r_toe_pos = positions[:, r_toe_idx, :]
    
    # Calculate kinematics
    l_ankle_vel, l_ankle_acc = calculate_kinematics(l_ankle_pos, fps)
    l_toe_vel, l_toe_acc = calculate_kinematics(l_toe_pos, fps)
    r_ankle_vel, r_ankle_acc = calculate_kinematics(r_ankle_pos, fps)
    r_toe_vel, r_toe_acc = calculate_kinematics(r_toe_pos, fps)
    
    # Detect joint-level support signals.
    l_ankle_contact = detect_foot_contacts(l_ankle_pos, l_ankle_vel, l_ankle_acc)
    l_toe_contact = detect_foot_contacts(l_toe_pos, l_toe_vel, l_toe_acc)
    r_ankle_contact = detect_foot_contacts(r_ankle_pos, r_ankle_vel, r_ankle_acc)
    r_toe_contact = detect_foot_contacts(r_toe_pos, r_toe_vel, r_toe_acc)

    # The ankle joint never reaches the floor in this dataset, but during stance the
    # whole foot should be stabilized. Promote the foot-level support phase to both
    # ankle and toe channels so downstream retargeting penalizes slip on the full foot.
    left_foot_contact = binary_closing(
        np.logical_or(l_ankle_contact, l_toe_contact), structure=np.ones(5)
    )
    left_foot_contact = binary_opening(left_foot_contact, structure=np.ones(3))

    right_foot_contact = binary_closing(
        np.logical_or(r_ankle_contact, r_toe_contact), structure=np.ones(5)
    )
    right_foot_contact = binary_opening(right_foot_contact, structure=np.ones(3))

    left_foot_contacts = np.repeat(left_foot_contact[:, None], 2, axis=1).astype(float)
    right_foot_contacts = np.repeat(right_foot_contact[:, None], 2, axis=1).astype(float)
    
    return left_foot_contacts, right_foot_contacts


def extract_keypoints_for_retargeting(
    input_file: Path,
    output_file: Path,
    fps: int = 200,
    output_fps: int = 30,
) -> None:
    """
    Extract keypoints from overground motion data for lower-body retargeting.
    
    Args:
        input_file: Path to input .npy file from treadmill2overground.py
        output_file: Path to output .npy file for retargeting
        fps: Input motion capture frame rate
        output_fps: Output frame rate for retargeting
    """
    print(f"Loading motion from: {input_file}")
    positions = np.load(input_file)
    
    n_frames_orig, n_joints, _ = positions.shape
    print(f"Loaded {n_frames_orig} frames, {n_joints} joints")
    
    # Resample if needed.
    if fps != output_fps:
        indices = get_resample_indices(n_frames_orig, fps, output_fps)
        positions = positions[indices]
        print(f"Resampled {n_frames_orig} -> {positions.shape[0]} frames at {output_fps} Hz")
    
    n_frames = positions.shape[0]
    
    # Estimate orientations
    print("Estimating joint orientations...")
    orientations = estimate_orientations(positions)
    
    # Extract foot contacts at original FPS, then downsample
    print("Extracting foot contacts...")
    positions_orig = np.load(input_file)  # Reload at original FPS
    left_contacts, right_contacts = extract_foot_contacts(positions_orig, fps)
    
    # Resample contacts to the same target timestamps.
    if fps != output_fps:
        contact_indices = get_resample_indices(len(left_contacts), fps, output_fps)
        left_contacts = left_contacts[contact_indices]
        right_contacts = right_contacts[contact_indices]
    
    # Ensure same length
    min_len = min(n_frames, len(left_contacts), len(right_contacts))
    positions = positions[:min_len]
    orientations = orientations[:min_len]
    left_contacts = left_contacts[:min_len]
    right_contacts = right_contacts[:min_len]
    
    # Save in retargeter-compatible format
    output_data = {
        "fps": output_fps,
        "source_fps": fps,
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": left_contacts,
        "right_foot_contacts": right_contacts,
    }
    
    print(f"Saving keypoints to: {output_file}")
    print(f"  - positions: {positions.shape}")
    print(f"  - orientations: {orientations.shape}")
    print(f"  - left_foot_contacts: {left_contacts.shape}")
    print(f"  - right_foot_contacts: {right_contacts.shape}")
    
    np.save(output_file, output_data)


def extract_keypoints_for_pyroki(
    input_file: Path,
    output_file: Path,
    fps: int = 200,
    output_fps: int = 30,
) -> None:
    """Backward-compatible alias for the old retarget entrypoint name."""
    extract_keypoints_for_retargeting(
        input_file=input_file,
        output_file=output_file,
        fps=fps,
        output_fps=output_fps,
    )


@app.command()
def main(
    input_file: Path = typer.Argument(
        ..., exists=True, help="Input .npy file from treadmill2overground.py"
    ),
    output_file: Path = typer.Argument(
        ..., help="Output .npy file for lower-body retargeting"
    ),
    fps: int = typer.Option(200, "--fps", "-f", help="Input frame rate (Hz)"),
    output_fps: int = typer.Option(30, "--output-fps", help="Output frame rate (Hz)"),
):
    """
    Extract keypoints from overground motion data for lower-body retargeting.
    
    Converts joint position data from treadmill2overground.py into the format
    expected by the lower-body retargeting scripts.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    extract_keypoints_for_retargeting(
        input_file=input_file,
        output_file=output_file,
        fps=fps,
        output_fps=output_fps,
    )
    
    print("✅ Keypoint extraction complete!")


if __name__ == "__main__":
    app()
