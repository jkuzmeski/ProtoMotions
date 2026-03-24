# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Retarget lower-body biomechanics keypoints into ProtoMotions-compatible NPZ files."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from protomotions.components.pose_lib import (
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_XML = (
    REPO_ROOT
    / "HumanRetargeting"
    / "rescale"
    / "smpl_humanoid_lower_body_adjusted_pd.xml"
)
LOWER_BODY_KEYPOINT_NAMES = [
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
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(LOWER_BODY_KEYPOINT_NAMES)}
KEYPOINT_TO_BODY_NAMES = [
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
SKELETON_EDGES = [
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_BLEND_ALPHAS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


def _resolve_model_xml(urdf_path: str | None) -> Path:
    if urdf_path:
        candidate = (
            REPO_ROOT
            / "protomotions"
            / "data"
            / "assets"
            / "mjcf"
            / f"{Path(urdf_path).stem}.xml"
        )
        if candidate.exists():
            return candidate
    return DEFAULT_MODEL_XML


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return fallback.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    x_axis = _normalize(matrix[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_raw = matrix[:, 1] - x_axis * np.dot(matrix[:, 1], x_axis)
    y_axis = _normalize(y_raw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z_axis = _normalize(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis), y_axis)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)


def _make_segment_rotation(
    segment_vector: np.ndarray,
    x_hint: np.ndarray,
    y_hint: np.ndarray,
) -> np.ndarray:
    z_axis = -_normalize(segment_vector, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    x_proj = x_hint - z_axis * np.dot(x_hint, z_axis)
    if np.linalg.norm(x_proj) < 1e-8:
        x_proj = y_hint - z_axis * np.dot(y_hint, z_axis)
    if np.linalg.norm(x_proj) < 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(basis, z_axis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_proj = basis - z_axis * np.dot(basis, z_axis)

    x_axis = _normalize(x_proj, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    x_axis = _normalize(np.cross(y_axis, z_axis), x_axis)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)


def _compute_global_rotations(positions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    num_frames = positions.shape[0]
    global_rotations = np.zeros((num_frames, len(LOWER_BODY_KEYPOINT_NAMES), 3, 3), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_positions = positions[frame_idx]
        frame_orientations = orientations[frame_idx]

        pelvis_rot = _orthonormalize(frame_orientations[KEYPOINT_INDEX["pelvis"]])
        pelvis_x = pelvis_rot[:, 0]
        pelvis_y = pelvis_rot[:, 1]

        left_hip_rot = _make_segment_rotation(
            frame_positions[KEYPOINT_INDEX["left_knee"]] - frame_positions[KEYPOINT_INDEX["left_hip"]],
            pelvis_x,
            pelvis_y,
        )
        right_hip_rot = _make_segment_rotation(
            frame_positions[KEYPOINT_INDEX["right_knee"]] - frame_positions[KEYPOINT_INDEX["right_hip"]],
            pelvis_x,
            pelvis_y,
        )
        left_knee_rot = _make_segment_rotation(
            frame_positions[KEYPOINT_INDEX["left_ankle"]] - frame_positions[KEYPOINT_INDEX["left_knee"]],
            left_hip_rot[:, 0],
            pelvis_y,
        )
        right_knee_rot = _make_segment_rotation(
            frame_positions[KEYPOINT_INDEX["right_ankle"]] - frame_positions[KEYPOINT_INDEX["right_knee"]],
            right_hip_rot[:, 0],
            pelvis_y,
        )
        left_ankle_rot = _orthonormalize(frame_orientations[KEYPOINT_INDEX["left_ankle"]])
        right_ankle_rot = _orthonormalize(frame_orientations[KEYPOINT_INDEX["right_ankle"]])
        left_toe_rot = _orthonormalize(frame_orientations[KEYPOINT_INDEX["left_foot"]])
        right_toe_rot = _orthonormalize(frame_orientations[KEYPOINT_INDEX["right_foot"]])

        global_rotations[frame_idx] = np.stack(
            [
                pelvis_rot,
                left_hip_rot,
                left_knee_rot,
                left_ankle_rot,
                left_toe_rot,
                right_hip_rot,
                right_knee_rot,
                right_ankle_rot,
                right_toe_rot,
            ],
            axis=0,
        )

    return global_rotations


def _save_contact_labels(output_path: Path, keypoint_data: dict) -> None:
    left_contacts = np.asarray(keypoint_data["left_foot_contacts"], dtype=np.float32)
    right_contacts = np.asarray(keypoint_data["right_foot_contacts"], dtype=np.float32)
    foot_contacts = np.stack(
        [
            np.mean(left_contacts, axis=1),
            np.mean(right_contacts, axis=1),
        ],
        axis=-1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        foot_contacts=foot_contacts,
        left_foot_contacts=left_contacts,
        right_foot_contacts=right_contacts,
    )
    print(f"Saved contact labels to {output_path}")


def _stabilize_root_quaternions(root_quats: torch.Tensor) -> torch.Tensor:
    root_quats = torch.nn.functional.normalize(root_quats, dim=-1)
    if root_quats.shape[0] <= 1:
        return root_quats

    stabilized = root_quats.clone()
    for frame_idx in range(1, stabilized.shape[0]):
        if torch.dot(stabilized[frame_idx - 1], stabilized[frame_idx]) < 0.0:
            stabilized[frame_idx] = -stabilized[frame_idx]
    return stabilized


def _select_continuous_joint_angles(
    joint_angles: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    raw_angles = joint_angles.detach().cpu().numpy().astype(np.float32, copy=False)
    lower = lower_limits.detach().cpu().numpy().astype(np.float32, copy=False)
    upper = upper_limits.detach().cpu().numpy().astype(np.float32, copy=False)

    wrapped = ((raw_angles + np.pi) % (2.0 * np.pi)) - np.pi
    stabilized = np.empty_like(wrapped)
    branch_offsets = np.arange(-2, 3, dtype=np.float32) * (2.0 * np.pi)

    for dof_idx in range(wrapped.shape[1]):
        lower_limit = lower[dof_idx]
        upper_limit = upper[dof_idx]
        initial_target = 0.5 * (lower_limit + upper_limit)

        for frame_idx in range(wrapped.shape[0]):
            base_angle = wrapped[frame_idx, dof_idx]
            candidates = base_angle + branch_offsets
            valid_candidates = candidates[
                (candidates >= lower_limit - 1e-5) & (candidates <= upper_limit + 1e-5)
            ]
            target = initial_target if frame_idx == 0 else stabilized[frame_idx - 1, dof_idx]

            if valid_candidates.size == 0:
                chosen = np.clip(base_angle, lower_limit, upper_limit)
            else:
                chosen = valid_candidates[np.argmin(np.abs(valid_candidates - target))]
            stabilized[frame_idx, dof_idx] = chosen

    stabilized_tensor = torch.from_numpy(stabilized).to(
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    )
    return torch.clamp(stabilized_tensor, min=lower_limits, max=upper_limits)


def _build_initial_qpos(
    positions: np.ndarray,
    orientations: np.ndarray,
    *,
    kinematic_info,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_rotations = _compute_global_rotations(positions, orientations)
    root_pos = torch.from_numpy(positions[:, KEYPOINT_INDEX["pelvis"]]).to(device=device, dtype=dtype)
    global_rotations_torch = torch.from_numpy(global_rotations).to(device=device, dtype=dtype)

    joint_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=global_rotations_torch,
    )
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        multi_dof_decomposition_method="euler_xyz",
    )

    lower = kinematic_info.dof_limits_lower.to(device=device, dtype=dtype)
    upper = kinematic_info.dof_limits_upper.to(device=device, dtype=dtype)
    root_rot_wxyz = _stabilize_root_quaternions(qpos[:, 3:7])
    joint_angles = _select_continuous_joint_angles(
        joint_angles=qpos[:, 7:],
        lower_limits=lower,
        upper_limits=upper,
    )
    initial_qpos = torch.cat([qpos[:, :3], root_rot_wxyz, joint_angles], dim=-1)
    return initial_qpos, global_rotations_torch


def _resolve_optimizer_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _build_retarget_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(
        (len(LOWER_BODY_KEYPOINT_NAMES), len(LOWER_BODY_KEYPOINT_NAMES)),
        device=device,
        dtype=dtype,
    )
    for start_name, end_name in SKELETON_EDGES:
        start_idx = KEYPOINT_INDEX[start_name]
        end_idx = KEYPOINT_INDEX[end_name]
        mask[start_idx, end_idx] = 1.0
        mask[end_idx, start_idx] = 1.0
    return mask


def _smooth_sequence(sequence: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 1 or sequence.shape[0] <= 2:
        return sequence

    pad = window_size // 2
    padded = torch.cat(
        [
            sequence[:1].expand(pad, -1),
            sequence,
            sequence[-1:].expand(pad, -1),
        ],
        dim=0,
    )
    smoothed = F.avg_pool1d(
        padded.transpose(0, 1).unsqueeze(0),
        kernel_size=window_size,
        stride=1,
    )
    return smoothed.squeeze(0).transpose(0, 1)


def _smooth_root_quaternions(root_quats: torch.Tensor, window_size: int) -> torch.Tensor:
    stabilized = _stabilize_root_quaternions(F.normalize(root_quats, dim=-1))
    if window_size <= 1 or stabilized.shape[0] <= 2:
        return stabilized
    smoothed = _smooth_sequence(stabilized, window_size=window_size)
    return _stabilize_root_quaternions(F.normalize(smoothed, dim=-1))


def _compose_qpos(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_angles: torch.Tensor,
    *,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    stabilized_root = _stabilize_root_quaternions(F.normalize(root_quat, dim=-1))
    stabilized_joints = _select_continuous_joint_angles(
        joint_angles=joint_angles,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )
    return torch.cat([root_pos, stabilized_root, stabilized_joints], dim=-1)


def _build_reference_qpos(
    initial_qpos: torch.Tensor,
    *,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    window_size: int = DEFAULT_SMOOTHING_WINDOW,
) -> torch.Tensor:
    reference_root_pos = _smooth_sequence(initial_qpos[:, :3], window_size=window_size)
    reference_root_quat = _smooth_root_quaternions(initial_qpos[:, 3:7], window_size=window_size)
    reference_joint_angles = _smooth_sequence(initial_qpos[:, 7:], window_size=window_size)
    return _compose_qpos(
        reference_root_pos,
        reference_root_quat,
        reference_joint_angles,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def _compute_orientation_loss(
    predicted_rotations: torch.Tensor,
    target_rotations: torch.Tensor,
) -> torch.Tensor:
    relative_rot = torch.matmul(predicted_rotations.transpose(-1, -2), target_rotations)
    trace = relative_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return (1.0 - trace / 3.0).mean()


def _compute_world_transforms_autograd(
    kinematic_info,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_pos = kinematic_info.local_pos.to(device=root_pos.device, dtype=root_pos.dtype)
    local_rot_ref_mat = kinematic_info.local_rot_ref_mat.to(device=root_pos.device, dtype=root_pos.dtype)

    world_positions = []
    world_rotations = []
    for body_idx, parent_idx in enumerate(kinematic_info.parent_indices):
        if parent_idx == -1:
            world_positions.append(root_pos)
            world_rotations.append(joint_rot_mats[:, 0, :, :])
            continue

        parent_pos = world_positions[parent_idx]
        parent_rot = world_rotations[parent_idx]
        offset = torch.matmul(parent_rot, local_pos[body_idx].view(1, 3, 1)).squeeze(-1)
        effective_local_rot = torch.matmul(
            local_rot_ref_mat[body_idx].view(1, 3, 3),
            joint_rot_mats[:, body_idx, :, :],
        )
        world_positions.append(parent_pos + offset)
        world_rotations.append(torch.matmul(parent_rot, effective_local_rot))

    return torch.stack(world_positions, dim=1), torch.stack(world_rotations, dim=1)


def _compute_trajectory_losses(
    qpos: torch.Tensor,
    *,
    reference_qpos: torch.Tensor,
    target_positions: torch.Tensor,
    target_rotations: torch.Tensor,
    left_contacts: torch.Tensor,
    right_contacts: torch.Tensor,
    kinematic_info,
    body_indices: torch.Tensor,
    left_body_indices: torch.Tensor,
    right_body_indices: torch.Tensor,
    pair_mask: torch.Tensor,
    fps: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    world_pos, world_rot = _compute_world_transforms_autograd(
        kinematic_info=kinematic_info,
        root_pos=qpos[:, :3],
        joint_rot_mats=joint_rot_mats,
    )

    body_pos = world_pos[:, body_indices]
    body_rot = world_rot[:, body_indices]

    delta_target = target_positions[:, :, None, :] - target_positions[:, None, :, :]
    delta_robot = body_pos[:, :, None, :] - body_pos[:, None, :, :]
    pair_denom = pair_mask.sum().clamp_min(1.0)

    left_foot_pos = world_pos[:, left_body_indices]
    right_foot_pos = world_pos[:, right_body_indices]
    foot_pos = torch.cat([left_foot_pos, right_foot_pos], dim=1)
    foot_contacts = torch.cat([left_contacts, right_contacts], dim=1)

    world_vel = (world_pos[1:] - world_pos[:-1]) * fps if world_pos.shape[0] > 1 else torch.zeros_like(world_pos)
    world_acc = world_vel[1:] - world_vel[:-1] if world_vel.shape[0] > 1 else torch.zeros_like(world_vel)
    body_vel = (body_pos[1:] - body_pos[:-1]) * fps if body_pos.shape[0] > 1 else torch.zeros_like(body_pos)
    target_body_vel = (
        (target_positions[1:] - target_positions[:-1]) * fps
        if target_positions.shape[0] > 1
        else torch.zeros_like(target_positions)
    )
    foot_vel = (foot_pos[1:] - foot_pos[:-1]) * fps if foot_pos.shape[0] > 1 else torch.zeros_like(foot_pos)
    foot_vel_contacts = foot_contacts[1:] if foot_pos.shape[0] > 1 else foot_contacts
    left_contact_mean = left_contacts.mean(dim=-1)
    right_contact_mean = right_contacts.mean(dim=-1)
    pelvis_rot_delta = body_rot[1:, 0] - body_rot[:-1, 0] if body_rot.shape[0] > 1 else torch.zeros_like(body_rot[:, 0])
    pelvis_rot_acc = (
        pelvis_rot_delta[1:] - pelvis_rot_delta[:-1]
        if pelvis_rot_delta.shape[0] > 1
        else torch.zeros_like(pelvis_rot_delta)
    )

    joint_angles = qpos[:, 7:]
    joint_vel = joint_angles[1:] - joint_angles[:-1]
    joint_acc = joint_vel[1:] - joint_vel[:-1] if joint_vel.shape[0] > 1 else torch.zeros_like(joint_vel)
    root_vel = qpos[1:, :3] - qpos[:-1, :3]
    root_acc = root_vel[1:] - root_vel[:-1] if root_vel.shape[0] > 1 else torch.zeros_like(root_vel)

    lower = kinematic_info.dof_limits_lower.to(device=qpos.device, dtype=qpos.dtype)
    upper = kinematic_info.dof_limits_upper.to(device=qpos.device, dtype=qpos.dtype)

    losses = {
        "position": F.mse_loss(body_pos, target_positions),
        "relative_position": (
            ((delta_robot - delta_target).pow(2) * pair_mask[None, :, :, None]).sum() / (pair_denom * 3.0)
        ),
        "orientation": _compute_orientation_loss(body_rot, target_rotations),
        "body_velocity": world_vel.pow(2).mean() if world_vel.numel() > 0 else qpos.new_tensor(0.0),
        "body_acceleration": world_acc.pow(2).mean() if world_acc.numel() > 0 else qpos.new_tensor(0.0),
        "body_velocity_match": (
            F.mse_loss(body_vel, target_body_vel) if body_vel.numel() > 0 else qpos.new_tensor(0.0)
        ),
        "root_orientation_smoothness": (
            pelvis_rot_delta.pow(2).mean() if pelvis_rot_delta.numel() > 0 else qpos.new_tensor(0.0)
        ),
        "root_orientation_acceleration": (
            pelvis_rot_acc.pow(2).mean() if pelvis_rot_acc.numel() > 0 else qpos.new_tensor(0.0)
        ),
        "joint_smoothness": joint_vel.pow(2).mean() if joint_vel.numel() > 0 else qpos.new_tensor(0.0),
        "joint_acceleration": joint_acc.pow(2).mean() if joint_acc.numel() > 0 else qpos.new_tensor(0.0),
        "root_smoothness": root_vel.pow(2).mean() if root_vel.numel() > 0 else qpos.new_tensor(0.0),
        "root_acceleration": root_acc.pow(2).mean() if root_acc.numel() > 0 else qpos.new_tensor(0.0),
        "foot_velocity": (
            (foot_vel.pow(2).sum(dim=-1) * foot_vel_contacts).sum() / foot_vel_contacts.sum().clamp_min(1.0)
            if foot_vel.numel() > 0
            else qpos.new_tensor(0.0)
        ),
        "foot_height": (
            (foot_pos[..., 2].pow(2) * foot_contacts).sum() / foot_contacts.sum().clamp_min(1.0)
        ),
        "foot_level": (
            (((left_foot_pos[:, 0, 2] - left_foot_pos[:, 1, 2]).pow(2) * left_contact_mean).sum()
            + ((right_foot_pos[:, 0, 2] - right_foot_pos[:, 1, 2]).pow(2) * right_contact_mean).sum())
            / (left_contact_mean.sum() + right_contact_mean.sum()).clamp_min(1.0)
        ),
        "joint_limit": (
            F.relu(lower - joint_angles).pow(2).mean()
            + F.relu(joint_angles - upper).pow(2).mean()
        ),
        "joint_deviation": (joint_angles - reference_qpos[:, 7:]).pow(2).mean(),
        "root_deviation": (qpos[:, :3] - reference_qpos[:, :3]).pow(2).mean(),
        "root_orientation_deviation": (qpos[:, 3:7] - reference_qpos[:, 3:7]).pow(2).mean(),
    }

    weights = {
        "position": 14.0,
        "relative_position": 8.0,
        "orientation": 3.0,
        "body_velocity": 2.5,
        "body_acceleration": 20.0,
        "body_velocity_match": 6.0,
        "root_orientation_smoothness": 8.0,
        "root_orientation_acceleration": 20.0,
        "joint_smoothness": 10.0,
        "joint_acceleration": 45.0,
        "root_smoothness": 6.0,
        "root_acceleration": 20.0,
        "foot_velocity": 14.0,
        "foot_height": 28.0,
        "foot_level": 15.0,
        "joint_limit": 200.0,
        "joint_deviation": 2.5,
        "root_deviation": 4.0,
        "root_orientation_deviation": 6.0,
    }
    total_loss = sum(weights[name] * value for name, value in losses.items())
    return total_loss, losses


def _optimize_trajectory(
    reference_qpos: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    target_rotations: torch.Tensor,
    left_contacts: torch.Tensor,
    right_contacts: torch.Tensor,
    kinematic_info,
    fps: int,
    steps: int,
    learning_rate: float,
    log_every: int,
) -> torch.Tensor:
    device = reference_qpos.device
    dtype = reference_qpos.dtype

    body_indices = torch.tensor(
        [kinematic_info.body_names.index(name) for name in KEYPOINT_TO_BODY_NAMES],
        device=device,
        dtype=torch.long,
    )
    left_body_indices = torch.tensor(
        [kinematic_info.body_names.index(name) for name in ("L_Ankle", "L_Toe")],
        device=device,
        dtype=torch.long,
    )
    right_body_indices = torch.tensor(
        [kinematic_info.body_names.index(name) for name in ("R_Ankle", "R_Toe")],
        device=device,
        dtype=torch.long,
    )
    pair_mask = _build_retarget_mask(device=device, dtype=dtype)

    lower = kinematic_info.dof_limits_lower.to(device=device, dtype=dtype)
    upper = kinematic_info.dof_limits_upper.to(device=device, dtype=dtype)

    root_pos = torch.nn.Parameter(reference_qpos[:, :3].clone())
    root_quat = torch.nn.Parameter(reference_qpos[:, 3:7].clone())
    joint_angles = torch.nn.Parameter(reference_qpos[:, 7:].clone())

    optimizer = torch.optim.Adam([root_pos, root_quat, joint_angles], lr=learning_rate)

    best_total = float("inf")
    best_qpos = reference_qpos.detach().clone()

    for step_idx in range(steps):
        optimizer.zero_grad()
        current_root_quat = F.normalize(root_quat, dim=-1)
        qpos = torch.cat([root_pos, current_root_quat, joint_angles], dim=-1)
        total_loss, loss_terms = _compute_trajectory_losses(
            qpos,
            reference_qpos=reference_qpos,
            target_positions=target_positions,
            target_rotations=target_rotations,
            left_contacts=left_contacts,
            right_contacts=right_contacts,
            kinematic_info=kinematic_info,
            body_indices=body_indices,
            left_body_indices=left_body_indices,
            right_body_indices=right_body_indices,
            pair_mask=pair_mask,
            fps=fps,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([root_pos, root_quat, joint_angles], max_norm=10.0)
        optimizer.step()

        with torch.no_grad():
            joint_angles.clamp_(min=lower, max=upper)
            root_quat.copy_(F.normalize(root_quat, dim=-1))

            loss_value = float(total_loss.detach().cpu())
            if loss_value < best_total:
                best_total = loss_value
                best_qpos = _compose_qpos(
                    root_pos.detach().clone(),
                    root_quat.detach().clone(),
                    joint_angles.detach().clone(),
                    lower_limits=lower,
                    upper_limits=upper,
                )

            if log_every > 0 and (step_idx == 0 or (step_idx + 1) % log_every == 0 or step_idx == steps - 1):
                summary = ", ".join(
                    f"{name}={float(value.detach().cpu()):.4f}"
                    for name, value in loss_terms.items()
                    if name in {
                        "position",
                        "orientation",
                        "body_velocity",
                        "body_acceleration",
                        "foot_velocity",
                        "joint_smoothness",
                    }
                )
                print(
                    f"      opt step {step_idx + 1:04d}/{steps}: total={loss_value:.4f} | {summary}"
                )

    return _compose_qpos(
        best_qpos[:, :3],
        best_qpos[:, 3:7],
        best_qpos[:, 7:],
        lower_limits=lower,
        upper_limits=upper,
    )


def _compute_motion_quality_metrics(
    qpos: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    kinematic_info,
    body_indices: torch.Tensor,
    fps: int,
) -> tuple[float, float, float]:
    _, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    world_pos, _ = _compute_world_transforms_autograd(
        kinematic_info=kinematic_info,
        root_pos=qpos[:, :3],
        joint_rot_mats=joint_rot_mats,
    )
    body_pos = world_pos[:, body_indices]
    body_vel = (world_pos[1:] - world_pos[:-1]) * fps if world_pos.shape[0] > 1 else torch.zeros_like(world_pos)
    joint_step = qpos[1:, 7:] - qpos[:-1, 7:] if qpos.shape[0] > 1 else torch.zeros_like(qpos[:, 7:])
    position_loss = float(F.mse_loss(body_pos, target_positions).detach().cpu())
    max_body_velocity = float(body_vel.abs().max().detach().cpu()) if body_vel.numel() > 0 else 0.0
    max_joint_step = float(joint_step.abs().max().detach().cpu()) if joint_step.numel() > 0 else 0.0
    return position_loss, max_body_velocity, max_joint_step


def _blend_trajectory_candidate(
    reference_qpos: torch.Tensor,
    optimized_qpos: torch.Tensor,
    *,
    alpha: float,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    blended_root_pos = torch.lerp(reference_qpos[:, :3], optimized_qpos[:, :3], alpha)
    blended_root_quat = torch.lerp(reference_qpos[:, 3:7], optimized_qpos[:, 3:7], alpha)
    blended_joint_angles = torch.lerp(reference_qpos[:, 7:], optimized_qpos[:, 7:], alpha)
    return _compose_qpos(
        blended_root_pos,
        blended_root_quat,
        blended_joint_angles,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def _select_trajectory_candidate(
    reference_qpos: torch.Tensor,
    optimized_qpos: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    kinematic_info,
    body_indices: torch.Tensor,
    fps: int,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    blend_alphas: tuple[float, ...] = DEFAULT_BLEND_ALPHAS,
    max_velocity_threshold: float = 15.0,
) -> torch.Tensor:
    reference_pos_loss, reference_velocity, reference_joint_step = _compute_motion_quality_metrics(
        reference_qpos,
        target_positions=target_positions,
        kinematic_info=kinematic_info,
        body_indices=body_indices,
        fps=fps,
    )
    velocity_budget = max(
        reference_velocity,
        min(max_velocity_threshold - 0.25, reference_velocity * 1.1 + 0.5),
    )
    joint_step_budget = reference_joint_step * 1.15 + 0.05

    candidates = [(0.0, reference_qpos)]
    candidates.extend(
        (
            alpha,
            _blend_trajectory_candidate(
                reference_qpos,
                optimized_qpos,
                alpha=alpha,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            ),
        )
        for alpha in blend_alphas
    )

    best_candidate = reference_qpos
    best_alpha = 0.0
    best_pos_loss = reference_pos_loss
    fallback_score = float("inf")

    for alpha, candidate_qpos in candidates:
        position_loss, max_body_velocity, max_joint_step = _compute_motion_quality_metrics(
            candidate_qpos,
            target_positions=target_positions,
            kinematic_info=kinematic_info,
            body_indices=body_indices,
            fps=fps,
        )
        within_budget = (
            max_body_velocity <= velocity_budget + 1e-6
            and max_joint_step <= joint_step_budget + 1e-6
        )
        if within_budget and position_loss < best_pos_loss:
            best_candidate = candidate_qpos
            best_alpha = alpha
            best_pos_loss = position_loss

        velocity_over = max(0.0, (max_body_velocity - velocity_budget) / max(velocity_budget, 1e-6))
        joint_step_over = max(0.0, (max_joint_step - joint_step_budget) / max(joint_step_budget, 1e-6))
        score = position_loss + 0.5 * velocity_over**2 + 0.25 * joint_step_over**2
        if score < fallback_score:
            fallback_score = score
            fallback_candidate = candidate_qpos
            fallback_alpha = alpha
            fallback_metrics = (position_loss, max_body_velocity, max_joint_step)

    if best_alpha == 0.0 and best_pos_loss == reference_pos_loss:
        best_candidate = fallback_candidate
        best_alpha = fallback_alpha
        best_pos_loss, best_velocity, best_joint_step = fallback_metrics
    else:
        _, best_velocity, best_joint_step = _compute_motion_quality_metrics(
            best_candidate,
            target_positions=target_positions,
            kinematic_info=kinematic_info,
            body_indices=body_indices,
            fps=fps,
        )

    print(
        "      selected candidate "
        f"alpha={best_alpha:.2f} "
        f"pos={best_pos_loss:.5f} "
        f"max_rb_vel={best_velocity:.3f} "
        f"max_joint_step={best_joint_step:.3f} "
        f"(budget vel={velocity_budget:.3f}, joint={joint_step_budget:.3f})"
    )
    return best_candidate


def _retarget_motion(
    keypoint_path: Path,
    output_path: Path,
    *,
    model_xml: Path,
    optimizer_device: str,
    optimizer_steps: int,
    optimizer_lr: float,
    retarget_fps: int,
    retarget_mode: str,
    log_every: int,
) -> None:
    keypoint_data = np.load(keypoint_path, allow_pickle=True).item()
    positions = np.asarray(keypoint_data["positions"], dtype=np.float32)
    orientations = np.asarray(keypoint_data["orientations"], dtype=np.float32)
    device = _resolve_optimizer_device(optimizer_device)
    dtype = torch.float32
    kinematic_info = extract_kinematic_info(str(model_xml)).to(device, dtype=dtype)

    initial_qpos, global_rotations_torch = _build_initial_qpos(
        positions,
        orientations,
        kinematic_info=kinematic_info,
        device=device,
        dtype=dtype,
    )
    lower = kinematic_info.dof_limits_lower.to(device=device, dtype=dtype)
    upper = kinematic_info.dof_limits_upper.to(device=device, dtype=dtype)
    reference_qpos = _build_reference_qpos(
        initial_qpos,
        lower_limits=lower,
        upper_limits=upper,
    )
    body_indices = torch.tensor(
        [kinematic_info.body_names.index(name) for name in KEYPOINT_TO_BODY_NAMES],
        device=device,
        dtype=torch.long,
    )
    target_positions = torch.from_numpy(positions).to(device=device, dtype=dtype)
    left_contacts = torch.from_numpy(
        np.asarray(keypoint_data["left_foot_contacts"], dtype=np.float32)
    ).to(device=device, dtype=dtype)
    right_contacts = torch.from_numpy(
        np.asarray(keypoint_data["right_foot_contacts"], dtype=np.float32)
    ).to(device=device, dtype=dtype)

    if retarget_mode == "trajectory":
        print(
            f"Optimizing trajectory on {device} for {keypoint_path.name} "
            f"({initial_qpos.shape[0]} frames, {optimizer_steps} steps)..."
        )
        optimized_qpos = _optimize_trajectory(
            reference_qpos=reference_qpos,
            target_positions=target_positions,
            target_rotations=global_rotations_torch,
            left_contacts=left_contacts,
            right_contacts=right_contacts,
            kinematic_info=kinematic_info,
            fps=retarget_fps,
            steps=optimizer_steps,
            learning_rate=optimizer_lr,
            log_every=log_every,
        )
        qpos = _select_trajectory_candidate(
            reference_qpos=reference_qpos,
            optimized_qpos=optimized_qpos,
            target_positions=target_positions,
            kinematic_info=kinematic_info,
            body_indices=body_indices,
            fps=retarget_fps,
            lower_limits=lower,
            upper_limits=upper,
        )
    else:
        qpos = reference_qpos

    root_rot_wxyz = _stabilize_root_quaternions(qpos[:, 3:7])
    joint_angles = qpos[:, 7:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_frame_pos=qpos[:, :3].cpu().numpy().astype(np.float32),
        base_frame_wxyz=root_rot_wxyz.cpu().numpy().astype(np.float32),
        joint_angles=joint_angles.cpu().numpy().astype(np.float32),
        joint_names=np.asarray(kinematic_info.dof_names),
    )
    print(f"Saved retargeted motion to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower-body retargeting from extracted keypoints")
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Accepted for compatibility. Visualization is not implemented in this script.",
    )
    parser.add_argument(
        "--keypoints-folder-path",
        type=str,
        required=True,
        help="Path to the folder containing extracted lower-body keypoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./retargeted_output_motions",
        help="Directory to save retargeted motions.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Optional URDF path. When it matches a generated subject asset, the paired MJCF is used.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Accepted for backward compatibility. Ignored by this script.",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Accepted for compatibility. Keypoints are assumed to already be sampled at target FPS.",
    )
    parser.add_argument(
        "--retarget-fps",
        type=int,
        default=30,
        help="Accepted for compatibility. Output frame rate is inherited from the input keypoints.",
    )
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=-1,
        help="Accepted for compatibility. This script keeps the full input sequence.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip motions that already have corresponding output files.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="treadmill",
        help="Accepted for backward compatibility with older retarget commands.",
    )
    parser.add_argument(
        "--save-contacts-only",
        action="store_true",
        help="Only save smoothed left/right foot contact labels.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints-folder-path}/contacts.",
    )
    parser.add_argument(
        "--retarget-mode",
        type=str,
        choices=("trajectory", "framewise"),
        default="trajectory",
        help="Use a multi-frame trajectory optimizer or the deterministic framewise fallback.",
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=40,
        help="Number of optimization steps for trajectory retargeting.",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=0.015,
        help="Learning rate for trajectory retargeting.",
    )
    parser.add_argument(
        "--optimizer-device",
        type=str,
        default="auto",
        help="Optimization device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print optimization losses every N steps.",
    )
    args = parser.parse_args()

    if args.visualize:
        print("Visualization is not implemented for the lower-body local retargeter. Continuing without it.")

    keypoint_paths = sorted(glob.glob(os.path.join(args.keypoints_folder_path, "*.npy")))
    if not keypoint_paths:
        print(f"No .npy files found in {args.keypoints_folder_path}. Exiting.")
        return

    if args.save_contacts_only:
        contacts_dir = Path(args.contacts_dir or Path(args.keypoints_folder_path) / "contacts")
        for keypoint_path_str in keypoint_paths:
            keypoint_path = Path(keypoint_path_str)
            output_path = contacts_dir / f"{keypoint_path.stem}_contacts.npz"
            if args.skip_existing and output_path.exists():
                print(f"Skipping existing contact labels: {output_path.name}")
                continue
            keypoint_data = np.load(keypoint_path, allow_pickle=True).item()
            _save_contact_labels(output_path, keypoint_data)
        return

    model_xml = _resolve_model_xml(args.urdf_path)
    print(f"Using lower-body model XML: {model_xml}")

    output_dir = Path(args.output_dir)
    for keypoint_path_str in keypoint_paths:
        keypoint_path = Path(keypoint_path_str)
        output_path = output_dir / f"{keypoint_path.stem}_retargeted.npz"
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing retargeted motion: {output_path.name}")
            continue
        _retarget_motion(
            keypoint_path,
            output_path,
            model_xml=model_xml,
            optimizer_device=args.optimizer_device,
            optimizer_steps=args.optimizer_steps,
            optimizer_lr=args.optimizer_lr,
            retarget_fps=args.retarget_fps,
            retarget_mode=args.retarget_mode,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
