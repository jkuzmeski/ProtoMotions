#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Retarget lower-body biomechanics keypoints with a real JAX/PyRoki solve."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import TypedDict
from xml.etree import ElementTree as ET

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy


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
ROBOT_LINK_NAMES = [
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
DIRECT_PAIRS = [
    ("pelvis", "left_hip", 1.0),
    ("pelvis", "right_hip", 1.0),
    ("left_hip", "left_knee", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("right_ankle", "right_foot", 1.0),
]
ORIENTATION_SOURCE_KEYPOINT_NAMES = [
    "pelvis",
]
ORIENTATION_ROBOT_LINK_NAMES = [
    "Pelvis",
]
# The lower-body biomechanics source does not expose a stable ankle-foot SO(3)
# target. Keep explicit orientation supervision on the pelvis only, and let the
# distal leg pose be shaped by the positional/contact terms instead.
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(LOWER_BODY_KEYPOINT_NAMES)}


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    orientation_alignment: float
    root_pose_alignment: float
    root_smoothness: float
    joint_smoothness: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float


def _mjcf_path_from_urdf_path(urdf_path: str | os.PathLike[str]) -> Path:
    urdf_path = Path(urdf_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    mjcf_path = repo_root / "protomotions" / "data" / "assets" / "mjcf" / f"{urdf_path.stem}.xml"
    if not mjcf_path.exists():
        raise FileNotFoundError(
            f"Could not find matching MJCF for URDF {urdf_path}. Expected {mjcf_path}"
        )
    return mjcf_path


def _load_joint_limits_from_mjcf(
    mjcf_path: str | os.PathLike[str],
    joint_names: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(mjcf_path).getroot()
    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        range_attr = joint.attrib.get("range")
        if name is None or range_attr is None:
            continue
        lower_deg, upper_deg = (float(v) for v in range_attr.split())
        limits_by_name[name] = (np.deg2rad(lower_deg), np.deg2rad(upper_deg))

    lower_limits = []
    upper_limits = []
    missing = []
    for name in joint_names:
        limits = limits_by_name.get(name)
        if limits is None:
            missing.append(name)
            lower_limits.append(-np.pi)
            upper_limits.append(np.pi)
        else:
            lower_limits.append(limits[0])
            upper_limits.append(limits[1])
    if missing:
        raise ValueError(
            f"Missing MJCF joint limits for joints: {missing}. MJCF path: {mjcf_path}"
        )
    return (
        np.asarray(lower_limits, dtype=np.float32),
        np.asarray(upper_limits, dtype=np.float32),
    )


def _build_rest_weights(joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    weights = np.full((len(joint_names),), 0.02, dtype=np.float32)
    for i, name in enumerate(joint_names):
        if name.startswith("L_Toe") or name.startswith("R_Toe"):
            weights[i] = 1.0
        elif "Knee_x" in name or "Knee_z" in name:
            weights[i] = 0.5
        elif "Hip_z" in name or "Ankle_z" in name:
            weights[i] = 0.15
        elif "Hip_x" in name or "Ankle_x" in name:
            weights[i] = 0.08
        elif "Knee_y" in name or "Hip_y" in name or "Ankle_y" in name:
            weights[i] = 0.01
    return weights


def _compute_root_frame_alignment_offset(robot: pk.Robot) -> np.ndarray:
    """Map semantic pelvis axes to the robot root frame.

    The source pelvis orientation is built from the left/right hip span plus world up.
    The lower-body robot root frame is not expressed in that same semantic basis, so we
    apply a constant offset derived from the robot's zero-pose geometry.
    """
    joint_cfg = np.zeros((len(robot.joints.actuated_names),), dtype=np.float32)
    fk = np.asarray(robot.forward_kinematics(cfg=joint_cfg))
    link_names = list(robot.links.names)
    left_hip_pos = fk[link_names.index("L_Hip"), 4:]
    right_hip_pos = fk[link_names.index("R_Hip"), 4:]

    x_axis = left_hip_pos - right_hip_pos
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    z_axis = np.cross(x_axis, y_axis)

    semantic_root_rot = np.column_stack([x_axis, y_axis, z_axis]).astype(
        np.float32, copy=False
    )
    return semantic_root_rot.T.copy()


def _apply_crossfade(contact_flags: np.ndarray, window_size: int = 5) -> np.ndarray:
    smoothed = np.zeros_like(contact_flags, dtype=np.float32)
    for i in range(len(contact_flags)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(contact_flags), i + window_size // 2 + 1)
        smoothed[i] = np.mean(contact_flags[start_idx:end_idx])
    return smoothed


def _pad_or_trim(data: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        return data
    current_frames = data.shape[0]
    if current_frames >= target_frames:
        return data[:target_frames]
    pad = np.repeat(data[-1:], target_frames - current_frames, axis=0)
    return np.concatenate([data, pad], axis=0)


def load_motion_data(
    motion_path: str,
    source_type: str,
    subsample_factor: int,
    target_raw_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    del source_type  # Lower-body keypoints already match the target semantics.
    print(f"Loading motion from: {motion_path}")
    motion_data = np.load(motion_path, allow_pickle=True).item()

    raw_positions = np.asarray(motion_data["positions"], dtype=np.float32)
    raw_orientations = np.asarray(motion_data["orientations"], dtype=np.float32)
    raw_left_contacts = np.asarray(
        motion_data["left_foot_contacts"], dtype=np.float32
    )
    raw_right_contacts = np.asarray(
        motion_data["right_foot_contacts"], dtype=np.float32
    )

    if raw_positions.shape[1] != len(LOWER_BODY_KEYPOINT_NAMES):
        raise ValueError(
            f"Expected {len(LOWER_BODY_KEYPOINT_NAMES)} lower-body keypoints, "
            f"got {raw_positions.shape[1]}"
        )

    original_raw_frames = raw_positions.shape[0]
    original_subsampled_count = raw_positions[::subsample_factor].shape[0]
    if target_raw_frames > 0:
        target_subsampled_frames = len(range(0, target_raw_frames, subsample_factor))
        num_timesteps = min(original_subsampled_count, target_subsampled_frames)
    else:
        num_timesteps = original_subsampled_count

    positions = _pad_or_trim(raw_positions, target_raw_frames)
    orientations = _pad_or_trim(raw_orientations, target_raw_frames)
    left_contacts = _pad_or_trim(raw_left_contacts, target_raw_frames)
    right_contacts = _pad_or_trim(raw_right_contacts, target_raw_frames)

    left_contact_avg = np.mean(left_contacts.astype(np.float32), axis=1, keepdims=True)
    right_contact_avg = np.mean(
        right_contacts.astype(np.float32), axis=1, keepdims=True
    )

    left_contact = _apply_crossfade(left_contact_avg)[::subsample_factor]
    right_contact = _apply_crossfade(right_contact_avg)[::subsample_factor]
    keypoints = positions[::subsample_factor]
    keypoint_orientations = orientations[::subsample_factor]

    return (
        keypoints,
        keypoint_orientations,
        left_contact,
        right_contact,
        num_timesteps,
    )


def save_contact_labels(
    output_path: str | os.PathLike[str],
    left_foot_contact: np.ndarray,
    right_foot_contact: np.ndarray,
    num_timesteps: int,
) -> None:
    foot_contacts = np.stack(
        [
            left_foot_contact[:num_timesteps].squeeze(-1),
            right_foot_contact[:num_timesteps].squeeze(-1),
        ],
        axis=-1,
    )
    np.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    joint_vel = (var_values[var_joints_curr] - var_values[var_joints_prev]) / dt
    excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)
    return excess_vel.flatten() * weight


@jaxls.Cost.create_factory
def joint_limit_cost(
    var_values: jaxls.VarValues,
    var_joints: jaxls.Var[jnp.ndarray],
    lower_limits: jnp.ndarray,
    upper_limits: jnp.ndarray,
    weight: float,
) -> jax.Array:
    joints = var_values[var_joints]
    lower_violation = jnp.maximum(lower_limits - joints, 0.0)
    upper_violation = jnp.maximum(joints - upper_limits, 0.0)
    return jnp.concatenate([lower_violation, upper_violation]) * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    target_keypoints_curr: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    retarget_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    robot_cfg_curr = var_values[var_robot_cfg_curr]
    robot_cfg_prev = var_values[var_robot_cfg_prev]

    T_world_link_curr = T_world_root_curr @ jaxlie.SE3(
        robot.forward_kinematics(cfg=robot_cfg_curr)
    )
    T_world_link_prev = T_world_root_prev @ jaxlie.SE3(
        robot.forward_kinematics(cfg=robot_cfg_prev)
    )

    link_pos_curr = T_world_link_curr.translation()[retarget_indices]
    link_pos_prev = T_world_link_prev.translation()[retarget_indices]

    left_ankle_idx = KEYPOINT_INDEX["left_ankle"]
    left_foot_idx = KEYPOINT_INDEX["left_foot"]
    right_ankle_idx = KEYPOINT_INDEX["right_ankle"]
    right_foot_idx = KEYPOINT_INDEX["right_foot"]

    left_contact_weight = left_foot_contact[0]
    right_contact_weight = right_foot_contact[0]

    left_ankle_target = target_keypoints_curr[left_ankle_idx]
    left_foot_target = target_keypoints_curr[left_foot_idx]
    right_ankle_target = target_keypoints_curr[right_ankle_idx]
    right_foot_target = target_keypoints_curr[right_foot_idx]

    left_ankle_anchor = left_contact_weight * (
        link_pos_curr[left_ankle_idx] - left_ankle_target
    )
    left_foot_anchor = left_contact_weight * (
        link_pos_curr[left_foot_idx] - left_foot_target
    )
    right_ankle_anchor = right_contact_weight * (
        link_pos_curr[right_ankle_idx] - right_ankle_target
    )
    right_foot_anchor = right_contact_weight * (
        link_pos_curr[right_foot_idx] - right_foot_target
    )

    left_ankle_vel = left_contact_weight * (
        link_pos_curr[left_ankle_idx] - link_pos_prev[left_ankle_idx]
    )
    left_foot_vel = left_contact_weight * (
        link_pos_curr[left_foot_idx] - link_pos_prev[left_foot_idx]
    )
    right_ankle_vel = right_contact_weight * (
        link_pos_curr[right_ankle_idx] - link_pos_prev[right_ankle_idx]
    )
    right_foot_vel = right_contact_weight * (
        link_pos_curr[right_foot_idx] - link_pos_prev[right_foot_idx]
    )

    left_z_consistency = left_contact_weight * (
        link_pos_curr[left_ankle_idx, 2] - link_pos_curr[left_foot_idx, 2]
    )
    right_z_consistency = right_contact_weight * (
        link_pos_curr[right_ankle_idx, 2] - link_pos_curr[right_foot_idx, 2]
    )

    return (
        jnp.concatenate(
            [
                left_ankle_anchor.flatten(),
                left_foot_anchor.flatten(),
                right_ankle_anchor.flatten(),
                right_foot_anchor.flatten(),
                left_ankle_vel.flatten(),
                left_foot_vel.flatten(),
                right_ankle_vel.flatten(),
                right_foot_vel.flatten(),
                jnp.array([left_z_consistency]),
                jnp.array([right_z_consistency]),
            ]
        )
        * weight
    )


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    retarget_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root = var_values[var_Ts_world_root]
    robot_cfg = var_values[var_robot_cfg]
    T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    link_rot = T_world_link.rotation().as_matrix()[retarget_indices]

    left_ankle_idx = KEYPOINT_INDEX["left_ankle"]
    right_ankle_idx = KEYPOINT_INDEX["right_ankle"]
    left_foot_idx = KEYPOINT_INDEX["left_foot"]
    right_foot_idx = KEYPOINT_INDEX["right_foot"]

    left_contact_weight = left_foot_contact[0]
    right_contact_weight = right_foot_contact[0]

    left_tilt = left_contact_weight * (
        jnp.concatenate(
            [
                link_rot[left_ankle_idx, 2, :2],
                link_rot[left_foot_idx, 2, :2],
            ]
        )
    )
    right_tilt = right_contact_weight * (
        jnp.concatenate(
            [
                link_rot[right_ankle_idx, 2, :2],
                link_rot[right_foot_idx, 2, :2],
            ]
        )
    )
    return jnp.concatenate([left_tilt, right_tilt]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    retarget_indices: jnp.ndarray,
    retarget_mask: jnp.ndarray,
    source_orientation_indices: jnp.ndarray,
    robot_orientation_indices: jnp.ndarray,
    joint_lower_limits: jnp.ndarray,
    joint_upper_limits: jnp.ndarray,
    joint_rest_weights: jnp.ndarray,
    root_orientation_offset: jnp.ndarray,
    weights: RetargetingWeights,
    subsample_factor: int = 1,
    input_fps: float = 30.0,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    timesteps = target_keypoints.shape[0]
    n_retarget = len(LOWER_BODY_KEYPOINT_NAMES)

    class SimplifiedJointsScaleVarLowerBody(
        jaxls.Var[jax.Array],
        default_factory=lambda: jnp.ones((n_retarget, n_retarget)),
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = SimplifiedJointsScaleVarLowerBody(jnp.zeros(timesteps))

    root_init_values = jaxlie.SE3(
        jnp.concatenate(
            [
                jaxlie.SO3.from_matrix(target_orientations[:, 0] @ root_orientation_offset).wxyz,
                target_keypoints[:, 0],
            ],
            axis=-1,
        )
    )

    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarLowerBody,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        robot_pos = T_world_link.translation()[retarget_indices]

        delta_target = keypoints[:, None] - keypoints[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        position_scale = var_values[var_joints_scale][..., None]
        residual_position_delta = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(n_retarget)[..., None])
            * retarget_mask[..., None]
        )

        delta_target_normalized = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (
            delta_target_normalized * delta_robot_normalized
        ).sum(axis=-1)
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(n_retarget))
            * retarget_mask
        )

        return (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: SimplifiedJointsScaleVarLowerBody,
    ) -> jax.Array:
        res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
        res_1 = (
            var_values[var_joints_scale] - var_values[var_joints_scale].T
        ).flatten() * 100.0
        res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarLowerBody,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        del var_joints_scale  # Kept for upstream script parity.
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        robot_pos = T_world_link.translation()[retarget_indices]
        return (robot_pos - keypoints).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def orientation_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        target_orientations_curr: jnp.ndarray,
    ) -> jax.Array:
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        robot_rot = T_world_link.rotation().as_matrix()[robot_orientation_indices]

        target_rot = target_orientations_curr[source_orientation_indices]
        pelvis_target = target_rot[0] @ root_orientation_offset
        target_rot = target_rot.at[0].set(pelvis_target)
        target_so3 = jaxlie.SO3.from_matrix(target_rot)
        robot_so3 = jaxlie.SO3.from_matrix(robot_rot)
        return ((target_so3.inverse() @ robot_so3).log().flatten()) * weights["orientation_alignment"]

    @jaxls.Cost.create_factory
    def root_smoothness_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    @jaxls.Cost.create_factory
    def root_pose_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        target_root_wxyz_xyz: jnp.ndarray,
    ) -> jax.Array:
        current = var_values[var_Ts_world_root]
        target = jaxlie.SE3(target_root_wxyz_xyz)
        return (target.inverse() @ current).log().flatten() * weights["root_pose_alignment"]

    costs: list[jaxls.Cost] = [
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_joints_scale),
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
        ),
        orientation_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_orientations,
        ),
        root_pose_alignment_cost(
            var_Ts_world_root,
            root_init_values.wxyz_xyz,
        ),
        joint_limit_cost(
            var_joints,
            joint_lower_limits[None, :],
            joint_upper_limits[None, :],
            100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness_cost(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            joint_rest_weights[None],
        ),
        joint_vel_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            20.0,
            subsample_factor / input_fps,
            weights["joint_vel_limit"],
        ),
    ]

    for t in range(1, timesteps):
        costs.append(
            foot_contact_cost(
                jaxls.SE3Var(t),
                jaxls.SE3Var(t - 1),
                robot.joint_var_cls(t),
                robot.joint_var_cls(t - 1),
                robot,
                target_keypoints[t],
                left_foot_contact[t],
                right_foot_contact[t],
                retarget_indices,
                weights["foot_contact"],
            )
        )

    for t in range(timesteps):
        costs.append(
            foot_tilt_cost(
                jaxls.SE3Var(t),
                robot.joint_var_cls(t),
                robot,
                left_foot_contact[t],
                right_foot_contact[t],
                retarget_indices,
                weights["foot_tilt"],
            )
        )

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_joints_scale]
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints,
                    var_Ts_world_root.with_value(root_init_values),
                    var_joints_scale,
                ]
            ),
            termination=jaxls.TerminationConfig(max_iterations=500),
        )
    )
    return solution[var_Ts_world_root], solution[var_joints]


def main() -> None:
    script_dir = Path(__file__).parent.resolve()
    default_urdf = (
        script_dir
        / "../protomotions/data/assets/urdf/for_retargeting/smpl_humanoid_lower_body_subject_S_GENERIC.urdf"
    )

    parser = argparse.ArgumentParser(
        description="Retarget lower-body biomechanics keypoints with JAX/PyRoki."
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Accepted for compatibility. Visualization is not implemented here.",
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
        default=str(default_urdf),
        help="Path to the subject-specific lower-body URDF.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Optional mesh directory passed through to yourdfpy.",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Subsample factor for keypoints and contacts.",
    )
    parser.add_argument(
        "--retarget-fps",
        type=float,
        default=30.0,
        help="FPS of the input keypoint sequence after extraction.",
    )
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=-1,
        help="Pad or trim raw frames before subsampling. -1 keeps the full sequence.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip motions whose output files already exist.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="treadmill",
        help="Accepted for CLI compatibility. Lower-body PyRoki uses the provided keypoints directly.",
    )
    parser.add_argument(
        "--save-contacts-only",
        action="store_true",
        help="Skip retargeting and only save smoothed foot-contact labels.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints-folder-path}/contacts.",
    )
    args = parser.parse_args()

    if args.visualize:
        print(
            "Visualization is not implemented for the lower-body PyRoki retargeter. "
            "Continuing without it."
        )

    keypoints_folder_path = args.keypoints_folder_path
    motion_paths = sorted(glob.glob(os.path.join(keypoints_folder_path, "*.npy")))
    if not motion_paths:
        print(f"No .npy files found in {keypoints_folder_path}. Exiting.")
        return

    if args.save_contacts_only:
        contacts_dir = (
            Path(args.contacts_dir)
            if args.contacts_dir is not None
            else Path(args.keypoints_folder_path) / "contacts"
        )
        contacts_dir.mkdir(parents=True, exist_ok=True)
        for i, motion_path in enumerate(motion_paths):
            print(
                f"Processing motion {i + 1}/{len(motion_paths)}: "
                f"{os.path.basename(motion_path)}"
            )
            output_path = contacts_dir / f"{Path(motion_path).stem}_contacts.npz"
            if args.skip_existing and output_path.exists():
                print(f"Output file {output_path.name} already exists, skipping...")
                continue
            _, _, left_contact, right_contact, num_timesteps = load_motion_data(
                motion_path,
                args.source_type,
                args.subsample_factor,
                args.target_raw_frames,
            )
            save_contact_labels(output_path, left_contact, right_contact, num_timesteps)
        return

    urdf = yourdfpy.URDF.load(args.urdf_path, mesh_dir=args.mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    mjcf_path = _mjcf_path_from_urdf_path(args.urdf_path)
    robot_link_names = list(robot.links.names)
    retarget_indices = jnp.array([robot_link_names.index(name) for name in ROBOT_LINK_NAMES])
    joint_lower_limits_np, joint_upper_limits_np = _load_joint_limits_from_mjcf(
        mjcf_path, robot.joints.actuated_names
    )
    retarget_mask = jnp.zeros((len(LOWER_BODY_KEYPOINT_NAMES), len(LOWER_BODY_KEYPOINT_NAMES)))
    for link_a, link_b, weight in DIRECT_PAIRS:
        idx_a = KEYPOINT_INDEX[link_a]
        idx_b = KEYPOINT_INDEX[link_b]
        retarget_mask = retarget_mask.at[idx_a, idx_b].set(weight)
        retarget_mask = retarget_mask.at[idx_b, idx_a].set(weight)
    source_orientation_indices = jnp.array(
        [KEYPOINT_INDEX[name] for name in ORIENTATION_SOURCE_KEYPOINT_NAMES]
    )
    robot_orientation_indices = jnp.array(
        [robot_link_names.index(name) for name in ORIENTATION_ROBOT_LINK_NAMES]
    )
    root_orientation_offset = jnp.asarray(_compute_root_frame_alignment_offset(robot))

    weights: RetargetingWeights = {
        "local_alignment": 2.0,
        "global_alignment": 6.0,
        "orientation_alignment": 0.5,
        "root_pose_alignment": 20.0,
        "root_smoothness": 1.0,
        "joint_smoothness": 4.0,
        "joint_rest_penalty": 0.05,
        "joint_vel_limit": 20.0,
        "foot_contact": 30.0,
        "foot_tilt": 2.0,
    }
    joint_rest_weights_np = _build_rest_weights(robot.joints.actuated_names) * weights["joint_rest_penalty"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running in non-visualize mode. Retargeting all motions and saving to disk.")
    for i, motion_path in enumerate(motion_paths):
        print(
            f"Processing motion {i + 1}/{len(motion_paths)}: "
            f"{os.path.basename(motion_path)}"
        )
        output_path = output_dir / f"{Path(motion_path).stem}_retargeted.npz"
        if args.skip_existing and output_path.exists():
            print(f"Output file {output_path.name} already exists, skipping...")
            continue

        (
            keypoints,
            keypoint_orientations,
            left_contact,
            right_contact,
            num_timesteps,
        ) = load_motion_data(
            motion_path,
            args.source_type,
            args.subsample_factor,
            args.target_raw_frames,
        )

        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=jnp.asarray(keypoints),
            target_orientations=jnp.asarray(keypoint_orientations),
            left_foot_contact=jnp.asarray(left_contact),
            right_foot_contact=jnp.asarray(right_contact),
            retarget_indices=retarget_indices,
            retarget_mask=retarget_mask,
            source_orientation_indices=source_orientation_indices,
            robot_orientation_indices=robot_orientation_indices,
            joint_lower_limits=jnp.asarray(joint_lower_limits_np),
            joint_upper_limits=jnp.asarray(joint_upper_limits_np),
            joint_rest_weights=jnp.asarray(joint_rest_weights_np),
            root_orientation_offset=root_orientation_offset,
            weights=weights,
            subsample_factor=args.subsample_factor,
            input_fps=args.retarget_fps,
        )
        joints_np = np.array(joints[:num_timesteps], dtype=np.float32)
        joints_np = np.clip(joints_np, joint_lower_limits_np, joint_upper_limits_np)

        np.savez_compressed(
            output_path,
            base_frame_pos=np.array(Ts_world_root.wxyz_xyz[:num_timesteps, 4:]),
            base_frame_wxyz=np.array(Ts_world_root.wxyz_xyz[:num_timesteps, :4]),
            joint_angles=joints_np,
            joint_names=np.asarray(robot.joints.actuated_names),
        )
        print(f"Saved retargeted motion to {output_path}")


if __name__ == "__main__":
    main()
