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
"""PyRoki-based retargeting for lower-body biomechanics keypoints.

This script keeps the existing lower-body CLI surface used by the biomechanics
pipeline, but the retarget solve itself is implemented with upstream PyRoki/JAX.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = (
    REPO_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "urdf"
    / "for_retargeting"
    / "smpl_humanoid_lower_body_subject_S_GENERIC.urdf"
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
    ("left_hip", "left_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("pelvis", "right_hip", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("right_ankle", "right_foot", 1.0),
]


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    orientation_alignment: float
    root_smoothness: float
    joint_smoothness: float
    joint_reference: float
    root_reference: float
    joint_vel_limit: float
    limit_cost: float
    foot_contact: float
    foot_tilt: float


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return fallback.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    x_axis = _normalize(matrix[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_raw = matrix[:, 1] - x_axis * np.dot(matrix[:, 1], x_axis)
    y_axis = _normalize(y_raw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z_axis = _normalize(
        np.cross(x_axis, y_axis),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    y_axis = _normalize(np.cross(z_axis, x_axis), y_axis)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)


def _rotation_between_vectors(
    source_vector: np.ndarray,
    target_vector: np.ndarray,
    fallback_axis: np.ndarray,
) -> np.ndarray:
    source = _normalize(source_vector, fallback_axis)
    target = _normalize(target_vector, fallback_axis)
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = fallback_axis - source * np.dot(fallback_axis, source)
        axis = _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return R.from_rotvec(np.pi * axis).as_matrix().astype(np.float32, copy=False)

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return R.from_rotvec(angle * axis).as_matrix().astype(np.float32, copy=False)


def _trusted_target_rotations(orientations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_rotations = np.tile(
        np.eye(3, dtype=np.float32),
        (orientations.shape[0], len(ROBOT_LINK_NAMES), 1, 1),
    )
    orientation_mask = np.zeros(len(ROBOT_LINK_NAMES), dtype=np.float32)

    source_to_target = {
        "pelvis": ("Pelvis", 1.0),
        "left_ankle": ("L_Ankle", 0.2),
        "left_foot": ("L_Toe", 0.08),
        "right_ankle": ("R_Ankle", 0.2),
        "right_foot": ("R_Toe", 0.08),
    }
    for source_name, (target_name, weight) in source_to_target.items():
        source_idx = KEYPOINT_INDEX[source_name]
        target_idx = ROBOT_LINK_NAMES.index(target_name)
        target_rotations[:, target_idx] = np.asarray(
            [_orthonormalize(matrix) for matrix in orientations[:, source_idx]],
            dtype=np.float32,
        )
        orientation_mask[target_idx] = weight

    return target_rotations, orientation_mask


def _compute_zero_pose_reference(robot: pk.Robot) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    zero_cfg = np.zeros(robot.joints.num_actuated_joints, dtype=np.float32)
    zero_fk = jaxlie.SE3(robot.forward_kinematics(zero_cfg))
    zero_positions = {
        name: np.asarray(zero_fk.translation()[robot.links.names.index(name)], dtype=np.float32)
        for name in ROBOT_LINK_NAMES
    }
    segment_defaults = {
        "L_Hip": zero_positions["L_Knee"] - zero_positions["L_Hip"],
        "L_Knee": zero_positions["L_Ankle"] - zero_positions["L_Knee"],
        "L_Ankle": zero_positions["L_Toe"] - zero_positions["L_Ankle"],
        "R_Hip": zero_positions["R_Knee"] - zero_positions["R_Hip"],
        "R_Knee": zero_positions["R_Ankle"] - zero_positions["R_Knee"],
        "R_Ankle": zero_positions["R_Toe"] - zero_positions["R_Ankle"],
    }
    return zero_positions, segment_defaults


def _smooth_contact_channels(
    contacts: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    contacts = np.asarray(contacts, dtype=np.float32)
    smoothed = np.zeros_like(contacts)
    for channel_idx in range(contacts.shape[1]):
        for t in range(contacts.shape[0]):
            start_idx = max(0, t - window_size // 2)
            end_idx = min(contacts.shape[0], t + window_size // 2 + 1)
            smoothed[t, channel_idx] = np.mean(
                contacts[start_idx:end_idx, channel_idx]
            )
    return smoothed


def _crossfade_foot_contacts(
    contacts: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Collapse ankle/toe contacts to a shared foot weight, like the G1 retargeter."""
    contacts = np.asarray(contacts, dtype=np.float32)
    averaged = contacts.mean(axis=1, keepdims=True)
    smoothed = _smooth_contact_channels(averaged, window_size=window_size)
    return np.repeat(smoothed, contacts.shape[1], axis=1)


def _pad_or_trim(array: np.ndarray, target_len: int) -> np.ndarray:
    if target_len < 0 or array.shape[0] == target_len:
        return array
    if array.shape[0] > target_len:
        return array[:target_len]
    if array.shape[0] == 0:
        raise ValueError("cannot pad empty motion arrays")
    pad = np.repeat(array[-1:], target_len - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


def load_motion_data(
    motion_path: Path,
    subsample_factor: int,
    target_raw_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    motion_data = np.load(motion_path, allow_pickle=True).item()

    raw_positions = np.asarray(motion_data["positions"], dtype=np.float32)
    raw_orientations = np.asarray(motion_data["orientations"], dtype=np.float32)
    raw_left_contacts = np.asarray(
        motion_data["left_foot_contacts"], dtype=np.float32
    )
    raw_right_contacts = np.asarray(
        motion_data["right_foot_contacts"], dtype=np.float32
    )

    effective_raw_frames = (
        raw_positions.shape[0] if target_raw_frames < 0 else target_raw_frames
    )

    positions = _pad_or_trim(raw_positions, effective_raw_frames)
    orientations = _pad_or_trim(raw_orientations, effective_raw_frames)
    left_contacts = _pad_or_trim(raw_left_contacts, effective_raw_frames)
    right_contacts = _pad_or_trim(raw_right_contacts, effective_raw_frames)

    left_contacts = _crossfade_foot_contacts(left_contacts)
    right_contacts = _crossfade_foot_contacts(right_contacts)

    num_timesteps = raw_positions[::subsample_factor].shape[0]

    return (
        positions[::subsample_factor],
        orientations[::subsample_factor],
        left_contacts[::subsample_factor],
        right_contacts[::subsample_factor],
        num_timesteps,
    )


def save_contact_labels(
    output_path: Path,
    left_foot_contact: np.ndarray,
    right_foot_contact: np.ndarray,
    num_timesteps: int,
) -> None:
    left_contacts = np.asarray(left_foot_contact[:num_timesteps], dtype=np.float32)
    right_contacts = np.asarray(right_foot_contact[:num_timesteps], dtype=np.float32)
    foot_contacts = np.stack(
        [left_contacts.mean(axis=1), right_contacts.mean(axis=1)],
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


def _unwrap_joint_trajectory(joint_angles: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(joint_angles, axis=0)
    return np.clip(unwrapped, -2.0 * np.pi, 2.0 * np.pi).astype(np.float32, copy=False)


def _project_joint_trajectory_to_limits(
    joint_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Project solved hinge angles onto the nearest valid periodic branch.

    PyRoki optimizes in a periodic angle space, while the downstream MotionLib
    path expects every saved DOF sample to already live inside the model limits.
    Resolve any residual 2*pi branch ambiguity first, then clamp tiny numerical
    spillover back to the legal interval.
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    lower_limits = np.asarray(lower_limits, dtype=np.float32)
    upper_limits = np.asarray(upper_limits, dtype=np.float32)

    wrapped = ((joint_angles + np.pi) % (2.0 * np.pi)) - np.pi
    projected = np.empty_like(wrapped)
    branch_offsets = np.arange(-2, 3, dtype=np.float32) * (2.0 * np.pi)

    for dof_idx in range(wrapped.shape[1]):
        lower = lower_limits[dof_idx]
        upper = upper_limits[dof_idx]

        for frame_idx in range(wrapped.shape[0]):
            raw_angle = joint_angles[frame_idx, dof_idx]
            candidates = wrapped[frame_idx, dof_idx] + branch_offsets
            valid_candidates = candidates[
                (candidates >= lower - 1e-5) & (candidates <= upper + 1e-5)
            ]

            if valid_candidates.size == 0:
                projected[frame_idx, dof_idx] = np.clip(raw_angle, lower, upper)
            else:
                projected[frame_idx, dof_idx] = valid_candidates[
                    np.argmin(np.abs(valid_candidates - raw_angle))
                ]

    return projected.astype(np.float32, copy=False)


def _build_initial_guesses(
    positions: np.ndarray,
    orientations: np.ndarray,
    robot: pk.Robot,
) -> tuple[jaxlie.SE3, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    trusted_rotations, orientation_mask = _trusted_target_rotations(orientations)
    _, segment_defaults = _compute_zero_pose_reference(robot)

    joint_cfg = np.zeros(
        (positions.shape[0], robot.joints.num_actuated_joints),
        dtype=np.float32,
    )
    pelvis_idx = KEYPOINT_INDEX["pelvis"]
    left_hip_idx = KEYPOINT_INDEX["left_hip"]
    left_knee_idx = KEYPOINT_INDEX["left_knee"]
    left_ankle_idx = KEYPOINT_INDEX["left_ankle"]
    left_toe_idx = KEYPOINT_INDEX["left_foot"]
    right_hip_idx = KEYPOINT_INDEX["right_hip"]
    right_knee_idx = KEYPOINT_INDEX["right_knee"]
    right_ankle_idx = KEYPOINT_INDEX["right_ankle"]
    right_toe_idx = KEYPOINT_INDEX["right_foot"]

    for frame_idx in range(positions.shape[0]):
        root_rotation = trusted_rotations[frame_idx, ROBOT_LINK_NAMES.index("Pelvis")]

        dof_idx = 0
        for side, indices in (
            (
                "L",
                (left_hip_idx, left_knee_idx, left_ankle_idx, left_toe_idx),
            ),
            (
                "R",
                (right_hip_idx, right_knee_idx, right_ankle_idx, right_toe_idx),
            ),
        ):
            hip_idx, knee_idx, ankle_idx, toe_idx = indices

            thigh_world = positions[frame_idx, knee_idx] - positions[frame_idx, hip_idx]
            hip_local_rot = _rotation_between_vectors(
                segment_defaults[f"{side}_Hip"],
                root_rotation.T @ thigh_world,
                fallback_axis=root_rotation[:, 1],
            )
            joint_cfg[frame_idx, dof_idx : dof_idx + 3] = R.from_matrix(
                hip_local_rot
            ).as_euler("xyz", degrees=False)
            dof_idx += 3

            hip_global_rot = root_rotation @ hip_local_rot
            shank_world = positions[frame_idx, ankle_idx] - positions[frame_idx, knee_idx]
            knee_local_rot = _rotation_between_vectors(
                segment_defaults[f"{side}_Knee"],
                hip_global_rot.T @ shank_world,
                fallback_axis=hip_global_rot[:, 1],
            )
            joint_cfg[frame_idx, dof_idx : dof_idx + 3] = R.from_matrix(
                knee_local_rot
            ).as_euler("xyz", degrees=False)
            dof_idx += 3

            knee_global_rot = hip_global_rot @ knee_local_rot
            ankle_global_rot = trusted_rotations[
                frame_idx, ROBOT_LINK_NAMES.index(f"{side}_Ankle")
            ]
            ankle_local_rot = knee_global_rot.T @ ankle_global_rot
            ankle_local_rot = _orthonormalize(ankle_local_rot)
            joint_cfg[frame_idx, dof_idx : dof_idx + 3] = R.from_matrix(
                ankle_local_rot
            ).as_euler("xyz", degrees=False)
            dof_idx += 3

            toe_global_rot = trusted_rotations[
                frame_idx, ROBOT_LINK_NAMES.index(f"{side}_Toe")
            ]
            toe_local_rot = ankle_global_rot.T @ toe_global_rot
            toe_local_rot = _orthonormalize(toe_local_rot)
            joint_cfg[frame_idx, dof_idx : dof_idx + 3] = R.from_matrix(
                toe_local_rot
            ).as_euler("xyz", degrees=False)
            dof_idx += 3

    joint_cfg = _unwrap_joint_trajectory(joint_cfg)
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
    joint_cfg = np.clip(joint_cfg, lower[None, :], upper[None, :])

    root_values = []
    for frame_idx in range(positions.shape[0]):
        root_values.append(
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(trusted_rotations[frame_idx, ROBOT_LINK_NAMES.index("Pelvis")]),
                jnp.asarray(positions[frame_idx, pelvis_idx]),
            )
        )

    root_init_values = jaxlie.SE3(
        jnp.stack([transform.wxyz_xyz for transform in root_values], axis=0)
    )
    return (
        root_init_values,
        jnp.asarray(joint_cfg),
        jnp.asarray(trusted_rotations),
        jnp.asarray(orientation_mask),
    )


def _build_retarget_mask() -> jnp.ndarray:
    n_retarget = len(LOWER_BODY_KEYPOINT_NAMES)
    mask = jnp.zeros((n_retarget, n_retarget))
    for link_a, link_b, weight in DIRECT_PAIRS:
        idx_a = KEYPOINT_INDEX[link_a]
        idx_b = KEYPOINT_INDEX[link_b]
        mask = mask.at[idx_a, idx_b].set(weight)
        mask = mask.at[idx_b, idx_a].set(weight)
    return mask


@jaxls.Cost.factory
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


@jaxls.Cost.factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_T_world_root_curr: jaxls.SE3Var,
    var_T_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    foot_link_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root_curr = var_values[var_T_world_root_curr]
    T_world_root_prev = var_values[var_T_world_root_prev]

    T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(var_values[var_robot_cfg_curr]))
    T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(var_values[var_robot_cfg_prev]))

    T_world_link_curr = T_world_root_curr @ T_root_link_curr
    T_world_link_prev = T_world_root_prev @ T_root_link_prev

    robot_positions_curr = T_world_link_curr.translation()
    robot_positions_prev = T_world_link_prev.translation()

    left_ankle_idx, left_toe_idx, right_ankle_idx, right_toe_idx = foot_link_indices
    left_ankle_curr = robot_positions_curr[left_ankle_idx]
    left_toe_curr = robot_positions_curr[left_toe_idx]
    right_ankle_curr = robot_positions_curr[right_ankle_idx]
    right_toe_curr = robot_positions_curr[right_toe_idx]

    left_ankle_prev = robot_positions_prev[left_ankle_idx]
    left_toe_prev = robot_positions_prev[left_toe_idx]
    right_ankle_prev = robot_positions_prev[right_ankle_idx]
    right_toe_prev = robot_positions_prev[right_toe_idx]

    left_ankle_vel = (left_ankle_curr - left_ankle_prev) * left_foot_contact[0]
    left_toe_vel = (left_toe_curr - left_toe_prev) * left_foot_contact[1]
    right_ankle_vel = (right_ankle_curr - right_ankle_prev) * right_foot_contact[0]
    right_toe_vel = (right_toe_curr - right_toe_prev) * right_foot_contact[1]

    left_pair_weight = jnp.maximum(left_foot_contact[0], left_foot_contact[1])
    right_pair_weight = jnp.maximum(right_foot_contact[0], right_foot_contact[1])
    left_z_consistency = left_pair_weight * (left_ankle_curr[2] - left_toe_curr[2])
    right_z_consistency = right_pair_weight * (right_ankle_curr[2] - right_toe_curr[2])

    return (
        jnp.concatenate(
            [
                left_ankle_vel.flatten(),
                left_toe_vel.flatten(),
                right_ankle_vel.flatten(),
                right_toe_vel.flatten(),
                jnp.array([left_z_consistency]),
                jnp.array([right_z_consistency]),
            ]
        )
        * weight
    )


@jaxls.Cost.factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    ankle_link_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root = var_values[var_T_world_root]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(var_values[var_robot_cfg]))
    T_world_link = T_world_root @ T_root_link

    left_ankle_idx, right_ankle_idx = ankle_link_indices
    ankle_rotations = T_world_link.rotation().as_matrix()

    left_ankle_rot = ankle_rotations[left_ankle_idx]
    right_ankle_rot = ankle_rotations[right_ankle_idx]

    left_weight = jnp.maximum(left_foot_contact[0], left_foot_contact[1])
    right_weight = jnp.maximum(right_foot_contact[0], right_foot_contact[1])

    left_residual = left_weight * (left_ankle_rot[2, 2] - 1.0)
    right_residual = right_weight * (right_ankle_rot[2, 2] - 1.0)
    return jnp.array([left_residual, right_residual]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_positions: jnp.ndarray,
    target_rotations: jnp.ndarray,
    orientation_mask: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    root_init_values: jaxlie.SE3,
    joint_init_values: jnp.ndarray,
    robot_link_indices: jnp.ndarray,
    retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    dt: float,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    timesteps = target_positions.shape[0]
    n_retarget = len(robot_link_indices)

    class LowerBodyScaleVar(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_T_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = LowerBodyScaleVar(jnp.zeros(timesteps))

    foot_link_indices = jnp.array(
        [
            robot.links.names.index("L_Ankle"),
            robot.links.names.index("L_Toe"),
            robot.links.names.index("R_Ankle"),
            robot.links.names.index("R_Toe"),
        ]
    )
    ankle_link_indices = jnp.array(
        [
            robot.links.names.index("L_Ankle"),
            robot.links.names.index("R_Ankle"),
        ]
    )

    @jaxls.Cost.factory
    def local_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: LowerBodyScaleVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = var_values[var_T_world_root] @ T_root_link

        robot_pos = T_world_link.translation()[robot_link_indices]
        delta_target = keypoints[:, None] - keypoints[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        position_scale = var_values[var_joints_scale][..., None]
        residual_position_delta = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(delta_target.shape[0])[..., None])
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
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * retarget_mask
        )

        return (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )

    @jaxls.Cost.factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: LowerBodyScaleVar,
    ) -> jax.Array:
        res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
        res_1 = (
            var_values[var_joints_scale] - var_values[var_joints_scale].T
        ).flatten() * 100.0
        res_2 = jnp.clip(-var_values[var_joints_scale], min=0.0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.factory
    def global_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = var_values[var_T_world_root] @ T_root_link
        robot_pos = T_world_link.translation()[robot_link_indices]
        return (robot_pos - keypoints).flatten() * weights["global_alignment"]

    @jaxls.Cost.factory
    def orientation_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        rotations_world: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = var_values[var_T_world_root] @ T_root_link
        robot_rot = T_world_link.rotation().as_matrix()[robot_link_indices]
        relative_rot = jnp.einsum(
            "nij,njk->nik",
            jnp.swapaxes(robot_rot, 1, 2),
            rotations_world,
        )
        return (
            jaxlie.SO3.from_matrix(relative_rot).log().flatten()
            * jnp.repeat(orientation_mask, 3)
            * weights["orientation_alignment"]
        )

    @jaxls.Cost.factory
    def root_reference_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        target_root: jaxlie.SE3,
    ) -> jax.Array:
        return (
            (var_values[var_T_world_root].inverse() @ target_root).log().flatten()
            * weights["root_reference"]
        )

    @jaxls.Cost.factory
    def joint_reference_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        reference_cfg: jnp.ndarray,
    ) -> jax.Array:
        return (
            (var_values[var_robot_cfg] - reference_cfg).flatten()
            * weights["joint_reference"]
        )

    @jaxls.Cost.factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_T_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (
            (var_values[var_T_world_root].inverse() @ var_values[var_T_world_root_prev])
            .log()
            .flatten()
            * weights["root_smoothness"]
        )

    costs: list[jaxls.Cost] = [
        local_alignment_cost(
            var_T_world_root,
            var_joints,
            var_joints_scale,
            target_positions,
        ),
        scale_regularization(var_joints_scale),
        global_alignment_cost(var_T_world_root, var_joints, target_positions),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            weights["limit_cost"],
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
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
                left_foot_contact[t],
                right_foot_contact[t],
                foot_link_indices,
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
                ankle_link_indices,
                weights["foot_tilt"],
            )
        )

    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[var_joints, var_T_world_root, var_joints_scale],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(joint_init_values),
                    var_T_world_root.with_value(root_init_values),
                    var_joints_scale,
                ]
            ),
            verbose=False,
            termination=jaxls.TerminationConfig(max_iterations=400),
        )
    )
    return solution[var_T_world_root], solution[var_joints]


def _run_single_motion(
    robot: pk.Robot,
    motion_path: Path,
    output_path: Path,
    *,
    subsample_factor: int,
    target_raw_frames: int,
    retarget_fps: int,
    weights: RetargetingWeights,
    robot_link_indices: jnp.ndarray,
    retarget_mask: jnp.ndarray,
) -> None:
    (
        keypoints,
        orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    ) = load_motion_data(
        motion_path=motion_path,
        subsample_factor=subsample_factor,
        target_raw_frames=target_raw_frames,
    )

    (
        root_init_values,
        joint_init_values,
        target_rotations,
        orientation_mask,
    ) = _build_initial_guesses(
        positions=keypoints,
        orientations=orientations,
        robot=robot,
    )

    pelvis_speed_mps = 0.0
    if keypoints.shape[0] > 1:
        pelvis_speed_mps = float(
            np.linalg.norm(np.diff(keypoints[:, KEYPOINT_INDEX["pelvis"]], axis=0), axis=1).mean()
            * retarget_fps
        )
    motion_scale = min(max(pelvis_speed_mps / 1.5, 1.0), 3.0)
    tracking_scale = min(np.sqrt(motion_scale), 1.75)
    regularization_scale = 1.0 / motion_scale
    contact_scale = 1.0 / np.sqrt(motion_scale)
    motion_weights = RetargetingWeights(
        local_alignment=weights["local_alignment"] * tracking_scale,
        global_alignment=weights["global_alignment"] * tracking_scale,
        orientation_alignment=weights["orientation_alignment"] * regularization_scale,
        root_smoothness=weights["root_smoothness"] * regularization_scale,
        joint_smoothness=weights["joint_smoothness"] * regularization_scale,
        joint_reference=weights["joint_reference"] * regularization_scale,
        root_reference=weights["root_reference"] * regularization_scale,
        joint_vel_limit=weights["joint_vel_limit"] * regularization_scale,
        limit_cost=weights["limit_cost"],
        foot_contact=weights["foot_contact"] * contact_scale,
        foot_tilt=weights["foot_tilt"] * contact_scale,
    )
    print(
        f"Retarget tuning for {motion_path.name}: pelvis_speed={pelvis_speed_mps:.2f} m/s, "
        f"motion_scale={motion_scale:.2f}, tracking_scale={tracking_scale:.2f}"
    )

    Ts_world_root, joints = solve_retargeting(
        robot=robot,
        target_positions=jnp.asarray(keypoints),
        target_rotations=jnp.asarray(target_rotations),
        orientation_mask=jnp.asarray(orientation_mask),
        left_foot_contact=jnp.asarray(left_foot_contact),
        right_foot_contact=jnp.asarray(right_foot_contact),
        root_init_values=root_init_values,
        joint_init_values=joint_init_values,
        robot_link_indices=robot_link_indices,
        retarget_mask=retarget_mask,
        weights=motion_weights,
        dt=float(subsample_factor) / float(retarget_fps),
    )

    joint_angles = _project_joint_trajectory_to_limits(
        np.asarray(joints[:num_timesteps], dtype=np.float32),
        lower_limits=np.asarray(robot.joints.lower_limits, dtype=np.float32),
        upper_limits=np.asarray(robot.joints.upper_limits, dtype=np.float32),
    )
    max_joint_correction = float(
        np.max(np.abs(joint_angles - np.asarray(joints[:num_timesteps], dtype=np.float32)))
    )
    if max_joint_correction > 1e-6:
        print(
            f"Projected solved joint angles back into URDF limits for {motion_path.name} "
            f"(max correction {max_joint_correction:.6f} rad)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_frame_pos=np.asarray(Ts_world_root.wxyz_xyz[:num_timesteps, 4:], dtype=np.float32),
        base_frame_wxyz=np.asarray(Ts_world_root.wxyz_xyz[:num_timesteps, :4], dtype=np.float32),
        joint_angles=joint_angles,
        joint_names=np.asarray(robot.joints.actuated_names),
    )
    print(f"Saved retargeted motion to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyRoki-based lower-body retargeting from extracted keypoints"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Accepted for compatibility. Visualization is not implemented in this batch wrapper.",
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
        default=str(DEFAULT_URDF_PATH),
        help="Path to the lower-body URDF used by PyRoki.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Optional mesh directory forwarded to yourdfpy when loading the URDF.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Accepted for compatibility; custom weight files are not implemented.",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Subsample factor for the keypoints.",
    )
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=450,
        help="Target raw frames before subsampling. Pass -1 to use the full motion.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip motions that already have retargeted outputs.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="treadmill",
        help="Accepted for compatibility; lower-body wrapper expects extracted keypoints.",
    )
    parser.add_argument(
        "--retarget-fps",
        type=int,
        default=30,
        help="Nominal FPS used when evaluating velocity-related costs.",
    )
    parser.add_argument(
        "--save-contacts-only",
        action="store_true",
        help="Skip retargeting and only save processed foot contact labels.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints_folder_path}/contacts.",
    )
    args = parser.parse_args()

    keypoints_folder_path = Path(args.keypoints_folder_path)
    motion_paths = [
        Path(path)
        for path in sorted(glob.glob(str(keypoints_folder_path / "*.npy")))
    ]
    if not motion_paths:
        raise FileNotFoundError(f"No keypoint files found in {keypoints_folder_path}")

    if args.save_contacts_only:
        contacts_dir = (
            Path(args.contacts_dir)
            if args.contacts_dir is not None
            else keypoints_folder_path / "contacts"
        )
        contacts_dir.mkdir(parents=True, exist_ok=True)
        for motion_path in motion_paths:
            output_path = contacts_dir / f"{motion_path.stem}_contacts.npz"
            if args.skip_existing and output_path.exists():
                print(f"Skipping existing contact labels: {output_path}")
                continue

            _, _, left_contacts, right_contacts, num_timesteps = load_motion_data(
                motion_path=motion_path,
                subsample_factor=args.subsample_factor,
                target_raw_frames=args.target_raw_frames,
            )
            save_contact_labels(
                output_path=output_path,
                left_foot_contact=left_contacts,
                right_foot_contact=right_contacts,
                num_timesteps=num_timesteps,
            )
        return

    urdf = yourdfpy.URDF.load(args.urdf_path, mesh_dir=args.mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_link_indices = jnp.asarray(
        [robot.links.names.index(name) for name in ROBOT_LINK_NAMES],
        dtype=jnp.int32,
    )
    retarget_mask = _build_retarget_mask()

    weights = RetargetingWeights(
        local_alignment=4.0,
        global_alignment=8.0,
        orientation_alignment=0.0,
        root_smoothness=1.0,
        joint_smoothness=0.75,
        joint_reference=0.0,
        root_reference=0.0,
        joint_vel_limit=0.0,
        limit_cost=25.0,
        foot_contact=12.0,
        foot_tilt=0.5,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for motion_path in motion_paths:
        output_path = output_dir / f"{motion_path.stem}_retargeted.npz"
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing retargeted motion: {output_path}")
            continue

        _run_single_motion(
            robot=robot,
            motion_path=motion_path,
            output_path=output_path,
            subsample_factor=args.subsample_factor,
            target_raw_frames=args.target_raw_frames,
            retarget_fps=args.retarget_fps,
            weights=weights,
            robot_link_indices=robot_link_indices,
            retarget_mask=retarget_mask,
        )


if __name__ == "__main__":
    main()
