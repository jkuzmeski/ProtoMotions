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

import numpy as np
import torch

from protomotions.components.pose_lib import (
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    left_contacts = np.mean(np.asarray(keypoint_data["left_foot_contacts"], dtype=np.float32), axis=1)
    right_contacts = np.mean(np.asarray(keypoint_data["right_foot_contacts"], dtype=np.float32), axis=1)
    foot_contacts = np.stack([left_contacts, right_contacts], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path}")


def _retarget_motion(
    keypoint_path: Path,
    output_path: Path,
    *,
    model_xml: Path,
) -> None:
    keypoint_data = np.load(keypoint_path, allow_pickle=True).item()
    positions = np.asarray(keypoint_data["positions"], dtype=np.float32)
    orientations = np.asarray(keypoint_data["orientations"], dtype=np.float32)

    global_rotations = _compute_global_rotations(positions, orientations)
    root_pos = torch.from_numpy(positions[:, KEYPOINT_INDEX["pelvis"]])
    global_rotations_torch = torch.from_numpy(global_rotations)

    kinematic_info = extract_kinematic_info(str(model_xml))
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

    joint_angles = qpos[:, 7:]
    lower = kinematic_info.dof_limits_lower.to(dtype=joint_angles.dtype)
    upper = kinematic_info.dof_limits_upper.to(dtype=joint_angles.dtype)
    joint_angles = torch.clamp(joint_angles, min=lower, max=upper)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_frame_pos=qpos[:, :3].cpu().numpy().astype(np.float32),
        base_frame_wxyz=qpos[:, 3:7].cpu().numpy().astype(np.float32),
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
        help="Accepted for compatibility with the PyRoki pipeline. Ignored by this script.",
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
        help="Accepted for compatibility with the original PyRoki scripts.",
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
        _retarget_motion(keypoint_path, output_path, model_xml=model_xml)


if __name__ == "__main__":
    main()
