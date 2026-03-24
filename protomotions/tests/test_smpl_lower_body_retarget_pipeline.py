# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from HumanRetargeting.biomechanics_retarget.treadmill2overground import estimate_ground_height
from HumanRetargeting.biomechanics_retarget.batch_retarget_lower_body import (
    _build_reference_qpos,
    _compute_world_transforms_autograd,
    _select_continuous_joint_angles,
)
from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
)


def test_select_continuous_joint_angles_prefers_nearest_valid_periodic_branch():
    raw_joint_angles = torch.tensor(
        [
            [6.2726130],
            [0.0137840],
            [6.2817564],
            [0.0009052],
            [6.2824898],
        ],
        dtype=torch.float32,
    )
    lower_limits = torch.tensor([-0.17453292], dtype=torch.float32)
    upper_limits = torch.tensor([2.61799383], dtype=torch.float32)

    stabilized = _select_continuous_joint_angles(
        joint_angles=raw_joint_angles,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    expected = torch.tensor(
        [
            [-0.0105724],
            [0.0137840],
            [-0.0014289],
            [0.0009052],
            [-0.0006957],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(stabilized, expected, atol=1e-4)
    assert torch.all(stabilized >= lower_limits)
    assert torch.all(stabilized <= upper_limits)


def test_estimate_ground_height_ignores_non_foot_outlier():
    joint_centers = np.zeros((2, 9, 3), dtype=np.float32)
    joint_centers[:, 0, 2] = np.array([-0.25, -0.2], dtype=np.float32)
    joint_centers[:, 3, 2] = np.array([0.01, 0.02], dtype=np.float32)
    joint_centers[:, 4, 2] = np.array([0.0, 0.01], dtype=np.float32)
    joint_centers[:, 7, 2] = np.array([0.015, 0.025], dtype=np.float32)
    joint_centers[:, 8, 2] = np.array([0.005, 0.02], dtype=np.float32)

    ground_height = estimate_ground_height(joint_centers)

    assert abs(ground_height - 0.00035) < 1e-4


def test_autograd_world_transforms_matches_reference_fk():
    model_xml = (
        "protomotions/data/assets/mjcf/"
        "smpl_humanoid_lower_body_subject_S_GENERIC.xml"
    )
    kinematic_info = extract_kinematic_info(model_xml)

    qpos = torch.zeros((2, kinematic_info.nq), dtype=torch.float32)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    qpos[1, 0] = 0.1
    qpos[1, 7:10] = torch.tensor([0.05, 0.1, -0.03], dtype=torch.float32)
    qpos.requires_grad_(True)

    root_pos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    ref_pos, ref_rot = compute_forward_kinematics_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos.detach(),
        joint_rot_mats=joint_rot_mats.detach(),
    )
    test_pos, test_rot = _compute_world_transforms_autograd(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
    )

    assert torch.allclose(test_pos.detach(), ref_pos, atol=1e-6)
    assert torch.allclose(test_rot.detach(), ref_rot, atol=1e-6)

    loss = test_pos.sum() + test_rot.sum()
    loss.backward()
    assert qpos.grad is not None


def test_build_reference_qpos_smooths_joint_steps_and_normalizes_root_quaternions():
    lower_limits = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    upper_limits = torch.tensor([1.0, 1.0], dtype=torch.float32)
    initial_qpos = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.8, 0.8],
            [0.1, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, -0.8],
            [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.8, 0.8],
            [0.3, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.8, -0.8],
            [0.4, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.8, 0.8],
        ],
        dtype=torch.float32,
    )

    reference_qpos = _build_reference_qpos(
        initial_qpos,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        window_size=3,
    )

    initial_max_joint_step = (initial_qpos[1:, 7:] - initial_qpos[:-1, 7:]).abs().max()
    reference_max_joint_step = (reference_qpos[1:, 7:] - reference_qpos[:-1, 7:]).abs().max()

    assert reference_max_joint_step < initial_max_joint_step
    assert torch.allclose(reference_qpos[:, 3:7].norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(reference_qpos[:, 7:] >= lower_limits)
    assert torch.all(reference_qpos[:, 7:] <= upper_limits)
