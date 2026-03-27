# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""Observation helpers for fixed-speed locomotion tasks.

The observation exposes the target speed together with the fixed forward heading
in the robot's heading-aligned frame, mirroring the style used by the steering
observation helper.
"""

import torch
from torch import Tensor

from protomotions.envs.obs.observation_component import ObservationComponentConfig
from protomotions.utils import rotations


def speed_obs_factory() -> ObservationComponentConfig:
    """Factory for the fixed-speed observation component."""
    return ObservationComponentConfig(
        function=compute_speed_obs,
        variables={
            "root_rot": "current_state_root_rot",
            "tar_dir": "tar_dir",
            "tar_speed": "tar_speed",
        },
    )


def compute_speed_obs(
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
) -> Tensor:
    """Compute speed-control observations in the robot's heading-aligned frame.

    Returns:
        Observation tensor with shape ``[num_envs, 3]``:
        ``[tar_speed, local_forward_heading_x, local_forward_heading_y]``.
    """
    if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
        raise ValueError(f"root_rot must have shape [num_envs, 4], got {root_rot.shape}")
    if tar_dir.ndim != 2 or tar_dir.shape[-1] != 2:
        raise ValueError(f"tar_dir must have shape [num_envs, 2], got {tar_dir.shape}")
    if tar_speed.ndim != 1:
        raise ValueError(f"tar_speed must have shape [num_envs], got {tar_speed.shape}")
    if root_rot.shape[0] != tar_dir.shape[0] or root_rot.shape[0] != tar_speed.shape[0]:
        raise ValueError("root_rot, tar_dir, and tar_speed must have matching batch sizes")

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)

    heading_rot = rotations.calc_heading_quat_inv(root_rot, True)
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)[..., 0:2]

    return torch.cat([tar_speed.unsqueeze(-1), local_tar_dir], dim=-1)

