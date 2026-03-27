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
"""Fixed-speed, fixed-heading control component.

This component provides a deterministic locomotion target with:
- A constant forward heading in world frame
- A constant target speed
- Standing-reset semantics to let the robot settle before motion begins

It does not depend on the motion manager.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class SpeedControlConfig(ControlComponentConfig):
    """Configuration for the fixed-speed control component.

    Attributes:
        target_speed: Desired forward speed in meters per second.
        speed_source: Source of target speed. ``fixed`` uses ``target_speed`` for all
            environments, while ``motion_file`` resolves one speed per sampled motion
            from metadata sidecars with filename fallback.
        heading_theta: Fixed world-frame heading angle in radians.
            ``0.0`` means forward along +X.
        standing_reset_steps: Number of post-reset steps to hold speed at zero.
            This lets the robot stand before the fixed-speed target activates.
    """

    _target_: str = "protomotions.envs.control.speed_control.SpeedControl"

    target_speed: float = 1.0
    speed_source: str = "fixed"
    heading_theta: float = 0.0
    standing_reset_steps: int = 0


class SpeedControl(ControlComponent):
    """Deterministic speed control with a fixed heading and fixed speed."""

    config: SpeedControlConfig

    def __init__(self, config: SpeedControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config = config
        self._validate_config()

        num_envs = self.env.num_envs
        device = self.env.device

        self._tar_dir_theta = torch.full(
            (num_envs,),
            float(self.config.heading_theta),
            device=device,
            dtype=torch.float32,
        )
        self._tar_dir = torch.zeros(num_envs, 2, device=device, dtype=torch.float32)
        self._tar_speed = torch.full(
            (num_envs,),
            float(self.config.target_speed),
            device=device,
            dtype=torch.float32,
        )
        self._commanded_speed = self._tar_speed.clone()
        self._standing_steps_remaining = torch.zeros(
            num_envs, device=device, dtype=torch.int64
        )
        self._motion_id_to_speed = self._build_motion_id_to_speed()

        self._set_heading(torch.arange(num_envs, device=device, dtype=torch.long))
        if self.config.standing_reset_steps > 0:
            self._tar_speed.zero_()

    def _validate_config(self) -> None:
        if not math.isfinite(self.config.target_speed):
            raise ValueError("SpeedControlConfig.target_speed must be finite")
        if self.config.target_speed < 0:
            raise ValueError("SpeedControlConfig.target_speed must be non-negative")
        if self.config.speed_source not in {"fixed", "motion_file"}:
            raise ValueError(
                "SpeedControlConfig.speed_source must be 'fixed' or 'motion_file'"
            )
        if not math.isfinite(self.config.heading_theta):
            raise ValueError("SpeedControlConfig.heading_theta must be finite")
        if self.config.standing_reset_steps < 0:
            raise ValueError("SpeedControlConfig.standing_reset_steps must be non-negative")

    def _build_motion_id_to_speed(self) -> torch.Tensor | None:
        if self.config.speed_source != "motion_file":
            return None

        motion_lib = getattr(self.env, "motion_lib", None)
        if motion_lib is None or len(getattr(motion_lib, "motion_files", ())) == 0:
            raise ValueError(
                "SpeedControlConfig.speed_source='motion_file' requires env.motion_lib with motion_files"
            )

        from HumanRetargeting.biomechanics_retarget.subject_profiles import (
            load_json_metadata,
            resolve_trial_speed_mps,
        )

        speeds = []
        for motion_file in motion_lib.motion_files:
            motion_path = Path(str(motion_file))
            metadata_path = motion_path.parent / "metadata" / f"{motion_path.stem}.json"
            metadata = load_json_metadata(metadata_path)
            speed_mps = resolve_trial_speed_mps(
                motion_path.stem,
                speed_mps=metadata.get("speed_mps"),
                metadata=metadata,
            )
            if speed_mps is None:
                raise ValueError(
                    f"Could not resolve speed for motion {motion_path} from metadata or filename"
                )
            speeds.append(float(speed_mps))

        return torch.tensor(speeds, device=self.env.device, dtype=torch.float32)

    def _resolve_commanded_speeds(self, env_ids: Tensor) -> Tensor:
        if self.config.speed_source == "fixed":
            return torch.full(
                (len(env_ids),),
                float(self.config.target_speed),
                device=self.env.device,
                dtype=torch.float32,
            )

        motion_manager = getattr(self.env, "motion_manager", None)
        if motion_manager is None or self._motion_id_to_speed is None:
            raise ValueError(
                "SpeedControlConfig.speed_source='motion_file' requires env.motion_manager and loaded motion speeds"
            )

        motion_ids = motion_manager.motion_ids[env_ids].long()
        return self._motion_id_to_speed[motion_ids]

    def _set_heading(self, env_ids: Tensor) -> None:
        theta = float(self.config.heading_theta)
        self._tar_dir_theta[env_ids] = theta
        self._tar_dir[env_ids, 0] = math.cos(theta)
        self._tar_dir[env_ids, 1] = math.sin(theta)

    def _activate_speed(self, env_ids: Tensor) -> None:
        self._tar_speed[env_ids] = self._commanded_speed[env_ids]
        self._standing_steps_remaining[env_ids] = 0

    def reset(self, env_ids: Tensor):
        """Reset the target for the given environments."""
        if len(env_ids) == 0:
            return

        self._set_heading(env_ids)
        self._commanded_speed[env_ids] = self._resolve_commanded_speeds(env_ids)

        if self.config.standing_reset_steps > 0:
            self._standing_steps_remaining[env_ids] = int(self.config.standing_reset_steps)
            self._tar_speed[env_ids] = 0.0
        else:
            self._activate_speed(env_ids)

    def step(self):
        """Advance standing-reset state and keep the target deterministic."""
        if self.config.standing_reset_steps <= 0:
            return

        standing_mask = self._standing_steps_remaining > 0
        if not torch.any(standing_mask):
            return

        self._standing_steps_remaining[standing_mask] -= 1
        ready_mask = standing_mask & (self._standing_steps_remaining == 0)
        if torch.any(ready_mask):
            self._activate_speed(ready_mask.nonzero(as_tuple=False).flatten())

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Speed control does not impose its own resets or terminations."""
        device = self.env.device
        num_envs = self.env.num_envs
        return (
            torch.zeros(num_envs, dtype=torch.bool, device=device),
            torch.zeros(num_envs, dtype=torch.bool, device=device),
        )

    def get_context(self) -> Dict[str, Any]:
        """Expose the deterministic target state for observation/reward functions."""
        return {
            "tar_dir": self._tar_dir,
            "tar_dir_theta": self._tar_dir_theta,
            "tar_speed": self._tar_speed,
            "commanded_tar_speed": self._commanded_speed,
            "standing_reset_steps_remaining": self._standing_steps_remaining,
            "is_standing": self._standing_steps_remaining > 0,
        }
