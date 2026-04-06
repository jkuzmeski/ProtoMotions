# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from protomotions.robot_configs.base import ControlConfig, ControlType, RobotAssetConfig
from protomotions.robot_configs.smpl_lower_body import (
    REPO_ROOT,
    SmplLowerBodyRobotConfig,
    _lower_body_control_overrides,
)


@dataclass
class SmplLowerBodyEllipsoidFeetRobotConfig(SmplLowerBodyRobotConfig):
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root=str(REPO_ROOT / "protomotions" / "data" / "assets"),
            asset_file_name="mjcf/smpl_humanoid_lower_body_ellipsoid_feet.xml",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=_lower_body_control_overrides(),
        )
    )
