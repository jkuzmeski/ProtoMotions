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
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabPhysXParams, IsaacLabSimParams
from protomotions.simulator.newton.config import NewtonSimParams


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = REPO_ROOT / "protomotions" / "data" / "assets"
DEFAULT_RESCALE_ROOT = REPO_ROOT / "HumanRetargeting" / "rescale"
_HEIGHT_NAME_RE = re.compile(
    r"^smpl_lower_body(?:_(?P<height_cm>\d+)cm)?(?P<contact_pads>_contact_pads)?(?P<torque>_torque)?$"
)
_SUBJECT_NAME_RE = re.compile(r"^smpl_lower_body_subject_(?P<subject_id>.+)$")


def _repo_root_from_assets_root(assets_root: Path) -> Path:
    try:
        return assets_root.parents[2]
    except IndexError as exc:
        raise ValueError(f"Could not infer repo root from assets root {assets_root}") from exc


def _relative_to(base_dir: Path, target: Path) -> str:
    try:
        return str(target.relative_to(base_dir).as_posix())
    except ValueError:
        return str(target.as_posix())


def _resolve_asset_paths(
    *,
    asset_stem: str,
    assets_root: Path,
    repo_root: Path,
    allow_rescale_fallback: bool,
) -> tuple[str, str, str | None]:
    mjcf_path = assets_root / "mjcf" / f"{asset_stem}.xml"
    usda_path = assets_root / "usd" / f"{asset_stem}.usda"
    if mjcf_path.exists():
        return (
            str(assets_root),
            _relative_to(assets_root, mjcf_path),
            _relative_to(assets_root, usda_path) if usda_path.exists() else None,
        )

    if allow_rescale_fallback:
        rescale_xml = repo_root / "HumanRetargeting" / "rescale" / f"{asset_stem}.xml"
        rescale_usda = repo_root / "HumanRetargeting" / "rescale" / f"{asset_stem}.usda"
        if rescale_xml.exists():
            return (
                str(repo_root),
                _relative_to(repo_root, rescale_xml),
                _relative_to(repo_root, rescale_usda) if rescale_usda.exists() else None,
            )

    raise FileNotFoundError(
        f"Could not find lower-body MJCF for asset stem {asset_stem!r} under "
        f"{assets_root / 'mjcf'}"
    )


def _lower_body_control_overrides() -> dict[str, ControlInfo]:
    return {
        ".*_Hip_.*": ControlInfo(
            stiffness=250.0,
            damping=25.0,
            effort_limit=500.0,
            velocity_limit=100.0,
        ),
        ".*_Knee_.*": ControlInfo(
            stiffness=200.0,
            damping=20.0,
            effort_limit=500.0,
            velocity_limit=100.0,
        ),
        ".*_Ankle_.*": ControlInfo(
            stiffness=150.0,
            damping=15.0,
            effort_limit=500.0,
            velocity_limit=100.0,
        ),
        ".*_Toe_.*": ControlInfo(
            stiffness=75.0,
            damping=8.0,
            effort_limit=500.0,
            velocity_limit=100.0,
        ),
    }


@dataclass
class SmplLowerBodyRobotConfig(RobotConfig):
    trackable_bodies_subset: str = "all"
    contact_bodies: list[str] = field(
        default_factory=lambda: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )
    non_termination_contact_bodies: list[str] = field(
        default_factory=lambda: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )
    common_naming_to_robot_body_names: dict[str, list[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["L_Ankle", "L_Toe"],
            "all_right_foot_bodies": ["R_Ankle", "R_Toe"],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": ["Pelvis"],
            "torso_body_name": ["Pelvis"],
        }
    )
    default_root_height: float = 0.95
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="HumanRetargeting/rescale/smpl_humanoid_lower_body_adjusted_pd.xml",
            asset_root=str(REPO_ROOT),
            usd_asset_file_name="HumanRetargeting/rescale/smpl_humanoid_lower_body_adjusted_pd.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
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
    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(fps=60, decimation=2, substeps=2),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=4,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(fps=60, decimation=2, substeps=2),
            newton=NewtonSimParams(fps=120, decimation=4),
        )
    )


class SmplLowerBodyConfigFactory:
    """Create lower-body robot configs from asset metadata or conventional names."""

    @classmethod
    def create(
        cls,
        *,
        height_cm: int | None = None,
        variant: str = "adjusted_pd",
        asset_root: str | Path | None = None,
        contact_pads: bool = False,
        subject_id: str | None = None,
    ) -> SmplLowerBodyRobotConfig:
        assets_root = Path(asset_root).resolve() if asset_root is not None else DEFAULT_ASSETS_ROOT
        repo_root = _repo_root_from_assets_root(assets_root)

        if subject_id is not None:
            return cls._create_subject_config(subject_id=subject_id, assets_root=assets_root)

        asset_stem = cls._asset_stem_for_variant(
            height_cm=height_cm,
            variant=variant,
            contact_pads=contact_pads,
        )
        return cls._create_from_asset_stem(
            asset_stem=asset_stem,
            assets_root=assets_root,
            repo_root=repo_root,
            default_root_height=0.95 if height_cm is None else 0.95 * height_cm / 170.0,
            control_type=ControlType.TORQUE if variant == "adjusted_torque" else ControlType.BUILT_IN_PD,
            allow_rescale_fallback=True,
        )

    @classmethod
    def from_robot_name(
        cls,
        robot_name: str,
        *,
        repo_root: Path | None = None,
    ) -> SmplLowerBodyRobotConfig:
        repo_root = repo_root.resolve() if repo_root is not None else REPO_ROOT
        assets_root = repo_root / "protomotions" / "data" / "assets"

        subject_match = _SUBJECT_NAME_RE.fullmatch(robot_name)
        if subject_match is not None:
            return cls._create_subject_config(
                subject_id=subject_match.group("subject_id"),
                assets_root=assets_root,
            )

        height_match = _HEIGHT_NAME_RE.fullmatch(robot_name)
        if height_match is None:
            raise ValueError(f"Invalid lower-body robot name: {robot_name}")

        height_cm = height_match.group("height_cm")
        contact_pads = bool(height_match.group("contact_pads"))
        variant = "adjusted_torque" if height_match.group("torque") else "adjusted_pd"
        return cls.create(
            height_cm=int(height_cm) if height_cm is not None else None,
            variant=variant,
            asset_root=assets_root,
            contact_pads=contact_pads,
        )

    @classmethod
    def _asset_stem_for_variant(
        cls,
        *,
        height_cm: int | None,
        variant: str,
        contact_pads: bool,
    ) -> str:
        stem = f"smpl_humanoid_lower_body_{variant}"
        if height_cm is not None:
            stem += f"_height_{height_cm}cm"
        if contact_pads:
            stem += "_contact_pads"
        return stem

    @classmethod
    def _create_subject_config(
        cls,
        *,
        subject_id: str,
        assets_root: Path,
    ) -> SmplLowerBodyRobotConfig:
        asset_stem = f"smpl_humanoid_lower_body_subject_{subject_id}"
        metadata_path = assets_root / "subjects" / f"{asset_stem}.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Subject metadata not found for {subject_id!r}: {metadata_path}. "
                "Run the HumanRetargeting pipeline with --subject-profile first."
            )

        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        config = cls._create_from_asset_stem(
            asset_stem=asset_stem,
            assets_root=assets_root,
            repo_root=_repo_root_from_assets_root(assets_root),
            default_root_height=float(metadata.get("default_root_height", 0.95)),
            control_type=ControlType.BUILT_IN_PD,
            allow_rescale_fallback=False,
        )
        config.contact_bodies = list(
            metadata.get("contact_bodies", ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"])
        )
        config.non_termination_contact_bodies = list(
            metadata.get(
                "non_termination_contact_bodies",
                ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"],
            )
        )
        config.default_root_height = float(metadata.get("default_root_height", config.default_root_height))
        return config

    @classmethod
    def _create_from_asset_stem(
        cls,
        *,
        asset_stem: str,
        assets_root: Path,
        repo_root: Path,
        default_root_height: float,
        control_type: ControlType,
        allow_rescale_fallback: bool,
    ) -> SmplLowerBodyRobotConfig:
        asset_root, mjcf_rel_path, usd_rel_path = _resolve_asset_paths(
            asset_stem=asset_stem,
            assets_root=assets_root,
            repo_root=repo_root,
            allow_rescale_fallback=allow_rescale_fallback,
        )
        return SmplLowerBodyRobotConfig(
            default_root_height=default_root_height,
            asset=RobotAssetConfig(
                asset_root=asset_root,
                asset_file_name=mjcf_rel_path,
                usd_asset_file_name=usd_rel_path,
                usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                angular_damping=0.0,
                linear_damping=0.0,
            ),
            control=ControlConfig(
                control_type=control_type,
                override_control_info=_lower_body_control_overrides(),
            ),
        )


def build_smpl_lower_body_robot_config(robot_name: str, **updates: Any) -> SmplLowerBodyRobotConfig:
    config = SmplLowerBodyConfigFactory.from_robot_name(robot_name)
    if updates:
        config.update_fields(**updates)
    return config
