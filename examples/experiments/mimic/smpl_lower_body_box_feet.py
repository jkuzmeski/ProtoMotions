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
"""Flat-ground Newton mimic baseline for SMPL lower body with box feet.

This uses the default ``smpl_lower_body`` asset line, whose ankle and toe
collision geoms are already boxes in the adjusted lower-body MJCF.

Expected CLI:
    --robot-name smpl_lower_body
    --simulator newton
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

import argparse


def _load_base_mimic_mlp():
    module_path = Path(__file__).with_name("mlp.py")
    spec = spec_from_file_location("mimic_mlp_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base experiment from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MIMIC_MLP = _load_base_mimic_mlp()
agent_config = _BASE_MIMIC_MLP.agent_config
apply_inference_overrides = _BASE_MIMIC_MLP.apply_inference_overrides
env_config = _BASE_MIMIC_MLP.env_config
motion_lib_config = _BASE_MIMIC_MLP.motion_lib_config
scene_lib_config = _BASE_MIMIC_MLP.scene_lib_config


def _validate_args(args: argparse.Namespace) -> None:
    if args.robot_name != "smpl_lower_body":
        raise ValueError(
            "This experiment requires --robot-name smpl_lower_body because the "
            "default lower-body MJCF already provides the box-foot baseline."
        )
    if args.simulator != "newton":
        raise ValueError("This experiment requires --simulator newton.")


def terrain_config(args: argparse.Namespace):
    """Build a flat terrain config compatible with Newton's friction rules."""
    from protomotions.components.terrains.config import CombineMode, TerrainConfig

    _validate_args(args)

    terrain_cfg = TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        border_size=40.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
    terrain_cfg.sim_config.combine_mode = CombineMode.MAX
    return terrain_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Use the default lower-body box feet with standard point contacts."""
    _validate_args(args)

    robot_cfg.update_fields(
        contact_bodies=["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )
    simulator_cfg.sim.pressure_field_feet = False
