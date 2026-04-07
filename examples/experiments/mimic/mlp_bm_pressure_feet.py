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
"""BeyondMimic teacher experiment with Newton pressure-field feet enabled."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

import argparse


def _load_base_experiment():
    module_path = Path(__file__).with_name("mlp_bm.py")
    spec = spec_from_file_location("mimic_mlp_bm_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base experiment from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_EXPERIMENT = _load_base_experiment()
agent_config = _BASE_EXPERIMENT.agent_config
apply_inference_overrides = _BASE_EXPERIMENT.apply_inference_overrides
env_config = _BASE_EXPERIMENT.env_config
motion_lib_config = _BASE_EXPERIMENT.motion_lib_config
scene_lib_config = _BASE_EXPERIMENT.scene_lib_config
terrain_config = _BASE_EXPERIMENT.terrain_config


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Expose the main pressure-field knobs for teacher runs."""
    parser.add_argument(
        "--pressure-field-foot-kh",
        type=float,
        default=2.5e7,
        help="Hydroelastic stiffness for the configured foot contact bodies.",
    )
    parser.add_argument(
        "--pressure-field-foot-sdf-max-resolution",
        type=int,
        default=32,
        help="Sparse SDF resolution for the configured foot contact bodies.",
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Reuse the BeyondMimic setup and enable foot-only pressure-field contact."""
    _BASE_EXPERIMENT.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    if args.simulator != "newton":
        raise ValueError("This experiment requires --simulator newton.")

    simulator_cfg.sim.pressure_field_feet = True
    simulator_cfg.sim.pressure_field_foot_kh = args.pressure_field_foot_kh
    simulator_cfg.sim.pressure_field_foot_sdf_max_resolution = (
        args.pressure_field_foot_sdf_max_resolution
    )
