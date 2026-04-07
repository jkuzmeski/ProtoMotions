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
"""Biomechanics evaluation config for SMPL lower-body ellipsoid feet with point contact."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import argparse

from protomotions.robot_configs.base import RobotConfig


def _load_base_experiment():
    module_path = Path(__file__).with_name("smpl_lower_body_ellipsoid_feet_point.py")
    spec = spec_from_file_location(
        "smpl_lower_body_ellipsoid_feet_point_base", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base experiment from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_EXPERIMENT = _load_base_experiment()
configure_robot_and_simulator = _BASE_EXPERIMENT.configure_robot_and_simulator
motion_lib_config = _BASE_EXPERIMENT.motion_lib_config
scene_lib_config = _BASE_EXPERIMENT.scene_lib_config
terrain_config = _BASE_EXPERIMENT.terrain_config


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Expose evaluator-specific controls for contact analysis exports."""
    parser.add_argument(
        "--heading-theta",
        type=float,
        default=0.0,
        help="Fixed world-frame heading angle used by the speed control during evaluation.",
    )
    parser.add_argument(
        "--standing-reset-steps",
        type=int,
        default=0,
        help="Number of post-reset steps to hold zero speed before gait evaluation.",
    )
    parser.add_argument(
        "--biomechanics-episodes-per-speed",
        type=int,
        default=20,
        help="Episodes to collect for each target speed during biomechanics evaluation.",
    )
    parser.add_argument(
        "--biomechanics-max-eval-steps",
        type=int,
        default=0,
        help="Optional hard cap on evaluation steps. Use 0 to keep the environment default.",
    )
    parser.add_argument(
        "--biomechanics-waveform-num-points",
        type=int,
        default=100,
        help="Number of normalized gait-phase samples exported per waveform.",
    )
    parser.add_argument(
        "--contact-analysis-num-bins-x",
        type=int,
        default=24,
        help="Number of pressure-map bins along the foot length.",
    )
    parser.add_argument(
        "--contact-analysis-num-bins-y",
        type=int,
        default=12,
        help="Number of pressure-map bins along the foot width.",
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    """Add motion-derived speed control to the flat-ground mimic setup."""
    from protomotions.envs.control.speed_control import SpeedControlConfig

    env_cfg = _BASE_EXPERIMENT.env_config(robot_cfg, args)
    control_components = dict(getattr(env_cfg, "control_components", {}) or {})
    control_components["speed"] = SpeedControlConfig(
        speed_source="motion_file",
        heading_theta=float(getattr(args, "heading_theta", 0.0)),
        standing_reset_steps=int(getattr(args, "standing_reset_steps", 0)),
    )
    env_cfg.control_components = control_components
    return env_cfg


def agent_config(robot_config: RobotConfig, env_cfg, args: argparse.Namespace):
    """Swap the standard mimic evaluator for the biomechanics evaluator."""
    from protomotions.agents.evaluators.config import BiomechanicsEvaluatorConfig

    agent_cfg = _BASE_EXPERIMENT.agent_config(robot_config, env_cfg, args)
    max_eval_steps = int(getattr(args, "biomechanics_max_eval_steps", 0))
    agent_cfg.evaluator = BiomechanicsEvaluatorConfig(
        episodes_per_speed=int(getattr(args, "biomechanics_episodes_per_speed", 20)),
        max_eval_steps=max_eval_steps if max_eval_steps > 0 else None,
        waveform_num_points=int(getattr(args, "biomechanics_waveform_num_points", 100)),
        contact_analysis_num_bins_x=int(getattr(args, "contact_analysis_num_bins_x", 24)),
        contact_analysis_num_bins_y=int(getattr(args, "contact_analysis_num_bins_y", 12)),
    )
    return agent_cfg


apply_inference_overrides = _BASE_EXPERIMENT.apply_inference_overrides
