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
"""Bootstrap variant of BeyondMimic-style lower-body tracking."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig, ControlType
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    ObservationNoiseDomainRandomizationConfig,
    DomainRandomizationConfig,
)

import argparse


def _load_base_mlp_bm():
    module_path = Path(__file__).with_name("mlp_bm.py")
    spec = spec_from_file_location("mlp_bm_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base experiment from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MLP_BM = _load_base_mlp_bm()
agent_config = _BASE_MLP_BM.agent_config
apply_inference_overrides = _BASE_MLP_BM.apply_inference_overrides
motion_lib_config = _BASE_MLP_BM.motion_lib_config
scene_lib_config = _BASE_MLP_BM.scene_lib_config
terrain_config = _BASE_MLP_BM.terrain_config


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    """Build a safer bootstrap environment for lower-body teacher training."""

    env_cfg = _BASE_MLP_BM.env_config(robot_cfg, args)

    # Bootstrap from motion start instead of random phase. Random-phase resets are
    # useful later, but they make early lower-body contact stabilization much
    # harder and amplify rare pathological impact states across large env counts.
    env_cfg.motion_manager.init_start_prob = 1.0

    # Give the reference reset a slightly larger vertical cushion so feet do not
    # start directly on the collision boundary.
    env_cfg.ref_respawn_offset = 0.03

    return env_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure a safer bootstrap setup for lower-body teacher training."""

    robot_cfg.control.control_type = ControlType.BUILT_IN_PD
    # The untrained PPO actor produces effectively random target offsets in the
    # first few epochs. Keep the bootstrap action scale small so the lower-body
    # PD controller does not immediately drive the feet into the ground.
    robot_cfg.control.action_scale = 0.25

    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    # Keep only the lower-risk domain randomization terms during bootstrap.
    # Pushes and COM offsets are the two most likely to knock the lower-body
    # tracker into pathological foot-ground contact states early in training.
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.6, 3.0),
            dynamic_friction_range=(0.6, 3.0),
            restitution_range=(0.0, 1.0),
            body_names=[".*"],
            body_indices=None,
        ),
        observation_noise=ObservationNoiseDomainRandomizationConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
    )
