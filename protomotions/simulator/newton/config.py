# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
from dataclasses import dataclass, field
from typing import Optional
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class NewtonSimParams(SimParams):
    """Newton/MuJoCo solver parameters."""

    solver: str = field(
        default="newton",
        metadata={"help": "Constraint solver: 'newton', 'cg', or 'direct'."}
    )
    integrator: str = field(
        default="implicitfast",
        metadata={"help": "Integrator: 'euler', 'implicit', or 'implicitfast'."}
    )
    iterations: int = field(
        default=100,
        metadata={"help": "Max solver iterations."}
    )
    ls_iterations: int = field(
        default=50,
        metadata={"help": "Line search iterations."}
    )
    ls_parallel: bool = field(
        default=True,
        metadata={"help": "Run line search in parallel."}
    )
    impratio: float = field(
        default=10.0,
        metadata={"help": "Implicit integration ratio."}
    )
    njmax: int = field(
        default=450,
        metadata={"help": "Max constraint Jacobian rows."}
    )
    nconmax: int = field(
        default=300,
        metadata={"help": "Max contacts."}
    )
    nccdmax: Optional[int] = field(
        default=None,
        metadata={"help": "Max CCD contact candidates per world. Defaults to nconmax when unset."}
    )
    naccdmax: Optional[int] = field(
        default=None,
        metadata={"help": "Global CCD contact candidate budget. Defaults to nccdmax * num_worlds when unset."}
    )
    cone: str = field(
        default="pyramidal",
        metadata={"help": "Friction cone: 'pyramidal' or 'elliptic'."}
    )
    ccd_iterations: int = field(
        default=200,
        metadata={"help": "CCD (continuous collision detection) iterations."}
    )
    use_mujoco_contacts: bool = field(
        default=False,
        metadata={
            "help": "Use MuJoCo-generated contacts inside SolverMuJoCo. Disable this to generate contacts via Newton's collision pipeline and feed them into step()."
        },
    )
    max_epa_workspace_iterations: Optional[int] = field(
        default=128,
        metadata={
            "help": "Cap for MuJoCo-Warp EPA scratch/workspace iterations. Keep this separate from ccd_iterations to avoid runaway GPU allocations in large batched runs."
        },
    )
    raise_on_mujoco_warning: bool = field(
        default=False,
        metadata={
            "help": "Raise immediately when MuJoCo emits a runtime warning during a Newton solver step. Enable this for bridge/debug investigations; it disables CUDA graph replay so warnings can fail fast at the offending step."
        },
    )
    raise_on_nonfinite: bool = field(
        default=False,
        metadata={
            "help": "Raise immediately when non-finite (NaN/Inf) values are detected in simulator state. Writes a diagnostic .pt dump to output/nonfinite_dumps/ before crashing. Use this to debug physics explosions instead of silently clamping and resetting."
        },
    )


@dataclass
class NewtonSimulatorConfig(SimulatorConfig):
    """Configuration specific to Newton simulator."""

    _target_: str = "protomotions.simulator.newton.simulator.NewtonSimulator"
    sim: NewtonSimParams = field(default_factory=NewtonSimParams)  # Override sim type
    w_last: bool = True  # Newton uses xyzw quaternions
    viewer_backend: str = field(
        default="gl",
        metadata={"help": "Newton viewer backend to use when headless=False: 'gl' or 'viser'."},
    )
    viewer_port: int = field(
        default=8097,
        metadata={"help": "Port used by the Newton viser viewer server."},
    )
    viewer_max_worlds: Optional[int] = field(
        default=16,
        metadata={
            "help": "Maximum number of Newton worlds to send to the viser viewer. "
            "None renders all worlds; lower values reduce browser-side choppiness."
        },
    )
    camera_follow_leash_xy: float = field(
        default=0.35,
        metadata={
            "help": "XY dead-zone radius in meters before camera anchor starts moving."
        },
    )
    camera_follow_vertical_tau: float = field(
        default=0.8,
        metadata={
            "help": "Vertical (Z) anchor smoothing time constant in seconds."
        },
    )
    camera_follow_max_speed: float = field(
        default=6.0,
        metadata={
            "help": "Maximum camera translation speed in meters per second."
        },
    )
    camera_follow_tau_target: float = field(
        default=0.12,
        metadata={
            "help": "Camera look-at target smoothing time constant in seconds."
        },
    )
