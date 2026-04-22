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
"""Configuration classes for evaluators."""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )
    max_foot_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Stop each episode after this many foot steps (left heel strikes). "
                "None = disabled (run for max_eval_steps or motion length). "
                "Useful for collecting a fixed number of gait cycles per motion."
            ),
            "min": 1,
        }
    )
    left_strike_min_interval_sec: float = field(
        default=0.20,
        metadata={
            "help": "Minimum time between accepted left-foot strikes to debounce contact chatter.",
            "min": 0.0,
        },
    )


@dataclass
class BiomechanicsEvaluatorConfig(EvaluatorConfig):
    """Configuration for speed-conditioned biomechanics evaluation."""

    _target_: str = "protomotions.agents.evaluators.biomechanics_evaluator.BiomechanicsEvaluator"
    target_speeds: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": (
                "Target forward speeds to evaluate in m/s. "
                "If omitted, the evaluator derives speeds from the motion metadata "
                "or the speed/steering control config."
            )
        },
    )
    episodes_per_speed: int = field(
        default=20,
        metadata={"help": "Number of episodes to run per target speed.", "min": 1},
    )
    max_eval_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Hard cap on steps per episode. None falls back to the environment's "
                "max_episode_length."
            )
        },
    )
    left_strike_min_interval_sec: float = field(
        default=0.20,
        metadata={
            "help": "Minimum time between accepted left-foot strikes to debounce contact chatter.",
            "min": 0.0,
        },
    )
    burn_in_speed_tolerance: float = field(
        default=0.10,
        metadata={
            "help": "Relative tolerance for cycle-mean forward speed during burn-in gating.",
            "min": 0.0,
        },
    )
    burn_in_consecutive_cycles: int = field(
        default=2,
        metadata={
            "help": "Number of consecutive in-range left-anchored cycles required to exit burn-in.",
            "min": 1,
        },
    )
    success_post_burn_in_cycles: int = field(
        default=10,
        metadata={
            "help": "Number of left-anchored cycles required after burn-in before success is recorded.",
            "min": 1,
        },
    )
    waveform_num_points: int = field(
        default=100,
        metadata={
            "help": "Number of normalized phase points used when exporting cycle waveforms.",
            "min": 8,
        },
    )
