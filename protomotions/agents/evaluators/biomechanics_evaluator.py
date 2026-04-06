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
"""Speed-conditioned biomechanics evaluator.

This evaluator runs a locomotion policy at one or more target speeds and
collects cycle-level gait metrics from simulator contact state.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import BiomechanicsEvaluatorConfig
from protomotions.envs.control.speed_control import SpeedControl
from protomotions.envs.control.steering_control import SteeringControl
from protomotions.utils import rotations


@dataclass
class CycleRecord:
    env_id: int
    episode_index: int
    target_speed: float
    cycle_index: int
    cycle_type: str
    duration_s: float
    mean_forward_speed_mps: float
    stride_length_m: float
    cadence_steps_per_min: float
    feature_waveforms: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    env_id: int
    episode_index: int
    target_speed: float
    success: bool
    burn_in_complete: bool
    steps: int
    total_cycles: int
    post_burn_in_cycles: int
    failure_reason: str
    cycles: List[CycleRecord] = field(default_factory=list)


@dataclass
class SpeedResult:
    target_speed: float
    episodes_requested: int
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def cycles(self) -> List[CycleRecord]:
        cycles: List[CycleRecord] = []
        for episode in self.episodes:
            cycles.extend(episode.cycles)
        return cycles


@dataclass
class _EpisodeTracker:
    env_id: int
    episode_index: int
    target_speed: float
    prev_left_contact: bool = False
    last_strike_step: int = -10**9
    cycle_started: bool = False
    burn_in_consecutive: int = 0
    burn_in_complete: bool = False
    post_burn_in_cycles: int = 0
    cycle_index: int = 0
    steps: int = 0
    finished: bool = False
    success: bool = False
    failure_reason: str = ""
    cycle_buffers: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    cycles: List[CycleRecord] = field(default_factory=list)


class BiomechanicsEvaluator(BaseEvaluator):
    """Speed-conditioned gait evaluator for biomechanics analysis."""

    _CYCLE_NORMALIZED_JOINTS: Tuple[str, ...] = ("hip", "knee", "ankle")

    def __init__(self, agent: Any, fabric: Any, config: BiomechanicsEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self.config: BiomechanicsEvaluatorConfig = config
        self._speed_results: List[SpeedResult] = []
        self._feature_names: List[str] = []
        self._dof_feature_indices: Dict[str, int] = {}
        self._primary_speed_control: Optional[Any] = None
        self._target_speeds: List[float] = []
        self._export_root: Optional[Path] = None

        self._cached_robot_state = None
        self._cached_progress_buf = None
        self._cached_reset_buf = None
        self._cached_terminate_buf = None
        self._cached_respawn_root_offset = None
        self._cached_state_history = None
        self._cached_motion_ids = None
        self._cached_motion_times = None
        self._cached_env_actions = None
        self._cached_speed_control_state = None

    @property
    def speed_control_component(self) -> Any:
        if self._primary_speed_control is None:
            raise RuntimeError(
                "BiomechanicsEvaluator requires a 'speed' or 'steering' control component."
            )
        return self._primary_speed_control

    def _register_plugins(self) -> None:
        """No plugin metrics are needed for the biomechanics path."""

    def _get_speed_control_component(self) -> Any:
        control_manager = getattr(self.env, "control_manager", None)
        if control_manager is None:
            raise RuntimeError("BiomechanicsEvaluator requires env.control_manager")

        component = control_manager.components.get("speed")
        if component is None:
            component = control_manager.components.get("steering")
        if component is None:
            raise RuntimeError(
                "BiomechanicsEvaluator requires a speed control component named 'speed' "
                "or a steering control component named 'steering'."
            )
        if not isinstance(component, (SpeedControl, SteeringControl)):
            raise TypeError(
                "Expected control component to be SpeedControl or SteeringControl, "
                f"got {type(component)}"
            )
        return component

    def _resolve_target_speeds(self) -> List[float]:
        if self.config.target_speeds:
            return sorted({float(speed) for speed in self.config.target_speeds})

        motion_lib = getattr(self.env, "motion_lib", None)
        motion_files = getattr(motion_lib, "motion_files", ()) if motion_lib is not None else ()
        if motion_files:
            from HumanRetargeting.biomechanics_retarget.subject_profiles import (
                load_json_metadata,
                resolve_trial_speed_mps,
            )

            target_speeds = set()
            for motion_file in motion_files:
                motion_path = Path(str(motion_file))
                metadata_path = motion_path.parent / "metadata" / f"{motion_path.stem}.json"
                metadata = load_json_metadata(metadata_path)
                speed_mps = resolve_trial_speed_mps(
                    motion_path.stem,
                    speed_mps=metadata.get("speed_mps"),
                    metadata=metadata,
                )
                if speed_mps is not None:
                    target_speeds.add(float(speed_mps))
            if target_speeds:
                return sorted(target_speeds)

        control = self._get_speed_control_component()
        if isinstance(control, SpeedControl):
            return [float(control.config.target_speed)]

        min_speed = float(control.config.tar_speed_min)
        max_speed = float(control.config.tar_speed_max)
        if math.isclose(min_speed, max_speed):
            return [min_speed]
        mid_speed = 0.5 * (min_speed + max_speed)
        return sorted({min_speed, mid_speed, max_speed})

    def _resolve_max_eval_steps(self) -> int:
        if self.config.max_eval_steps is not None:
            return int(self.config.max_eval_steps)
        return int(self.env.max_episode_length)

    def _feature_side(self, name: str) -> Optional[str]:
        if re.search(r"(?:^|[^a-z])(left|l)(?:[^a-z]|$)", name):
            return "left"
        if re.search(r"(?:^|[^a-z])(right|r)(?:[^a-z]|$)", name):
            return "right"
        return None

    def _feature_axis(self, name: str, joint: str) -> Optional[str]:
        if any(token in name for token in ("pitch", "flex")):
            return "flex"
        if any(token in name for token in ("roll", "add", "abd", "abduction")):
            return "add"
        if any(token in name for token in ("yaw", "rot", "twist")):
            return "rot"
        if joint == "knee":
            return "flex"
        if joint in ("hip", "ankle"):
            return "flex"
        return None

    def _infer_dof_features(self) -> Dict[str, int]:
        dof_names = list(self.env.robot_config.kinematic_info.dof_names)
        feature_candidates: List[Tuple[str, int]] = []

        for dof_idx, dof_name in enumerate(dof_names):
            name = dof_name.lower()
            joint = None
            if "hip" in name:
                joint = "hip"
            elif "knee" in name:
                joint = "knee"
            elif "ankle" in name:
                joint = "ankle"
            else:
                continue

            axis = self._feature_axis(name, joint)
            if axis is None:
                continue

            side = self._feature_side(name)
            feature_name = f"{side + '_' if side else ''}{joint}_{axis}"
            feature_candidates.append((feature_name, dof_idx))

        # Keep the first index per feature so the evaluator stays robust across
        # robot families that may have duplicate semantic names.
        feature_indices: Dict[str, int] = {}
        for feature_name, dof_idx in feature_candidates:
            feature_indices.setdefault(feature_name, dof_idx)
        return feature_indices

    def _build_feature_names(self) -> List[str]:
        names = ["pelvis_flex", "pelvis_add", "pelvis_rot"]
        names.extend(sorted(self._dof_feature_indices.keys()))
        return names

    def _cycle_normalized_joint_plot_specs(
        self, feature_names: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, Dict[str, str]]]:
        available = set(self._feature_names if feature_names is None else feature_names)
        specs: List[Tuple[str, Dict[str, str]]] = []

        for joint_name in self._CYCLE_NORMALIZED_JOINTS:
            side_features: Dict[str, str] = {}
            left_feature = f"left_{joint_name}_flex"
            right_feature = f"right_{joint_name}_flex"
            if left_feature in available:
                side_features["Left"] = left_feature
            if right_feature in available:
                side_features["Right"] = right_feature
            if side_features:
                specs.append((joint_name.capitalize(), side_features))

        return specs

    def _cache_eval_state(self) -> None:
        self._cached_robot_state = self.env.simulator.get_robot_state()
        self._cached_progress_buf = self.env.progress_buf.clone()
        self._cached_reset_buf = self.env.reset_buf.clone()
        self._cached_terminate_buf = self.env.terminate_buf.clone()
        self._cached_respawn_root_offset = self.env.respawn_root_offset.clone()
        self._cached_env_actions = self.env.simulator.get_current_actions()

        if self.env.state_history is not None:
            self._cached_state_history = self.env.state_history.save_state()
        else:
            self._cached_state_history = None

        if self.env.motion_manager is not None:
            self._cached_motion_ids = self.env.motion_manager.motion_ids.clone()
            self._cached_motion_times = self.env.motion_manager.motion_times.clone()
        else:
            self._cached_motion_ids = None
            self._cached_motion_times = None

        control = self._get_speed_control_component()
        self._cached_speed_control_state = {}
        for attribute_name in (
            "_heading_change_steps",
            "_tar_dir_theta",
            "_tar_dir",
            "_tar_face_dir",
            "_tar_speed",
            "_prev_root_pos",
            "_standing_steps_remaining",
        ):
            if hasattr(control, attribute_name):
                self._cached_speed_control_state[attribute_name] = getattr(
                    control, attribute_name
                ).clone()

    def _restore_eval_state(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)

        if self._cached_robot_state is not None:
            self.env.simulator.reset_envs(self._cached_robot_state, None, env_ids)

        if self._cached_state_history is not None:
            self.env.state_history.load_state(self._cached_state_history)

        if self._cached_motion_ids is not None:
            self.env.motion_manager.motion_ids = self._cached_motion_ids.clone()
        if self._cached_motion_times is not None:
            self.env.motion_manager.motion_times = self._cached_motion_times.clone()

        if self._cached_progress_buf is not None:
            self.env.progress_buf.copy_(self._cached_progress_buf)
        if self._cached_reset_buf is not None:
            self.env.reset_buf.copy_(self._cached_reset_buf)
        if self._cached_terminate_buf is not None:
            self.env.terminate_buf.copy_(self._cached_terminate_buf)
        if self._cached_respawn_root_offset is not None:
            self.env.respawn_root_offset.copy_(self._cached_respawn_root_offset)

        if self._cached_env_actions is not None:
            sim_target = getattr(self.env.simulator.config, "_target_", "").lower()
            if "isaacgym" in sim_target:
                self.env.simulator.step(self._cached_env_actions, markers_callback=None)

        control = self._get_speed_control_component()
        if self._cached_speed_control_state is not None:
            for attribute_name, value in self._cached_speed_control_state.items():
                getattr(control, attribute_name).copy_(value)

        self._cached_robot_state = None
        self._cached_progress_buf = None
        self._cached_reset_buf = None
        self._cached_terminate_buf = None
        self._cached_respawn_root_offset = None
        self._cached_state_history = None
        self._cached_motion_ids = None
        self._cached_motion_times = None
        self._cached_env_actions = None
        self._cached_speed_control_state = None

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, Optional[float]]:
        self.agent.eval()
        metrics = self.initialize_eval()
        if not metrics:
            return {}, None

        try:
            self._speed_results = self._run_all_speeds()
            evaluation_log, score = self._build_logs_and_export()
            self.eval_count += 1
            return evaluation_log, score
        finally:
            self.cleanup_after_evaluation()

    def initialize_eval(self) -> Dict:
        if self.fabric.global_rank != 0:
            return {}
        self._target_speeds = self._resolve_target_speeds()
        self._primary_speed_control = self._get_speed_control_component()
        self._dof_feature_indices = self._infer_dof_features()
        self._feature_names = self._build_feature_names()
        self._export_root = self.root_dir / "results" / "biomechanics"
        self._speed_results = []
        self._cache_eval_state()
        return {"biomechanics_eval": 1.0}

    def run_evaluation(self, metrics: Dict) -> None:
        self._speed_results = self._run_all_speeds()

    def process_eval_results(self, metrics: Dict) -> Tuple[Dict, Optional[float]]:
        return self._build_logs_and_export()

    def _run_all_speeds(self) -> List[SpeedResult]:
        speed_results: List[SpeedResult] = []
        for target_speed in self._target_speeds:
            speed_results.append(self._run_single_speed(target_speed))
        return speed_results

    def _prepare_speed_batch(self, env_ids: torch.Tensor, target_speed: float) -> None:
        force_default_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        self.env.reset(
            env_ids,
            sample_flat=True,
            force_default_mask=force_default_mask,
            disable_motion_resample=True,
        )
        self._apply_target_speed(env_ids, target_speed)

    def _apply_target_speed(self, env_ids: torch.Tensor, target_speed: float) -> None:
        control = self.speed_control_component
        control._tar_speed[env_ids] = target_speed
        control._tar_dir_theta[env_ids] = 0.0
        forward_dir = torch.zeros(
            len(env_ids), 2, device=self.device, dtype=control._tar_dir.dtype
        )
        forward_dir[:, 0] = 1.0
        control._tar_dir[env_ids] = forward_dir
        if hasattr(control, "_tar_face_dir"):
            control._tar_face_dir[env_ids] = forward_dir.to(control._tar_face_dir.dtype)
        if hasattr(control, "_heading_change_steps"):
            control._heading_change_steps[env_ids] = (
                self.env.progress_buf[env_ids] + self._resolve_max_eval_steps() * 10
            )
        if hasattr(control, "_standing_steps_remaining"):
            control._standing_steps_remaining[env_ids] = 0

    def _initial_left_contact(self, env_ids: torch.Tensor) -> Dict[int, bool]:
        current_state = self.env.simulator.get_robot_state()
        left_contact = self._get_left_contact(current_state)
        return {int(env_id): bool(left_contact[int(env_id)].item()) for env_id in env_ids}

    def _compute_step_signals(self, current_state) -> Dict[str, torch.Tensor]:
        root_rot = current_state.root_rot
        root_lin_vel = current_state.rigid_body_vel[:, 0, :]

        heading_rot = rotations.calc_heading_quat(root_rot, True)
        forward_axis = torch.zeros(
            root_rot.shape[0], 3, device=self.device, dtype=root_rot.dtype
        )
        forward_axis[:, 0] = 1.0
        heading_dir = rotations.quat_rotate(heading_rot, forward_axis, True)
        root_forward_speed = (root_lin_vel[:, :2] * heading_dir[:, :2]).sum(dim=-1)

        roll, pitch, yaw = rotations.get_euler_xyz(root_rot, True)
        pelvis_flex = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        pelvis_add = torch.atan2(torch.sin(roll), torch.cos(roll))
        pelvis_rot = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        signals: Dict[str, torch.Tensor] = {
            "root_forward_speed": root_forward_speed,
            "pelvis_flex": pelvis_flex,
            "pelvis_add": pelvis_add,
            "pelvis_rot": pelvis_rot,
        }

        if current_state.dof_pos is not None:
            for feature_name, dof_idx in self._dof_feature_indices.items():
                signals[feature_name] = current_state.dof_pos[:, dof_idx]

        return signals

    def _get_left_contact(self, current_state) -> torch.Tensor:
        body_names = self.env.robot_config.kinematic_info.body_names
        left_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_left_foot_bodies", []
        )
        if not left_bodies:
            raise RuntimeError("No left foot bodies available for strike detection")

        left_indices = [body_names.index(name) for name in left_bodies if name in body_names]
        if not left_indices:
            raise RuntimeError(
                f"Could not map left foot bodies {left_bodies} to body indices"
            )

        contacts = current_state.rigid_body_contacts[:, left_indices].bool()
        return contacts.any(dim=-1)

    def _resample_cycle(self, values: List[float], num_points: int) -> np.ndarray:
        if len(values) == 0:
            return np.zeros((num_points,), dtype=np.float32)
        if len(values) == 1:
            return np.full((num_points,), values[0], dtype=np.float32)

        src_phase = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        dst_phase = np.linspace(0.0, 1.0, num=num_points, dtype=np.float32)
        return np.interp(dst_phase, src_phase, np.asarray(values, dtype=np.float32)).astype(
            np.float32
        )

    def _finalize_cycle(
        self,
        tracker: _EpisodeTracker,
        cycle_buffers: Dict[str, List[float]],
    ) -> Optional[CycleRecord]:
        if "root_forward_speed" not in cycle_buffers:
            return None

        num_frames = len(cycle_buffers["root_forward_speed"])
        if num_frames < 2:
            return None

        duration_s = (num_frames - 1) * float(self.env.dt)
        if duration_s <= 0:
            return None

        mean_forward_speed = float(np.mean(cycle_buffers["root_forward_speed"]))
        stride_length = mean_forward_speed * duration_s
        cadence_steps_per_min = 120.0 / duration_s

        waveforms = {
            feature_name: self._resample_cycle(values, self.config.waveform_num_points)
            for feature_name, values in cycle_buffers.items()
        }

        cycle_type = "burn_in" if not tracker.burn_in_complete else "post_burn_in"
        return CycleRecord(
            env_id=tracker.env_id,
            episode_index=tracker.episode_index,
            target_speed=tracker.target_speed,
            cycle_index=tracker.cycle_index,
            cycle_type=cycle_type,
            duration_s=duration_s,
            mean_forward_speed_mps=mean_forward_speed,
            stride_length_m=stride_length,
            cadence_steps_per_min=cadence_steps_per_min,
            feature_waveforms=waveforms,
        )

    def _record_cycle(self, tracker: _EpisodeTracker, cycle_record: CycleRecord) -> None:
        tracker.cycles.append(cycle_record)
        tracker.cycle_index += 1

        if abs(tracker.target_speed) < 1e-6:
            tol = 0.10
        else:
            tol = abs(tracker.target_speed) * float(self.config.burn_in_speed_tolerance)
        in_range = abs(cycle_record.mean_forward_speed_mps - tracker.target_speed) <= tol

        if not tracker.burn_in_complete:
            if in_range:
                tracker.burn_in_consecutive += 1
            else:
                tracker.burn_in_consecutive = 0

            if tracker.burn_in_consecutive >= self.config.burn_in_consecutive_cycles:
                tracker.burn_in_complete = True
                tracker.post_burn_in_cycles = 0
        else:
            tracker.post_burn_in_cycles += 1
            if tracker.post_burn_in_cycles >= self.config.success_post_burn_in_cycles:
                tracker.finished = True
                tracker.success = True
                tracker.failure_reason = ""

    def _run_episode_batch(
        self,
        target_speed: float,
        episode_start_index: int,
        episode_count: int,
    ) -> List[EpisodeResult]:
        if episode_count <= 0:
            return []

        batch_env_count = min(self.env.num_envs, episode_count)
        batch_env_ids = torch.arange(batch_env_count, device=self.device, dtype=torch.long)
        self._prepare_speed_batch(batch_env_ids, target_speed)
        initial_contacts = self._initial_left_contact(batch_env_ids)

        trackers: Dict[int, _EpisodeTracker] = {}
        for idx, env_id in enumerate(batch_env_ids.tolist()):
            trackers[env_id] = _EpisodeTracker(
                env_id=env_id,
                episode_index=episode_start_index + idx,
                target_speed=target_speed,
                prev_left_contact=initial_contacts[env_id],
            )

        completed: List[EpisodeResult] = []
        next_episode_index = episode_start_index + batch_env_count
        episodes_started = batch_env_count
        max_steps = self._resolve_max_eval_steps()
        min_interval_steps = max(
            1,
            round(float(self.config.left_strike_min_interval_sec) / float(self.env.dt)),
        )

        while len(completed) < episode_count and trackers:
            obs = self.env.get_obs()
            obs_td = self.agent.obs_dict_to_tensordict(
                self.agent.add_agent_info_to_obs(obs)
            )
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            _, _, dones, terminated, _ = self.env.step(actions)

            current_state = self.env.simulator.get_robot_state()
            signals = self._compute_step_signals(current_state)
            left_contact = self._get_left_contact(current_state)

            finished_env_ids: List[int] = []
            for env_id, tracker in trackers.items():
                if tracker.finished:
                    finished_env_ids.append(env_id)
                    continue

                env_t = int(env_id)
                tracker.steps += 1
                current_left_contact = bool(left_contact[env_t].item())
                strike = (
                    (not tracker.prev_left_contact)
                    and current_left_contact
                    and (tracker.steps - tracker.last_strike_step) >= min_interval_steps
                )

                if tracker.cycle_started:
                    for feature_name, feature_values in signals.items():
                        tracker.cycle_buffers[feature_name].append(
                            float(feature_values[env_t].item())
                        )

                    if strike:
                        cycle_record = self._finalize_cycle(
                            tracker, tracker.cycle_buffers
                        )
                        if cycle_record is not None:
                            self._record_cycle(tracker, cycle_record)

                        tracker.cycle_buffers = defaultdict(list)
                        for feature_name, feature_values in signals.items():
                            tracker.cycle_buffers[feature_name].append(
                                float(feature_values[env_t].item())
                            )
                        tracker.last_strike_step = tracker.steps
                elif strike:
                    tracker.cycle_started = True
                    tracker.cycle_buffers = defaultdict(list)
                    for feature_name, feature_values in signals.items():
                        tracker.cycle_buffers[feature_name].append(
                            float(feature_values[env_t].item())
                        )
                    tracker.last_strike_step = tracker.steps

                tracker.prev_left_contact = current_left_contact

                if bool(dones[env_t].item()) or bool(terminated[env_t].item()):
                    tracker.finished = True
                    tracker.success = False
                    tracker.failure_reason = "env_terminated"

                if tracker.steps >= max_steps and not tracker.finished:
                    tracker.finished = True
                    tracker.success = False
                    tracker.failure_reason = "max_eval_steps"

                if tracker.finished:
                    completed.append(
                        EpisodeResult(
                            env_id=tracker.env_id,
                            episode_index=tracker.episode_index,
                            target_speed=tracker.target_speed,
                            success=tracker.success,
                            burn_in_complete=tracker.burn_in_complete,
                            steps=tracker.steps,
                            total_cycles=tracker.cycle_index,
                            post_burn_in_cycles=tracker.post_burn_in_cycles,
                            failure_reason=tracker.failure_reason,
                            cycles=tracker.cycles,
                        )
                    )
                    finished_env_ids.append(env_id)

            for env_id in finished_env_ids:
                tracker = trackers.pop(env_id, None)
                if tracker is None:
                    continue

                if len(completed) >= episode_count:
                    continue

                if episodes_started < episode_count:
                    new_episode_index = next_episode_index
                    next_episode_index += 1
                    episodes_started += 1
                    self._prepare_speed_batch(
                        torch.tensor([env_id], device=self.device, dtype=torch.long),
                        target_speed,
                    )
                    initial_contact = self._initial_left_contact(
                        torch.tensor([env_id], device=self.device, dtype=torch.long)
                    )[env_id]
                    trackers[env_id] = _EpisodeTracker(
                        env_id=env_id,
                        episode_index=new_episode_index,
                        target_speed=target_speed,
                        prev_left_contact=initial_contact,
                    )

        return completed[:episode_count]

    def _run_single_speed(self, target_speed: float) -> SpeedResult:
        episodes_requested = self.config.episodes_per_speed
        episodes: List[EpisodeResult] = []
        episode_start_index = 0

        while len(episodes) < episodes_requested:
            remaining = episodes_requested - len(episodes)
            batch_results = self._run_episode_batch(
                target_speed=target_speed,
                episode_start_index=episode_start_index,
                episode_count=remaining,
            )
            episodes.extend(batch_results)
            episode_start_index += len(batch_results)
            if not batch_results:
                break

        return SpeedResult(
            target_speed=target_speed,
            episodes_requested=episodes_requested,
            episodes=episodes,
        )

    def _speed_tag(self, target_speed: float) -> str:
        return f"{target_speed:.2f}".replace("-", "m").replace(".", "p")

    def _summarize_speed_result(self, speed_result: SpeedResult) -> Dict[str, float]:
        episodes = speed_result.episodes
        total = max(len(episodes), 1)
        successes = sum(1 for episode in episodes if episode.success)
        burn_in_successes = sum(1 for episode in episodes if episode.burn_in_complete)

        all_post_burn_in_cycles = [
            cycle
            for episode in episodes
            for cycle in episode.cycles
            if cycle.cycle_type == "post_burn_in"
        ]

        stride_lengths = [cycle.stride_length_m for cycle in all_post_burn_in_cycles]
        cadences = [cycle.cadence_steps_per_min for cycle in all_post_burn_in_cycles]
        cycle_speeds = [
            cycle.mean_forward_speed_mps for cycle in all_post_burn_in_cycles
        ]

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            arr = np.asarray(values, dtype=np.float32)
            return float(arr.mean()), float(arr.std())

        stride_mean, stride_std = _mean_std(stride_lengths)
        cadence_mean, cadence_std = _mean_std(cadences)
        speed_mean, speed_std = _mean_std(cycle_speeds)
        post_burn_in_cycles = len(all_post_burn_in_cycles)
        cycle_counts = [episode.post_burn_in_cycles for episode in episodes]
        cycle_count_mean, cycle_count_std = _mean_std(cycle_counts)
        episode_steps = [episode.steps for episode in episodes]
        steps_mean, steps_std = _mean_std(episode_steps)

        return {
            "target_speed_mps": float(speed_result.target_speed),
            "episodes_requested": float(speed_result.episodes_requested),
            "episodes_completed": float(len(episodes)),
            "success_rate": float(successes / total),
            "burn_in_success_rate": float(burn_in_successes / total),
            "post_burn_in_cycles_collected": float(post_burn_in_cycles),
            "mean_stride_length_m": stride_mean,
            "std_stride_length_m": stride_std,
            "mean_cadence_steps_per_min": cadence_mean,
            "std_cadence_steps_per_min": cadence_std,
            "mean_cycle_speed_mps": speed_mean,
            "std_cycle_speed_mps": speed_std,
            "mean_post_burn_in_cycles_per_episode": cycle_count_mean,
            "std_post_burn_in_cycles_per_episode": cycle_count_std,
            "mean_episode_steps": steps_mean,
            "std_episode_steps": steps_std,
        }

    def _build_waveform_exports(
        self, cycles: List[CycleRecord], feature_names: Optional[List[str]] = None
    ) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        if feature_names is None:
            feature_names = sorted(
                {name for cycle in cycles for name in cycle.feature_waveforms.keys()}
            )

        phase = np.linspace(0.0, 1.0, self.config.waveform_num_points, dtype=np.float32)

        if not cycles:
            mean_std_exports: Dict[str, np.ndarray] = {}
            for feature_name in feature_names:
                mean_std_exports[f"mean__{feature_name}"] = np.zeros(
                    (self.config.waveform_num_points,), dtype=np.float32
                )
                mean_std_exports[f"std__{feature_name}"] = np.zeros(
                    (self.config.waveform_num_points,), dtype=np.float32
                )
            return feature_names, phase, mean_std_exports

        raw_cycles: Dict[str, List[np.ndarray]] = {name: [] for name in feature_names}
        for cycle in cycles:
            for feature_name in feature_names:
                if feature_name in cycle.feature_waveforms:
                    raw_cycles[feature_name].append(cycle.feature_waveforms[feature_name])

        mean_std_exports: Dict[str, np.ndarray] = {}
        for feature_name in feature_names:
            if raw_cycles[feature_name]:
                stacked = np.stack(raw_cycles[feature_name], axis=0)
                mean_std_exports[f"mean__{feature_name}"] = stacked.mean(axis=0).astype(
                    np.float32
                )
                mean_std_exports[f"std__{feature_name}"] = stacked.std(axis=0).astype(
                    np.float32
                )
            else:
                mean_std_exports[f"mean__{feature_name}"] = np.zeros(
                    (self.config.waveform_num_points,), dtype=np.float32
                )
                mean_std_exports[f"std__{feature_name}"] = np.zeros(
                    (self.config.waveform_num_points,), dtype=np.float32
                )

        return feature_names, phase, mean_std_exports

    def _export_speed_result(self, speed_result: SpeedResult) -> Dict[str, Any]:
        speed_tag = self._speed_tag(speed_result.target_speed)
        speed_dir = self._export_root / speed_tag
        speed_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_speed_result(speed_result)
        summary["speed_tag"] = speed_tag

        post_burn_in_cycles = [
            cycle
            for episode in speed_result.episodes
            for cycle in episode.cycles
            if cycle.cycle_type == "post_burn_in"
        ]

        all_cycles = [
            cycle for episode in speed_result.episodes for cycle in episode.cycles
        ]

        feature_names, phase, waveform_exports = self._build_waveform_exports(
            post_burn_in_cycles, feature_names=self._feature_names
        )

        cycles_npz: Dict[str, np.ndarray] = {
            "phase": phase,
            "feature_names": np.array(feature_names, dtype=np.str_),
            "cycle_env_id": np.array(
                [cycle.env_id for cycle in all_cycles], dtype=np.int32
            ),
            "cycle_episode_index": np.array(
                [cycle.episode_index for cycle in all_cycles], dtype=np.int32
            ),
            "cycle_index": np.array(
                [cycle.cycle_index for cycle in all_cycles], dtype=np.int32
            ),
            "cycle_type": np.array(
                [cycle.cycle_type for cycle in all_cycles], dtype=np.str_
            ),
            "cycle_target_speed_mps": np.array(
                [cycle.target_speed for cycle in all_cycles], dtype=np.float32
            ),
            "cycle_duration_s": np.array(
                [cycle.duration_s for cycle in all_cycles], dtype=np.float32
            ),
            "cycle_mean_forward_speed_mps": np.array(
                [cycle.mean_forward_speed_mps for cycle in all_cycles], dtype=np.float32
            ),
            "cycle_stride_length_m": np.array(
                [cycle.stride_length_m for cycle in all_cycles], dtype=np.float32
            ),
            "cycle_cadence_steps_per_min": np.array(
                [cycle.cadence_steps_per_min for cycle in all_cycles], dtype=np.float32
            ),
        }

        if all_cycles:
            for feature_name in feature_names:
                cycles_npz[f"raw__{feature_name}"] = np.stack(
                    [cycle.feature_waveforms[feature_name] for cycle in all_cycles], axis=0
                ).astype(np.float32)
        else:
            for feature_name in feature_names:
                cycles_npz[f"raw__{feature_name}"] = np.zeros(
                    (0, self.config.waveform_num_points), dtype=np.float32
                )

        np.savez_compressed(speed_dir / "cycles.npz", **cycles_npz)
        np.savez_compressed(
            speed_dir / "waveforms.npz",
            phase=phase,
            feature_names=np.array(feature_names, dtype=np.str_),
            **waveform_exports,
        )

        self._save_speed_summary_plot(
            speed_dir / "waveforms.png",
            speed_result.target_speed,
            feature_names,
            phase,
            waveform_exports,
        )
        self._log_cycle_normalized_joint_figure(
            speed_tag=speed_tag,
            target_speed=speed_result.target_speed,
            phase=phase,
            waveform_exports=waveform_exports,
            post_burn_in_cycle_count=len(post_burn_in_cycles),
            feature_names=feature_names,
        )

        with open(speed_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return summary

    def _save_speed_summary_plot(
        self,
        output_path: Path,
        target_speed: float,
        feature_names: List[str],
        phase: np.ndarray,
        waveform_exports: Dict[str, np.ndarray],
    ) -> None:
        if not feature_names:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        num_features = len(feature_names)
        num_cols = 3
        num_rows = math.ceil(num_features / num_cols)
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(5.0 * num_cols, 2.6 * num_rows),
            squeeze=False,
        )
        fig.suptitle(f"Biomechanics waveforms @ {target_speed:.2f} m/s", fontsize=14)

        for idx, feature_name in enumerate(feature_names):
            ax = axes[idx // num_cols][idx % num_cols]
            mean = waveform_exports.get(f"mean__{feature_name}")
            std = waveform_exports.get(f"std__{feature_name}")
            if mean is None or std is None:
                continue
            ax.plot(phase, mean, color="black", linewidth=1.5)
            ax.fill_between(phase, mean - std, mean + std, color="tab:blue", alpha=0.2)
            ax.set_title(feature_name)
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.2)

        for idx in range(num_features, num_rows * num_cols):
            axes[idx // num_cols][idx % num_cols].axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _log_cycle_normalized_joint_figure(
        self,
        speed_tag: str,
        target_speed: float,
        phase: np.ndarray,
        waveform_exports: Dict[str, np.ndarray],
        post_burn_in_cycle_count: int,
        feature_names: Optional[Iterable[str]] = None,
    ) -> None:
        if not self.fabric.loggers:
            return

        figure = self._create_cycle_normalized_joint_figure(
            target_speed=target_speed,
            phase=phase,
            waveform_exports=waveform_exports,
            post_burn_in_cycle_count=post_burn_in_cycle_count,
            feature_names=feature_names,
        )
        if figure is None:
            return

        tag = f"eval/biomechanics/cycle_normalized_joints/{speed_tag}"
        self._log_tensorboard_figure(tag, figure)

    def _create_cycle_normalized_joint_figure(
        self,
        target_speed: float,
        phase: np.ndarray,
        waveform_exports: Dict[str, np.ndarray],
        post_burn_in_cycle_count: int,
        feature_names: Optional[Iterable[str]] = None,
    ):
        joint_specs = self._cycle_normalized_joint_plot_specs(feature_names)
        if not joint_specs:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        phase_percent = np.asarray(phase, dtype=np.float32) * 100.0
        fig, axes = plt.subplots(
            1,
            len(joint_specs),
            figsize=(5.2 * len(joint_specs), 4.0),
            squeeze=False,
        )
        fig.suptitle(
            f"Cycle-normalized lower-body joints @ {target_speed:.2f} m/s",
            fontsize=14,
        )

        if post_burn_in_cycle_count <= 0:
            for ax, (joint_label, _) in zip(axes[0], joint_specs):
                ax.set_title(joint_label)
                ax.set_xlim(0.0, 100.0)
                ax.set_xlabel("Gait cycle (%)")
                ax.set_ylabel("Angle (deg)")
                ax.text(
                    0.5,
                    0.5,
                    "No post-burn-in cycles",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.grid(True, alpha=0.2)
            fig.tight_layout()
            return fig

        side_colors = {"Left": "tab:blue", "Right": "tab:orange"}
        for ax, (joint_label, side_features) in zip(axes[0], joint_specs):
            for side_label, feature_name in side_features.items():
                mean = waveform_exports.get(f"mean__{feature_name}")
                std = waveform_exports.get(f"std__{feature_name}")
                if mean is None or std is None:
                    continue
                mean_deg = np.rad2deg(np.asarray(mean, dtype=np.float32))
                std_deg = np.rad2deg(np.asarray(std, dtype=np.float32))
                color = side_colors.get(side_label, "black")
                ax.plot(
                    phase_percent,
                    mean_deg,
                    label=side_label,
                    color=color,
                    linewidth=1.8,
                )
                ax.fill_between(
                    phase_percent,
                    mean_deg - std_deg,
                    mean_deg + std_deg,
                    color=color,
                    alpha=0.18,
                )

            ax.set_title(joint_label)
            ax.set_xlim(0.0, 100.0)
            ax.set_xlabel("Gait cycle (%)")
            ax.set_ylabel("Angle (deg)")
            ax.grid(True, alpha=0.2)
            if side_features:
                ax.legend(frameon=False)

        fig.tight_layout()
        return fig

    def _log_tensorboard_figure(self, tag: str, figure: Any) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        step = getattr(self.agent, "current_epoch", self.eval_count)
        try:
            for logger in self.fabric.loggers:
                experiment = getattr(logger, "experiment", None)
                if experiment is None or not hasattr(experiment, "add_figure"):
                    continue
                experiment.add_figure(tag, figure, global_step=step, close=False)
        finally:
            if plt is not None:
                plt.close(figure)

    def _build_logs_and_export(self) -> Tuple[Dict[str, float], Optional[float]]:
        if self._export_root is None:
            self._export_root = self.root_dir / "results" / "biomechanics"
        self._export_root.mkdir(parents=True, exist_ok=True)

        log: Dict[str, float] = {}
        top_level_success_rates = []
        top_level_stride_lengths = []
        top_level_cadences = []
        top_level_cycle_counts = []

        speed_summaries: Dict[str, Dict[str, float]] = {}
        for speed_result in self._speed_results:
            summary = self._export_speed_result(speed_result)
            speed_tag = summary["speed_tag"]
            speed_summaries[speed_tag] = summary

            log[f"eval/biomechanics/{speed_tag}/success_rate"] = summary[
                "success_rate"
            ]
            log[f"eval/biomechanics/{speed_tag}/mean_stride_length_m"] = summary[
                "mean_stride_length_m"
            ]
            log[f"eval/biomechanics/{speed_tag}/mean_cadence_steps_per_min"] = summary[
                "mean_cadence_steps_per_min"
            ]
            log[f"eval/biomechanics/{speed_tag}/post_burn_in_cycles"] = summary[
                "post_burn_in_cycles_collected"
            ]

            top_level_success_rates.append(summary["success_rate"])
            top_level_stride_lengths.append(summary["mean_stride_length_m"])
            top_level_cadences.append(summary["mean_cadence_steps_per_min"])
            top_level_cycle_counts.append(summary["mean_post_burn_in_cycles_per_episode"])

        overall_summary = {
            "episodes_per_speed": self.config.episodes_per_speed,
            "success_post_burn_in_cycles": self.config.success_post_burn_in_cycles,
            "burn_in_consecutive_cycles": self.config.burn_in_consecutive_cycles,
            "burn_in_speed_tolerance": self.config.burn_in_speed_tolerance,
            "left_strike_min_interval_sec": self.config.left_strike_min_interval_sec,
            "waveform_num_points": self.config.waveform_num_points,
            "target_speeds_mps": self._target_speeds,
            "speed_summaries": speed_summaries,
        }

        with open(self._export_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, indent=2, sort_keys=True)

        if top_level_success_rates:
            log["eval/biomechanics/success_rate"] = float(
                np.mean(np.asarray(top_level_success_rates, dtype=np.float32))
            )
            log["eval/biomechanics/mean_stride_length_m"] = float(
                np.mean(np.asarray(top_level_stride_lengths, dtype=np.float32))
            )
            log["eval/biomechanics/mean_cadence_steps_per_min"] = float(
                np.mean(np.asarray(top_level_cadences, dtype=np.float32))
            )
            log["eval/biomechanics/mean_post_burn_in_cycles_per_episode"] = float(
                np.mean(np.asarray(top_level_cycle_counts, dtype=np.float32))
            )

        score = log.get("eval/biomechanics/success_rate", None)
        return log, score

    def cleanup_after_evaluation(self) -> None:
        if self.fabric.global_rank == 0:
            self._restore_eval_state()
