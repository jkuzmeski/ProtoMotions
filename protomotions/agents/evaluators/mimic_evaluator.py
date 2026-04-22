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
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from torch import Tensor
import math
from dataclasses import dataclass

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""
    motion_ids: Tensor  # which motion each env is tracking
    frame_limits: Tensor  # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        super().__init__(agent, fabric, config)

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        metrics = {}

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=self.device
        )

        # GRF buffers: 3 force components (x, y, z) per foot
        metrics["grf_left"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, 3, device=self.device
        )
        metrics["grf_right"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, 3, device=self.device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache env state for restoration."""
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib.get_motion_length(None)
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        self._init_eval_component_buffers(num_motions)

        # Cache env + motion manager state (restored in cleanup_after_evaluation)
        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()

        return self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps
        )

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if self._motion_failed is None:
            return

        failed_motions = torch.nonzero(self._motion_failed).flatten().tolist()
        success_motions = torch.nonzero(~self._motion_failed).flatten().tolist()

        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        if failure_discount != 0:
            new_weights[failed_motions] /= failure_discount
        else:
            new_weights[failed_motions] = 1.0
        self.env.motion_manager.update_sampling_weights(new_weights)

    def _get_left_foot_indices(self) -> List[int]:
        """Resolve left-foot body indices for step detection."""
        body_names = list(self.env.robot_config.kinematic_info.body_names)
        left_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_left_foot_bodies", []
        )
        if not left_bodies:
            raise RuntimeError("No left foot bodies available for step detection")
        indices = [body_names.index(n) for n in left_bodies if n in body_names]
        if not indices:
            raise RuntimeError(
                f"Could not map left foot bodies {left_bodies} to body indices"
            )
        return indices

    def _get_left_contact(self, left_indices: List[int], env_ids: torch.Tensor) -> torch.Tensor:
        """Return a [len(env_ids)] bool tensor: True where any left foot body is in contact."""
        current_state = self.env.simulator.get_robot_state()
        return current_state.rigid_body_contacts[env_ids][:, left_indices].bool().any(dim=-1)

    def evaluate_episode(self, env_ids: torch.Tensor, max_steps: int) -> None:
        """Run a single episode batch, optionally with EMA action smoothing.

        When eval_action_ema_alpha is set, actions are low-pass filtered to
        simulate deployment conditions. Motions that fail under EMA get higher
        sampling weight, creating curriculum pressure toward smooth policies.

        When max_foot_steps is set, each env independently stops after the
        configured number of left-foot heel strikes.
        """
        ema_alpha = self.config.eval_action_ema_alpha
        max_foot_steps = self.config.max_foot_steps

        # --- Foot-step tracking setup (fully vectorized) ---
        left_indices: Optional[List[int]] = None
        min_interval_steps: int = 1
        prev_left_contact: Optional[torch.Tensor] = None
        step_counts: Optional[torch.Tensor] = None
        last_strike_step: Optional[torch.Tensor] = None
        all_done: Optional[torch.Tensor] = None

        if max_foot_steps is not None:
            left_indices = self._get_left_foot_indices()
            min_interval_steps = max(
                1,
                round(float(self.config.left_strike_min_interval_sec) / float(self.env.dt)),
            )
            n = env_ids.shape[0]
            step_counts = torch.zeros(n, dtype=torch.long, device=self.device)
            last_strike_step = torch.full((n,), -(10**9), dtype=torch.long, device=self.device)
            all_done = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        # Snapshot initial contact state after reset
        if max_foot_steps is not None:
            prev_left_contact = self._get_left_contact(left_indices, env_ids)

        prev_actions = None

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            # Apply EMA smoothing (deployment simulation)
            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

            # --- Vectorized foot-step counting ---
            if max_foot_steps is not None:
                cur_contact = self._get_left_contact(left_indices, env_ids)
                # Strike = rising edge + debounce interval elapsed + not already done
                strike = (
                    (~prev_left_contact)
                    & cur_contact
                    & ((step_idx - last_strike_step) >= min_interval_steps)
                    & (~all_done)
                )
                step_counts[strike] += 1
                last_strike_step[strike] = step_idx
                all_done |= step_counts >= max_foot_steps
                prev_left_contact = cur_contact

                if all_done.all():
                    break

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        batches = self._build_eval_batches()
        num_motions = self.motion_lib.num_motions()
        for batch_idx, (env_ids, motion_ids) in enumerate(batches):
            start = batch_idx * self.num_envs
            end = min(start + self.num_envs, num_motions)
            print(f"Evaluating motions {start} to {end}, out of total {num_motions}")
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt).floor().long().clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.
        
        Returns:
            List of (env_ids, motion_ids) tuples
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        num_motions = self.motion_lib.num_motions()
        batches = []
        for start in range(0, num_motions, self.num_envs):
            end = min(start + self.num_envs, num_motions)
            motion_ids = torch.arange(start, end, device=self.device)
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)
            batches.append((env_ids, motion_ids))
        return batches

    # --- Hook overrides ---
    
    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset."""
        self.motion_manager.motion_ids[env_ids] = self._episode_ctx.motion_ids
        self.motion_manager.motion_times[env_ids] = 0.0
    
    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}
    
    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Filter by frame limits and check failures only for active clips."""
        still_active = self._episode_ctx.frame_limits > step_idx
        if still_active.any():
            active_env_ids = env_ids[still_active]
            active_motion_ids = self._episode_ctx.motion_ids[still_active]
            self._check_evaluation_failures(active_env_ids, active_motion_ids)
    
    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics and GRF each step."""
        self._record_trajectory_step(
            self._metrics, extras, env_ids, self._episode_ctx.motion_ids, actions
        )
        # Collect ground reaction forces from the same raw exported state used for contact segmentation.
        self._record_grf_step(
            self._metrics, extras, env_ids, self._episode_ctx.motion_ids
        )

    def _get_foot_body_indices(self) -> Tuple[List[int], List[int]]:
        """Return (left_indices, right_indices) into body_names for foot bodies."""
        body_names = list(self.env.robot_config.kinematic_info.body_names)
        left_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_left_foot_bodies", []
        )
        right_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_right_foot_bodies", []
        )
        left_idx = [body_names.index(n) for n in left_bodies if n in body_names]
        right_idx = [body_names.index(n) for n in right_bodies if n in body_names]
        return left_idx, right_idx

    def _record_grf_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
    ) -> None:
        """Record per-foot GRF from the raw exported robot-state contact-force tensor."""
        if "grf_left" not in metrics or "grf_right" not in metrics:
            return
        raw_forces = extras.get("raw/rigid_body_contact_forces")
        if raw_forces is None:
            return

        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        forces = raw_forces[active_env_ids].view(-1, num_bodies, 3)

        left_idx, right_idx = self._get_foot_body_indices()
        if left_idx:
            # Sum forces across all left-foot bodies → [num_active_envs, 3]
            left_grf = forces[:, left_idx, :].sum(dim=1).detach()
            metrics["grf_left"].update(active_motion_ids, left_grf)
        if right_idx:
            right_grf = forces[:, right_idx, :].sum(dim=1).detach()
            metrics["grf_right"].update(active_motion_ids, right_grf)

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(active_motion_ids, actions[active_env_ids].detach())

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach())

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process results and update motion sampling weights."""
        to_log, success_rate = super().process_eval_results()
        self._update_motion_sampling_weights()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(self._metrics, epoch=self.agent.current_epoch)

            # Generate per-motion plots
            self._plot_all_motions(self._metrics)

        return to_log, success_rate

    def cleanup_after_evaluation(self) -> None:
        """Restore env and motion manager state after evaluation."""
        self.motion_manager.motion_ids = self._cached_motion_ids
        self.motion_manager.motion_times = self._cached_motion_times
        self.env.restore_state(self._env_snapshot)
        
        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times
        super().cleanup_after_evaluation()

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Only plot metrics that were actually collected
        eval_metric_keys = list(self.config.evaluation_components.keys())
        available_keys = [k for k in eval_metric_keys if k in metrics]

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=available_keys if available_keys else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

    def _plot_all_motions(self, metrics: Dict[str, MotionMetrics]) -> None:
        """Generate gait-cycle-segmented comparison plots for every motion.

        For each motion, produces:
        1. **Timeseries**: predicted vs reference DOF angles with contact shading.
        2. **Full gait cycle** (heel-strike to heel-strike): mean ± std across cycles.
        3. **Stance phase** (heel-strike to toe-off): mean ± std across cycles.

        Left foot strikes/toe-offs are detected from the **predicted** contact
        data stored in ``metrics["rigid_body_contacts"]``.  Reference waveforms
        come from ``self.motion_lib``.
        """
        if not metrics or "dof_pos" not in metrics:
            print("No dof_pos metric available for plotting")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        num_motions = self.motion_lib.num_motions()
        motion_files = getattr(self.motion_lib, "motion_files", None)
        dof_names = list(self.env.robot_config.kinematic_info.dof_names)
        body_names = list(self.env.robot_config.kinematic_info.body_names)
        dt = self.env.dt
        num_dofs = len(dof_names)
        cycle_pts = 101  # normalized phase resolution

        # Resolve left / right foot body indices for contact segmentation
        left_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_left_foot_bodies", []
        )
        right_bodies = self.env.robot_config.common_naming_to_robot_body_names.get(
            "all_right_foot_bodies", []
        )
        left_indices = [body_names.index(n) for n in left_bodies if n in body_names]
        right_indices = [body_names.index(n) for n in right_bodies if n in body_names]

        # Debounce interval (frames)
        min_interval = max(
            1, round(float(self.config.left_strike_min_interval_sec) / dt)
        )

        output_dir = self.root_dir / "results" / "per_motion_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        has_contacts = "rigid_body_contacts" in metrics
        has_ref_contacts = (
            self.motion_lib.contacts is not None
            and self.motion_lib.contacts.numel() > 0
        )
        has_grf = "grf_left" in metrics and "grf_right" in metrics
        grf_labels = ["X (N)", "Y (N)", "Z (N)"]

        for motion_id in range(num_motions):
            if motion_files is not None:
                from pathlib import Path as _Path
                motion_name = _Path(str(motion_files[motion_id])).stem
            else:
                motion_name = f"motion_{motion_id}"

            n_frames = metrics["dof_pos"].frame_counts[motion_id].item()
            if n_frames == 0:
                continue

            # --- Extract predicted DOF data ---
            pred_dof = metrics["dof_pos"].data[motion_id, :n_frames, :].cpu().numpy()

            # --- Extract reference DOF data ---
            ref_start = self.motion_lib.length_starts[motion_id].item()
            ref_n = min(
                int(self.motion_lib.motion_num_frames[motion_id].item()), n_frames
            )
            ref_dof = self.motion_lib.dps[ref_start: ref_start + ref_n].cpu().numpy()

            # --- Extract predicted contacts for left foot ---
            pred_left_contact = None
            pred_right_contact = None
            if has_contacts and left_indices:
                c = metrics["rigid_body_contacts"].data[motion_id, :n_frames, :]
                pred_left_contact = c[:, left_indices].cpu().numpy().any(axis=1).astype(bool)
            if has_contacts and right_indices:
                c = metrics["rigid_body_contacts"].data[motion_id, :n_frames, :]
                pred_right_contact = c[:, right_indices].cpu().numpy().any(axis=1).astype(bool)

            # --- Extract reference contacts for left foot ---
            ref_left_contact = None
            if has_ref_contacts and left_indices:
                rc = self.motion_lib.contacts[ref_start: ref_start + ref_n, :]
                ref_left_contact = rc[:, left_indices].cpu().numpy().any(axis=1) > 0.5

            # --- Extract GRF data [n_frames, 3] ---
            grf_left = None
            grf_right = None
            if has_grf:
                grf_left = metrics["grf_left"].data[motion_id, :n_frames, :].cpu().numpy()
                grf_right = metrics["grf_right"].data[motion_id, :n_frames, :].cpu().numpy()

            # --- Detect heel strikes and toe offs from predicted contacts ---
            hs_frames = []  # heel-strike frame indices
            to_frames = []  # toe-off frame indices
            if pred_left_contact is not None:
                last_hs = -(10**9)
                for i in range(1, len(pred_left_contact)):
                    # Heel strike = rising edge
                    if not pred_left_contact[i - 1] and pred_left_contact[i]:
                        if (i - last_hs) >= min_interval:
                            hs_frames.append(i)
                            last_hs = i
                    # Toe off = falling edge
                    if pred_left_contact[i - 1] and not pred_left_contact[i]:
                        to_frames.append(i)

            # --- Segment gait cycles and stance phases ---
            def _resample(values, num_points):
                if len(values) < 2:
                    return None
                src = np.linspace(0, 1, len(values), dtype=np.float32)
                dst = np.linspace(0, 1, num_points, dtype=np.float32)
                return np.interp(dst, src, np.asarray(values, dtype=np.float32))

            # Full cycle: HS → next HS  (DOFs + GRF)
            pred_cycles = [[] for _ in range(num_dofs)]
            ref_cycles = [[] for _ in range(num_dofs)]
            # GRF cycles: 6 channels (left xyz, right xyz)
            n_grf_ch = 6
            grf_cycle_data = [[] for _ in range(n_grf_ch)]
            for ci in range(len(hs_frames) - 1):
                s, e = hs_frames[ci], hs_frames[ci + 1]
                if e - s < 4:
                    continue
                for d in range(num_dofs):
                    w = _resample(pred_dof[s:e, d], cycle_pts)
                    if w is not None:
                        pred_cycles[d].append(w)
                    if ref_dof.shape[0] >= e and d < ref_dof.shape[1]:
                        wr = _resample(ref_dof[s:e, d], cycle_pts)
                        if wr is not None:
                            ref_cycles[d].append(wr)
                if grf_left is not None and grf_right is not None:
                    for ax_i in range(3):
                        wl = _resample(grf_left[s:e, ax_i], cycle_pts)
                        wr = _resample(grf_right[s:e, ax_i], cycle_pts)
                        if wl is not None:
                            grf_cycle_data[ax_i].append(wl)
                        if wr is not None:
                            grf_cycle_data[3 + ax_i].append(wr)

            # Stance phase: HS → next TO after that HS  (DOFs + GRF)
            pred_stance = [[] for _ in range(num_dofs)]
            ref_stance = [[] for _ in range(num_dofs)]
            grf_stance_data = [[] for _ in range(n_grf_ch)]
            for hs in hs_frames:
                tos_after = [t for t in to_frames if t > hs]
                if not tos_after:
                    continue
                to = tos_after[0]
                if to - hs < 3:
                    continue
                for d in range(num_dofs):
                    w = _resample(pred_dof[hs:to, d], cycle_pts)
                    if w is not None:
                        pred_stance[d].append(w)
                    if ref_dof.shape[0] >= to and d < ref_dof.shape[1]:
                        wr = _resample(ref_dof[hs:to, d], cycle_pts)
                        if wr is not None:
                            ref_stance[d].append(wr)
                if grf_left is not None and grf_right is not None:
                    for ax_i in range(3):
                        wl = _resample(grf_left[hs:to, ax_i], cycle_pts)
                        wr = _resample(grf_right[hs:to, ax_i], cycle_pts)
                        if wl is not None:
                            grf_stance_data[ax_i].append(wl)
                        if wr is not None:
                            grf_stance_data[3 + ax_i].append(wr)

            # ============================================================
            # FIGURE 1: Timeseries (predicted vs reference, contact shading)
            # ============================================================
            time_arr = np.arange(n_frames) * dt
            n_grf_rows = 3 if grf_left is not None else 0
            total_rows = num_dofs + n_grf_rows
            fig_ts, axes_ts = plt.subplots(
                total_rows, 1, figsize=(14, 2.5 * total_rows), squeeze=False
            )
            fig_ts.suptitle(f"{motion_name} — Predicted vs Reference", fontsize=14)

            for d in range(num_dofs):
                ax = axes_ts[d][0]
                ax.plot(time_arr, np.degrees(pred_dof[:, d]), color="tab:blue",
                        linewidth=1.0, label="Predicted")
                if d < ref_dof.shape[1]:
                    ref_time = np.arange(ref_n) * dt
                    ax.plot(ref_time, np.degrees(ref_dof[:, d]), color="tab:orange",
                            linewidth=1.0, alpha=0.8, label="Reference")
                if pred_left_contact is not None:
                    self._shade_contact(ax, pred_left_contact, dt, color="blue", alpha=0.07)
                if pred_right_contact is not None:
                    self._shade_contact(ax, pred_right_contact, dt, color="red", alpha=0.05)
                ax.set_ylabel(f"{dof_names[d]} (deg)", fontsize=8)
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=7)
                if d == 0:
                    ax.legend(fontsize=8, loc="upper right")

            # GRF timeseries rows
            if grf_left is not None:
                for ax_i in range(3):
                    ax = axes_ts[num_dofs + ax_i][0]
                    ax.plot(time_arr, grf_left[:, ax_i], color="tab:blue",
                            linewidth=0.8, label="Left foot")
                    ax.plot(time_arr, grf_right[:, ax_i], color="tab:red",
                            linewidth=0.8, alpha=0.8, label="Right foot")
                    if pred_left_contact is not None:
                        self._shade_contact(ax, pred_left_contact, dt, color="blue", alpha=0.07)
                    if pred_right_contact is not None:
                        self._shade_contact(ax, pred_right_contact, dt, color="red", alpha=0.05)
                    ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=8)
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(labelsize=7)
                    if ax_i == 0:
                        ax.legend(fontsize=8, loc="upper right")

            axes_ts[total_rows - 1][0].set_xlabel("Time (s)", fontsize=9)
            fig_ts.tight_layout()
            fig_ts.savefig(output_dir / f"{motion_name}_timeseries.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ts)

            if grf_left is not None:
                fig_grf_ts, axes_grf_ts = plt.subplots(
                    3, 1, figsize=(14, 7.5), squeeze=False
                )
                fig_grf_ts.suptitle(f"{motion_name} — Ground Reaction Forces", fontsize=14)
                for ax_i in range(3):
                    ax = axes_grf_ts[ax_i][0]
                    ax.plot(time_arr, grf_left[:, ax_i], color="tab:blue",
                            linewidth=0.9, label="Left foot")
                    ax.plot(time_arr, grf_right[:, ax_i], color="tab:red",
                            linewidth=0.9, alpha=0.85, label="Right foot")
                    if pred_left_contact is not None:
                        self._shade_contact(ax, pred_left_contact, dt, color="blue", alpha=0.07)
                    if pred_right_contact is not None:
                        self._shade_contact(ax, pred_right_contact, dt, color="red", alpha=0.05)
                    ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=9)
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(labelsize=8)
                    if ax_i == 0:
                        ax.legend(fontsize=8, loc="upper right")
                axes_grf_ts[2][0].set_xlabel("Time (s)", fontsize=9)
                fig_grf_ts.tight_layout()
                fig_grf_ts.savefig(output_dir / f"{motion_name}_forces.png", dpi=150, bbox_inches="tight")
                plt.close(fig_grf_ts)

            # ============================================================
            # FIGURE 2: Full gait cycle (HS → HS), mean ± std
            # ============================================================
            has_dof_cycles = any(len(c) >= 2 for c in pred_cycles)
            has_grf_cycles = any(len(c) >= 2 for c in grf_cycle_data)
            if has_dof_cycles or has_grf_cycles:
                phase_pct = np.linspace(0, 100, cycle_pts)
                n_cycle_grf_rows = 3 if has_grf_cycles else 0
                gc_rows = num_dofs + n_cycle_grf_rows
                fig_gc, axes_gc = plt.subplots(
                    gc_rows, 1, figsize=(10, 2.5 * gc_rows), squeeze=False
                )
                fig_gc.suptitle(
                    f"{motion_name} — Full Gait Cycle (HS→HS), "
                    f"{len(pred_cycles[0])} cycles",
                    fontsize=13,
                )
                for d in range(num_dofs):
                    ax = axes_gc[d][0]
                    self._plot_mean_std_band(
                        ax, phase_pct, pred_cycles[d], color="tab:blue", label="Predicted"
                    )
                    self._plot_mean_std_band(
                        ax, phase_pct, ref_cycles[d], color="tab:orange", label="Reference"
                    )
                    ax.set_ylabel(f"{dof_names[d]} (deg)", fontsize=8)
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(labelsize=7)
                    if d == 0:
                        ax.legend(fontsize=8, loc="upper right")
                # GRF cycle rows
                if has_grf_cycles:
                    for ax_i in range(3):
                        ax = axes_gc[num_dofs + ax_i][0]
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_cycle_data[ax_i],
                            color="tab:blue", label="Left foot"
                        )
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_cycle_data[3 + ax_i],
                            color="tab:red", label="Right foot"
                        )
                        ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=8)
                        ax.grid(True, alpha=0.2)
                        ax.tick_params(labelsize=7)
                        if ax_i == 0:
                            ax.legend(fontsize=8, loc="upper right")
                axes_gc[gc_rows - 1][0].set_xlabel("Gait cycle (%)", fontsize=9)
                fig_gc.tight_layout()
                fig_gc.savefig(output_dir / f"{motion_name}_gait_cycle.png", dpi=150, bbox_inches="tight")
                plt.close(fig_gc)

                if has_grf_cycles:
                    fig_grf_gc, axes_grf_gc = plt.subplots(
                        3, 1, figsize=(10, 7.5), squeeze=False
                    )
                    fig_grf_gc.suptitle(
                        f"{motion_name} — GRF Full Gait Cycle (HS→HS)",
                        fontsize=13,
                    )
                    for ax_i in range(3):
                        ax = axes_grf_gc[ax_i][0]
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_cycle_data[ax_i],
                            color="tab:blue", label="Left foot"
                        )
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_cycle_data[3 + ax_i],
                            color="tab:red", label="Right foot"
                        )
                        ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=9)
                        ax.grid(True, alpha=0.2)
                        ax.tick_params(labelsize=8)
                        if ax_i == 0:
                            ax.legend(fontsize=8, loc="upper right")
                    axes_grf_gc[2][0].set_xlabel("Gait cycle (%)", fontsize=9)
                    fig_grf_gc.tight_layout()
                    fig_grf_gc.savefig(output_dir / f"{motion_name}_forces_gait_cycle.png", dpi=150, bbox_inches="tight")
                    plt.close(fig_grf_gc)

            # ============================================================
            # FIGURE 3: Stance phase (HS → TO), mean ± std
            # ============================================================
            has_dof_stance = any(len(c) >= 2 for c in pred_stance)
            has_grf_stance = any(len(c) >= 2 for c in grf_stance_data)
            if has_dof_stance or has_grf_stance:
                phase_pct = np.linspace(0, 100, cycle_pts)
                n_stance_grf_rows = 3 if has_grf_stance else 0
                st_rows = num_dofs + n_stance_grf_rows
                fig_st, axes_st = plt.subplots(
                    st_rows, 1, figsize=(10, 2.5 * st_rows), squeeze=False
                )
                fig_st.suptitle(
                    f"{motion_name} — Stance Phase (HS→TO), "
                    f"{len(pred_stance[0])} phases",
                    fontsize=13,
                )
                for d in range(num_dofs):
                    ax = axes_st[d][0]
                    self._plot_mean_std_band(
                        ax, phase_pct, pred_stance[d], color="tab:blue", label="Predicted"
                    )
                    self._plot_mean_std_band(
                        ax, phase_pct, ref_stance[d], color="tab:orange", label="Reference"
                    )
                    ax.set_ylabel(f"{dof_names[d]} (deg)", fontsize=8)
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(labelsize=7)
                    if d == 0:
                        ax.legend(fontsize=8, loc="upper right")
                # GRF stance rows
                if has_grf_stance:
                    for ax_i in range(3):
                        ax = axes_st[num_dofs + ax_i][0]
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_stance_data[ax_i],
                            color="tab:blue", label="Left foot"
                        )
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_stance_data[3 + ax_i],
                            color="tab:red", label="Right foot"
                        )
                        ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=8)
                        ax.grid(True, alpha=0.2)
                        ax.tick_params(labelsize=7)
                        if ax_i == 0:
                            ax.legend(fontsize=8, loc="upper right")
                axes_st[st_rows - 1][0].set_xlabel("Stance phase (%)", fontsize=9)
                fig_st.tight_layout()
                fig_st.savefig(output_dir / f"{motion_name}_stance.png", dpi=150, bbox_inches="tight")
                plt.close(fig_st)

                if has_grf_stance:
                    fig_grf_st, axes_grf_st = plt.subplots(
                        3, 1, figsize=(10, 7.5), squeeze=False
                    )
                    fig_grf_st.suptitle(
                        f"{motion_name} — GRF Stance Phase (HS→TO)",
                        fontsize=13,
                    )
                    for ax_i in range(3):
                        ax = axes_grf_st[ax_i][0]
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_stance_data[ax_i],
                            color="tab:blue", label="Left foot"
                        )
                        self._plot_mean_std_band_raw(
                            ax, phase_pct, grf_stance_data[3 + ax_i],
                            color="tab:red", label="Right foot"
                        )
                        ax.set_ylabel(f"GRF {grf_labels[ax_i]}", fontsize=9)
                        ax.grid(True, alpha=0.2)
                        ax.tick_params(labelsize=8)
                        if ax_i == 0:
                            ax.legend(fontsize=8, loc="upper right")
                    axes_grf_st[2][0].set_xlabel("Stance phase (%)", fontsize=9)
                    fig_grf_st.tight_layout()
                    fig_grf_st.savefig(output_dir / f"{motion_name}_forces_stance.png", dpi=150, bbox_inches="tight")
                    plt.close(fig_grf_st)

        print(f"Per-motion plots saved to: {output_dir}")

    @staticmethod
    def _shade_contact(ax, contact_bool: np.ndarray, dt: float, color: str, alpha: float):
        """Shade time regions where contact is active."""
        in_contact = False
        start = 0.0
        for i, c in enumerate(contact_bool):
            if c and not in_contact:
                start = i * dt
                in_contact = True
            elif not c and in_contact:
                ax.axvspan(start, i * dt, color=color, alpha=alpha)
                in_contact = False
        if in_contact:
            ax.axvspan(start, len(contact_bool) * dt, color=color, alpha=alpha)

    @staticmethod
    def _plot_mean_std_band(ax, x: np.ndarray, cycles: list, color: str, label: str):
        """Plot mean ± 1 std band from a list of resampled cycle arrays (in degrees)."""
        if len(cycles) < 1:
            return
        stacked = np.stack([np.degrees(c) for c in cycles], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=1.5, label=f"{label} (n={len(cycles)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    @staticmethod
    def _plot_mean_std_band_raw(ax, x: np.ndarray, cycles: list, color: str, label: str):
        """Plot mean ± 1 std band without unit conversion (for forces in Newtons)."""
        if len(cycles) < 1:
            return
        stacked = np.stack(cycles, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=1.5, label=f"{label} (n={len(cycles)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    def _save_predicted_motion_lib(
        self, metrics: Dict[str, MotionMetrics], epoch: int
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a "predicted" version of MotionLib where unknown fields are copied
        from the ground-truth self.motion_lib.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()

        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * self.env.dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * self.env.dt

        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                f = min(f, data.shape[1])
                per_motion.append(data[m, :f].detach().clone())
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        # Pack predicted contacts from metrics
        contacts_data = metrics[
            "rigid_body_contacts"
        ].data  # [num_motions, max_frames, num_bodies]
        contacts_list = []
        for m in range(num_motions):
            f = motion_num_frames[m].item()
            # Clamp to available frames
            f = min(f, contacts_data.shape[1])
            # Convert float contacts to bool for consistency with MotionLib format
            contacts_list.append(contacts_data[m, :f].bool().detach().clone())
        contacts = torch.cat(contacts_list, dim=0)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": contacts,  # Always save predicted contacts
        }

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
