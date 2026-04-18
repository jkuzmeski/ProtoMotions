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
import os
import traceback
import torch
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
    ProjectileConfig,
)
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)
from protomotions.simulator.newton.config import NewtonSimulatorConfig
import warp as wp
import newton
from newton import JointTargetMode
from newton.selection import ArticulationView
from newton import Contacts
from newton.sensors import SensorContact
from newton.solvers import SolverNotifyFlags
import copy
import logging

log = logging.getLogger(__name__)


wp.config.enable_backward = False
wp.config.quiet = True


@wp.kernel
def compute_pd_torques_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    pd_targets: wp.array(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    torque_limits: wp.array(dtype=wp.float32),
    q_stride: int,
    qd_stride: int,
    q_dof_start: int,
    qd_dof_start: int,
    num_dofs: int,
):
    """Compute PD torques for explicit PD control (CUDA graph compatible)."""
    tid = wp.tid()
    env_id = tid // num_dofs
    dof_id = tid % num_dofs

    q_idx = env_id * q_stride + q_dof_start + dof_id
    qd_idx = env_id * qd_stride + qd_dof_start + dof_id

    pos = joint_q[q_idx]
    vel = joint_qd[qd_idx]
    target = pd_targets[tid]

    torque = kp[dof_id] * (target - pos) - kd[dof_id] * vel
    torque = wp.clamp(torque, -torque_limits[dof_id], torque_limits[dof_id])

    joint_f[qd_idx] = torque


@wp.kernel
def apply_torques_kernel(
    joint_f: wp.array(dtype=wp.float32),
    torques: wp.array(dtype=wp.float32),
    qd_stride: int,
    qd_dof_start: int,
    num_dofs: int,
):
    """Copy pre-computed torques to joint_f (CUDA graph compatible)."""
    tid = wp.tid()
    env_id = tid // num_dofs
    dof_id = tid % num_dofs

    qd_idx = env_id * qd_stride + qd_dof_start + dof_id
    joint_f[qd_idx] = torques[tid]


class NewtonSimulator(Simulator):
    """Newton physics engine wrapper for our simulation framework."""

    config: NewtonSimulatorConfig

    def __init__(
        self,
        config: NewtonSimulatorConfig,
        robot_config,
        terrain,
        device: torch.device,
        scene_lib,
        custom_key_handlers: Optional[Dict[str, callable]] = None,
    ) -> None:
        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        self._custom_key_handlers = custom_key_handlers or {}
        self._any_key_pressed = False  # used to avoid repeating the same key press

        # Configure timing
        self.sim_time = 0.0
        self.sim_dt = 1.0 / self.config.sim.fps
        self.decimation = self.config.sim.decimation
        self.frame_dt = self.sim_dt * self.decimation

        self._contact_sensors = {}
        self._contact_forces = {}  # Store contact forces per body
        self.contacts = None  # Initialized after solver/sensors are set up
        self._camera_initialized = False
        self._needs_state_sync = False
        self._last_reset_root_pos: Optional[torch.Tensor] = None
        self._last_reset_root_rot: Optional[torch.Tensor] = None
        self._last_reset_root_vel: Optional[torch.Tensor] = None
        self._last_reset_root_ang_vel: Optional[torch.Tensor] = None
        self._last_reset_dof_pos: Optional[torch.Tensor] = None
        self._last_reset_dof_vel: Optional[torch.Tensor] = None
        self._last_reset_sim_time: Optional[torch.Tensor] = None

    def _get_builder_joint_keys(self) -> list[str]:
        """Return builder joint names in the old ProtoMotions matching format.

        Older Newton builds exposed ``joint_key`` directly. The local fork stores
        full joint labels in ``joint_label``; the basename after the final slash
        preserves the compound key format ProtoMotions expects.
        """
        if hasattr(self.robot, "joint_key"):
            return list(self.robot.joint_key)
        if hasattr(self.robot, "joint_label"):
            return [label.rsplit("/", 1)[-1] for label in self.robot.joint_label]
        raise AttributeError("Newton ModelBuilder has neither 'joint_key' nor 'joint_label'")

    def _get_robot_articulation_pattern(self) -> str:
        """Resolve the articulation selector pattern for the finalized model.

        Newton's local API drifted from builder-side ``articulation_key`` to
        ``articulation_label``. Some builds also preserve hierarchical labels
        after replication, so hard-coding ``"robot"`` is brittle. Prefer the
        exact canonical label when present and fall back to simple wildcard
        forms that still select one robot articulation per world.
        """
        labels = list(getattr(self.model, "articulation_label", []))
        if not labels:
            return "robot"

        if "robot" in labels:
            return "robot"

        leaf_labels = [label.rsplit("/", 1)[-1] for label in labels]
        if all(label == "robot" for label in leaf_labels):
            return "*/robot"

        if all(label.startswith("robot") for label in leaf_labels):
            return "robot*"

        unique_labels = set(labels)
        if len(unique_labels) == 1:
            return next(iter(unique_labels))

        unique_leaf_labels = set(leaf_labels)
        if len(unique_leaf_labels) == 1:
            return f"*/{next(iter(unique_leaf_labels))}"

        raise KeyError(
            "Unable to resolve robot articulation pattern from labels "
            f"{labels!r}; expected a stable per-world robot articulation"
        )

    def _get_model_body_pattern(self, body_name: str) -> str:
        """Resolve a body selector pattern against finalized Newton body labels."""
        body_labels = list(getattr(self.model, "body_label", []))
        if body_name in body_labels:
            return body_name

        if any(label.rsplit("/", 1)[-1] == body_name for label in body_labels):
            return f"*/{body_name}"

        raise KeyError(
            f"Unable to resolve body pattern for '{body_name}' from labels {body_labels!r}"
        )

    def _create_simulation(self) -> None:
        """Create the Newton simulation environment."""
        self._create_envs()
        self._zero_passive_forces()
        self._setup_robot()
        self._setup_sim()
        if self.robot_config.contact_bodies is not None:
            self._setup_contact_sensors()
        self._create_contacts()
        self._set_robot_friction_to_terrain()
        self._apply_domain_randomization_if_needed()

        self.graph = None
        self.use_cuda_graph = False

        fail_fast_warnings = bool(
            getattr(self.config.sim, "raise_on_mujoco_warning", False)
        )
        can_use_cuda_graph = (
            wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        )

        if can_use_cuda_graph and fail_fast_warnings:
            print(
                "[INFO] CUDA graph disabled: fail-fast MuJoCo warning escalation "
                "requires uncaptured Newton solver steps"
            )

        if can_use_cuda_graph and not fail_fast_warnings:
            print(f"[INFO] Using CUDA graph ({self.control_type.name})")
            self.use_cuda_graph = True
            zeros = torch.zeros(
                self.num_envs,
                1,
                self.robot_config.number_of_actions,
                device=self.device,
                dtype=torch.float32,
            )

            if self.control_type == ControlType.BUILT_IN_PD:
                self.robot_view.set_attribute(
                    "joint_target_pos",
                    self.control,
                    wp.from_torch(zeros, dtype=wp.float32),
                )
            else:
                self._update_pd_targets(zeros.squeeze(1))

            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            print(f"[INFO] {self.control_type.name} mode (no CUDA graph)")

    def _create_envs(self) -> None:
        """Creates environments and loads robot assets.

        Follows the Newton G1 example pattern: configure joint properties
        on the builder BEFORE finalize, then use replicate() for multi-env.
        """
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        asset_path = os.path.join(asset_root, asset_file)

        print(f"Loading robot from: {asset_path}")

        # 1. Create articulation builder
        self.robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(self.robot)
        self.robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig()
        self.robot.default_shape_cfg.mu = 1.0

        # 2. Load MJCF
        self.robot.add_mjcf(
            asset_path,
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=False,
            floating=not self.robot_config.asset.fix_base_link,
            enable_self_collisions=self.robot_config.asset.self_collisions,
        )

        # 3. Set per-DOF joint properties ON THE BUILDER (before finalize)
        self._configure_builder_joint_properties()

        if hasattr(self.robot, "articulation_label"):
            self.robot.articulation_label = ["robot"]
        if hasattr(self.robot, "articulation_key"):
            self.robot.articulation_key = ["robot"]
        self.robot.approximate_meshes("convex_hull")

        # 5. Add projectile free bodies to the builder (before replicate)
        self._proj_config = ProjectileConfig()
        proj_sizes = self._proj_config.get_sizes()
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=self._proj_config.density
        )
        for i in range(self._proj_config.num_projectiles):
            s = proj_sizes[i]
            xform = wp.transform(
                (0.0, 0.0, self._proj_config.hide_z),
                (0.0, 0.0, 0.0, 1.0),
            )
            body = self.robot.add_body(xform=xform)
            self.robot.add_shape_box(body=body, hx=s, hy=s, hz=s, cfg=shape_cfg)

        # 6. Replicate into worlds with ground plane
        builder = newton.ModelBuilder()
        builder.replicate(self.robot, self.num_envs)

        if self.terrain is not None:
            ground_cfg = newton.ModelBuilder.ShapeConfig(
                mu=self.terrain.sim_config.static_friction,
                restitution=self.terrain.sim_config.restitution,
            )
            builder.add_ground_plane(cfg=ground_cfg)
        else:
            builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        # Compute projectile joint_q/joint_qd offsets per world.
        # Per-world joint_q layout:
        #   [robot_free(7), robot_dofs(N), proj_0_free(7), ..., proj_{P-1}_free(7)]
        num_dofs = self.robot_config.number_of_actions
        is_floating = not self.robot_config.asset.fix_base_link
        self._proj_jq_offset = (7 if is_floating else 0) + num_dofs
        self._proj_jqd_offset = (6 if is_floating else 0) + num_dofs
        num_proj = self._proj_config.num_projectiles
        self._proj_q_stride = self._proj_jq_offset + num_proj * 7
        self._proj_qd_stride = self._proj_jqd_offset + num_proj * 6

    def _zero_passive_forces(self) -> None:
        """Zero out MuJoCo passive stiffness/damping loaded from MJCF.

        We manage PD control ourselves (via joint_target_ke/kd or explicit PD
        kernels), so passive forces from the MJCF would double-count.
        """
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return
        for attr_name in ("dof_passive_stiffness", "dof_passive_damping"):
            attr = getattr(mujoco_attrs, attr_name, None)
            if attr is not None:
                attr.zero_()

    def _configure_builder_joint_properties(self) -> None:
        """Set joint stiffness, damping, armature, and actuator mode on the builder.

        This must be called BEFORE finalize() so MuJoCo creates actuators
        correctly. Builder DOFs 0-5 are the floating base (skip); DOFs 6+
        are the actuated joint DOFs.
        """
        # Build mapping from our DOF names to builder DOF indices.
        # The builder joint key list contains compound names like
        # "L_Hip_x_L_Hip_y_L_Hip_z" for multi-DOF joints.
        common_dof_names = list(self._dof_names)  # copy
        builder_joint_keys = self._get_builder_joint_keys()
        is_floating = not self.robot_config.asset.fix_base_link
        dof_offset = 6 if is_floating else 0  # skip free-joint DOFs

        # Walk through our DOF names, matching them to builder joint keys.
        builder_dof_idx = dof_offset
        while len(common_dof_names) > 0:
            common_dof_name = common_dof_names[0]

            # Check if it's a direct match
            if common_dof_name in builder_joint_keys:
                # Single-DOF joint
                info = self.robot_config.control.control_info[common_dof_name]
                self._set_builder_dof_properties(builder_dof_idx, info)
                builder_dof_idx += 1
                common_dof_names.pop(0)
            else:
                # Multi-DOF joint: find the compound key containing this name
                multi_dof_key = None
                for key in builder_joint_keys:
                    if common_dof_name in key:
                        multi_dof_key = key
                        break
                assert multi_dof_key is not None, (
                    f"No joint key match found for {common_dof_name} "
                    f"in {builder_joint_keys}"
                )

                # Consume all DOF names that belong to this compound joint
                while (
                    len(common_dof_names) > 0 and common_dof_names[0] in multi_dof_key
                ):
                    info = self.robot_config.control.control_info[common_dof_names[0]]
                    self._set_builder_dof_properties(builder_dof_idx, info)
                    builder_dof_idx += 1
                    common_dof_names.pop(0)

    def _set_builder_dof_properties(self, dof_idx: int, info) -> None:
        """Set a single builder DOF's properties from a ControlInfo entry."""
        if self.control_type == ControlType.BUILT_IN_PD:
            self.robot.joint_target_ke[dof_idx] = info.stiffness
            self.robot.joint_target_kd[dof_idx] = info.damping
            self.robot.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)
        else:
            # PROPORTIONAL / TORQUE: we apply forces ourselves
            self.robot.joint_target_ke[dof_idx] = 0.0
            self.robot.joint_target_kd[dof_idx] = 0.0
            self.robot.joint_target_mode[dof_idx] = int(JointTargetMode.NONE)

        if info.armature is not None:
            self.robot.joint_armature[dof_idx] = info.armature
        if info.friction is not None:
            self.robot.joint_friction[dof_idx] = info.friction
        if info.effort_limit is not None:
            self.robot.joint_effort_limit[dof_idx] = info.effort_limit
        if info.velocity_limit is not None:
            self.robot.joint_velocity_limit[dof_idx] = info.velocity_limit

    def _setup_robot(self) -> None:
        """Setup robot view and capture default states.

        Joint properties (ke/kd/armature/act_mode) are already set on the
        builder before finalize — see _configure_builder_joint_properties().
        """
        common_dof_names = copy.deepcopy(self._dof_names)
        builder_joint_keys = self._get_builder_joint_keys()
        newton_dof_names = {}

        while len(common_dof_names) > 0:
            common_dof_name = common_dof_names[0]
            if common_dof_name in builder_joint_keys:
                newton_dof_names[common_dof_name] = common_dof_name
                common_dof_names.pop(0)
            else:
                multi_dof_name = None
                for newton_dof_name in builder_joint_keys:
                    if common_dof_name in newton_dof_name:
                        multi_dof_name = newton_dof_name
                        break
                assert (
                    multi_dof_name is not None
                ), f"No joint key match found for {common_dof_name} in {builder_joint_keys}"

                newton_dof_names[multi_dof_name] = []
                while (
                    len(common_dof_names) > 0 and common_dof_names[0] in multi_dof_name
                ):
                    newton_dof_names[multi_dof_name].append(common_dof_names[0])
                    common_dof_names.pop(0)

        self._newton_dof_names = newton_dof_names

        self.robot_view = ArticulationView(
            self.model,
            pattern=self._get_robot_articulation_pattern(),
            include_joints=list(self._newton_dof_names.keys()),
            include_links=self._body_names,
        )

        self.default_body_transforms = (
            wp.to_torch(self.robot_view.get_link_transforms(self.model))
            .squeeze(1)
            .clone()
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        self.default_body_velocities = (
            wp.to_torch(self.robot_view.get_link_velocities(self.model))
            .squeeze(1)
            .clone()
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        self.default_root_transforms = (
            wp.to_torch(self.robot_view.get_root_transforms(self.model))
            .squeeze(1)
            .clone()
        )
        self.default_root_velocities = (
            wp.to_torch(self.robot_view.get_root_velocities(self.model))
            .squeeze(1)
            .clone()
        )
        self.default_dof_positions = (
            wp.to_torch(self.robot_view.get_dof_positions(self.model))
            .squeeze(1)
            .clone()
        )
        self.default_dof_velocities = (
            wp.to_torch(self.robot_view.get_dof_velocities(self.model))
            .squeeze(1)
            .clone()
        )

        self._setup_explicit_pd_arrays()

    def _setup_explicit_pd_arrays(self) -> None:
        """Setup persistent Warp arrays for explicit PD control."""
        num_dofs = self.robot_config.number_of_actions
        self._pd_num_dofs = num_dofs

        is_floating = not self.robot_config.asset.fix_base_link
        num_proj = self._proj_config.num_projectiles
        self._pd_q_stride = (7 if is_floating else 0) + num_dofs + num_proj * 7
        self._pd_qd_stride = (6 if is_floating else 0) + num_dofs + num_proj * 6
        self._pd_q_dof_start = 7 if is_floating else 0
        self._pd_qd_dof_start = 6 if is_floating else 0

        device_str = (
            str(self.device) if not isinstance(self.device, str) else self.device
        )
        self._pd_targets_wp = wp.zeros(
            self.num_envs * num_dofs, dtype=wp.float32, device=device_str
        )

        kp_list = []
        kd_list = []
        torque_limits_list = []
        for dof_name in self.robot_view.joint_names:
            common_dof_names = self._newton_dof_names[dof_name]
            if not isinstance(common_dof_names, list):
                common_dof_names = [common_dof_names]
            for common_dof_name in common_dof_names:
                kp_list.append(
                    self.robot_config.control.control_info[common_dof_name].stiffness
                )
                kd_list.append(
                    self.robot_config.control.control_info[common_dof_name].damping
                )
                limit = self.robot_config.control.control_info[
                    common_dof_name
                ].effort_limit
                torque_limits_list.append(limit if limit is not None else 1000.0)

        self._pd_kp_wp = wp.from_torch(
            torch.tensor(kp_list, device=self.device, dtype=torch.float32)
        )
        self._pd_kd_wp = wp.from_torch(
            torch.tensor(kd_list, device=self.device, dtype=torch.float32)
        )
        self._pd_torque_limits_wp = wp.from_torch(
            torch.tensor(torque_limits_list, device=self.device, dtype=torch.float32)
        )

    def _setup_sim(self) -> None:
        """Creates simulation using config parameters."""
        sim_params = self.config.sim
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver=sim_params.solver,
            integrator=sim_params.integrator,
            njmax=sim_params.njmax,
            nconmax=sim_params.nconmax,
            nccdmax=sim_params.nccdmax,
            naccdmax=sim_params.naccdmax,
            iterations=sim_params.iterations,
            ls_iterations=sim_params.ls_iterations,
            ls_parallel=sim_params.ls_parallel,
            impratio=sim_params.impratio,
            cone=sim_params.cone,
            ccd_iterations=sim_params.ccd_iterations,
            max_epa_workspace_iterations=sim_params.max_epa_workspace_iterations,
            raise_on_mujoco_warning=sim_params.raise_on_mujoco_warning,
        )

        self.viewer = None
        if not self.headless:
            viewer_backend = getattr(self.config, "viewer_backend", "gl")
            if viewer_backend == "gl":
                self.viewer = newton.viewer.ViewerGL()
                self.viewer.vsync = True
            elif viewer_backend == "viser":
                viewer_port = getattr(self.config, "viewer_port", 8097)
                self.viewer = newton.viewer.ViewerViser(port=viewer_port)
                self.viewer.show_static = True
            else:
                raise ValueError(
                    f"Unsupported Newton viewer_backend '{viewer_backend}'. "
                    "Expected 'gl' or 'viser'."
                )
            self.viewer.set_model(self.model)
            if viewer_backend == "viser":
                self._setup_viser_ground_visual()
                viewer_max_worlds = getattr(self.config, "viewer_max_worlds", 16)
                if viewer_max_worlds is not None:
                    self.viewer.set_visible_worlds(
                        range(min(viewer_max_worlds, self.model.world_count))
                    )

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

    def _setup_viser_ground_visual(self) -> None:
        """Add a viewer-only ground reference for the Viser backend.

        Some Newton/Viser combinations do not reliably display the built-in
        plane primitive. A scene grid keeps ground orientation readable even
        when the physics ground is not visible.
        """
        if getattr(self.config, "viewer_backend", "gl") != "viser":
            return

        server = getattr(self.viewer, "_server", None)
        if server is None or not hasattr(server, "scene"):
            return

        add_grid = getattr(server.scene, "add_grid", None)
        if add_grid is None:
            return

        grid_path = "/protomotions_ground"
        try:
            add_grid(
                grid_path,
                plane="xy",
                infinite_grid=True,
                cell_color=(170, 170, 170),
                cell_thickness=1.0,
                cell_size=0.5,
                section_color=(120, 120, 120),
                section_thickness=1.5,
                section_size=2.0,
                fade_distance=80.0,
                shadow_opacity=0.2,
                plane_color=(245, 245, 245),
                plane_opacity=0.12,
            )
        except TypeError:
            # Fallback for older viser signatures.
            try:
                add_grid(grid_path, width=2.0, height=2.0, cell_size=0.5)
            except TypeError:
                log.warning(
                    "ViewerViser scene.add_grid signature is unsupported; "
                    "skipping viewer-only ground grid."
                )

    def _apply_domain_randomization_if_needed(self) -> None:
        """Apply friction and center of mass domain randomization.

        Newton/MuJoCo uses:
        - shape_material_mu for friction coefficient (single value, not static/dynamic)
        - shape_material_restitution for restitution
        - body_com for center of mass offsets

        After modifying these, we must call solver.notify_model_changed() with
        the appropriate flags so MuJoCo updates its internal model.
        """
        if self._domain_randomization is None:
            return

        notify_flags = 0

        # Apply friction randomization
        if "friction" in self._domain_randomization:
            # Get current friction values via ArticulationView (scoped to robot)
            # Shape: (num_envs, 1, num_shapes_per_robot)
            mu_wp = self.robot_view.get_attribute("shape_material_mu", self.model)
            rest_wp = self.robot_view.get_attribute(
                "shape_material_restitution", self.model
            )
            current_friction = wp.to_torch(mu_wp)
            current_restitution = wp.to_torch(rest_wp)

            # Get body indices that should be randomized
            body_indices = self._domain_randomization["friction"]["body_indices"]
            static_friction = self._domain_randomization["friction"]["static_friction"]
            restitution = self._domain_randomization["friction"]["restitution"]

            num_buckets = static_friction.shape[0] if static_friction is not None else 0

            if num_buckets > 0:
                # Build body name → local shape indices mapping via ArticulationView
                link_name_to_idx = {
                    name: i for i, name in enumerate(self.robot_view.link_names)
                }

                for idx, local_body_idx in enumerate(body_indices):
                    body_name = self._body_names[local_body_idx]
                    link_idx = link_name_to_idx.get(body_name)
                    if link_idx is None:
                        continue
                    local_shape_indices = self.robot_view.link_shapes[link_idx]
                    if not local_shape_indices:
                        continue

                    # Generate random bucket assignment per env for this body type
                    bucket_ids = torch.randint(
                        0, num_buckets, (self.num_envs,), device=self.device
                    )

                    # Vectorized assignment across all envs at once
                    if static_friction is not None:
                        friction_values = static_friction[bucket_ids, idx]
                        current_friction[:, 0, local_shape_indices] = (
                            friction_values.unsqueeze(1)
                        )

                    if restitution is not None:
                        restitution_values = restitution[bucket_ids, idx]
                        current_restitution[:, 0, local_shape_indices] = (
                            restitution_values.unsqueeze(1)
                        )

                # Write back through ArticulationView
                self.robot_view.set_attribute("shape_material_mu", self.model, mu_wp)
                self.robot_view.set_attribute(
                    "shape_material_restitution", self.model, rest_wp
                )

                notify_flags |= SolverNotifyFlags.SHAPE_PROPERTIES
                print(
                    f"[INFO] Applied friction domain randomization to {len(body_indices)} body types"
                )

        # Apply center of mass randomization
        if "center_of_mass" in self._domain_randomization:
            # Get current body COM values via ArticulationView
            # Shape: (num_envs, 1, num_links, 3)
            com_wp = self.robot_view.get_attribute("body_com", self.model)
            current_com = wp.to_torch(com_wp)

            body_indices = self._domain_randomization["center_of_mass"]["body_indices"]
            com_offsets = self._domain_randomization["center_of_mass"]["com"]

            # Build body name → link index mapping
            link_name_to_idx = {
                name: i for i, name in enumerate(self.robot_view.link_names)
            }

            for idx, local_body_idx in enumerate(body_indices):
                body_name = self._body_names[local_body_idx]
                link_idx = link_name_to_idx.get(body_name)
                if link_idx is None:
                    continue

                # com_offsets shape: [num_envs, num_matching_bodies, 3]
                # current_com shape: (num_envs, 1, num_links, 3)
                offsets = com_offsets[:, idx].to(current_com.device)
                current_com[:, 0, link_idx] += offsets

            # Write back through ArticulationView
            self.robot_view.set_attribute("body_com", self.model, com_wp)

            notify_flags |= SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            print(
                f"[INFO] Applied center of mass domain randomization to {len(body_indices)} body types"
            )

        # Notify solver of changes so MuJoCo updates its internal model
        if notify_flags != 0:
            self.solver.notify_model_changed(notify_flags)

    def _set_robot_friction_to_terrain(self) -> None:
        """Set robot shape friction/restitution to terrain values as baseline.

        This ensures a consistent friction floor before domain randomization.
        DR will override with randomized values for specific bodies if configured.

        Uses ArticulationView get/set_attribute API (not direct model.assign)
        to ensure Newton's internal solver state stays consistent.
        """
        if self.terrain is None:
            return

        terrain_friction = self.terrain.sim_config.static_friction
        terrain_restitution = self.terrain.sim_config.restitution

        # Get robot shape materials via ArticulationView (scoped to robot only)
        mu_wp = self.robot_view.get_attribute("shape_material_mu", self.model)
        rest_wp = self.robot_view.get_attribute(
            "shape_material_restitution", self.model
        )

        # Modify values via torch (writes through to underlying warp memory)
        mu_torch = wp.to_torch(mu_wp)
        rest_torch = wp.to_torch(rest_wp)
        mu_torch[:] = terrain_friction
        rest_torch[:] = terrain_restitution

        # Write back through ArticulationView
        self.robot_view.set_attribute("shape_material_mu", self.model, mu_wp)
        self.robot_view.set_attribute("shape_material_restitution", self.model, rest_wp)
        self.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Returns the ordering of bodies and DOFs in the simulation."""
        joint_names = self.robot_view.joint_names
        dof_names = []

        for joint_name in joint_names:
            if type(self._newton_dof_names[joint_name]) is list:
                dof_names.extend(self._newton_dof_names[joint_name])
            else:
                dof_names.append(self._newton_dof_names[joint_name])
        return SimBodyOrdering(
            body_names=self.robot_view.body_names,
            dof_names=dof_names,
        )

    def _setup_markers(
        self, visualization_markers: Dict[str, VisualizationMarkerConfig]
    ) -> None:
        """Setup visualization markers."""
        return

    def _setup_contact_sensors(self) -> None:
        """Setup contact sensors for each contact body."""
        if (
            self.robot_config.contact_bodies is None
            or len(self.robot_config.contact_bodies) == 0
        ):
            return

        print(
            f"[INFO] Setting up contact sensors for bodies: {self.robot_config.contact_bodies}"
        )

        # Create a contact sensor for each specified contact body
        for body_name in self.robot_config.contact_bodies:
            body_pattern = self._get_model_body_pattern(body_name)
            # Create sensor that detects contacts between this body and anything
            # The sensor will aggregate contacts across all environments
            sensor = SensorContact(
                self.model, sensing_obj_bodies=body_pattern, verbose=False
            )
            self._contact_sensors[body_name] = sensor

            self._contact_forces[body_name] = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=torch.float32
            )

        print(
            f"[INFO] Contact sensors setup complete for {len(self._contact_sensors)} bodies"
        )

    def _create_contacts(self) -> None:
        """Create Contacts object with correct capacity and requested attributes."""
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

    def _simulate(self) -> None:
        """Run physics simulation for one frame (decimation substeps)."""
        for substep_idx in range(self.decimation):
            self.state_0.clear_forces()
            if self.control_type == ControlType.PROPORTIONAL:
                self._apply_pd_kernel(self.state_0)
            elif self.control_type == ControlType.TORQUE:
                self._apply_torques_kernel_method()
            if self.viewer:
                self.viewer.apply_forces(self.state_0)
            try:
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )
            except Exception as exc:
                self._raise_on_solver_failure(
                    source="solver_step",
                    substep_idx=substep_idx,
                    exc=exc,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.decimation % 2 != 0:
            self.state_0.assign(self.state_1)

    @staticmethod
    def _serialize_debug_value(value):
        """Convert MuJoCo/Newton runtime values into debug-friendly Python types."""
        if value is None:
            return None
        if hasattr(value, "numpy"):
            try:
                value = value.numpy()
            except Exception:
                pass
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, tuple):
            return [NewtonSimulator._serialize_debug_value(v) for v in value]
        if isinstance(value, list):
            return [NewtonSimulator._serialize_debug_value(v) for v in value]
        return repr(value)

    def _get_solver_runtime_options(self) -> Dict[str, object]:
        """Read the effective MuJoCo runtime options from the active solver."""
        option_names = (
            "timestep",
            "iterations",
            "ls_iterations",
            "ccd_iterations",
            "sdf_iterations",
            "sdf_initpoints",
            "solver",
            "integrator",
            "cone",
            "jacobian",
            "impratio",
            "tolerance",
            "ls_tolerance",
            "ccd_tolerance",
            "run_collision_detection",
        )
        runtime_options: Dict[str, object] = {}
        for model_name in ("mj_model", "mjw_model"):
            solver_model = getattr(self.solver, model_name, None)
            if solver_model is None or not hasattr(solver_model, "opt"):
                continue
            model_options = {}
            for option_name in option_names:
                option_value = getattr(solver_model.opt, option_name, None)
                if option_value is None:
                    continue
                model_options[option_name] = self._serialize_debug_value(option_value)
            if model_options:
                runtime_options[model_name] = model_options
        return runtime_options

    def _compute_body_ground_clearances(
        self,
        body_pos: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return terrain heights and clearances for each body if available."""
        if self.terrain is None or not hasattr(self.terrain, "get_ground_heights"):
            return None, None

        try:
            ground_heights = self.terrain.get_ground_heights(
                body_pos.reshape(-1, body_pos.shape[-1])
            ).view(self.num_envs, body_pos.shape[1])
        except Exception:
            return None, None

        clearances = body_pos[:, :, 2] - ground_heights
        return ground_heights, clearances

    def _get_rigid_body_contact_forces_snapshot(self) -> torch.Tensor:
        """Collect latest per-body contact forces into a dense tensor."""
        rigid_body_contact_forces = torch.zeros(
            self.num_envs, len(self._body_names), 3, device=self.device
        )
        for body_name, contact_force in self._contact_forces.items():
            if body_name not in self._body_names:
                continue
            body_idx = self._body_names.index(body_name)
            rigid_body_contact_forces[:, body_idx, :] = contact_force
        return rigid_body_contact_forces

    def _build_solver_failure_candidates(
        self,
        root_transforms: torch.Tensor,
        root_velocities: torch.Tensor,
        dof_vel: torch.Tensor,
        body_pos: Optional[torch.Tensor] = None,
        body_clearances: Optional[torch.Tensor] = None,
        contact_forces: Optional[torch.Tensor] = None,
        limit: int = 10,
    ) -> list[Dict[str, object]]:
        """Rank likely-problematic envs for solver failures."""
        if self.num_envs == 0:
            return []

        root_z = root_transforms[:, 2]
        lin_speed = torch.linalg.vector_norm(root_velocities[:, :3], dim=1)
        ang_speed = torch.linalg.vector_norm(root_velocities[:, 3:], dim=1)
        max_dof_vel = (
            dof_vel.abs().max(dim=1).values
            if dof_vel.numel() > 0
            else torch.zeros(self.num_envs, device=root_transforms.device)
        )

        min_clearance = None
        min_body_idx = None
        clearance_term = torch.zeros_like(root_z)
        if body_clearances is not None and body_clearances.numel() > 0:
            min_clearance, min_body_idx = body_clearances.min(dim=1)
            clearance_term = (-min_clearance).clamp_min(0.0)

        max_contact_force = None
        max_contact_idx = None
        if contact_forces is not None and contact_forces.numel() > 0:
            contact_magnitudes = torch.linalg.vector_norm(contact_forces, dim=-1)
            max_contact_force, max_contact_idx = contact_magnitudes.max(dim=1)

        # Favor penetrations first, then fast-moving envs with low body/root height.
        score = (
            10.0 * clearance_term
            + 0.25 * (-root_z).clamp_min(0.0)
            + 0.05 * lin_speed
            + 0.02 * ang_speed
            + 0.01 * max_dof_vel
        )
        top_k = min(limit, self.num_envs)
        candidate_env_ids = torch.topk(score, k=top_k).indices.detach().cpu().tolist()

        candidates: list[Dict[str, object]] = []
        for env_id in candidate_env_ids:
            candidate: Dict[str, object] = {
                "env_id": int(env_id),
                "score": float(score[env_id].item()),
                "steps_since_reset": int(self._steps_since_reset[env_id].item()),
                "root_z": float(root_z[env_id].item()),
                "root_pos": root_transforms[env_id, :3].detach().cpu().tolist(),
                "root_lin_speed": float(lin_speed[env_id].item()),
                "root_ang_speed": float(ang_speed[env_id].item()),
                "max_abs_dof_vel": float(max_dof_vel[env_id].item()),
            }
            if min_clearance is not None and min_body_idx is not None:
                body_idx = int(min_body_idx[env_id].item())
                candidate["min_body_clearance"] = float(min_clearance[env_id].item())
                candidate["min_body_name"] = self._body_names[body_idx]
                if body_pos is not None:
                    candidate["min_body_pos"] = (
                        body_pos[env_id, body_idx].detach().cpu().tolist()
                    )
            if max_contact_force is not None and max_contact_idx is not None:
                contact_body_idx = int(max_contact_idx[env_id].item())
                candidate["max_contact_force"] = float(max_contact_force[env_id].item())
                candidate["max_contact_force_body"] = self._body_names[contact_body_idx]
            if self._last_reset_root_pos is not None:
                candidate["time_since_reset"] = float(
                    self.sim_time - float(self._last_reset_sim_time[env_id].item())
                )
                candidate["last_reset_root_pos"] = (
                    self._last_reset_root_pos[env_id].detach().cpu().tolist()
                )
            candidates.append(candidate)
        return candidates

    def _write_solver_failure_debug_dump(
        self,
        source: str,
        substep_idx: int,
        exc: Exception,
    ) -> Path:
        """Write a fail-fast debug dump for solver-step failures and warnings."""
        if wp.get_device().is_cuda:
            wp.synchronize()

        root_transforms = wp.to_torch(
            self.robot_view.get_root_transforms(self.state_0)
        ).squeeze(1)
        root_velocities = wp.to_torch(
            self.robot_view.get_root_velocities(self.state_0)
        ).squeeze(1)
        body_pos, body_rot, body_vel, body_ang_vel = self._read_bodies_state_tensors()
        body_ground_heights, body_clearances = self._compute_body_ground_clearances(
            body_pos
        )
        dof_pos = (
            wp.to_torch(self.robot_view.get_dof_positions(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )
        dof_vel = (
            wp.to_torch(self.robot_view.get_dof_velocities(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )
        rigid_body_contact_forces = self._get_rigid_body_contact_forces_snapshot()
        solver_failure_candidates = self._build_solver_failure_candidates(
            root_transforms=root_transforms,
            root_velocities=root_velocities,
            dof_vel=dof_vel,
            body_pos=body_pos,
            body_clearances=body_clearances,
            contact_forces=rigid_body_contact_forces,
            limit=10,
        )

        solver_capacity = {
            "max_contact_count": int(self.solver.get_max_contact_count()),
            "use_cuda_graph": bool(self.use_cuda_graph),
        }
        for data_name, attr_names in (
            ("mjw_data", ("nacon", "naconmax", "naccd", "naccdmax")),
            ("mj_data", ("ncon", "nconmax", "nefc", "njmax")),
        ):
            solver_data = getattr(self.solver, data_name, None)
            if solver_data is None:
                continue
            data_snapshot = {}
            for attr_name in attr_names:
                if hasattr(solver_data, attr_name):
                    data_snapshot[attr_name] = self._serialize_debug_value(
                        getattr(solver_data, attr_name)
                    )
            if data_snapshot:
                solver_capacity[data_name] = data_snapshot

        debug_payload = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "source": source,
            "substep_idx": int(substep_idx),
            "sim_time": float(self.sim_time),
            "frame_dt": float(self.frame_dt),
            "sim_dt": float(self.sim_dt),
            "decimation": int(self.decimation),
            "num_envs": int(self.num_envs),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "solver_config": {
                "solver": self.config.sim.solver,
                "integrator": self.config.sim.integrator,
                "iterations": int(self.config.sim.iterations),
                "ls_iterations": int(self.config.sim.ls_iterations),
                "ls_parallel": bool(self.config.sim.ls_parallel),
                "impratio": float(self.config.sim.impratio),
                "njmax": int(self.config.sim.njmax),
                "nconmax": int(self.config.sim.nconmax),
                "nccdmax": None if self.config.sim.nccdmax is None else int(self.config.sim.nccdmax),
                "naccdmax": None if self.config.sim.naccdmax is None else int(self.config.sim.naccdmax),
                "max_epa_workspace_iterations": None if self.config.sim.max_epa_workspace_iterations is None else int(self.config.sim.max_epa_workspace_iterations),
                "cone": self.config.sim.cone,
                "ccd_iterations": int(self.config.sim.ccd_iterations),
                "raise_on_mujoco_warning": bool(self.config.sim.raise_on_mujoco_warning),
            },
            "solver_runtime_options": self._get_solver_runtime_options(),
            "solver_capacity": solver_capacity,
            "actions": {
                "current": self._common_actions.detach().cpu().clone(),
                "previous": self._previous_actions.detach().cpu().clone(),
                "prev_prev": self._prev_prev_actions.detach().cpu().clone(),
            },
            "solver_failure_candidates": solver_failure_candidates,
            "state_before_step": {
                "root_transforms": root_transforms.detach().cpu().clone(),
                "root_velocities": root_velocities.detach().cpu().clone(),
                "rigid_body_pos": body_pos.detach().cpu().clone(),
                "rigid_body_rot": body_rot.detach().cpu().clone(),
                "rigid_body_vel": body_vel.detach().cpu().clone(),
                "rigid_body_ang_vel": body_ang_vel.detach().cpu().clone(),
                "rigid_body_contact_forces": rigid_body_contact_forces.detach().cpu().clone(),
                "body_ground_heights": None if body_ground_heights is None else body_ground_heights.detach().cpu().clone(),
                "body_ground_clearances": None if body_clearances is None else body_clearances.detach().cpu().clone(),
                "body_names": list(self._body_names),
                "dof_pos": dof_pos.detach().cpu().clone(),
                "dof_vel": dof_vel.detach().cpu().clone(),
                "joint_q": wp.to_torch(self.state_0.joint_q).detach().cpu().clone(),
                "joint_qd": wp.to_torch(self.state_0.joint_qd).detach().cpu().clone(),
            },
        }

        dump_dir = Path("output/sim_failure_dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        dump_path = dump_dir / (
            f"newton_solver_failure_{timestamp}_substep{substep_idx}_src_{source}.pt"
        )
        torch.save(debug_payload, dump_path)
        return dump_path

    def _raise_on_solver_failure(
        self,
        source: str,
        substep_idx: int,
        exc: Exception,
    ) -> None:
        """Crash immediately with a solver failure summary and state dump."""
        dump_path: Optional[Path] = None
        dump_error = None
        try:
            dump_path = self._write_solver_failure_debug_dump(
                source=source,
                substep_idx=substep_idx,
                exc=exc,
            )
        except Exception as dump_exc:  # pragma: no cover - best effort diagnostics
            dump_error = str(dump_exc)

        candidate_summaries = []
        try:
            root_transforms = wp.to_torch(
                self.robot_view.get_root_transforms(self.state_0)
            ).squeeze(1)
            root_velocities = wp.to_torch(
                self.robot_view.get_root_velocities(self.state_0)
            ).squeeze(1)
            body_pos, _, _, _ = self._read_bodies_state_tensors()
            _, body_clearances = self._compute_body_ground_clearances(body_pos)
            dof_vel = (
                wp.to_torch(self.robot_view.get_dof_velocities(self.state_0))
                .squeeze(1)
                .view(self.num_envs, -1)
            )
            contact_forces = self._get_rigid_body_contact_forces_snapshot()
            candidate_summaries = self._build_solver_failure_candidates(
                root_transforms=root_transforms,
                root_velocities=root_velocities,
                dof_vel=dof_vel,
                body_pos=body_pos,
                body_clearances=body_clearances,
                contact_forces=contact_forces,
                limit=3,
            )
        except Exception:
            candidate_summaries = []

        summary_lines = [
            "Newton solver step failed.",
            f"source={source}",
            f"sim_time={self.sim_time:.6f}s frame_dt={self.frame_dt:.6f}s sim_dt={self.sim_dt:.6f}s",
            f"substep={substep_idx + 1}/{self.decimation}",
            f"use_cuda_graph={self.use_cuda_graph}",
            (
                f"solver_budget(nconmax={int(self.config.sim.nconmax)}, "
                f"nccdmax={self.config.sim.nccdmax if self.config.sim.nccdmax is not None else 'auto'}, "
                f"njmax={int(self.config.sim.njmax)}, "
                f"ccd_iterations={int(self.config.sim.ccd_iterations)})"
            ),
            f"exception={type(exc).__name__}: {exc}",
        ]
        if candidate_summaries:
            summary_lines.append(f"candidate_envs={candidate_summaries}")
        if dump_path is not None:
            summary_lines.append(f"diagnostic_dump={dump_path}")
        if dump_error is not None:
            summary_lines.append(f"diagnostic_dump_error={dump_error}")

        msg = " | ".join(summary_lines)
        log.error(msg)
        raise RuntimeError(msg) from exc

    def _update_contact_sensors(self) -> None:
        """Update contact sensors after physics step. Must be called outside CUDA graph."""
        if len(self._contact_sensors) > 0:
            self.solver.update_contacts(self.contacts, self.state_0)
            for body_name, sensor in self._contact_sensors.items():
                if hasattr(sensor, "update"):
                    sensor.update(self.state_0, self.contacts)
                else:
                    sensor.eval(self.contacts)
                # Store the net contact force for this body (across all environments)
                # Newer Newton builds expose total_force, older ones expose
                # net_force with a singleton body axis.
                force_wp = None
                if hasattr(sensor, "total_force") and sensor.total_force is not None:
                    force_wp = sensor.total_force
                elif hasattr(sensor, "net_force") and sensor.net_force is not None:
                    force_wp = sensor.net_force

                if force_wp is not None:
                    net_force = wp.to_torch(force_wp).clone()
                    # Squeeze the body dimension if present (shape [N, 1, 3] -> [N, 3])
                    if net_force.dim() == 3 and net_force.shape[1] == 1:
                        net_force = net_force.squeeze(1)
                    self._contact_forces[body_name] = net_force

    def _sync_state_reads_if_needed(self) -> None:
        """Synchronize Warp work once before reading simulator state tensors."""
        if not self._needs_state_sync:
            return

        if wp.get_device().is_cuda:
            wp.synchronize()
        self._needs_state_sync = False

    def _reset_solver_worlds_from_state(self, env_ids: torch.Tensor) -> None:
        """Clear per-world MuJoCo runtime state after direct Newton state teleports."""
        if env_ids is None or env_ids.numel() == 0:
            return
        reset_worlds = getattr(self.solver, "reset_worlds", None)
        if reset_worlds is None:
            return
        reset_worlds(self.state_0, env_ids)

    def _get_nonfinite_env_ids(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Return env IDs where any provided tensor contains non-finite values."""
        bad_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for tensor in tensors:
            finite_mask = torch.isfinite(tensor)
            if tensor.dim() >= 2:
                finite_mask = finite_mask.all(dim=tuple(range(1, tensor.dim())))
            bad_envs |= ~finite_mask
        return torch.nonzero(bad_envs, as_tuple=False).flatten()

    def _summarize_nonfinite_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        first_bad_env: int,
    ) -> str:
        """Return a compact summary of non-finite values for a state tensor."""
        if tensor.dim() == 0:
            is_finite = bool(torch.isfinite(tensor).item())
            return f"{name}: finite={is_finite}"

        if tensor.shape[0] != self.num_envs:
            return f"{name}: unexpected leading shape {tuple(tensor.shape)}"

        finite_mask = torch.isfinite(tensor)
        if tensor.dim() >= 2:
            env_is_finite = finite_mask.all(dim=tuple(range(1, tensor.dim())))
            env_slice = tensor[first_bad_env]
            env_finite = torch.isfinite(env_slice)
        else:
            env_is_finite = finite_mask
            env_slice = tensor[first_bad_env : first_bad_env + 1]
            env_finite = torch.isfinite(env_slice)

        bad_env_count = int((~env_is_finite).sum().item())
        env_bad_count = int((~env_finite).sum().item())
        env_total_count = int(env_finite.numel())

        if env_finite.any():
            max_abs = float(env_slice[env_finite].abs().max().item())
            max_abs_str = f"{max_abs:.3e}"
        else:
            max_abs_str = "nan"

        return (
            f"{name}: bad_envs={bad_env_count}/{self.num_envs}, "
            f"env[{first_bad_env}] nonfinite={env_bad_count}/{env_total_count}, "
            f"env[{first_bad_env}] max|finite|={max_abs_str}"
        )

    def _assert_finite_reset_inputs(
        self,
        new_states: ResetState,
        env_ids: torch.Tensor,
    ) -> None:
        """Fail fast when reset inputs contain non-finite values."""
        fields = {
            "reset.root_pos": new_states.root_pos,
            "reset.root_rot": new_states.root_rot,
            "reset.root_vel": new_states.root_vel,
            "reset.root_ang_vel": new_states.root_ang_vel,
            "reset.dof_pos": new_states.dof_pos,
            "reset.dof_vel": new_states.dof_vel,
        }

        for name, tensor in fields.items():
            if tensor is None:
                continue

            finite_mask = torch.isfinite(tensor)
            if tensor.dim() >= 2:
                row_finite = finite_mask.all(dim=tuple(range(1, tensor.dim())))
            else:
                row_finite = finite_mask

            if row_finite.all():
                continue

            bad_local_ids = torch.nonzero(~row_finite, as_tuple=False).flatten()
            bad_env_ids = env_ids[bad_local_ids]
            bad_element_count = int((~finite_mask[bad_local_ids]).sum().item())
            msg = (
                f"Non-finite reset input detected in {name}. "
                f"bad_envs={bad_env_ids.numel()}/{env_ids.numel()} "
                f"(first 10 env ids: {bad_env_ids[:10].detach().cpu().tolist()}) "
                f"bad_elements={bad_element_count}"
            )
            log.error(msg)
            raise AssertionError(msg)

    def _record_last_reset_state(
        self,
        new_states: ResetState,
        env_ids: torch.Tensor,
    ) -> None:
        """Store last applied reset state for post-mortem diagnostics."""
        if self._last_reset_root_pos is None:
            self._last_reset_root_pos = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=new_states.root_pos.dtype
            )
            self._last_reset_root_rot = torch.zeros(
                self.num_envs, 4, device=self.device, dtype=new_states.root_rot.dtype
            )
            self._last_reset_root_vel = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=new_states.root_vel.dtype
            )
            self._last_reset_root_ang_vel = torch.zeros(
                self.num_envs,
                3,
                device=self.device,
                dtype=new_states.root_ang_vel.dtype,
            )
            self._last_reset_dof_pos = torch.zeros(
                self.num_envs,
                new_states.dof_pos.shape[-1],
                device=self.device,
                dtype=new_states.dof_pos.dtype,
            )
            self._last_reset_dof_vel = torch.zeros(
                self.num_envs,
                new_states.dof_vel.shape[-1],
                device=self.device,
                dtype=new_states.dof_vel.dtype,
            )
            self._last_reset_sim_time = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            )

        self._last_reset_root_pos[env_ids] = new_states.root_pos.detach()
        self._last_reset_root_rot[env_ids] = new_states.root_rot.detach()
        self._last_reset_root_vel[env_ids] = new_states.root_vel.detach()
        self._last_reset_root_ang_vel[env_ids] = new_states.root_ang_vel.detach()
        self._last_reset_dof_pos[env_ids] = new_states.dof_pos.detach()
        self._last_reset_dof_vel[env_ids] = new_states.dof_vel.detach()
        self._last_reset_sim_time[env_ids] = float(self.sim_time)

    def _write_nonfinite_debug_dump(
        self,
        source: str,
        bad_env_ids: torch.Tensor,
        first_bad_env: int,
        tensors: Dict[str, torch.Tensor],
    ) -> Optional[Path]:
        """Write a debug dump for non-finite Newton states and return its path."""
        if wp.get_device().is_cuda:
            wp.synchronize()

        root_transforms = wp.to_torch(
            self.robot_view.get_root_transforms(self.state_0)
        ).squeeze(1)
        root_velocities = wp.to_torch(
            self.robot_view.get_root_velocities(self.state_0)
        ).squeeze(1)
        dof_pos = (
            wp.to_torch(self.robot_view.get_dof_positions(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )
        dof_vel = (
            wp.to_torch(self.robot_view.get_dof_velocities(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )

        joint_q = wp.to_torch(self.state_0.joint_q)
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        q_stride = joint_q.numel() // self.num_envs if self.num_envs > 0 else joint_q.numel()
        qd_stride = (
            joint_qd.numel() // self.num_envs if self.num_envs > 0 else joint_qd.numel()
        )
        q_start = first_bad_env * q_stride
        q_end = min((first_bad_env + 1) * q_stride, joint_q.numel())
        qd_start = first_bad_env * qd_stride
        qd_end = min((first_bad_env + 1) * qd_stride, joint_qd.numel())

        contact_snapshot = {}
        for body_name, contact_force in self._contact_forces.items():
            if contact_force.shape[0] > first_bad_env:
                contact_snapshot[body_name] = contact_force[first_bad_env].detach().cpu().clone()

        debug_payload = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "source": source,
            "sim_time": float(self.sim_time),
            "frame_dt": float(self.frame_dt),
            "sim_dt": float(self.sim_dt),
            "decimation": int(self.decimation),
            "num_envs": int(self.num_envs),
            "bad_env_ids": bad_env_ids.detach().cpu().clone(),
            "first_bad_env": int(first_bad_env),
            "steps_since_reset_first_bad_env": int(
                self._steps_since_reset[first_bad_env].item()
            ),
            "actions_first_bad_env": {
                "current": self._common_actions[first_bad_env].detach().cpu().clone(),
                "previous": self._previous_actions[first_bad_env].detach().cpu().clone(),
                "prev_prev": self._prev_prev_actions[first_bad_env].detach().cpu().clone(),
            },
            "state_first_bad_env": {
                "root_transforms": root_transforms[first_bad_env].detach().cpu().clone(),
                "root_velocities": root_velocities[first_bad_env].detach().cpu().clone(),
                "dof_pos": dof_pos[first_bad_env].detach().cpu().clone(),
                "dof_vel": dof_vel[first_bad_env].detach().cpu().clone(),
                "joint_q_flat": joint_q[q_start:q_end].detach().cpu().clone(),
                "joint_qd_flat": joint_qd[qd_start:qd_end].detach().cpu().clone(),
            },
            "nonfinite_masks_first_bad_env": {
                name: ~torch.isfinite(tensor[first_bad_env]).detach().cpu()
                for name, tensor in tensors.items()
            },
            "tensors_first_bad_env": {
                name: tensor[first_bad_env].detach().cpu().clone()
                for name, tensor in tensors.items()
            },
            "contact_forces_first_bad_env": contact_snapshot,
        }

        if self._last_reset_root_pos is not None:
            last_reset_root_pos = self._last_reset_root_pos[first_bad_env].detach().clone()
            reset_ground_height = None
            reset_root_clearance = None
            if self.terrain is not None and hasattr(self.terrain, "get_ground_heights"):
                try:
                    reset_ground_height = float(
                        self.terrain.get_ground_heights(
                            last_reset_root_pos.unsqueeze(0)
                        )
                        .reshape(-1)[0]
                        .item()
                    )
                    reset_root_clearance = float(
                        last_reset_root_pos[2].item() - reset_ground_height
                    )
                except Exception:
                    reset_ground_height = None
                    reset_root_clearance = None

            debug_payload["last_reset_first_bad_env"] = {
                "sim_time": float(self._last_reset_sim_time[first_bad_env].item()),
                "root_pos": last_reset_root_pos.cpu().clone(),
                "root_rot": self._last_reset_root_rot[first_bad_env].detach()
                .cpu()
                .clone(),
                "root_vel": self._last_reset_root_vel[first_bad_env].detach()
                .cpu()
                .clone(),
                "root_ang_vel": self._last_reset_root_ang_vel[first_bad_env]
                .detach()
                .cpu()
                .clone(),
                "dof_pos": self._last_reset_dof_pos[first_bad_env].detach()
                .cpu()
                .clone(),
                "dof_vel": self._last_reset_dof_vel[first_bad_env].detach()
                .cpu()
                .clone(),
                "ground_height": reset_ground_height,
                "root_clearance": reset_root_clearance,
            }

        dump_dir = Path("output/nonfinite_dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        dump_path = dump_dir / (
            f"newton_nonfinite_{timestamp}_env{first_bad_env}_src_{source}.pt"
        )
        torch.save(debug_payload, dump_path)
        return dump_path

    def _raise_on_nonfinite_envs(
        self,
        source: str,
        bad_env_ids: torch.Tensor,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Fail hard with rich diagnostics when non-finite state is detected."""
        if bad_env_ids.numel() == 0:
            return

        first_bad_env = int(bad_env_ids[0].item())

        dump_path: Optional[Path] = None
        dump_error = None
        try:
            dump_path = self._write_nonfinite_debug_dump(
                source=source,
                bad_env_ids=bad_env_ids,
                first_bad_env=first_bad_env,
                tensors=tensors,
            )
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            dump_error = str(exc)

        summary_lines = [
            "Non-finite Newton simulator state detected.",
            f"source={source}",
            f"sim_time={self.sim_time:.6f}s frame_dt={self.frame_dt:.6f}s",
            (
                f"bad_envs={bad_env_ids.numel()}/{self.num_envs}, "
                f"first_bad_env={first_bad_env}, "
                f"bad_env_ids(first10)={bad_env_ids[:10].tolist()}"
            ),
            f"steps_since_reset[first_bad_env]={int(self._steps_since_reset[first_bad_env].item())}",
        ]
        for name, tensor in tensors.items():
            summary_lines.append(
                self._summarize_nonfinite_tensor(
                    name=name,
                    tensor=tensor,
                    first_bad_env=first_bad_env,
                )
            )
        if dump_path is not None:
            summary_lines.append(f"diagnostic_dump={dump_path}")
        if dump_error is not None:
            summary_lines.append(f"diagnostic_dump_error={dump_error}")

        msg = " | ".join(summary_lines)
        log.error(msg)
        raise AssertionError(msg)

    def _read_bodies_state_tensors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read raw body state tensors from Newton without finite checks."""
        body_transforms = (
            wp.to_torch(self.robot_view.get_link_transforms(self.state_0))
            .squeeze(1)
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        body_pos = body_transforms[:, :, :3]
        body_rot = body_transforms[:, :, 3:]

        body_vel_transforms = (
            wp.to_torch(self.robot_view.get_link_velocities(self.state_0))
            .squeeze(1)
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        body_vel = body_vel_transforms[:, :, :3]
        body_ang_vel = body_vel_transforms[:, :, 3:]
        return body_pos, body_rot, body_vel, body_ang_vel

    def _physics_step(self) -> None:
        """Performs a physics simulation step."""
        # Update control targets before simulation
        if self.control_type == ControlType.BUILT_IN_PD:
            self._apply_control()
        elif self.control_type == ControlType.PROPORTIONAL:
            pd_tar = self._action_to_pd_targets(self._common_actions)
            if (
                self._domain_randomization is not None
                and "action_noise" in self._domain_randomization
            ):
                pd_tar[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]
            sim_targets = pd_tar[:, self.data_conversion.dof_convert_to_sim]
            self._update_pd_targets(sim_targets)
        elif self.control_type == ControlType.TORQUE:
            torques = self._action_to_torque_targets(self._common_actions)
            if (
                self._domain_randomization is not None
                and "action_noise" in self._domain_randomization
            ):
                torques[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]
            torques = torch.clip(
                torques, -self._torque_limits_common, self._torque_limits_common
            )
            sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
            self._update_torques(sim_torques)

        # Run simulation
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self._update_contact_sensors()
        self.sim_time += self.frame_dt
        self._needs_state_sync = True

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: ObjectState = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """Sets the state of specified environments using vectorized operations."""
        # assert new_object_states is None, "Newton does not yet support setting object states."

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_mask[env_ids] = True
        self._assert_finite_reset_inputs(new_states, env_ids)

        # Newton expects the state setter to be provided with the states for all envs.
        # The mask is used to determine which envs to apply the update to.
        root_state = wp.to_torch(self.robot_view.get_root_transforms(self.state_0)).squeeze(
            1
        )
        root_vel_state = wp.to_torch(
            self.robot_view.get_root_velocities(self.state_0)
        ).squeeze(1)
        dof_pos = (
            wp.to_torch(self.robot_view.get_dof_positions(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )
        dof_vel = (
            wp.to_torch(self.robot_view.get_dof_velocities(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )

        root_state = root_state.clone()
        root_vel_state = root_vel_state.clone()
        dof_pos = dof_pos.clone()
        dof_vel = dof_vel.clone()

        root_state[env_ids, :3] = new_states.root_pos
        root_state[env_ids, 3:] = new_states.root_rot
        root_vel_state[env_ids, :3] = new_states.root_vel
        root_vel_state[env_ids, 3:] = new_states.root_ang_vel
        dof_pos[env_ids] = new_states.dof_pos
        dof_vel[env_ids] = new_states.dof_vel

        root_state_3d = root_state.unsqueeze(1)
        root_vel_state_3d = root_vel_state.unsqueeze(1)
        dof_pos_3d = dof_pos.unsqueeze(1)
        dof_vel_3d = dof_vel.unsqueeze(1)

        # Set state_0 using ArticulationView
        self.robot_view.set_root_transforms(self.state_0, root_state_3d, mask=env_mask)
        self.robot_view.set_root_velocities(
            self.state_0, root_vel_state_3d, mask=env_mask
        )
        self.robot_view.set_dof_velocities(self.state_0, dof_vel_3d, mask=env_mask)

        self.robot_view.set_dof_positions(self.state_0, dof_pos_3d, mask=env_mask)

        # Also update state_1 to match state_0
        self.robot_view.set_root_transforms(self.state_1, root_state_3d, mask=env_mask)
        self.robot_view.set_root_velocities(
            self.state_1, root_vel_state_3d, mask=env_mask
        )
        self.robot_view.set_dof_velocities(self.state_1, dof_vel_3d, mask=env_mask)
        self.robot_view.set_dof_positions(self.state_1, dof_pos_3d, mask=env_mask)

        # Clear forces after reset
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        # Recompute forward kinematics to refresh derived body states
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        newton.eval_fk(
            self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1
        )
        self._record_last_reset_state(new_states, env_ids)
        self._reset_solver_worlds_from_state(env_ids)
        self._needs_state_sync = True

    # ===== Group 4: State Getters =====
    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns contact forces for robot bodies."""
        self._sync_state_reads_if_needed()

        # Initialize with zeros for all bodies
        rigid_body_contact_forces = torch.zeros(
            self.num_envs, len(self._body_names), 3, device=self.device
        )

        # Populate contact forces from sensors
        if len(self._contact_sensors) > 0:
            for body_name, contact_force in self._contact_forces.items():
                # Find the index of this body in the body_names list
                if body_name in self._body_names:
                    body_idx = self._body_names.index(body_name)
                    rigid_body_contact_forces[:, body_idx, :] = contact_force

        if env_ids is not None:
            rigid_body_contact_forces = rigid_body_contact_forces[env_ids]

        return RobotState(
            rigid_body_contact_forces=rigid_body_contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_contact_binary(
        self, env_ids: Optional[torch.Tensor] = None, force_threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Returns binary contact labels for robot bodies.

        A body is considered in contact if its contact force magnitude exceeds the threshold.

        Args:
            env_ids: Optional tensor of environment IDs to query
            force_threshold: Minimum contact force magnitude to consider as contact (default: 1.0 N)

        Returns:
            Binary tensor of shape [num_envs, num_bodies] where 1 indicates contact
        """
        # Get contact forces
        contact_state = self._get_simulator_bodies_contact_buf(env_ids=env_ids)
        contact_forces = (
            contact_state.rigid_body_contact_forces
        )  # [num_envs, num_bodies, 3]

        # Compute force magnitudes
        force_magnitudes = torch.norm(contact_forces, dim=-1)  # [num_envs, num_bodies]

        # Apply threshold to get binary labels
        contact_binary = (force_magnitudes > force_threshold).float()

        return contact_binary

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns the state of robot bodies."""
        self._sync_state_reads_if_needed()

        body_pos, body_rot, body_vel, body_ang_vel = self._read_bodies_state_tensors()

        bad_env_ids = self._get_nonfinite_env_ids(
            body_pos, body_rot, body_vel, body_ang_vel
        )
        if bad_env_ids.numel() > 0:
            self._raise_on_nonfinite_envs(
                source="bodies_state",
                bad_env_ids=bad_env_ids,
                tensors={
                    "rigid_body_pos": body_pos,
                    "rigid_body_rot": body_rot,
                    "rigid_body_vel": body_vel,
                    "rigid_body_ang_vel": body_ang_vel,
                },
            )

        if env_ids is not None:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]

        return RobotState(
            rigid_body_pos=body_pos,
            rigid_body_rot=body_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """Returns the root state of the robot."""
        self._sync_state_reads_if_needed()

        root_transforms = wp.to_torch(
            self.robot_view.get_root_transforms(self.state_0)
        ).squeeze(1)
        root_velocities = wp.to_torch(
            self.robot_view.get_root_velocities(self.state_0)
        ).squeeze(1)

        bad_env_ids = self._get_nonfinite_env_ids(root_transforms, root_velocities)
        if bad_env_ids.numel() > 0:
            self._raise_on_nonfinite_envs(
                source="root_state",
                bad_env_ids=bad_env_ids,
                tensors={
                    "root_transforms": root_transforms,
                    "root_velocities": root_velocities,
                },
            )

        if env_ids is not None:
            root_transforms = root_transforms[env_ids]
            root_velocities = root_velocities[env_ids]

        return RootOnlyState(
            root_pos=root_transforms[:, :3],
            root_rot=root_transforms[:, 3:],
            root_vel=root_velocities[:, :3],
            root_ang_vel=root_velocities[:, 3:],
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Returns empty object state (objects not supported in Newton)."""
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Returns contact forces for simulation objects."""
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_dof_forces(self, env_ids=None):
        """Returns the DOF forces."""
        self._sync_state_reads_if_needed()

        dof_forces = wp.to_torch(self.robot_view.get_dof_forces(self.control)).squeeze(
            1
        )
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return RobotState(
            dof_forces=dof_forces, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns the state of robot DOFs."""
        self._sync_state_reads_if_needed()

        dof_pos = (
            wp.to_torch(self.robot_view.get_dof_positions(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )

        dof_vel = (
            wp.to_torch(self.robot_view.get_dof_velocities(self.state_0))
            .squeeze(1)
            .view(self.num_envs, -1)
        )

        bad_env_ids = self._get_nonfinite_env_ids(dof_pos, dof_vel)
        if bad_env_ids.numel() > 0:
            self._raise_on_nonfinite_envs(
                source="dof_state",
                bad_env_ids=bad_env_ids,
                tensors={
                    "dof_pos": dof_pos,
                    "dof_vel": dof_vel,
                },
            )

        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]

        return RobotState(
            dof_pos=dof_pos, dof_vel=dof_vel, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from Newton's internal API for verification purposes only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in Newton's DOF ordering.
        """
        dof_limits_lower = wp.to_torch(
            self.robot_view.get_attribute("joint_limit_lower", self.model)
        )[0, 0]
        dof_limits_upper = wp.to_torch(
            self.robot_view.get_attribute("joint_limit_upper", self.model)
        )[0, 0]
        return dof_limits_lower, dof_limits_upper

    # ===== Group 5: Control & Computation Methods =====
    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Applies PD position targets using Newton's internal PD controller."""
        a_wp = wp.from_torch(
            pd_targets.unsqueeze(1), dtype=wp.float32, requires_grad=False
        )
        self.robot_view.set_attribute("joint_target_pos", self.control, a_wp)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        a_wp = wp.from_torch(
            torques.unsqueeze(1), dtype=wp.float32, requires_grad=False
        )
        self.robot_view.set_dof_forces(self.control, a_wp)

    def _apply_pd_kernel(self, state: newton.State) -> None:
        """Apply explicit PD control using Warp kernel."""
        wp.launch(
            kernel=compute_pd_torques_kernel,
            dim=self.num_envs * self._pd_num_dofs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.control.joint_f,
                self._pd_targets_wp,
                self._pd_kp_wp,
                self._pd_kd_wp,
                self._pd_torque_limits_wp,
                self._pd_q_stride,
                self._pd_qd_stride,
                self._pd_q_dof_start,
                self._pd_qd_dof_start,
                self._pd_num_dofs,
            ],
        )

    def _update_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Update PD targets in the persistent Warp array."""
        wp.copy(
            self._pd_targets_wp, wp.from_torch(pd_targets.view(-1), dtype=wp.float32)
        )

    def _apply_torques_kernel_method(self) -> None:
        """Apply direct torques using Warp kernel."""
        wp.launch(
            kernel=apply_torques_kernel,
            dim=self.num_envs * self._pd_num_dofs,
            inputs=[
                self.control.joint_f,
                self._pd_targets_wp,
                self._pd_qd_stride,
                self._pd_qd_dof_start,
                self._pd_num_dofs,
            ],
        )

    def _update_torques(self, torques: torch.Tensor) -> None:
        """Update torques in the persistent Warp array."""
        wp.copy(self._pd_targets_wp, wp.from_torch(torques.view(-1), dtype=wp.float32))

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to robot root by adding to current velocities."""
        self._sync_state_reads_if_needed()

        current_vel_3d = wp.to_torch(self.robot_view.get_root_velocities(self.state_0))
        current_vel = current_vel_3d.squeeze(1)
        new_vel = current_vel.clone()
        new_vel[env_ids, :3] += linear_velocity
        new_vel[env_ids, 3:6] += angular_velocity

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True
        new_vel_3d = new_vel.unsqueeze(1)
        self.robot_view.set_root_velocities(self.state_0, new_vel_3d, mask=env_mask)
        self.robot_view.set_root_velocities(self.state_1, new_vel_3d, mask=env_mask)

    # ===== Group 5b: Projectile Methods =====
    def _get_projectile_positions_rotations(self) -> tuple:
        """Return projectile (positions, rotations_xyzw) from Newton joint_q.

        Newton uses xyzw quaternions natively — no conversion needed.
        """
        self._sync_state_reads_if_needed()

        joint_q = wp.to_torch(self.state_0.joint_q)
        n_proj = self._proj_config.num_projectiles
        q_stride = self._proj_q_stride
        jq_off = self._proj_jq_offset

        pos = torch.zeros(self.num_envs, n_proj, 3, device=self.device)
        rot = torch.zeros(self.num_envs, n_proj, 4, device=self.device)
        for eid in range(self.num_envs):
            for pid in range(n_proj):
                qp = eid * q_stride + jq_off + pid * 7
                pos[eid, pid] = joint_q[qp : qp + 3]
                rot[eid, pid] = joint_q[qp + 3 : qp + 7]
        return pos, rot

    def _create_projectiles(self, config: ProjectileConfig) -> None:
        """Projectile bodies are already added to the builder during _create_envs."""
        # Already created via add_body/add_shape_box in _create_envs
        pass

    def _set_projectile_root_states(
        self,
        proj_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations_xyzw: torch.Tensor,
        velocities: torch.Tensor,
        ang_velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set projectile state by writing directly to joint_q/joint_qd arrays.

        Newton uses xyzw quaternions natively — no conversion needed.
        """
        joint_q = wp.to_torch(self.state_0.joint_q)
        joint_qd = wp.to_torch(self.state_0.joint_qd)

        q_stride = self._proj_q_stride
        qd_stride = self._proj_qd_stride
        jq_off = self._proj_jq_offset
        jqd_off = self._proj_jqd_offset

        for i in range(len(env_ids)):
            eid = env_ids[i].item()
            pid = proj_indices[i].item()

            qp = eid * q_stride + jq_off + pid * 7
            joint_q[qp : qp + 3] = positions[i]
            joint_q[qp + 3 : qp + 7] = rotations_xyzw[i]

            qv = eid * qd_stride + jqd_off + pid * 6
            joint_qd[qv : qv + 3] = velocities[i]
            joint_qd[qv + 3 : qv + 6] = ang_velocities[i]

        # Also update state_1 to match
        wp.copy(self.state_1.joint_q, self.state_0.joint_q)
        wp.copy(self.state_1.joint_qd, self.state_0.joint_qd)

        # Recompute forward kinematics to refresh body transforms
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        newton.eval_fk(
            self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1
        )
        self._reset_solver_worlds_from_state(env_ids)
        self._needs_state_sync = True

    # ===== Group 6: Rendering & Visualization =====
    def _init_camera(self) -> None:
        """Initializes camera."""
        char_root_pos = (
            self._get_simulator_root_state([self._camera_target["env"]])
            .root_pos.flatten()
            .cpu()
            .numpy()
        )

        cam_pos = char_root_pos + np.array([0, -5.0, 1])

        camera_target = char_root_pos + np.array([0, 0, 0.2])
        vector_to_target = camera_target - cam_pos
        normalized_vector_to_target = vector_to_target / np.linalg.norm(
            vector_to_target
        )
        pitch = np.rad2deg(np.arcsin(normalized_vector_to_target[2]))
        yaw = np.rad2deg(
            np.arctan2(normalized_vector_to_target[1], normalized_vector_to_target[0])
        )

        self.viewer.set_camera(wp.vec3(cam_pos.tolist()), pitch, yaw)
        self._cam_prev_char_pos = char_root_pos

    def _init_keyboard(self) -> None:
        """Initializes keyboard controls."""
        pass

    def _update_camera(self) -> None:
        """Updates camera position."""
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state([self._camera_target["env"]])
                .root_pos.flatten()
                .cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .flatten()
                .cpu()
                .numpy()
            )
            height_offset = 0

        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "pos"):
            cam_pos = np.array(self.viewer.camera.pos)
        elif hasattr(self.viewer, "_camera_request") and self.viewer._camera_request:
            cam_pos = np.array(self.viewer._camera_request[0], dtype=np.float64)
        else:
            return
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = char_root_pos + np.array([0, 0, height_offset])
        new_cam_pos = char_root_pos + cam_delta

        vector_to_target = new_cam_target - new_cam_pos
        normalized_vector_to_target = vector_to_target / np.linalg.norm(
            vector_to_target
        )
        pitch = np.rad2deg(np.arcsin(normalized_vector_to_target[2]))
        yaw = np.rad2deg(
            np.arctan2(normalized_vector_to_target[1], normalized_vector_to_target[0])
        )

        self.viewer.set_camera(wp.vec3(new_cam_pos.tolist()), pitch, yaw)
        self._cam_prev_char_pos = char_root_pos

    def close(self) -> None:
        """Closes the simulator and cleans up resources."""
        pass

    def _write_viewport_to_file(self, file_name: str) -> None:
        """Writes viewport to file."""
        pass

    def render(self) -> None:
        """Renders the current simulation state."""
        if not self.headless:
            if not self._camera_initialized:
                self._init_camera()
                self._camera_initialized = True
            else:
                self._update_camera()

            any_key_pressed = False
            if self.viewer.is_key_down("q"):
                sys.exit()
            elif self.viewer.is_key_down("j"):
                if not self._any_key_pressed:
                    self._throw_projectile()
                any_key_pressed = True
            elif self.viewer.is_key_down("l"):
                if not self._any_key_pressed:
                    self._toggle_video_record()
                any_key_pressed = True
            elif self.viewer.is_key_down(";"):
                if not self._any_key_pressed:
                    self._cancel_video_record()
                any_key_pressed = True
            elif self.viewer.is_key_down("o"):
                if not self._any_key_pressed:
                    self._toggle_camera_target()
                any_key_pressed = True
            elif self.viewer.is_key_down("m"):
                if not self._any_key_pressed:
                    self._toggle_markers()
                any_key_pressed = True
            elif self.viewer.is_key_down("r"):
                if not self._any_key_pressed:
                    self._requested_reset()
                any_key_pressed = True
            for key, handler in self._custom_key_handlers.items():
                if self.viewer.is_key_down(key):
                    if not self._any_key_pressed:
                        handler()
                    any_key_pressed = True

            self._any_key_pressed = any_key_pressed

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        super().render()

    def _write_viewport_to_file(self, file_name: str) -> None:
        import matplotlib.pyplot as plt

        viewport = self.viewer.get_frame().numpy()  # [H, W, 3] as uint8
        plt.imsave(file_name, viewport)

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Updates visualization markers."""
        pass
