# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
import socket
import torch
import numpy as np
from rich.progress import Progress
import sys
from typing import Dict, Optional, Tuple

from protomotions.components.scene_lib import SceneLib, MeshSceneObject
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
)
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.components.terrains.config import CombineMode
from protomotions.simulator.newton.config import NewtonSimulatorConfig
import openmesh
import warp as wp
import newton
from newton.selection import ArticulationView
from newton import Contacts
from newton.sensors import SensorContact
from newton.solvers import SolverNotifyFlags
import copy

from protomotions.simulator.newton.opengl_compat import (
    prepare_gl_viewer_compat,
)
from protomotions.simulator.newton.mujoco_compat import (
    install_mujoco_joint_arg_coercion,
    normalize_joint_dof_mujoco_attributes,
)

wp.config.enable_backward = False
wp.config.quiet = True


def _get_primary_local_ip() -> Optional[str]:
    """Best-effort resolution of the primary local IPv4 address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _shape_label_targets_body(shape_label: str, body_name: str) -> bool:
    """Return whether an imported Newton shape label belongs to the target body."""
    return (
        f"/{body_name}/" in shape_label
        or f"/{body_name}_geom_" in shape_label
        or shape_label.startswith(f"{body_name}_geom_")
    )


def _configure_pressure_field_foot_shapes(
    builder: newton.ModelBuilder,
    contact_bodies: list[str],
    sim_params,
) -> list[int]:
    """Mark the configured contact bodies as compliant pressure-field shapes."""
    if not getattr(sim_params, "pressure_field_feet", False):
        return []

    foot_shape_ids = [
        shape_id
        for shape_id, shape_label in enumerate(builder.shape_label)
        if any(_shape_label_targets_body(shape_label, body_name) for body_name in contact_bodies)
    ]
    if not foot_shape_ids:
        raise ValueError(f"No foot shapes matched contact bodies {contact_bodies!r}")

    hydro_flags = int(newton.ShapeFlags.HYDROELASTIC | newton.ShapeFlags.HYDROELASTIC_COMPLIANT)
    rigid_flag = int(newton.ShapeFlags.HYDROELASTIC_RIGID)
    workflow = int(newton.HydroelasticContactWorkflow.PRESSURE)

    for shape_id in foot_shape_ids:
        builder.shape_flags[shape_id] = (int(builder.shape_flags[shape_id]) | hydro_flags) & ~rigid_flag
        builder.shape_sdf_max_resolution[shape_id] = sim_params.pressure_field_foot_sdf_max_resolution
        builder.shape_sdf_target_voxel_size[shape_id] = None
        builder.shape_material_kh[shape_id] = sim_params.pressure_field_foot_kh
        builder.shape_hydroelastic_contact_workflow[shape_id] = workflow
        builder.shape_hydroelastic_contact_workflow_user_set[shape_id] = True

    return foot_shape_ids


def _quat_xyzw_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion into a 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in quat]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _transform_points_np(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply an xyzw rigid transform to row-major point coordinates."""
    rotation = _quat_xyzw_to_matrix_np(quat_xyzw)
    return points @ rotation.T + position[None, :]


def _make_aabb_corners_np(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Enumerate the eight corners of an axis-aligned bounding box."""
    return np.array(
        [
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], lower[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )


def _planar_axes_from_quat_xyzw(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame foot axes projected onto the flat-ground plane."""
    rotation = _quat_xyzw_to_matrix_np(quat_xyzw)
    x_axis = rotation[:, 0].astype(np.float32)
    x_axis[2] = 0.0
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1.0e-6:
        x_axis = rotation[:, 1].astype(np.float32)
        x_axis[2] = 0.0
        x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1.0e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis /= x_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1.0e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis /= y_norm
    return x_axis, y_axis


def _normalize_to_unit_interval(value: float, lower: float, upper: float) -> float:
    """Map a coordinate from [lower, upper] to [-1, 1] with graceful fallback."""
    span = upper - lower
    if abs(span) < 1.0e-6:
        return 0.0
    normalized = 2.0 * ((value - lower) / span) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


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


def convert_to_indexed_mesh(vertices, triangles):
    # Flatten the triangles tensor to get a single list of vertex indices
    vertex_indices = triangles.flatten()

    # Get the unique vertices and their corresponding indices
    unique_points, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)

    # Map the original vertex indices to the new unique point indices
    new_indices = inverse_indices[vertex_indices]

    # Reshape the new indices to form the indexed triangles
    indexed_triangles = new_indices.reshape(-1, 3)

    return unique_points, indexed_triangles


def create_plane_mesh(width: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple centered XY plane mesh for viewer-only flat ground."""
    half_width = 0.5 * width
    half_length = 0.5 * length
    vertices = np.array(
        [
            [-half_width, -half_length, 0.0],
            [half_width, -half_length, 0.0],
            [half_width, half_length, 0.0],
            [-half_width, half_length, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, indices


class NewtonSimulator(Simulator):
    """Newton physics engine wrapper for our simulation framework."""

    config: NewtonSimulatorConfig

    def __init__(
        self,
        config: NewtonSimulatorConfig,
        robot_config,
        terrain: Terrain,
        device: torch.device,
        scene_lib: SceneLib,
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
        self.contacts = Contacts(0, 0)  # Replaced with model.contacts() after sensor setup.
        self._camera_initialized = False
        self._viewer_camera_pos = None
        self._contact_analysis = None

    def _create_simulation(self) -> None:
        """Create the Newton simulation environment."""
        self._create_envs()
        self._setup_robot()
        self._setup_contact_analysis_metadata()
        self._setup_sim()
        if self.robot_config.contact_bodies is not None:
            self._setup_contact_sensors()
        self.contacts = self.model.contacts()
        self._set_robot_friction_to_minimum()
        self._apply_domain_randomization_if_needed()

        self.graph = None
        self.use_cuda_graph = False
        disable_cuda_graph = os.environ.get("PROTO_DISABLE_NEWTON_CUDA_GRAPH") == "1"

        if disable_cuda_graph:
            print("[INFO] Newton CUDA graph disabled by PROTO_DISABLE_NEWTON_CUDA_GRAPH=1")
        elif wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print(f"[INFO] Using CUDA graph ({self.control_type.name})")
            self.use_cuda_graph = True

            if self.control_type == ControlType.BUILT_IN_PD:
                zero_targets = wp.zeros_like(
                    self.robot_view.get_attribute("joint_target_pos", self.control)
                )
                self.robot_view.set_attribute("joint_target_pos", self.control, zero_targets)
            else:
                zeros = torch.zeros(
                    self.num_envs,
                    self.robot_config.number_of_actions,
                    device=self.device,
                    dtype=torch.float32,
                )
                self._update_pd_targets(zeros.squeeze(1))

            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            print(f"[INFO] {self.control_type.name} mode (no CUDA graph)")

    def _create_envs(self) -> None:
        """Creates environments and loads robot assets."""
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        asset_path = os.path.join(asset_root, asset_file)

        print(f"Loading robot from: {asset_path}")

        self.robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig()

        self.robot.add_mjcf(
            asset_path,
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=False,  # We manually filter out and report only the joints/bodies we are interested in.
            floating=not self.robot_config.asset.fix_base_link,
            enable_self_collisions=self.robot_config.asset.self_collisions,
        )
        if len(self.robot.articulation_label) == 1:
            self.robot.articulation_label[0] = "robot"
        pressure_field_shape_ids = _configure_pressure_field_foot_shapes(
            self.robot,
            self.robot_config.contact_bodies or [],
            self.config.sim,
        )
        if pressure_field_shape_ids:
            print(f"[INFO] Enabled pressure-field contact on {len(pressure_field_shape_ids)} foot shapes")
        self.robot.approximate_meshes("convex_hull")

        self._object_assets = {}
        self._static_object_poses = []
        if self.scene_lib.num_scenes() > 0:
            self._load_object_assets()

        builder = newton.ModelBuilder()
        builder.current_env_group = -1
        self._add_terrain(builder)
        for env_id in range(self.num_envs):
            builder.current_env_group = env_id
            builder.begin_world(label=f"world_{env_id}")
            builder.add_builder(self.robot)

            if self.scene_lib.num_scenes() > 0:
                scene = self.scene_lib.scenes[env_id]
                scene_offset_x, scene_offset_y = self.scene_lib.scene_offsets[env_id]

                env_static_object_poses = []
                for obj_idx, obj in enumerate(scene.objects):
                    object_asset = self._object_assets[obj.first_instance_id]
                    # Use initial translation and rotation from object definition
                    # obj.translation is shape (N, 3), obj.rotation is shape (N, 4)
                    obj_trans = obj.translation[0]  # Initial position
                    obj_rot = obj.rotation[0]  # Initial rotation (xyzw)
                    object_pose = torch.tensor(
                        [
                            scene_offset_x + obj_trans[0].item(),
                            scene_offset_y + obj_trans[1].item(),
                            obj_trans[2].item(),
                            obj_rot[0].item(),
                            obj_rot[1].item(),
                            obj_rot[2].item(),
                            obj_rot[3].item(),
                        ],
                        device=self.device,
                        dtype=torch.float,
                    )
                    if obj.options.fix_base_link:
                        env_static_object_poses.append(object_pose)

                    if isinstance(object_asset, newton.Mesh):
                        if not obj.options.fix_base_link:
                            # add_body is a convenience method that creates body + free joint + articulation
                            body_id = builder.add_body(xform=object_pose, key=f"object_{env_id}_{obj_idx}")
                            xform = None
                        else:
                            body_id = -1
                            xform = object_pose
                        builder.add_shape_mesh(
                            body=body_id, xform=xform, mesh=object_asset
                        )
                    else:
                        if not obj.options.fix_base_link:
                            # add_body is a convenience method that creates body + free joint + articulation
                            body_id = builder.add_body(xform=object_pose, key=f"object_{env_id}_{obj_idx}")
                            xform = None
                        else:
                            body_id = -1
                            xform = object_pose
                        builder.add_shape_box(
                            body=body_id,
                            xform=xform,
                            hx=obj.width / 2,
                            hy=obj.depth / 2,
                            hz=obj.height / 2,
                        )
                if len(env_static_object_poses) > 0:
                    env_static_object_poses = torch.stack(
                        env_static_object_poses, dim=0
                    )  # [num_static_objects, 7]
                    self._static_object_poses.append(env_static_object_poses)
            
            # End world after all objects for this env are added
            builder.end_world()

        if len(self._static_object_poses) > 0:
            self._static_object_poses = torch.stack(
                self._static_object_poses, dim=0
            )  # [num_envs, num_static_objects, 7]

        builder.approximate_meshes(method="coacd")
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

    def _load_object_assets(self) -> None:
        if self.scene_lib.num_scenes() > 0:
            self._object_names = []

            # Count assets
            asset_count = sum(
                1 for scene in self.scene_lib.scenes for obj in scene.objects
            )

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Loading object assets...",
                    total=asset_count,
                )

                # Iterate through all scenes and their objects
                for scene in self.scene_lib.scenes:
                    for obj in scene.objects:
                        # Skip if we've already processed this object type
                        if not obj.is_first_instance:
                            progress.update(task, advance=1)
                            continue

                        first_object_id = obj.first_instance_id
                        if isinstance(obj, MeshSceneObject):
                            object_name = os.path.splitext(
                                os.path.basename(obj.object_path)
                            )[0]
                            asset_path = obj.object_path

                            m = openmesh.read_trimesh(asset_path)
                            mesh_points = np.array(m.points())
                            mesh_indices = np.array(
                                m.face_vertex_indices(), dtype=np.int32
                            ).flatten()
                            asset = newton.Mesh(mesh_points, mesh_indices)
                        else:
                            object_name = obj.object_identifier
                            asset = obj

                        self._object_names.append(object_name)
                        self._object_assets[first_object_id] = asset

                        progress.update(task, advance=1)

            print(
                f"=========== Total number of unique objects is {len(self._object_assets)}"
            )

    def _setup_robot(self) -> None:
        """Setup robot view and control parameters."""
        common_dof_names = copy.deepcopy(self._dof_names)
        newton_dof_names = {}
        robot_joint_labels = [label.rsplit("/", 1)[-1] for label in self.robot.joint_label]

        while len(common_dof_names) > 0:
            common_dof_name = common_dof_names[0]
            if common_dof_name in robot_joint_labels:
                newton_dof_names[common_dof_name] = common_dof_name
                common_dof_names.pop(0)
            else:
                multi_dof_name = None
                for newton_dof_name in robot_joint_labels:
                    if common_dof_name in newton_dof_name:
                        multi_dof_name = newton_dof_name
                        break
                assert (
                    multi_dof_name is not None
                ), f"No joint label match found for {common_dof_name} in {robot_joint_labels}"

                newton_dof_names[multi_dof_name] = []
                while (
                    len(common_dof_names) > 0 and common_dof_names[0] in multi_dof_name
                ):
                    newton_dof_names[multi_dof_name].append(common_dof_names[0])
                    common_dof_names.pop(0)

        self._newton_dof_names = newton_dof_names

        self.robot_view = ArticulationView(
            self.model,
            pattern="robot",
            include_joints=list(self._newton_dof_names.keys()),
            include_links=self._body_names,
        )

        if self.scene_lib.num_scenes() > 0 and any(
            not obj.options.fix_base_link for obj in self.scene_lib.scenes[0].objects
        ):
            self.object_view = ArticulationView(
                self.model,
                pattern="object_*",
            )
        else:
            self.object_view = None

        joint_stiffness = []
        joint_damping = []
        joint_armature = []
        joint_friction = []
        joint_effort_limit = []
        joint_velocity_limit = []

        for dof_name in self.robot_view.joint_names:
            common_dof_names = self._newton_dof_names[dof_name]
            if not isinstance(common_dof_names, list):
                common_dof_names = [common_dof_names]
            for common_dof_name in common_dof_names:
                ke = (
                    self.robot_config.control.control_info[common_dof_name].stiffness
                    if self.control_type == ControlType.BUILT_IN_PD
                    else 0.0
                )
                joint_stiffness.append(ke)

                kd = (
                    self.robot_config.control.control_info[common_dof_name].damping
                    if self.control_type == ControlType.BUILT_IN_PD
                    else 0.0
                )
                joint_damping.append(kd)

                armature = self.robot_config.control.control_info[common_dof_name].armature
                joint_armature.append(armature)

                friction = self.robot_config.control.control_info[
                    common_dof_name
                ].friction
                if joint_friction is not None and friction is not None:
                    joint_friction.append(friction)
                else:
                    joint_friction = None

                effort_limit = self.robot_config.control.control_info[
                    common_dof_name
                ].effort_limit
                if joint_effort_limit is not None and effort_limit is not None:
                    joint_effort_limit.append(effort_limit)
                else:
                    joint_effort_limit = None

                velocity_limit = self.robot_config.control.control_info[
                    common_dof_name
                ].velocity_limit
                if joint_velocity_limit is not None and velocity_limit is not None:
                    joint_velocity_limit.append(velocity_limit)
                else:
                    joint_velocity_limit = None

                # if self.robot_config.control.control_type == ControlType.BUILT_IN_PD:
                #     joint_dof_mode.append(newton.JointMode.TARGET_POSITION)
                # else:
                #     joint_dof_mode.append(newton.JointMode.NONE)

        joint_stiffness = wp.from_torch(
            torch.tensor(joint_stiffness, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(self.num_envs, 1, -1)
        )
        joint_damping = wp.from_torch(
            torch.tensor(joint_damping, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(self.num_envs, 1, -1)
        )
        joint_armature = wp.from_torch(
            torch.tensor(joint_armature, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(self.num_envs, 1, -1)
        )
        if joint_friction is not None:
            joint_friction = wp.from_torch(
                torch.tensor(joint_friction, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(self.num_envs, 1, -1)
            )
        if joint_effort_limit is not None:
            joint_effort_limit = wp.from_torch(
                torch.tensor(
                    joint_effort_limit, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(self.num_envs, 1, -1)
            )
        if joint_velocity_limit is not None:
            joint_velocity_limit = wp.from_torch(
                torch.tensor(
                    joint_velocity_limit, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(self.num_envs, 1, -1)
            )

        self.robot_view.set_attribute("joint_target_ke", self.model, joint_stiffness)
        self.robot_view.set_attribute("joint_target_kd", self.model, joint_damping)
        self.robot_view.set_attribute("joint_armature", self.model, joint_armature)
        if joint_friction is not None:
            self.robot_view.set_attribute("joint_friction", self.model, joint_friction)
        if joint_effort_limit is not None:
            self.robot_view.set_attribute(
                "joint_effort_limit", self.model, joint_effort_limit
            )
        if joint_velocity_limit is not None:
            self.robot_view.set_attribute(
                "joint_velocity_limit", self.model, joint_velocity_limit
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
        self.default_root_transforms = wp.to_torch(
            self.robot_view.get_root_transforms(self.model)
        ).squeeze(1).clone()
        self.default_root_velocities = wp.to_torch(
            self.robot_view.get_root_velocities(self.model)
        ).squeeze(1).clone()
        self.default_dof_positions = wp.to_torch(
            self.robot_view.get_dof_positions(self.model)
        ).squeeze(1).clone()
        self.default_dof_velocities = wp.to_torch(
            self.robot_view.get_dof_velocities(self.model)
        ).squeeze(1).clone()

        self._setup_explicit_pd_arrays()

    def _setup_explicit_pd_arrays(self) -> None:
        """Setup persistent Warp arrays for explicit PD control."""
        num_dofs = self.robot_config.number_of_actions
        self._pd_num_dofs = num_dofs

        is_floating = not self.robot_config.asset.fix_base_link
        self._pd_q_stride = (7 if is_floating else 0) + num_dofs
        self._pd_qd_stride = (6 if is_floating else 0) + num_dofs
        self._pd_q_dof_start = 7 if is_floating else 0
        self._pd_qd_dof_start = 6 if is_floating else 0

        device_str = str(self.device) if not isinstance(self.device, str) else self.device
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
                limit = self.robot_config.control.control_info[common_dof_name].effort_limit
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
        install_mujoco_joint_arg_coercion()
        normalize_joint_dof_mujoco_attributes(self.model)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver=sim_params.solver,
            integrator=sim_params.integrator,
            njmax=sim_params.njmax,
            nconmax=sim_params.nconmax,
            iterations=sim_params.iterations,
            ls_iterations=sim_params.ls_iterations,
            ls_parallel=sim_params.ls_parallel,
            impratio=sim_params.impratio,
            cone=sim_params.cone,
        )

        # MuJoCo exposes CCD iterations as a solver option, but Newton does not yet
        # forward it through SolverMuJoCo's constructor. Set it explicitly here so
        # contact-heavy experiments can tune CCD convergence from repo config/CLI.
        ccd_iterations = getattr(sim_params, "ccd_iterations", None)
        if ccd_iterations is not None and hasattr(self.solver.mjw_model.opt, "ccd_iterations"):
            ccd_iterations_opt = self.solver.mjw_model.opt.ccd_iterations
            if hasattr(ccd_iterations_opt, "fill_"):
                ccd_iterations_opt.fill_(ccd_iterations)
            elif isinstance(ccd_iterations_opt, int):
                self.solver.mjw_model.opt.ccd_iterations = int(ccd_iterations)
            else:
                ccd_iterations_torch = wp.to_torch(ccd_iterations_opt)
                ccd_iterations_torch[:] = ccd_iterations
                self.solver.mjw_model.opt.ccd_iterations = wp.from_torch(
                    ccd_iterations_torch, dtype=wp.int32
                )

        geom_margin = wp.to_torch(self.solver.mjw_model.geom_margin)
        geom_margin[:] = 0.01
        self.solver.mjw_model.geom_margin = wp.from_torch(geom_margin, dtype=wp.float32)

        geom_gap = wp.to_torch(self.solver.mjw_model.geom_gap)
        geom_gap[:] = 0.01
        self.solver.mjw_model.geom_gap = wp.from_torch(geom_gap, dtype=wp.float32)

        self.viewer = None
        if not self.headless:
            if self.config.viewer_backend == "gl":
                prepare_gl_viewer_compat()
                self.viewer = newton.viewer.ViewerGL()
                self.viewer.vsync = True
            elif self.config.viewer_backend == "viser":
                self.viewer = newton.viewer.ViewerViser(port=self.config.viewer_port)
                self.viewer.show_static = True
                self._setup_viser_ground_visual()
                local_ip = _get_primary_local_ip()
                print(
                    f"[INFO] Newton viser URL inside WSL: http://127.0.0.1:{self.config.viewer_port}"
                )
                if local_ip is not None:
                    print(
                        f"[INFO] Newton viser URL from Windows/LAN: http://{local_ip}:{self.config.viewer_port}"
                    )
            else:
                raise ValueError(
                    f"Unsupported Newton viewer backend: {self.config.viewer_backend}"
                )
            self.viewer.set_model(self.model)

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _setup_viser_ground_visual(self) -> None:
        """Add a viewer-only ground reference for the viser backend."""
        if self.config.viewer_backend != "viser":
            return

        server = getattr(self.viewer, "_server", None)
        if server is None:
            return

        server.scene.add_grid(
            "/protomotions_ground",
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
            # Get current friction values
            current_friction = wp.to_torch(self.model.shape_material_mu).clone()
            current_restitution = wp.to_torch(self.model.shape_material_restitution).clone()

            # Get shape-to-body mapping
            shape_body = wp.to_torch(self.model.shape_body)

            # Get body indices that should be randomized
            body_indices = self._domain_randomization["friction"]["body_indices"]
            static_friction = self._domain_randomization["friction"]["static_friction"]
            restitution = self._domain_randomization["friction"]["restitution"]

            num_buckets = static_friction.shape[0] if static_friction is not None else 0

            if num_buckets > 0:
                # In Newton multi-world, bodies are replicated per world
                # We need to find the global body index for each env's body
                num_bodies_per_robot = len(self._body_names)

                # Match IsaacLab: generate bucket_ids per body type (each body type
                # gets independent random bucket assignment across environments)
                for idx, local_body_idx in enumerate(body_indices):
                    # Generate new bucket assignment for this body type
                    bucket_ids = torch.randint(0, num_buckets, (self.num_envs,), device=self.device)

                    for env_idx in range(self.num_envs):
                        bucket_id = bucket_ids[env_idx].item()
                        global_body_idx = env_idx * num_bodies_per_robot + local_body_idx

                        # Find all shapes belonging to this body
                        shape_mask = shape_body == global_body_idx
                        shape_indices = torch.where(shape_mask)[0]

                        # Apply friction from the bucket using body-specific index
                        if static_friction is not None and len(shape_indices) > 0:
                            friction_value = static_friction[bucket_id, idx].item()
                            current_friction[shape_indices] = friction_value

                        if restitution is not None and len(shape_indices) > 0:
                            restitution_value = restitution[bucket_id, idx].item()
                            current_restitution[shape_indices] = restitution_value

                # Update model attributes
                self.model.shape_material_mu.assign(
                    wp.from_torch(current_friction, dtype=wp.float32)
                )
                self.model.shape_material_restitution.assign(
                    wp.from_torch(current_restitution, dtype=wp.float32)
                )

                notify_flags |= SolverNotifyFlags.SHAPE_PROPERTIES
                print(
                    f"[INFO] Applied friction domain randomization to {len(body_indices)} body types"
                )

        # Apply center of mass randomization
        if "center_of_mass" in self._domain_randomization:
            # Get current body COM values
            current_com = wp.to_torch(self.model.body_com).clone()

            body_indices = self._domain_randomization["center_of_mass"]["body_indices"]
            com_offsets = self._domain_randomization["center_of_mass"]["com"]

            # In Newton multi-world, bodies are replicated per world
            num_bodies_per_robot = len(self._body_names)

            for env_idx in range(self.num_envs):
                for idx, local_body_idx in enumerate(body_indices):
                    global_body_idx = env_idx * num_bodies_per_robot + local_body_idx

                    # Add the COM offset for this environment
                    # com_offsets shape: [num_envs, num_matching_bodies, 3]
                    offset = com_offsets[env_idx, idx]
                    current_com[global_body_idx] += offset.to(current_com.device)

            # Update model attribute
            self.model.body_com.assign(wp.from_torch(current_com, dtype=wp.vec3))

            notify_flags |= SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            print(
                f"[INFO] Applied center of mass domain randomization to {len(body_indices)} body types"
            )

        # Notify solver of changes so MuJoCo updates its internal model
        if notify_flags != 0:
            self.solver.notify_model_changed(notify_flags)

    def _set_robot_friction_to_minimum(self) -> None:
        """Set robot shape friction/restitution to terrain values.

        This ensures stable contact. DR will override with correct values if configured.
        """
        if self.terrain is None:
            return

        terrain_friction = self.terrain.sim_config.static_friction
        terrain_restitution = self.terrain.sim_config.restitution

        current_friction = wp.to_torch(self.model.shape_material_mu).clone()
        current_restitution = wp.to_torch(self.model.shape_material_restitution).clone()
        shape_body = wp.to_torch(self.model.shape_body)

        robot_shape_mask = shape_body >= 0
        robot_shape_indices = torch.where(robot_shape_mask)[0]

        if len(robot_shape_indices) > 0:
            current_friction[robot_shape_indices] = terrain_friction
            current_restitution[robot_shape_indices] = terrain_restitution
            self.model.shape_material_mu.assign(
                wp.from_torch(current_friction, dtype=wp.float32)
            )
            self.model.shape_material_restitution.assign(
                wp.from_torch(current_restitution, dtype=wp.float32)
            )
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

    def _add_terrain(self, builder: newton.ModelBuilder) -> None:
        """Adds terrain with friction and restitution from terrain config."""
        if self.terrain is None:
            return

        assert (
            self.terrain.sim_config.combine_mode == CombineMode.MAX
        ), "Newton only supports max friction combine mode"

        # Ground material properties from terrain config
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self.terrain.sim_config.static_friction,
            restitution=self.terrain.sim_config.restitution,
            hydroelastic_type=(
                newton.HydroelasticType.RIGID
                if getattr(self.config.sim, "pressure_field_feet", False)
                else newton.HydroelasticType.NONE
            ),
        )

        print("Adding terrain")
        if (
            sum(self.terrain.config.terrain_proportions[:-1]) == 0
            and self.terrain.config.terrain_proportions[-1] == 1.0
        ):
            if self.config.viewer_backend == "viser" and not getattr(self.config.sim, "pressure_field_feet", False):
                # ViewerViser does not reliably display Newton's special plane primitive.
                # Use a large explicit mesh instead for flat-ground scenes.
                plane_vertices, plane_indices = create_plane_mesh(400.0, 400.0)
                ground_mesh = newton.Mesh(
                    vertices=plane_vertices[:, 0:3], indices=plane_indices
                )
                builder.add_shape_mesh(
                    body=-1, mesh=ground_mesh, cfg=ground_cfg, key="ground_plane"
                )
            else:
                # Pressure-field feet need a rigid hydroelastic plane, not a visual mesh surrogate.
                builder.add_ground_plane(cfg=ground_cfg)
        else:
            points, indices = convert_to_indexed_mesh(
                self.terrain.vertices, self.terrain.triangles
            )
            ground_mesh = newton.Mesh(vertices=points, indices=indices)
            builder.add_shape_mesh(body=-1, mesh=ground_mesh, cfg=ground_cfg, key="ground_plane")
        print("Terrain added")

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
            body_patterns = [body_name, f"*/{body_name}"]
            # Create sensor that detects contacts between this body and anything
            # The sensor will aggregate contacts across all environments
            sensor = SensorContact(
                self.model, sensing_obj_bodies=body_patterns, verbose=False
            )
            self._contact_sensors[body_name] = sensor

            self._contact_forces[body_name] = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=torch.float32
            )

        print(f"[INFO] Contact sensors setup complete for {len(self._contact_sensors)} bodies")

    def _setup_contact_analysis_metadata(self) -> None:
        """Precompute static foot metadata for GRF/CoP/pressure analysis."""
        left_bodies = list(
            self.robot_config.common_naming_to_robot_body_names.get(
                "all_left_foot_bodies", []
            )
        )
        right_bodies = list(
            self.robot_config.common_naming_to_robot_body_names.get(
                "all_right_foot_bodies", []
            )
        )
        if not left_bodies and not right_bodies:
            self._contact_analysis = None
            return

        num_bodies_per_robot = len(self._body_names)
        shape_body = wp.to_torch(self.model.shape_body).detach().cpu().numpy()
        shape_world = wp.to_torch(self.model.shape_world).detach().cpu().numpy()
        shape_transform = wp.to_torch(self.model.shape_transform).detach().cpu().numpy()
        shape_aabb_lower = (
            wp.to_torch(self.model.shape_collision_aabb_lower).detach().cpu().numpy()
        )
        shape_aabb_upper = (
            wp.to_torch(self.model.shape_collision_aabb_upper).detach().cpu().numpy()
        )
        default_body_transforms = self.default_body_transforms[0].detach().cpu().numpy()

        side_specs = [
            ("left", left_bodies),
            ("right", right_bodies),
        ]
        sides = []
        body_side_index = np.full((num_bodies_per_robot,), -1, dtype=np.int32)

        for side_index, (side_name, body_names) in enumerate(side_specs):
            local_body_indices = [
                self._body_names.index(body_name)
                for body_name in body_names
                if body_name in self._body_names
            ]
            if not local_body_indices:
                continue

            anchor_body_name = next(
                (
                    body_name
                    for body_name in body_names
                    if "ankle" in body_name.lower() and body_name in self._body_names
                ),
                body_names[0],
            )
            anchor_body_idx = self._body_names.index(anchor_body_name)
            body_side_index[local_body_indices] = side_index

            shape_ids = np.where(
                (shape_body >= 0)
                & (shape_body < num_bodies_per_robot)
                & np.isin(shape_body, np.asarray(local_body_indices, dtype=np.int32))
            )[0]
            if shape_ids.size == 0:
                continue

            anchor_transform = default_body_transforms[anchor_body_idx]
            anchor_pos = anchor_transform[:3]
            anchor_rotation = _quat_xyzw_to_matrix_np(anchor_transform[3:7])

            x_values = []
            y_values = []
            for shape_id in shape_ids.tolist():
                local_body_idx = int(shape_body[shape_id])
                body_transform = default_body_transforms[local_body_idx]
                shape_corners = _make_aabb_corners_np(
                    shape_aabb_lower[shape_id], shape_aabb_upper[shape_id]
                )
                corners_body = _transform_points_np(
                    shape_transform[shape_id, :3],
                    shape_transform[shape_id, 3:7],
                    shape_corners,
                )
                corners_world = _transform_points_np(
                    body_transform[:3],
                    body_transform[3:7],
                    corners_body,
                )
                corners_anchor = (corners_world - anchor_pos[None, :]) @ anchor_rotation
                x_values.append(corners_anchor[:, 0])
                y_values.append(corners_anchor[:, 1])

            bounds = np.array(
                [
                    float(np.min(np.concatenate(x_values))),
                    float(np.max(np.concatenate(x_values))),
                    float(np.min(np.concatenate(y_values))),
                    float(np.max(np.concatenate(y_values))),
                ],
                dtype=np.float32,
            )
            sides.append(
                {
                    "name": side_name,
                    "body_names": list(body_names),
                    "body_indices": np.asarray(local_body_indices, dtype=np.int32),
                    "anchor_body_name": anchor_body_name,
                    "anchor_body_idx": int(anchor_body_idx),
                    "support_bounds_xy": bounds,
                }
            )

        if not sides:
            self._contact_analysis = None
            return

        self._contact_analysis = {
            "num_bodies_per_robot": int(num_bodies_per_robot),
            "shape_body": shape_body.astype(np.int32),
            "shape_world": shape_world.astype(np.int32),
            "body_side_index": body_side_index,
            "sides": sides,
        }

    def get_contact_analysis_metadata(self) -> Optional[Dict[str, object]]:
        """Return static foot-analysis metadata for downstream plotting/export."""
        if self._contact_analysis is None:
            return None

        return {
            "side_names": [side["name"] for side in self._contact_analysis["sides"]],
            "anchor_body_names": [
                side["anchor_body_name"] for side in self._contact_analysis["sides"]
            ],
            "support_bounds_xy": [
                side["support_bounds_xy"].copy()
                for side in self._contact_analysis["sides"]
            ],
        }

    def get_contact_analysis_frame(
        self,
        current_state: Optional[RobotState] = None,
        *,
        num_bins_x: int = 24,
        num_bins_y: int = 12,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate the current foot-ground contact state into analysis-friendly outputs."""
        if self._contact_analysis is None or self.contacts.force is None:
            return None

        if current_state is None:
            current_state = self.get_robot_state()
        elif current_state.state_conversion != StateConversion.COMMON:
            current_state = current_state.clone().convert_to_common(self.data_conversion)

        if current_state.rigid_body_pos is None or current_state.rigid_body_rot is None:
            return None

        side_infos = self._contact_analysis["sides"]
        num_sides = len(side_infos)
        num_envs = current_state.rigid_body_pos.shape[0]

        grf_world = np.zeros((num_envs, num_sides, 3), dtype=np.float32)
        cop_world_weighted = np.zeros((num_envs, num_sides, 3), dtype=np.float32)
        cop_world = np.full((num_envs, num_sides, 3), np.nan, dtype=np.float32)
        cop_local = np.full((num_envs, num_sides, 2), np.nan, dtype=np.float32)
        cop_normalized = np.full((num_envs, num_sides, 2), np.nan, dtype=np.float32)
        pressure_maps = np.zeros(
            (num_envs, num_sides, num_bins_y, num_bins_x), dtype=np.float32
        )
        contact_counts = np.zeros((num_envs, num_sides), dtype=np.int32)
        stance_force_z = np.zeros((num_envs, num_sides), dtype=np.float32)

        rigid_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        if rigid_contact_count <= 0:
            return {
                "side_names": np.array(
                    [side["name"] for side in side_infos], dtype=np.str_
                ),
                "grf_world": grf_world,
                "cop_world": cop_world,
                "cop_local": cop_local,
                "cop_normalized": cop_normalized,
                "cop_valid": np.zeros((num_envs, num_sides), dtype=bool),
                "pressure_maps": pressure_maps,
                "contact_counts": contact_counts,
                "support_bounds_xy": np.stack(
                    [side["support_bounds_xy"] for side in side_infos], axis=0
                ),
            }

        shape0 = self.contacts.rigid_contact_shape0.numpy()[:rigid_contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:rigid_contact_count]
        point0 = self.contacts.rigid_contact_point0.numpy()[:rigid_contact_count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:rigid_contact_count]
        force = self.contacts.force.numpy()[:rigid_contact_count, :3]

        shape_body = self._contact_analysis["shape_body"]
        shape_world = self._contact_analysis["shape_world"]
        body_side_index = self._contact_analysis["body_side_index"]
        num_bodies_per_robot = self._contact_analysis["num_bodies_per_robot"]

        rigid_body_pos = current_state.rigid_body_pos.detach().cpu().numpy()
        rigid_body_rot = current_state.rigid_body_rot.detach().cpu().numpy()

        for contact_idx in range(rigid_contact_count):
            shape_idx_0 = int(shape0[contact_idx])
            shape_idx_1 = int(shape1[contact_idx])
            body_idx_0 = int(shape_body[shape_idx_0])
            body_idx_1 = int(shape_body[shape_idx_1])

            local_body_idx_0 = body_idx_0 % num_bodies_per_robot if body_idx_0 >= 0 else -1
            local_body_idx_1 = body_idx_1 % num_bodies_per_robot if body_idx_1 >= 0 else -1
            side_idx_0 = (
                int(body_side_index[local_body_idx_0]) if local_body_idx_0 >= 0 else -1
            )
            side_idx_1 = (
                int(body_side_index[local_body_idx_1]) if local_body_idx_1 >= 0 else -1
            )

            env_idx = -1
            side_idx = -1
            contact_force = None
            ground_point = None

            if side_idx_0 >= 0 and body_idx_1 < 0:
                env_idx = int(shape_world[shape_idx_0])
                side_idx = side_idx_0
                contact_force = force[contact_idx]
                ground_point = point1[contact_idx]
            elif side_idx_1 >= 0 and body_idx_0 < 0:
                env_idx = int(shape_world[shape_idx_1])
                side_idx = side_idx_1
                contact_force = -force[contact_idx]
                ground_point = point0[contact_idx]
            else:
                continue

            if env_idx < 0 or env_idx >= num_envs or side_idx < 0 or side_idx >= num_sides:
                continue

            contact_counts[env_idx, side_idx] += 1
            grf_world[env_idx, side_idx] += contact_force.astype(np.float32)

            fz = float(max(contact_force[2], 0.0))
            if fz <= 0.0:
                continue

            stance_force_z[env_idx, side_idx] += fz
            cop_world_weighted[env_idx, side_idx] += ground_point.astype(np.float32) * fz

            side_info = side_infos[side_idx]
            anchor_body_idx = side_info["anchor_body_idx"]
            anchor_pos = rigid_body_pos[env_idx, anchor_body_idx]
            anchor_rot = rigid_body_rot[env_idx, anchor_body_idx]
            x_axis, y_axis = _planar_axes_from_quat_xyzw(anchor_rot)
            delta = ground_point.astype(np.float32) - anchor_pos.astype(np.float32)
            local_x = float(np.dot(delta, x_axis))
            local_y = float(np.dot(delta, y_axis))

            x_min, x_max, y_min, y_max = side_info["support_bounds_xy"]
            bin_width = max((x_max - x_min) / float(num_bins_x), 1.0e-6)
            bin_height = max((y_max - y_min) / float(num_bins_y), 1.0e-6)
            bin_area = bin_width * bin_height

            bin_x = int(np.clip(np.floor((local_x - x_min) / bin_width), 0, num_bins_x - 1))
            bin_y = int(np.clip(np.floor((local_y - y_min) / bin_height), 0, num_bins_y - 1))
            pressure_maps[env_idx, side_idx, bin_y, bin_x] += np.float32(fz / bin_area)

        cop_valid = stance_force_z > 1.0e-6
        for env_idx in range(num_envs):
            for side_idx, side_info in enumerate(side_infos):
                if not cop_valid[env_idx, side_idx]:
                    continue

                weighted_force = stance_force_z[env_idx, side_idx]
                cop_world[env_idx, side_idx] = (
                    cop_world_weighted[env_idx, side_idx] / weighted_force
                ).astype(np.float32)

                anchor_body_idx = side_info["anchor_body_idx"]
                anchor_pos = rigid_body_pos[env_idx, anchor_body_idx]
                anchor_rot = rigid_body_rot[env_idx, anchor_body_idx]
                x_axis, y_axis = _planar_axes_from_quat_xyzw(anchor_rot)
                delta = cop_world[env_idx, side_idx] - anchor_pos.astype(np.float32)
                local_x = float(np.dot(delta, x_axis))
                local_y = float(np.dot(delta, y_axis))
                cop_local[env_idx, side_idx, 0] = local_x
                cop_local[env_idx, side_idx, 1] = local_y

                x_min, x_max, y_min, y_max = side_info["support_bounds_xy"]
                cop_normalized[env_idx, side_idx, 0] = _normalize_to_unit_interval(
                    local_x, x_min, x_max
                )
                cop_normalized[env_idx, side_idx, 1] = _normalize_to_unit_interval(
                    local_y, y_min, y_max
                )

        return {
            "side_names": np.array([side["name"] for side in side_infos], dtype=np.str_),
            "grf_world": grf_world,
            "cop_world": cop_world,
            "cop_local": cop_local,
            "cop_normalized": cop_normalized,
            "cop_valid": cop_valid,
            "pressure_maps": pressure_maps,
            "contact_counts": contact_counts,
            "support_bounds_xy": np.stack(
                [side["support_bounds_xy"] for side in side_infos], axis=0
            ).astype(np.float32),
        }

    def _simulate(self) -> None:
        """Run physics simulation for one frame (decimation substeps)."""
        for _ in range(self.decimation):
            self.state_0.clear_forces()
            if self.control_type == ControlType.PROPORTIONAL:
                self._apply_pd_kernel(self.state_0)
            elif self.control_type == ControlType.TORQUE:
                self._apply_torques_kernel_method()
            if self.viewer:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.decimation % 2 != 0:
            self.state_1.assign(self.state_0)

    def _update_contact_sensors(self) -> None:
        """Update contact sensors after physics step. Must be called outside CUDA graph."""
        if len(self._contact_sensors) > 0:
            self.solver.update_contacts(self.contacts, self.state_0)
            for body_name, sensor in self._contact_sensors.items():
                sensor.update(self.state_0, self.contacts)
                if sensor.total_force is not None:
                    self._contact_forces[body_name] = wp.to_torch(sensor.total_force).clone()

    def _physics_step(self) -> None:
        """Performs a physics simulation step."""
        # Update control targets before simulation
        if self.control_type == ControlType.BUILT_IN_PD:
            self._apply_control()
        elif self.control_type == ControlType.PROPORTIONAL:
            pd_tar = self._action_to_pd_targets(self._common_actions)
            if self._domain_randomization is not None and "action_noise" in self._domain_randomization:
                pd_tar[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]
            sim_targets = pd_tar[:, self.data_conversion.dof_convert_to_sim]
            self._update_pd_targets(sim_targets)
        elif self.control_type == ControlType.TORQUE:
            torques = self._action_to_torque_targets(self._common_actions)
            if self._domain_randomization is not None and "action_noise" in self._domain_randomization:
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

        # Refresh derived link transforms from the latest generalized state
        # before any state reads. Newton reset paths already do this explicitly.
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        self._update_contact_sensors()
        self.sim_time += self.frame_dt

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

        # Newton expects the state setter to be provided with the states for all envs.
        # The mask is used to determine which envs to apply the update to.
        robot_state = self._get_simulator_bodies_state()
        robot_dof_state = self._get_simulator_dof_state()
        robot_state.merge_fields_from(robot_dof_state)

        robot_state.root_pos[env_ids] = new_states.root_pos
        robot_state.root_rot[env_ids] = new_states.root_rot
        robot_state.root_vel[env_ids] = new_states.root_vel
        robot_state.root_ang_vel[env_ids] = new_states.root_ang_vel
        robot_state.dof_pos[env_ids] = new_states.dof_pos
        robot_state.dof_vel[env_ids] = new_states.dof_vel

        root_state = torch.cat([robot_state.root_pos, robot_state.root_rot], dim=1)
        root_vel_state = torch.cat(
            [robot_state.root_vel, robot_state.root_ang_vel], dim=1
        )

        root_state_3d = root_state.unsqueeze(1)
        root_vel_state_3d = root_vel_state.unsqueeze(1)
        dof_pos_3d = robot_state.dof_pos.unsqueeze(1)
        dof_vel_3d = robot_state.dof_vel.unsqueeze(1)

        # Set state_0 using ArticulationView
        self.robot_view.set_root_transforms(self.state_0, root_state_3d, mask=env_mask)
        self.robot_view.set_root_velocities(self.state_0, root_vel_state_3d, mask=env_mask)
        self.robot_view.set_dof_velocities(
            self.state_0, dof_vel_3d, mask=env_mask
        )

        self.robot_view.set_dof_positions(
            self.state_0, dof_pos_3d, mask=env_mask
        )

        # Also update state_1 to match state_0
        self.robot_view.set_root_transforms(self.state_1, root_state_3d, mask=env_mask)
        self.robot_view.set_root_velocities(self.state_1, root_vel_state_3d, mask=env_mask)
        self.robot_view.set_dof_velocities(
            self.state_1, dof_vel_3d, mask=env_mask
        )
        self.robot_view.set_dof_positions(
            self.state_1, dof_pos_3d, mask=env_mask
        )

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

    # ===== Group 4: State Getters =====
    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns contact forces for robot bodies."""
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
        body_transforms = wp.to_torch(
            self.robot_view.get_link_transforms(self.state_0)
        ).squeeze(1).view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        body_pos = body_transforms[:, :, :3]
        body_rot = body_transforms[:, :, 3:]

        body_vel_transforms = wp.to_torch(
            self.robot_view.get_link_velocities(self.state_0)
        ).squeeze(1).view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        body_vel = body_vel_transforms[:, :, :3]
        body_ang_vel = body_vel_transforms[:, :, 3:]

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
        root_transforms = wp.to_torch(self.robot_view.get_root_transforms(self.state_0)).squeeze(1)
        root_velocities = wp.to_torch(self.robot_view.get_root_velocities(self.state_0)).squeeze(1)

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
        """Returns the root state of simulation objects. Not implemented."""
        newton_root_pos = []
        newton_root_rot = []
        newton_root_vel = []
        newton_root_ang_vel = []

        if self.robot_view is not None:
            num_dynamic_objects = sum(
                not obj.options.fix_base_link
                for obj in self.scene_lib.scenes[0].objects
            )
            view_transforms = wp.to_torch(
                self.object_view.get_root_transforms(self.state_0)
            ).view(self.num_envs, num_dynamic_objects, -1)
            view_velocities = wp.to_torch(
                self.object_view.get_root_velocities(self.state_0)
            ).view(self.num_envs, num_dynamic_objects, -1)
        else:
            view_transforms = view_velocities = None

        articulated_object_index = 0
        static_object_index = 0
        for obj_idx in range(self.scene_lib.num_objects_per_scene):
            if self.scene_lib.scenes[0].objects[obj_idx].options.fix_base_link:
                newton_root_pos.append(
                    self._static_object_poses[:, static_object_index, :3]
                )
                newton_root_rot.append(
                    self._static_object_poses[:, static_object_index, 3:]
                )
                newton_root_vel.append(torch.zeros_like(newton_root_pos[-1]))
                newton_root_ang_vel.append(torch.zeros_like(newton_root_pos[-1]))
                static_object_index += 1
            else:
                newton_root_pos.append(view_transforms[:, articulated_object_index, :3])
                newton_root_rot.append(view_transforms[:, articulated_object_index, 3:])
                newton_root_vel.append(view_velocities[:, articulated_object_index, :3])
                newton_root_ang_vel.append(
                    view_velocities[:, articulated_object_index, 3:]
                )
                articulated_object_index += 1

        newton_root_pos = torch.stack(newton_root_pos, dim=1)
        newton_root_rot = torch.stack(newton_root_rot, dim=1)
        newton_root_vel = torch.stack(newton_root_vel, dim=1)
        newton_root_ang_vel = torch.stack(newton_root_ang_vel, dim=1)
        if env_ids is not None:
            newton_root_pos = newton_root_pos[env_ids]
            newton_root_rot = newton_root_rot[env_ids]
            newton_root_vel = newton_root_vel[env_ids]
            newton_root_ang_vel = newton_root_ang_vel[env_ids]
        return ObjectState(
            root_pos=newton_root_pos,
            root_rot=newton_root_rot,
            root_vel=newton_root_vel,
            root_ang_vel=newton_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Returns contact forces for simulation objects."""
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_dof_forces(self, env_ids=None):
        """Returns the DOF forces."""
        dof_forces = wp.to_torch(self.robot_view.get_dof_forces(self.control)).squeeze(1)
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return RobotState(
            dof_forces=dof_forces, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns the state of robot DOFs."""
        dof_pos = wp.to_torch(
            self.robot_view.get_dof_positions(self.state_0)
        ).squeeze(1).view(self.num_envs, -1)

        dof_vel = wp.to_torch(
            self.robot_view.get_dof_velocities(self.state_0)
        ).squeeze(1).view(self.num_envs, -1)

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
        a_wp = wp.from_torch(pd_targets.unsqueeze(1), dtype=wp.float32, requires_grad=False)
        self.robot_view.set_attribute("joint_target_pos", self.control, a_wp)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        a_wp = wp.from_torch(torques.unsqueeze(1), dtype=wp.float32, requires_grad=False)
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
        wp.copy(self._pd_targets_wp, wp.from_torch(pd_targets.view(-1), dtype=wp.float32))

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

        self._set_viewer_camera(cam_pos, camera_target, pitch, yaw)
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

        cam_pos = np.array(self._viewer_camera_pos)
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

        self._set_viewer_camera(new_cam_pos, new_cam_target, pitch, yaw)
        self._cam_prev_char_pos = char_root_pos

    def _set_viewer_camera(
        self,
        cam_pos: np.ndarray,
        cam_target: np.ndarray,
        pitch: float,
        yaw: float,
    ) -> None:
        """Track camera state locally and forward it to the active viewer backend."""
        self._viewer_camera_pos = np.array(cam_pos, dtype=np.float32)
        if self.config.viewer_backend == "viser":
            server = getattr(self.viewer, "_server", None)
            if server is not None:
                server.initial_camera.position = tuple(self._viewer_camera_pos.tolist())
                server.initial_camera.look_at = tuple(np.asarray(cam_target).tolist())
                server.initial_camera.up = (0.0, 0.0, 1.0)

                for client in server.get_clients().values():
                    client.camera.position = self._viewer_camera_pos
                    client.camera.look_at = np.asarray(cam_target, dtype=np.float64)
                    client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return

        self.viewer.set_camera(wp.vec3(self._viewer_camera_pos.tolist()), pitch, yaw)

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
                    self._push_robot()
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

        if self.config.viewer_backend != "gl":
            raise RuntimeError(
                "Viewport capture is only supported with the Newton GL viewer backend."
            )
        viewport = self.viewer.get_frame().numpy()  # [H, W, 3] as uint8
        plt.imsave(file_name, viewport)

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Updates visualization markers."""
        pass
