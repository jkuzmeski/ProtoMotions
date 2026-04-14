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
"""
Default Pose Visualizer for Humanoid Robots

This tool visualizes a robot in its default standing pose by:
1. Loading a specified robot from the robot config factory
2. Resetting it onto a flat terrain at the simulator's default pose
3. Holding that pose with the robot's default controller
"""

from typing import Dict, List
import argparse
from dataclasses import dataclass
import math

# Parse arguments first (argparse is safe, doesn't import torch)
parser = argparse.ArgumentParser(
    description="Default Pose Visualizer for Humanoid Robots"
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["isaacgym", "isaaclab", "newton"],
    default="newton",
    help="Simulator to use (isaacgym, isaaclab, newton)",
)
parser.add_argument(
    "--robot",
    type=str,
    default="smpl_lower_body_ellipsoid_feet",
    help="Robot to load from protomotions.robot_configs.factory.robot_config",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

from protomotions.simulator.base_simulator.config import (  # noqa: E402
    VisualizationMarkerConfig,
    MarkerConfig,
    MarkerState,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.simulator.base_simulator.utils import (  # noqa: E402
    convert_friction_for_simulator,
)


@dataclass
class RobotSpec:
    """Robot specification with body names for visualization"""

    # Body names to visualize (these are the rigid body names, not joint names)
    viz_bodies: List[str]


# Define robot specifications
ROBOT_SPECS = {
    "g1": RobotSpec(
        viz_bodies=[
            "pelvis",
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
    ),
    "rigv1": RobotSpec(
        viz_bodies=["Hips", "Spine2", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot"],
    ),
    "smpl": RobotSpec(
        viz_bodies=["Pelvis", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"],
    ),
    "smpl_lower_body": RobotSpec(
        viz_bodies=["Pelvis", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"],
    ),
    "smpl_lower_body_ellipsoid_feet": RobotSpec(
        viz_bodies=["Pelvis", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"],
    ),
}


class DefaultPoseVisualizer:
    def __init__(
        self,
        robot_name: str = "smpl_lower_body_ellipsoid_feet",
        num_envs: int = 1,
        simulator_type: str = "newton",
        headless: bool = False,
        cpu_only: bool = False,
        extra_simulator_params: dict = None,
    ):
        self.robot_name = robot_name
        self.num_envs = num_envs
        self.simulator_type = simulator_type
        self.headless = headless
        self.device = torch.device("cuda:0" if not cpu_only else "cpu")

        # Load robot configuration using factory function
        self.robot_cfg = robot_config(robot_name)
        self.robot_spec = self._resolve_robot_spec(robot_name)

        # Create simulator configuration using factory function
        self.simulator_cfg = simulator_config(
            simulator_type,
            self.robot_cfg,
            headless=headless,
            num_envs=num_envs,
            experiment_name="default_pose_viz",
        )

        # Keep the robot dynamic and let the default PD controller hold the nominal pose.
        self.robot_cfg.asset.disable_gravity = False
        self.robot_cfg.asset.fix_base_link = False
        self.robot_cfg.asset.self_collisions = False

        # Create visualization markers
        self.viz_markers = self._create_visualization_markers()

        terrain_grid_size = max(1, math.ceil(num_envs ** (1 / 3)))
        terrain_config = TerrainConfig(
            map_length=8.0,
            map_width=8.0,
            border_size=4.0,
            num_levels=terrain_grid_size,
            num_terrains=terrain_grid_size,
        )
        terrain_config, self.simulator_cfg = convert_friction_for_simulator(
            terrain_config, self.simulator_cfg
        )
        terrain = Terrain(
            config=terrain_config,
            num_envs=self.simulator_cfg.num_envs,
            device=self.device,
        )

        # Create empty scene_lib (no extra scene objects)
        from protomotions.components.scene_lib import SceneLib

        scene_lib = SceneLib.empty(
            num_envs=self.simulator_cfg.num_envs, device=self.device
        )

        # Get simulator class and instantiate
        SimulatorClass = get_class(self.simulator_cfg._target_)

        extra_params = extra_simulator_params or {}
        self.simulator = SimulatorClass(
            config=self.simulator_cfg,
            robot_config=self.robot_cfg,
            terrain=terrain,
            device=self.device,
            scene_lib=scene_lib,
            **extra_params,
        )

        # Initialize the simulator with visualization markers
        self.simulator._initialize_with_markers(self.viz_markers)

        print(f"Loaded {robot_name} robot using {simulator_type}")
        print(f"Robot config: {type(self.robot_cfg).__name__}")
        print(f"Number of actions: {self.robot_cfg.number_of_actions}")
        print(f"Number of DOFs: {self.robot_cfg.kinematic_info.num_dofs}")
        print(f"Visualizing bodies: {self.robot_spec.viz_bodies}")
        print("Press 'R' to reapply the default standing pose")

        self.simulator.user_requested_reset = True

    def _resolve_robot_spec(self, robot_name: str) -> RobotSpec:
        if robot_name in ROBOT_SPECS:
            return ROBOT_SPECS[robot_name]

        preferred_bodies = [
            "Pelvis",
            "torso_link",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
            "L_Toe",
            "R_Toe",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        body_names = self.robot_cfg.kinematic_info.body_names
        viz_bodies = [body for body in preferred_bodies if body in body_names]
        if not viz_bodies:
            viz_bodies = body_names[: min(6, len(body_names))]
        return RobotSpec(viz_bodies=viz_bodies)

    def _create_visualization_markers(self) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for specified body locations"""
        # Create one marker config for each body we want to visualize
        marker_configs = [
            MarkerConfig(size="regular") for _ in self.robot_spec.viz_bodies
        ]

        # Create a single visualization marker group for all bodies
        markers = {
            "body_markers": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=marker_configs
            )
        }

        return markers

    def _get_updated_marker_positions(self):
        """Update marker positions to follow the specified bodies"""
        if not self.viz_markers:
            return

        # this will convert to sim common ordering, which is the MJCF ordering
        current_state = self.simulator.get_bodies_state()

        idx_in_common = [
            self.simulator._body_names.index(body_name)
            for body_name in self.robot_spec.viz_bodies
        ]

        all_positions = (
            current_state.rigid_body_pos[:, idx_in_common, :].detach().clone()
        )
        all_orientations = (
            current_state.rigid_body_rot[:, idx_in_common, :].detach().clone()
        )

        # # surgery on the 1st marker
        # root_orientation = all_orientations[:, 0, :].detach().clone()
        # root_offset = torch.tensor([0.0, 0.1, 0.0], device=self.device)
        # root_offset = root_offset.repeat(self.num_envs, 1)
        # all_positions[:, 0, :] += quat_apply(root_orientation, root_offset, w_last=True)

        marker_states = {
            "body_markers": MarkerState(
                translation=all_positions, orientation=all_orientations
            )
        }

        return marker_states

    def run(self):
        """Main simulation loop"""
        step_count = 0

        # Parameters
        spacing = 4.0  # spacing between humanoids

        # Determine the grid size along each axis (cube root rounded up)
        grid_size = math.ceil(self.num_envs ** (1 / 3))

        # Create grid coordinates
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, device=self.device),
                torch.arange(grid_size, device=self.device),
                torch.arange(grid_size, device=self.device),
                indexing="ij",  # ensures x,y,z layout
            ),
            dim=-1,
        ).reshape(-1, 3)

        # Scale by spacing and offset each robot to its nominal standing height.
        root_positions = coords[: self.num_envs] * spacing
        terrain_xy = root_positions[:, :2]
        terrain_heights = self.simulator.terrain.get_ground_heights(terrain_xy).view(-1)
        root_positions[:, 2] = terrain_heights + self.robot_cfg.default_root_height
        while True:
            if self.simulator.user_requested_reset:
                default_state = self.simulator.get_default_robot_reset_state()
                default_state.root_pos[:] = root_positions
                env_ids = torch.arange(self.num_envs, device=self.device)
                self.simulator.reset_envs(
                    default_state, new_object_states=None, env_ids=env_ids
                )
                self.simulator.user_requested_reset = False

            _common_actions = torch.zeros(
                self.num_envs, self.robot_cfg.number_of_actions, device=self.device
            )

            marker_states = self._get_updated_marker_positions()

            self.simulator.step(_common_actions, markers_callback=lambda: marker_states)

            step_count += 1


def main():
    # Use the global args that were parsed early
    global args, AppLauncher

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Extra simulator parameters for IsaacLab
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    visualizer = DefaultPoseVisualizer(
        robot_name=args.robot,
        num_envs=args.num_envs,
        simulator_type=args.simulator,
        headless=args.headless,
        cpu_only=args.cpu_only,
        extra_simulator_params=extra_simulator_params,
    )

    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.simulator.close()


if __name__ == "__main__":
    main()
