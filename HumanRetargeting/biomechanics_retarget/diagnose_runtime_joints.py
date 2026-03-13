#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Runtime joint diagnostic for env_kinematic_playback.

Runs a short kinematic playback and prints:
  - Newton's actual joint ordering vs MJCF/COMMON ordering
  - The dof_convert_to_sim mapping
  - dof_pos values BEFORE and AFTER convert_to_sim at each step
  - The actual simulator joint state after reset

This helps identify if the COMMON→SIMULATOR reordering is the source of
mismatched legs.

Usage:
    cd D:\Biomotions\newton\ProtoMotions

    ..\.venv\Scripts\python.exe HumanRetargeting\biomechanics_retarget\diagnose_runtime_joints.py ^
        --experiment-path examples\experiments\mimic\mlp.py ^
        --motion-file HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\packaged_data\S_GENERIC.pt ^
        --robot-name smpl_lower_body_subject_S_GENERIC ^
        --simulator newton ^
        --num-envs 1 ^
        --headless ^
        --num-diagnostic-steps 5
"""

import argparse
import sys
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(
        description="Runtime joint diagnostic for kinematic playback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-name", type=str, required=True)
    parser.add_argument("--simulator", type=str, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument("--experiment-path", type=str, required=True)
    parser.add_argument("--scenes-file", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="diagnose_runtime")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--cpu-only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-diagnostic-steps", type=int, default=3,
        help="Number of steps to run and print diagnostics for",
    )
    return parser


parser = create_parser()
args, unknown_args = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402
AppLauncher = import_simulator_before_torch(args.simulator)

import importlib  # noqa: E402
import logging  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)


def deg(t):
    """Tensor to degrees numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy() * 180.0 / np.pi
    return np.asarray(t) * 180.0 / np.pi


def print_header(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    global parser, args

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"Error: Experiment file not found: {experiment_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    args_final = parser.parse_args()

    # Extra simulator parameters
    extra_simulator_params = {}
    if args_final.simulator == "isaaclab":
        app_launcher = AppLauncher(headless=args_final.headless)
        extra_simulator_params["app_launcher"] = app_launcher

    if args_final.seed is not None:
        torch.manual_seed(args_final.seed)
        np.random.seed(args_final.seed)

    from protomotions.utils.config_builder import build_standard_configs
    from protomotions.simulator.base_simulator.config import SimulatorConfig
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.robot_configs.base import RobotConfig

    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")
    configure_robot_and_simulator_fn = getattr(
        experiment_module, "configure_robot_and_simulator", None
    )

    configs = build_standard_configs(
        args=args_final,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=None,
    )

    robot_config: RobotConfig = configs["robot"]
    simulator_config: SimulatorConfig = configs["simulator"]
    terrain_config = configs["terrain"]
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config: EnvConfig = configs["env"]

    if args_final.motion_file is not None:
        motion_lib_config.motion_file = args_final.motion_file
    if args_final.scenes_file is not None:
        scene_lib_config.scene_file = args_final.scenes_file

    from protomotions.envs.control.kinematic_replay_control import KinematicReplayControlConfig
    env_config.show_terrain_markers = False
    env_config.control_components = {"kinematic_replay": KinematicReplayControlConfig()}
    env_config.termination_components = {}
    env_config.observation_components = {}
    env_config.reward_components = {}

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)

    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        save_dir=None,
        **extra_simulator_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    from protomotions.envs.base_env.env import BaseEnv
    env: BaseEnv = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # ===================================================================
    # DIAGNOSTIC: Print joint ordering info
    # ===================================================================
    print_header("JOINT ORDERING DIAGNOSTIC")

    sim = env.simulator

    # Common DOF names (from MJCF parse order)
    common_dof_names = sim._dof_names
    print(f"\nCOMMON dof names (from MJCF, {len(common_dof_names)} dofs):")
    for i, name in enumerate(common_dof_names):
        print(f"  [{i:2d}] {name}")

    # Simulator body ordering
    sim_ordering = sim._get_sim_body_ordering()
    sim_dof_names = sim_ordering.dof_names
    sim_body_names = sim_ordering.body_names
    print(f"\nSIMULATOR dof names (Newton's actual order, {len(sim_dof_names)} dofs):")
    for i, name in enumerate(sim_dof_names):
        print(f"  [{i:2d}] {name}")

    print(f"\nSIMULATOR body names ({len(sim_body_names)} bodies):")
    for i, name in enumerate(sim_body_names):
        print(f"  [{i:2d}] {name}")

    # The conversion mapping
    dc = sim.data_conversion
    print_header("DATA CONVERSION MAPPING")
    print(f"  sim_w_last (quat convention): {dc.sim_w_last}")
    print(f"  dof_convert_to_sim:    {dc.dof_convert_to_sim.cpu().tolist()}")
    print(f"  dof_convert_to_common: {dc.dof_convert_to_common.cpu().tolist()}")
    print(f"  body_convert_to_sim:    {dc.body_convert_to_sim.cpu().tolist()}")
    print(f"  body_convert_to_common: {dc.body_convert_to_common.cpu().tolist()}")

    # Check if reordering is identity
    dof_to_sim = dc.dof_convert_to_sim.cpu().tolist()
    is_identity = (dof_to_sim == list(range(len(dof_to_sim))))
    if is_identity:
        print("\n  ✓ dof_convert_to_sim is IDENTITY - no joint reordering")
    else:
        print("\n  ✗ dof_convert_to_sim is NOT identity - joints ARE reordered!")
        print("\n  Mapping detail (COMMON[i] → SIM position):")
        for i, sim_idx in enumerate(dof_to_sim):
            common_name = common_dof_names[i] if i < len(common_dof_names) else f"?{i}"
            sim_name = sim_dof_names[sim_idx] if sim_idx < len(sim_dof_names) else f"?{sim_idx}"
            match = "✓" if common_name == sim_name else "✗"
            print(f"    COMMON[{i:2d}] {common_name:<20} → SIM[{sim_idx:2d}] {sim_name:<20} {match}")

    body_to_sim = dc.body_convert_to_sim.cpu().tolist()
    body_identity = (body_to_sim == list(range(len(body_to_sim))))
    if body_identity:
        print("\n  ✓ body_convert_to_sim is IDENTITY - no body reordering")
    else:
        print("\n  ✗ body_convert_to_sim is NOT identity - bodies ARE reordered!")

    # ===================================================================
    # DIAGNOSTIC: Run a few steps and compare motion state vs applied state
    # ===================================================================
    print_header("RUNTIME STEP DIAGNOSTIC")

    env.reset()
    env.respawn_root_offset.zero_()

    num_steps = args_final.num_diagnostic_steps
    print(f"Running {num_steps} diagnostic steps...\n")

    for step_i in range(num_steps):
        print(f"\n{'─' * 60}")
        print(f"  STEP {step_i}")
        print(f"{'─' * 60}")

        # Get the reference state that KinematicReplayControl would compute
        motion_times = env.motion_manager.motion_times.clone()
        motion_ids = env.motion_manager.motion_ids.clone()

        ref_state = env.motion_lib.get_motion_state(motion_ids, motion_times)

        # Print COMMON-format dof_pos (what the motion lib returns)
        dof_common = ref_state.dof_pos[0].detach().cpu()
        print(f"\n  Motion state dof_pos (COMMON format, frame time={motion_times[0].item():.4f}s):")
        for i, name in enumerate(common_dof_names):
            val_deg = dof_common[i].item() * 180.0 / np.pi
            print(f"    [{i:2d}] {name:<20} = {dof_common[i].item():8.4f} rad  ({val_deg:8.2f} deg)")

        print(f"\n  Motion state root_pos: {ref_state.rigid_body_pos[0, 0].detach().cpu().numpy()}")
        print(f"  Motion state root_rot (xyzw): {ref_state.rigid_body_rot[0, 0].detach().cpu().numpy()}")

        # Now simulate what convert_to_sim does
        from protomotions.simulator.base_simulator.simulator_state import ResetState
        ref_reset = ResetState.from_robot_state(ref_state.clone())
        dof_before = ref_reset.dof_pos[0].detach().cpu().clone()

        # Apply conversion
        ref_reset_converted = ref_reset.convert_to_sim(dc)
        dof_after = ref_reset_converted.dof_pos[0].detach().cpu()

        if not is_identity:
            print(f"\n  dof_pos AFTER convert_to_sim (SIMULATOR format):")
            for i in range(len(sim_dof_names)):
                sim_name = sim_dof_names[i] if i < len(sim_dof_names) else f"sim_{i}"
                val_deg = dof_after[i].item() * 180.0 / np.pi
                print(f"    [{i:2d}] {sim_name:<20} = {dof_after[i].item():8.4f} rad  ({val_deg:8.2f} deg)")

            # Show the reordering effect
            print(f"\n  Reordering comparison:")
            print(f"    {'COMMON idx':>12} {'COMMON name':<20} {'value(deg)':>12}  →  {'SIM idx':>8} {'SIM name':<20} {'value(deg)':>12}")
            for common_i in range(len(common_dof_names)):
                sim_i = dof_to_sim[common_i]
                c_name = common_dof_names[common_i]
                s_name = sim_dof_names[sim_i] if sim_i < len(sim_dof_names) else "?"
                c_val = dof_before[common_i].item() * 180.0 / np.pi
                s_val = dof_after[sim_i].item() * 180.0 / np.pi
                match = "✓" if abs(c_val - s_val) < 0.01 else "✗ MISMATCH"
                print(f"    {common_i:>12} {c_name:<20} {c_val:12.4f}  →  {sim_i:>8} {s_name:<20} {s_val:12.4f}  {match}")
        else:
            print(f"\n  (dof_convert_to_sim is identity — no reordering applied)")

        # Actually step the env
        actions = torch.zeros(env.num_envs, robot_config.number_of_actions, device=device)
        env.step(actions)

        # Read back actual simulator state
        sim_state = env.simulator.get_robot_state()
        actual_dof = sim_state.dof_pos[0].detach().cpu()

        print(f"\n  Actual simulator dof_pos (after reset_envs):")
        # sim_state is in SIMULATOR format
        for i in range(min(len(sim_dof_names), actual_dof.shape[0])):
            sim_name = sim_dof_names[i] if i < len(sim_dof_names) else f"sim_{i}"
            val_deg = actual_dof[i].item() * 180.0 / np.pi
            print(f"    [{i:2d}] {sim_name:<20} = {actual_dof[i].item():8.4f} rad  ({val_deg:8.2f} deg)")

        # Body positions from simulator
        if sim_state.rigid_body_pos is not None:
            print(f"\n  Actual body positions (world XYZ):")
            rbp = sim_state.rigid_body_pos[0].detach().cpu()
            for i in range(min(len(sim_body_names), rbp.shape[0])):
                b_name = sim_body_names[i] if i < len(sim_body_names) else f"body_{i}"
                print(f"    [{i:2d}] {b_name:<20} = [{rbp[i, 0]:8.4f}, {rbp[i, 1]:8.4f}, {rbp[i, 2]:8.4f}]")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print_header("DIAGNOSTIC SUMMARY")

    if not is_identity:
        print("  ✗ Joint reordering detected between COMMON and SIMULATOR!")
        print("  This could cause legs to look wrong if the mapping is incorrect.")
        print()
        print("  COMMON order (MJCF/motion data):")
        for i, n in enumerate(common_dof_names):
            print(f"    [{i:2d}] {n}")
        print()
        print("  SIMULATOR order (Newton):")
        for i, n in enumerate(sim_dof_names):
            print(f"    [{i:2d}] {n}")
    else:
        print("  ✓ No joint reordering between COMMON and SIMULATOR.")
        print("  If legs still look wrong, the issue is upstream (NPZ/convert/package).")

    if not body_identity:
        print(f"\n  ✗ Body reordering also detected!")
        print(f"  This affects rigid_body_pos/rot interpretation.")
    else:
        print(f"\n  ✓ Body order is consistent.")

    if not dc.sim_w_last:
        print(f"\n  Note: Quaternion conversion xyzw→wxyz is applied for this simulator.")
    else:
        print(f"\n  ✓ Quaternion convention matches (w_last={dc.sim_w_last}).")

    env.close()
    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
