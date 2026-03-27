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
"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=1.0,
        help="Fixed forward speed for speed-only deployment experiments.",
    )
    parser.add_argument(
        "--heading-theta",
        type=float,
        default=0.0,
        help="Fixed world-frame heading angle for speed-only deployment experiments.",
    )
    parser.add_argument(
        "--standing-reset-steps",
        type=int,
        default=0,
        help="Post-reset steps to hold zero speed before activating the target speed.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--deployment-mode",
        action="store_true",
        default=False,
        help="Run the speed-only deployment path and reject motion-backed configs.",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402
import importlib.util  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from dataclasses import asdict  # noqa: E402
from protomotions.utils.config_utils import clean_dict_for_storage  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


def load_experiment_module(experiment_path: Path):
    """Load the checkpoint's copied experiment module for inference overrides."""
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {experiment_path}")

    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)
    return experiment_module


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    if args.deployment_mode and args.motion_file is not None:
        raise ValueError("Deployment mode forbids --motion-file")

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    experiment_config_path = checkpoint.parent / "experiment_config.py"
    experiment_module = None
    if experiment_config_path.exists():
        log.info(f"Loading experiment module from {experiment_config_path}")
        experiment_module = load_experiment_module(experiment_config_path)

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    if args.deployment_mode and experiment_module is None:
        raise ValueError(
            "Deployment mode requires experiment_config.py next to the checkpoint."
        )

    if args.deployment_mode:
        if not hasattr(experiment_module, "motion_lib_config") or not hasattr(
            experiment_module, "env_config"
        ):
            raise ValueError(
                "Deployment mode requires experiment module motion_lib_config() and env_config()."
            )
        motion_lib_config = experiment_module.motion_lib_config(args)
        env_config = experiment_module.env_config(robot_config, args)

    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    if experiment_module is not None:
        apply_inference_overrides_fn = getattr(
            experiment_module, "apply_inference_overrides", None
        )
        if apply_inference_overrides_fn is not None:
            log.info("Applying experiment inference overrides")
            apply_inference_overrides_fn(
                robot_config,
                simulator_config,
                env_config,
                agent_config,
                terrain_config,
                motion_lib_config,
                scene_lib_config,
                args,
            )

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    if args.deployment_mode:
        motion_file = getattr(motion_lib_config, "motion_file", None)
        if motion_file is not None:
            raise ValueError(
                "Deployment mode requires motion_lib.motion_file to be None"
            )
        control_components = getattr(env_config, "control_components", None) or {}
        if "masked_mimic" in control_components:
            raise ValueError(
                "Deployment mode refuses motion-backed masked_mimic control"
            )
        if "speed" not in control_components:
            raise ValueError("Deployment mode requires speed control")

        from protomotions.agents.masked_mimic.model import MaskedMimicModel
        from protomotions.agents.masked_mimic.agent import MaskedMimic

        MaskedMimicModel.forward = MaskedMimicModel.forward_inference

        class _NoOpOptimizer:
            def load_state_dict(self, state_dict):
                return None

            def state_dict(self):
                return {}

        def _deployment_create_optimizers(self, model):
            self.model = model
            self.maskedmimic_optimizer = _NoOpOptimizer()

        MaskedMimic.create_optimizers = _deployment_create_optimizers

    # Create fabric config for inference (simplified)
    fabric_config = FabricConfig(
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.to_fabric_kwargs())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False)

    if args.full_eval:
        agent.evaluator.eval_count = 0
        evaluation_log, evaluated_score = agent.evaluator.evaluate()
        
        # Print evaluation metrics
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key, value in sorted(evaluation_log.items()):
            print(f"  {key}: {value:.6f}")
        print("=" * 60)
        if evaluated_score is not None:
            print(f"  Overall Score: {evaluated_score:.6f}")
        print("=" * 60 + "\n")
    else:
        agent.evaluator.simple_test_policy(collect_metrics=True)


if __name__ == "__main__":
    main()
