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
"""MaskedMimic transformer experiment with a speed-conditioned prior.

This forks the BeyondMimic-flavored masked mimic baseline into a student that
conditions the prior on a deterministic speed command plus current/history
tokens. The encoder remains privileged and still sees motion-backed masked
pose targets during training.

Deployment mode switches the environment to fixed-speed control only and
refuses to proceed if any motion-backed control or motion file leaks through.
"""

import argparse

from protomotions.robot_configs.base import RobotConfig, ControlType
from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    ObservationNoiseDomainRandomizationConfig,
    PushDomainRandomizationConfig,
    SimulatorConfig,
)
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.masked_mimic.config import MaskedMimicAgentConfig


# Global configuration for masked mimic
NUM_FUTURE_STEPS = 5
TOTAL_STORED_HISTORICAL_STEPS = 3
NUM_HISTORICAL_CONDITIONED_STEPS = 3

DEPLOYMENT_ALLOWED_OBS_KEYS = {
    "noisy_reduced_coords_obs",
    "previous_actions",
    "speed_obs",
    "historical_pose_obs",
}

DEPLOYMENT_FORBIDDEN_OBS_KEYS = {
    "noisy_historical_reduced_coords_obs",
    "noisy_mimic_reduced_coords_target_poses",
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "masked_mimic_target_poses_masks",
    "masked_mimic_target_bodies_masks",
    "max_coords_obs",
    "historical_max_coords_obs",
    "mimic_max_coords_target_poses",
}


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training (e.g., results/mimic_mlp_bm/last.ckpt)",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=1.0,
        help="Fixed forward speed used by SpeedControl during training and deployment.",
    )
    parser.add_argument(
        "--heading-theta",
        type=float,
        default=0.0,
        help="Fixed world-frame heading angle used by SpeedControl.",
    )
    parser.add_argument(
        "--standing-reset-steps",
        type=int,
        default=0,
        help="Number of post-reset steps to hold the speed target at zero.",
    )
    parser.add_argument(
        "--deployment-mode",
        action="store_true",
        default=False,
        help="Build a speed-only deployment config with no motion-backed masked-mimic control.",
    )


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration with low friction settings for BeyondMimic."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=0.01,
            dynamic_friction=0.01,
            restitution=0.0,
            combine_mode=CombineMode.AVERAGE,
        )
    )
    return terrain_cfg


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    deployment_mode = bool(getattr(args, "deployment_mode", False))
    motion_file = getattr(args, "motion_file", None)
    if deployment_mode and motion_file is not None:
        raise ValueError(
            "Deployment mode forbids a motion file. Pass --motion-file only for the training run."
        )
    return MotionLibConfig(motion_file=None if deployment_mode else motion_file)


def _build_speed_control_config(
    args: argparse.Namespace,
    *,
    speed_source: str,
):
    from protomotions.envs.control.speed_control import SpeedControlConfig

    return SpeedControlConfig(
        target_speed=float(getattr(args, "target_speed", 1.0)),
        speed_source=speed_source,
        heading_theta=float(getattr(args, "heading_theta", 0.0)),
        standing_reset_steps=int(getattr(args, "standing_reset_steps", 0)),
    )


def _validate_deployment_env(env_cfg: EnvConfig) -> None:
    if env_cfg is None:
        raise ValueError("Deployment mode requires an environment config")

    control_components = getattr(env_cfg, "control_components", None) or {}
    if "masked_mimic" in control_components:
        raise ValueError(
            "Deployment mode must not instantiate masked_mimic control; use speed-only control instead."
        )
    if "speed" not in control_components:
        raise ValueError("Deployment mode requires speed control to be present")

    observation_components = getattr(env_cfg, "observation_components", None) or {}
    leaked_keys = [
        key for key in observation_components.keys() if key in DEPLOYMENT_FORBIDDEN_OBS_KEYS
    ]
    if leaked_keys:
        raise ValueError(
            f"Deployment mode leaked motion-backed observation components: {sorted(leaked_keys)}"
        )

    missing_required = DEPLOYMENT_ALLOWED_OBS_KEYS.difference(observation_components.keys())
    if missing_required:
        raise ValueError(
            f"Deployment mode is missing required speed-only observation components: {sorted(missing_required)}"
        )


def _configure_training_control_components(args: argparse.Namespace):
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig

    return {
        "speed": _build_speed_control_config(args, speed_source="motion_file"),
        "masked_mimic": MaskedMimicControlConfig(
            num_masked_future_steps=NUM_FUTURE_STEPS,
            num_future_steps=1,
            bootstrap_on_episode_end=True,
            time_alpha=2.0,
            time_beta=5.0,
            repeat_mask_probability=0.8,
            force_max_conditioned_bodies_prob=0.1,
            force_small_num_conditioned_bodies_prob=0.1,
            visible_target_pose_prob=0.8,
        ),
    }


def _configure_deployment_control_components(args: argparse.Namespace):
    return {"speed": _build_speed_control_config(args, speed_source="fixed")}


def _base_observation_components():
    from protomotions.envs.obs import (
        historical_actions_factory,
        historical_max_coords_obs_factory,
        historical_poses_with_time_reduced_coords_factory,
        max_coords_obs_factory,
        previous_actions_factory,
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
    )
    from protomotions.envs.obs.speed_obs_functions import speed_obs_factory

    return {
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(observation_noise=True),
        "noisy_historical_reduced_coords_obs": historical_reduced_coords_obs_factory(
            observation_noise=True
        ),
        "previous_actions": previous_actions_factory(),
        "historical_previous_actions": historical_actions_factory(),
        "speed_obs": speed_obs_factory(),
        "max_coords_obs": max_coords_obs_factory(),
        "historical_max_coords_obs": historical_max_coords_obs_factory(),
    }


def _deployment_observation_components():
    from protomotions.envs.obs import (
        historical_poses_with_time_reduced_coords_factory,
        previous_actions_factory,
        reduced_coords_obs_factory,
    )
    from protomotions.envs.obs.speed_obs_functions import speed_obs_factory

    return {
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(observation_noise=True),
        "previous_actions": previous_actions_factory(),
        "speed_obs": speed_obs_factory(),
        "historical_pose_obs": historical_poses_with_time_reduced_coords_factory(
            num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
            total_stored_historical_steps=TOTAL_STORED_HISTORICAL_STEPS,
        ),
    }


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.terminations import (
        anchor_ori_error_factory,
        anchor_pos_error_factory,
        relative_body_pos_error_factory,
        tracking_error_factory,
    )
    from protomotions.envs.rewards import (
        action_smoothness_physical_factory,
        global_anchor_ori_rew_factory,
        global_anchor_pos_rew_factory,
        global_body_ang_vel_rew_factory,
        global_body_lin_vel_rew_factory,
        relative_body_ori_rew_factory,
        relative_body_pos_rew_factory,
        soft_pos_limit_rew_factory,
    )

    deployment_mode = bool(getattr(args, "deployment_mode", False))

    if deployment_mode:
        control_components = _configure_deployment_control_components(args)
        observation_components = _deployment_observation_components()
        termination_components = {}
        reward_components = {}
        motion_manager = MimicMotionManagerConfig(
            init_start_prob=0.0,
            resample_on_reset=False,
            realign_motion_with_humanoid_on_each_step=False,
        )
        env_cfg = EnvConfig(
            max_episode_length=1000,
            num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,
            control_components=control_components,
            observation_components=observation_components,
            termination_components=termination_components,
            reward_components=reward_components,
            motion_manager=motion_manager,
        )
        _validate_deployment_env(env_cfg)
        return env_cfg

    control_components = _configure_training_control_components(args)

    conditionable_body_ids = [
        robot_cfg.kinematic_info.body_names.index(name)
        for name in robot_cfg.trackable_bodies_subset
    ]

    observation_components = _base_observation_components()

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.masked_mimic.utils import load_expert_configs

        expert_configs = load_expert_configs(expert_model_path)
        expert_env_config = expert_configs["env"]

        expert_history_steps = getattr(expert_env_config, "num_state_history_steps", 0)
        assert TOTAL_STORED_HISTORICAL_STEPS >= expert_history_steps, (
            f"Insufficient history: current={TOTAL_STORED_HISTORICAL_STEPS}, "
            f"expert requires={expert_history_steps}"
        )

        if hasattr(expert_env_config, "control_components") and expert_env_config.control_components:
            for ctrl_cfg in expert_env_config.control_components.values():
                expert_num_future = getattr(ctrl_cfg, "num_future_steps", None)
                if expert_num_future is not None:
                    masked_mimic_cfg = control_components["masked_mimic"]
                    if masked_mimic_cfg.num_future_steps < expert_num_future:
                        masked_mimic_cfg.num_future_steps = expert_num_future

    # Rebuild the masked-mimic target components with the conditionable body ids.
    from protomotions.envs.obs import (
        historical_actions_factory,
        historical_max_coords_obs_factory,
        historical_poses_with_time_reduced_coords_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        masked_mimic_target_poses_factory,
        mimic_target_poses_reduced_coords_factory,
        previous_actions_factory,
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
    )
    from protomotions.envs.obs.general import passthrough_float_factory
    from protomotions.envs.obs.masked_mimic_obs_functions import (
        target_masks_factory,
        target_time_offsets_factory,
    )
    from protomotions.envs.obs.speed_obs_functions import speed_obs_factory

    observation_components.update(
        {
            "noisy_reduced_coords_obs": reduced_coords_obs_factory(observation_noise=True),
            "noisy_historical_reduced_coords_obs": historical_reduced_coords_obs_factory(
                observation_noise=True
            ),
            "noisy_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(
                num_future_steps=1, observation_noise=True
            ),
            "previous_actions": previous_actions_factory(),
            "historical_previous_actions": historical_actions_factory(),
            "speed_obs": speed_obs_factory(),
            "masked_mimic_target_poses": masked_mimic_target_poses_factory(
                conditionable_body_ids=conditionable_body_ids
            ),
            "masked_mimic_target_masks": target_masks_factory(
                conditionable_body_ids=conditionable_body_ids
            ),
            "masked_mimic_target_times": target_time_offsets_factory(),
            "historical_pose_obs": historical_poses_with_time_reduced_coords_factory(
                num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
                total_stored_historical_steps=TOTAL_STORED_HISTORICAL_STEPS,
            ),
            "masked_mimic_target_poses_masks": passthrough_float_factory(
                variable="masked_mimic_target_poses_masks"
            ),
            "masked_mimic_target_bodies_masks": passthrough_float_factory(
                variable="masked_mimic_target_bodies_masks"
            ),
            "max_coords_obs": max_coords_obs_factory(),
            "historical_max_coords_obs": historical_max_coords_obs_factory(),
            "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(
                with_velocities=True
            ),
        }
    )

    termination_components = {
        "bad_ref_pos": anchor_pos_error_factory(threshold=0.5),
        "bad_ref_ori": anchor_ori_error_factory(threshold=0.8),
        "bad_motion_body_pos": relative_body_pos_error_factory(threshold=0.25),
        "tracking_error": tracking_error_factory(threshold=0.25),
    }

    reward_components = {
        "global_anchor_pos": global_anchor_pos_rew_factory(weight=0.5, sigma=0.3),
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
            use_density_weights=True,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
            use_density_weights=True,
        ),
        "body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0, sigma=1.0, use_density_weights=True
        ),
        "body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0, sigma=3.14, use_density_weights=True
        ),
        "action_rate": action_smoothness_physical_factory(weight=-0.1),
        "limits_dof_pos": soft_pos_limit_rew_factory(weight=-100.0),
    }

    return EnvConfig(
        ref_respawn_offset=0.01,
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
    """Build MaskedMimic agent configuration with a speed-conditioned prior."""
    from protomotions.agents.masked_mimic.config import (
        KLDScheduleConfig,
        MaskedMimicModelConfig,
        VaeConfig,
        VaeNoiseType,
    )
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ModuleOperationReshapeConfig,
        ObsProcessorConfig,
        TransformerConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import BiomechanicsEvaluatorConfig

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    encoder_config = ModuleContainerConfig(
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "masked_mimic_target_poses",
            "masked_mimic_target_bodies_masks",
            "masked_mimic_target_times",
            "masked_mimic_target_poses_masks",
        ],
        out_keys=["encoder_mu", "encoder_logvar"],
        models=[
            ObsProcessorConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["noisy_reduced_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["noisy_mimic_reduced_coords_target_poses"],
                out_keys=["noisy_mimic_reduced_coords_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "noisy_reduced_coords_obs_norm",
                    "noisy_mimic_reduced_coords_target_poses_norm",
                    "masked_mimic_target_poses_norm",
                    "masked_mimic_target_bodies_masks",
                    "masked_mimic_target_times_norm",
                    "masked_mimic_target_poses_masks",
                ],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(5)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_logvar"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    prior_config = ModuleContainerConfig(
        in_keys=["noisy_reduced_coords_obs", "speed_obs", "historical_pose_obs"],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            MLPWithConcatConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["current_state_token"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["speed_obs"],
                out_keys=["speed_token"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[
                    MLPLayerConfig(units=transformer_encoder_widths, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            TransformerConfig(
                in_keys=[
                    "current_state_token",
                    "speed_token",
                    "historical_pose_obs_token",
                ],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_logvar"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    trunk_config = ModuleContainerConfig(
        in_keys=["noisy_reduced_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["noisy_reduced_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "noisy_reduced_coords_obs_norm",
                    "previous_actions_norm",
                    "vae_latent",
                ],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=VaeConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VaeNoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = BiomechanicsEvaluatorConfig()

    expert_model_path = getattr(args, "expert_model_path", None)

    agent_config = MaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
    return agent_config


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MaskedMimicAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.envs.obs import reduced_coords_obs_factory
    from protomotions.utils.config_utils import import_experiment_relative_eval_overrides

    deployment_mode = bool(getattr(args, "deployment_mode", False))
    if deployment_mode:
        terrain_cfg.sim_config = TerrainSimConfig(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            combine_mode=CombineMode.AVERAGE,
        )
        simulator_cfg.domain_randomization = None

        if motion_lib_cfg is not None and getattr(motion_lib_cfg, "motion_file", None) is not None:
            raise ValueError(
                "Deployment mode refuses motion-backed configs: motion_lib.motion_file must be None."
            )

        if env_cfg is None:
            raise ValueError("Deployment mode requires an environment config")

        if hasattr(env_cfg, "control_components"):
            control_components = env_cfg.control_components or {}
            if "masked_mimic" in control_components:
                raise ValueError(
                    "Deployment mode must not instantiate masked_mimic control."
                )
            if "speed" not in control_components:
                raise ValueError("Deployment mode must keep speed control enabled")

        if hasattr(env_cfg, "observation_components"):
            env_cfg.observation_components["noisy_reduced_coords_obs"] = (
                reduced_coords_obs_factory(observation_noise=False)
            )
            observation_keys = set((env_cfg.observation_components or {}).keys())
            forbidden = sorted(observation_keys.intersection(DEPLOYMENT_FORBIDDEN_OBS_KEYS))
            if forbidden:
                raise ValueError(
                    "Deployment mode leaked motion-backed observation components: "
                    f"{forbidden}"
                )
            missing = DEPLOYMENT_ALLOWED_OBS_KEYS.difference(observation_keys)
            if missing:
                raise ValueError(
                    "Deployment mode is missing required speed-only observations: "
                    f"{sorted(missing)}"
                )

        if agent_cfg is not None:
            if hasattr(agent_cfg, "expert_model_path"):
                agent_cfg.expert_model_path = None
            agent_cfg.evaluator = EvaluatorConfig()
        return

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides("../mimic/mlp.py")
    apply_inference_overrides_fn(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )

    # Standard inference cleanup for training-mode checkpoints.
    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        agent_cfg.expert_model_path = None
        agent_cfg.evaluator = EvaluatorConfig()


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator with BeyondMimic domain randomization."""
    robot_cfg.control.control_type = ControlType.BUILT_IN_PD
    robot_cfg.control.action_scale = 1.0

    robot_cfg.update_fields(contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"])

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.01, 0.01),
            dof_names=[".*"],
            dof_indices=None,
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.0, 1.0),
            dynamic_friction_range=(0.0, 1.0),
            restitution_range=(0.0, 0.0),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_displacement_range=(-0.01, 0.01),
        ),
        observation_noise=ObservationNoiseDomainRandomizationConfig(
            noise_level=0.01,
            global_root_pos_noise_level=0.005,
            global_root_vel_noise_level=0.01,
            global_body_pos_noise_level=0.005,
            global_body_vel_noise_level=0.01,
            dof_pos_noise_level=0.005,
            dof_vel_noise_level=0.01,
            root_rot_noise_level=0.01,
            root_ang_vel_noise_level=0.01,
            body_rot_noise_level=0.01,
            body_ang_vel_noise_level=0.01,
            contact_body_pos_noise_level=0.005,
            contact_body_vel_noise_level=0.01,
            body_contact_force_noise_level=0.01,
            ground_heights_noise_level=0.005,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_s=(3.0, 5.0),
            push_vel_range=(-0.5, 0.5),
            push_strength_range=(0.5, 1.0),
        ),
    )
