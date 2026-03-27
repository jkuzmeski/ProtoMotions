#!/usr/bin/env python3
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
"""Benchmark Newton rollout collection throughput using deterministic mean actions.

This benchmark follows the same high-level collection path as training:
1. Build the exact experiment/env/agent stack
2. Reset done environments
3. Run policy forward passes
4. Apply deterministic ``mean_action`` PD targets
5. Step the environment and record rollout data

It intentionally excludes optimizer/backprop cost and focuses on local rollout
collection throughput for the Newton simulator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from lightning.fabric import Fabric
from tensordict import TensorDict

from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.utils.component_builder import build_all_components
from protomotions.utils.config_builder import build_standard_configs
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.torch_utils import seeding


DEFAULT_EXPERIMENT_PATH = "examples/experiments/mimic/mlp_bm_bootstrap.py"
DEFAULT_MOTION_FILE = (
    "HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/"
    "yaml_data/experiment_matrix/every_other.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Newton rollout collection throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="smpl_lower_body_subject_S_GENERIC",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        default=DEFAULT_EXPERIMENT_PATH,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="s_generic_teacher_every_other_rollout_bench",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=DEFAULT_MOTION_FILE,
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768,
        help="Required by experiment config; not used for optimization in this benchmark.",
    )
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=32768,
        help="Required by experiment config; rollout benchmark does not train.",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--benchmark-seconds", type=float, default=120.0)
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=32,
        help="Synchronize and check elapsed time every N rollout steps.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/tmp/protomotions_newton_rollout_bench",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional config overrides in key=value form.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only a single JSON object with benchmark results.",
    )
    args = parser.parse_args()
    args.simulator = "newton"
    return args


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_experiment_module(experiment_path: str) -> ModuleType:
    path = Path(experiment_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    spec = importlib.util.spec_from_file_location("benchmark_experiment_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_fabric(args: argparse.Namespace) -> Fabric:
    accelerator = "cuda" if args.device == "cuda" else "cpu"
    fabric = Fabric(accelerator=accelerator, devices=1)
    fabric.launch()
    fabric.seed_everything(args.seed)
    seeding(args.seed, torch_deterministic=False)
    return fabric


def _build_env_and_agent(args: argparse.Namespace) -> tuple[Fabric, Any, Any]:
    experiment_module = _load_experiment_module(args.experiment_path)

    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")
    configure_robot_and_simulator_fn = getattr(
        experiment_module,
        "configure_robot_and_simulator",
        None,
    )
    agent_config_fn = getattr(experiment_module, "agent_config", None)

    configs = build_standard_configs(
        args=args,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=agent_config_fn,
    )

    if args.overrides:
        from protomotions.utils.config_utils import (
            apply_config_overrides,
            parse_cli_overrides,
        )

        cli_overrides = parse_cli_overrides(args.overrides)
        if cli_overrides:
            apply_config_overrides(
                cli_overrides,
                configs["env"],
                configs["simulator"],
                configs["robot"],
                configs["agent"],
                terrain_config=configs["terrain"],
                motion_lib_config=configs["motion_lib"],
                scene_lib_config=configs["scene_lib"],
            )

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        configs["terrain"],
        configs["simulator"],
    )

    fabric = _build_fabric(args)

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=configs["scene_lib"],
        motion_lib_config=configs["motion_lib"],
        simulator_config=simulator_config,
        robot_config=configs["robot"],
        device=fabric.device,
    )

    EnvClass = get_class(configs["env"]._target_)
    env = EnvClass(
        config=configs["env"],
        robot_config=configs["robot"],
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    AgentClass = get_class(configs["agent"]._target_)
    agent = AgentClass(
        config=configs["agent"],
        env=env,
        fabric=fabric,
        root_dir=root_dir,
    )
    agent.setup()
    agent.eval()

    return fabric, env, agent


@torch.no_grad()
def _forward_mean_action(agent: Any, obs_td: TensorDict) -> tuple[TensorDict, torch.Tensor]:
    actor_td = agent.model._actor.mu(obs_td)
    mu = actor_td[agent.model._actor.config.mu_key]
    std = torch.exp(agent.model._actor.logstd)
    neglogp = -torch.distributions.Normal(mu, std).log_prob(mu).sum(dim=-1)

    actor_td["action"] = mu
    actor_td["mean_action"] = mu
    actor_td["neglogp"] = neglogp

    output_td = agent.model._critic(actor_td)
    return output_td, mu


@torch.no_grad()
def _initialize_rollout_buffer(agent: Any) -> None:
    agent.experience_buffer = ExperienceBuffer(
        agent.num_envs,
        agent.num_steps,
        device=agent.device,
    )

    obs = agent.add_agent_info_to_obs(agent.env.get_obs())
    obs_td = agent.obs_dict_to_tensordict(obs)

    for key, env_tensor in obs_td.items():
        agent.experience_buffer.register_key(
            key,
            shape=env_tensor.shape[1:],
            dtype=env_tensor.dtype,
        )

    output_td, _ = _forward_mean_action(agent, obs_td)
    agent.model_output_keys = agent.model.out_keys
    for key in agent.model_output_keys:
        value = output_td[key]
        if value.ndim == 1:
            agent.experience_buffer.register_key(key, dtype=value.dtype)
        else:
            agent.experience_buffer.register_key(
                key,
                shape=value.shape[1:],
                dtype=value.dtype,
            )

    agent.experience_buffer.register_key("rewards")
    if agent.config.normalize_rewards:
        agent.experience_buffer.register_key("unnormalized_rewards")
    agent.experience_buffer.register_key("dones", dtype=torch.long)
    agent.register_algorithm_experience_buffer_keys()


def _reset_experience_buffer_tracking(agent: Any) -> None:
    for key in agent.experience_buffer.store_dict:
        agent.experience_buffer.store_dict[key] = 0


@torch.no_grad()
def _collect_rollout_step(
    agent: Any,
    done_indices: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs, _ = agent.env.reset(done_indices)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    for key, env_tensor in obs_td.items():
        agent.experience_buffer.update_data(key, step, env_tensor)

    output_td, action = _forward_mean_action(agent, obs_td)
    for key in agent.model_output_keys:
        if key in output_td:
            agent.experience_buffer.update_data(key, step, output_td[key])

    agent.check_obs_for_nans(obs_td, action)

    next_obs, rewards, dones, terminated, extras = agent.env.step(action)
    if not torch.all(torch.isfinite(rewards)):
        raise ValueError(f"Non-finite rewards detected: {rewards}")

    next_obs = agent.add_agent_info_to_obs(next_obs)
    next_obs_td = agent.obs_dict_to_tensordict(next_obs)

    dones, terminated, extras = agent.post_env_step_modifications(
        dones,
        terminated,
        extras,
    )
    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

    agent.record_rollout_step(
        next_obs_td,
        action,
        rewards,
        dones,
        terminated,
        done_indices,
        extras,
        step,
    )
    agent.step_count += agent.get_step_count_increment()

    return done_indices, dones.long().sum()


def _run_warmup(agent: Any, warmup_steps: int) -> None:
    agent.fabric.call("before_play_steps", agent)
    done_indices = torch.arange(agent.num_envs, device=agent.device, dtype=torch.long)
    for step_idx in range(warmup_steps):
        if step_idx > 0 and step_idx % agent.num_steps == 0:
            _reset_experience_buffer_tracking(agent)
        done_indices, _ = _collect_rollout_step(agent, done_indices, step_idx % agent.num_steps)


def _run_benchmark(agent: Any, args: argparse.Namespace) -> dict[str, Any]:
    agent.fabric.call("before_play_steps", agent)
    done_indices = torch.arange(agent.num_envs, device=agent.device, dtype=torch.long)
    reset_count = torch.zeros((), device=agent.device, dtype=torch.long)
    rollout_steps = 0

    _synchronize(agent.device)
    start_time = time.perf_counter()

    while True:
        for _ in range(args.sync_interval):
            step_idx = rollout_steps % agent.num_steps
            if rollout_steps > 0 and step_idx == 0:
                _reset_experience_buffer_tracking(agent)
            done_indices, step_resets = _collect_rollout_step(agent, done_indices, step_idx)
            reset_count += step_resets
            rollout_steps += 1

        _synchronize(agent.device)
        elapsed = time.perf_counter() - start_time
        if elapsed >= args.benchmark_seconds:
            break

    total_time_s = time.perf_counter() - start_time
    samples_collected = rollout_steps * agent.num_envs

    return {
        "benchmark": "newton_rollout_mean_action",
        "robot_name": args.robot_name,
        "experiment_path": args.experiment_path,
        "motion_file": args.motion_file,
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "benchmark_seconds_requested": args.benchmark_seconds,
        "benchmark_time_s": total_time_s,
        "rollout_steps": rollout_steps,
        "samples_collected": samples_collected,
        "env_steps_per_s": rollout_steps / total_time_s,
        "samples_per_s": samples_collected / total_time_s,
        "reset_count": int(reset_count.item()),
        "device": str(agent.device),
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = _parse_args()

    setup_start = time.perf_counter()
    fabric = None
    env = None
    try:
        fabric, env, agent = _build_env_and_agent(args)
        _initialize_rollout_buffer(agent)
        _synchronize(agent.device)
        setup_time_s = time.perf_counter() - setup_start

        _run_warmup(agent, args.warmup_steps)
        benchmark_results = _run_benchmark(agent, args)
        benchmark_results["setup_time_s"] = setup_time_s
    finally:
        if env is not None:
            env.close()
        if fabric is not None and fabric.world_size > 1:
            fabric.barrier()

    if args.json:
        print(json.dumps(benchmark_results, sort_keys=True))
        return

    print("\nNewton Rollout Benchmark Results")
    print(json.dumps(benchmark_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
