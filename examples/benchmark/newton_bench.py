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
"""Benchmark Newton simulator throughput for research loops.

This is intended to be the fixed harness in an autoresearch-style loop:
- the benchmark stays stable
- the agent edits the Newton backend
- throughput is measured the same way every iteration
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from rich.progress import track


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Newton simulator collection throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-name", type=str, default="h1_2")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("sim", "robot_state", "collection_like"),
        default="collection_like",
        help=(
            "Benchmark mode. "
            "'sim' measures simulator.step only, "
            "'robot_state' adds one get_robot_state() call, "
            "'collection_like' mimics repeated collection-time state reads."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root-height", type=float, default=1.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only a single JSON object with benchmark results.",
    )
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_simulator(args: argparse.Namespace) -> tuple[NewtonSimulator, torch.Tensor]:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.terrains.terrain import Terrain
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.newton.config import (
        NewtonSimParams,
        NewtonSimulatorConfig,
    )
    from protomotions.simulator.newton.simulator import NewtonSimulator

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    robot_cfg = robot_config(args.robot_name)
    terrain = Terrain(
        config=TerrainConfig(),
        num_envs=args.num_envs,
        device=device,
    )
    scene_lib = SceneLib.empty(
        num_envs=args.num_envs,
        device=str(device),
        terrain=terrain,
    )
    simulator_config = NewtonSimulatorConfig(
        sim=NewtonSimParams(
            fps=args.fps,
            decimation=args.decimation,
        ),
        headless=True,
        robot=robot_cfg,
        num_envs=args.num_envs,
        experiment_name=f"newton_bench_{args.mode}",
    )

    simulator = NewtonSimulator(
        config=simulator_config,
        robot_config=robot_cfg,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
    )
    simulator._initialize_with_markers({})

    default_state = simulator.get_default_robot_reset_state()
    root_pos = torch.zeros(args.num_envs, 3, device=device)
    root_pos[:, :2] = terrain.sample_valid_locations(args.num_envs).view(-1, 2)
    root_pos[:, 2] = args.root_height
    default_state.root_pos[:] = root_pos
    simulator.reset_envs(
        default_state,
        env_ids=torch.arange(args.num_envs, device=device),
    )

    actions = torch.randn(
        args.num_envs,
        robot_cfg.number_of_actions,
        device=device,
    )
    return simulator, actions


def _collection_probe(simulator: NewtonSimulator, mode: str) -> None:
    if mode == "sim":
        return
    if mode == "robot_state":
        simulator.get_robot_state()
        return
    if mode == "collection_like":
        simulator.get_robot_state()
        simulator.get_root_state()
        simulator.get_robot_state()
        return
    raise ValueError(f"Unsupported benchmark mode: {mode}")


def _format_results(
    args: argparse.Namespace,
    total_time_s: float,
    simulator: NewtonSimulator,
) -> dict[str, Any]:
    return {
        "benchmark": "newton_collection",
        "mode": args.mode,
        "robot_name": args.robot_name,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "fps": args.fps,
        "decimation": args.decimation,
        "frame_dt": simulator.frame_dt,
        "total_time_s": total_time_s,
        "step_time_ms": (total_time_s / args.steps) * 1000.0,
        "steps_per_s": args.steps / total_time_s,
        "env_steps_per_s": (args.num_envs * args.steps) / total_time_s,
        "device": str(simulator.device),
    }


def main() -> None:
    args = _parse_args()
    simulator, actions = _build_simulator(args)
    device = torch.device(args.device)

    try:
        for _ in track(range(args.warmup_steps), description="Warmup"):
            simulator.step(actions)
            _collection_probe(simulator, args.mode)
        _synchronize(device)

        start_time = time.perf_counter()
        for _ in track(range(args.steps), description="Benchmark"):
            simulator.step(actions)
            _collection_probe(simulator, args.mode)
        _synchronize(device)
        total_time_s = time.perf_counter() - start_time
    finally:
        simulator.close()

    results = _format_results(args, total_time_s, simulator)
    if args.json:
        print(json.dumps(results, sort_keys=True))
        return

    print("\nNewton Benchmark Results")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
