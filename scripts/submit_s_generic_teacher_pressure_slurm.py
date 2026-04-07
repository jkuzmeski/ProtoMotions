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
#
"""Submit the S_GENERIC pressure-field teacher run through train_slurm.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SUBSET_CHOICES = (
    "all_8",
    "anchor_3",
    "every_other",
    "leave_edge_high",
    "leave_edge_low",
    "loo_15",
    "loo_20",
    "loo_25",
    "loo_30",
    "loo_35",
    "loo_40",
    "loo_45",
    "loo_50",
    "speed_2",
)


def _default_remote_dir_name(subset: str) -> str:
    return f"s_generic_{subset}_pressure_suite"


def _default_experiment_name(subset: str) -> str:
    return f"s_generic_teacher_{subset}_pressure"


def _motion_manifest_path(subset: str) -> str:
    return (
        "HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/"
        f"yaml_data/experiment_matrix/{subset}.yaml"
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the S_GENERIC pressure-field teacher job to Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--user", required=True, help="Cluster username")
    parser.add_argument("--subset", choices=SUBSET_CHOICES, default="every_other")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--remote-dir-name", default=None)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--training-max-steps", type=int, default=10_000_000_000_000)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--slurm-time", default="4:00:00")
    parser.add_argument("--account", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--array-size", type=int, default=5)
    parser.add_argument("--only-upload-code", action="store_true")
    parser.add_argument("--pressure-field-foot-kh", type=float, default=2.5e7)
    parser.add_argument(
        "--pressure-field-foot-sdf-max-resolution", type=int, default=32
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--sync-paths", nargs="*", default=[])
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional raw train_agent.py args appended after the pressure-field "
            "defaults. This option must be last."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()

    subset = args.subset
    experiment_name = args.experiment_name or _default_experiment_name(subset)
    remote_dir_name = args.remote_dir_name or _default_remote_dir_name(subset)

    cmd = [
        sys.executable,
        "protomotions/train_slurm.py",
        "--robot-name=smpl_lower_body_subject_S_GENERIC",
        "--simulator=newton",
        f"--num-envs={args.num_envs}",
        f"--batch-size={args.batch_size}",
        f"--motion-file={_motion_manifest_path(subset)}",
        "--experiment-path=examples/experiments/mimic/mlp_bm_pressure_feet.py",
        f"--experiment-name={experiment_name}",
        f"--user={args.user}",
        f"--training-max-steps={args.training_max_steps}",
        f"--ngpu={args.ngpu}",
        f"--nodes={args.nodes}",
        f"--seed={args.seed}",
        f"--slurm-time={args.slurm_time}",
        f"--array-size={args.array_size}",
        f"--remote-dir-name={remote_dir_name}",
    ]

    if args.account:
        cmd.append(f"--account={args.account}")
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.checkpoint:
        cmd.append(f"--checkpoint={args.checkpoint}")
    if args.use_wandb:
        cmd.append("--use-wandb")
    if args.only_upload_code:
        cmd.append("--only-upload-code")
    if args.sync_paths:
        cmd.extend(["--sync-paths", *args.sync_paths])
    if args.overrides:
        cmd.extend(["--overrides", *args.overrides])

    extra_args = [
        f"--pressure-field-foot-kh={args.pressure_field_foot_kh}",
        f"--pressure-field-foot-sdf-max-resolution={args.pressure_field_foot_sdf_max_resolution}",
        *args.extra_train_args,
    ]
    if extra_args:
        cmd.extend(["--extra-args", *extra_args])

    if args.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
