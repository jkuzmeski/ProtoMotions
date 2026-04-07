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
"""Compare GRF, normalized CoP, and pressure maps across biomechanics runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ContactRun:
    label: str
    path: Path
    data: dict[str, np.ndarray]


def _load_contact_npz(path: Path) -> Path:
    if path.is_dir():
        npz_path = path / "contact_analysis.npz"
    else:
        npz_path = path
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find contact analysis export at {npz_path}")
    return npz_path


def _parse_run_arg(run_arg: str) -> tuple[str, Path]:
    if "=" not in run_arg:
        raise ValueError(
            f"Invalid --run value {run_arg!r}. Use the form <label>=<path>."
        )
    label, raw_path = run_arg.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid --run value {run_arg!r}: empty label.")
    return label, Path(raw_path).expanduser().resolve()


def _load_runs(run_args: list[str]) -> list[ContactRun]:
    runs: list[ContactRun] = []
    for run_arg in run_args:
        label, path = _parse_run_arg(run_arg)
        npz_path = _load_contact_npz(path)
        with np.load(npz_path) as data:
            loaded = {key: data[key] for key in data.files}
        runs.append(ContactRun(label=label, path=npz_path, data=loaded))
    return runs


def _validate_runs(runs: list[ContactRun]) -> tuple[np.ndarray, list[str]]:
    if len(runs) < 2:
        raise ValueError("Provide at least two --run entries to compare.")

    reference_phase = runs[0].data["phase"]
    reference_sides = [str(name) for name in runs[0].data["side_names"].tolist()]
    for run in runs[1:]:
        if not np.array_equal(reference_phase, run.data["phase"]):
            raise ValueError(
                f"Phase grid mismatch between {runs[0].label!r} and {run.label!r}."
            )
        side_names = [str(name) for name in run.data["side_names"].tolist()]
        if side_names != reference_sides:
            raise ValueError(
                f"Foot side mismatch between {runs[0].label!r} and {run.label!r}: "
                f"{reference_sides} vs {side_names}"
            )
    return reference_phase, reference_sides


def _plot_waveform_comparison(
    output_path: Path,
    phase: np.ndarray,
    side_names: list[str],
    runs: list[ContactRun],
) -> None:
    fig, axes = plt.subplots(
        len(side_names),
        2,
        figsize=(12.0, 3.8 * max(len(side_names), 1)),
        squeeze=False,
    )

    colors = list(plt.get_cmap("tab10").colors)
    grf_axes = ("x", "y", "z")
    cop_axes = ("x", "y")

    for side_idx, side_name in enumerate(side_names):
        grf_ax = axes[side_idx][0]
        cop_ax = axes[side_idx][1]

        for run_idx, run in enumerate(runs):
            color = colors[run_idx % len(colors)]
            for axis_name in grf_axes:
                mean = run.data.get(f"mean__{side_name}_grf_{axis_name}")
                std = run.data.get(f"std__{side_name}_grf_{axis_name}")
                if mean is None or std is None:
                    continue
                label = f"{run.label} {axis_name.upper()}"
                grf_ax.plot(
                    phase,
                    mean,
                    linewidth=1.6,
                    color=color,
                    alpha=1.0 if axis_name == "z" else 0.7,
                    linestyle={"x": "--", "y": "-.", "z": "-"}[axis_name],
                    label=label,
                )
                grf_ax.fill_between(
                    phase,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.06 if axis_name == "z" else 0.03,
                )

            for axis_name in cop_axes:
                mean = run.data.get(f"mean__{side_name}_cop_{axis_name}_norm")
                std = run.data.get(f"std__{side_name}_cop_{axis_name}_norm")
                if mean is None or std is None:
                    continue
                label = f"{run.label} CoP {axis_name.upper()}"
                cop_ax.plot(
                    phase,
                    mean,
                    linewidth=1.6,
                    color=color,
                    alpha=1.0 if axis_name == "x" else 0.7,
                    linestyle={"x": "-", "y": "--"}[axis_name],
                    label=label,
                )
                cop_ax.fill_between(
                    phase,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.06 if axis_name == "x" else 0.03,
                )

        grf_ax.set_title(f"{side_name.title()} foot GRF")
        grf_ax.set_xlim(0.0, 1.0)
        grf_ax.set_xlabel("Cycle phase")
        grf_ax.set_ylabel("Force [N]")
        grf_ax.grid(True, alpha=0.2)
        grf_ax.legend(loc="upper right", fontsize=8, ncol=2)

        cop_ax.set_title(f"{side_name.title()} foot normalized CoP")
        cop_ax.set_xlim(0.0, 1.0)
        cop_ax.set_ylim(-1.05, 1.05)
        cop_ax.set_xlabel("Cycle phase")
        cop_ax.set_ylabel("Normalized position")
        cop_ax.grid(True, alpha=0.2)
        cop_ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_pressure_map_comparison(
    output_path: Path,
    side_names: list[str],
    runs: list[ContactRun],
) -> None:
    fig, axes = plt.subplots(
        len(runs),
        len(side_names),
        figsize=(4.8 * max(len(side_names), 1), 3.6 * len(runs)),
        squeeze=False,
    )

    vmax = 0.0
    for run in runs:
        for side_name in side_names:
            key = f"mean__{side_name}_pressure_map_pa"
            if key in run.data:
                vmax = max(vmax, float(np.max(run.data[key])))
    vmax = vmax if vmax > 0.0 else 1.0

    image = None
    for row_idx, run in enumerate(runs):
        for col_idx, side_name in enumerate(side_names):
            ax = axes[row_idx][col_idx]
            pressure_map = run.data.get(f"mean__{side_name}_pressure_map_pa")
            if pressure_map is None:
                pressure_map = np.zeros((1, 1), dtype=np.float32)
            image = ax.imshow(
                pressure_map,
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
            )
            title = f"{run.label}: {side_name.title()}"
            ax.set_title(title)
            ax.set_xlabel("Normalized fore-aft")
            ax.set_ylabel("Normalized medial-lateral")
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)

    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Pressure [Pa]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _compute_summary(runs: list[ContactRun], side_names: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for run in runs:
        run_summary: dict[str, object] = {
            "source": str(run.path),
            "sides": {},
        }
        for side_name in side_names:
            grf_std_xyz = [
                float(np.mean(run.data[f"std__{side_name}_grf_{axis_name}"]))
                for axis_name in ("x", "y", "z")
            ]
            cop_x = run.data[f"mean__{side_name}_cop_x_norm"]
            cop_y = run.data[f"mean__{side_name}_cop_y_norm"]
            cop_valid = run.data.get(f"mean__{side_name}_cop_valid")
            cop_path_length = float(
                np.sum(np.sqrt(np.diff(cop_x) ** 2 + np.diff(cop_y) ** 2))
            )
            pressure_peak = float(
                np.max(run.data.get(f"mean__{side_name}_pressure_map_pa", np.zeros((1, 1))))
            )
            run_summary["sides"][side_name] = {
                "mean_grf_std_x_n": grf_std_xyz[0],
                "mean_grf_std_y_n": grf_std_xyz[1],
                "mean_grf_std_z_n": grf_std_xyz[2],
                "cop_path_length_norm": cop_path_length,
                "cop_valid_fraction": (
                    float(np.mean(cop_valid)) if cop_valid is not None else None
                ),
                "pressure_peak_pa": pressure_peak,
            }
        summary[run.label] = run_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GRF, normalized CoP, and pressure maps across biomechanics runs. "
            "Each --run must be <label>=<speed_result_dir_or_contact_analysis.npz>."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Named run input, for example --run point=results/biomechanics/1p25",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison outputs. Defaults to the current directory.",
    )
    args = parser.parse_args()

    runs = _load_runs(args.run)
    phase, side_names = _validate_runs(runs)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path.cwd().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_waveform_comparison(
        output_dir / "comparison_contact_waveforms.png",
        phase,
        side_names,
        runs,
    )
    _plot_pressure_map_comparison(
        output_dir / "comparison_pressure_maps.png",
        side_names,
        runs,
    )

    summary = _compute_summary(runs, side_names)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
