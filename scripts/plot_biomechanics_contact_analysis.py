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
"""Regenerate GRF, normalized CoP, and pressure plots from biomechanics exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_contact_npz(path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    if path.is_dir():
        npz_path = path / "contact_analysis.npz"
        output_dir = path
    else:
        npz_path = path
        output_dir = path.parent

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find contact analysis export at {npz_path}")

    with np.load(npz_path) as data:
        loaded = {key: data[key] for key in data.files}
    return output_dir, loaded


def _plot_contact_waveforms(
    output_path: Path,
    phase: np.ndarray,
    data: dict[str, np.ndarray],
    side_names: list[str],
) -> None:
    fig, axes = plt.subplots(
        len(side_names),
        2,
        figsize=(11.0, 3.6 * max(len(side_names), 1)),
        squeeze=False,
    )

    grf_colors = {
        "x": "tab:red",
        "y": "tab:green",
        "z": "tab:blue",
    }
    cop_colors = {
        "x": "tab:orange",
        "y": "tab:purple",
    }

    for row_idx, side_name in enumerate(side_names):
        grf_ax = axes[row_idx][0]
        cop_ax = axes[row_idx][1]

        for axis_name in ("x", "y", "z"):
            mean = data.get(f"mean__{side_name}_grf_{axis_name}")
            std = data.get(f"std__{side_name}_grf_{axis_name}")
            if mean is None or std is None:
                continue
            grf_ax.plot(
                phase,
                mean,
                linewidth=1.5,
                color=grf_colors[axis_name],
                label=f"GRF {axis_name.upper()}",
            )
            grf_ax.fill_between(
                phase,
                mean - std,
                mean + std,
                color=grf_colors[axis_name],
                alpha=0.18,
            )
        grf_ax.set_title(f"{side_name.title()} foot GRF")
        grf_ax.set_xlim(0.0, 1.0)
        grf_ax.set_xlabel("Cycle phase")
        grf_ax.set_ylabel("Force [N]")
        grf_ax.grid(True, alpha=0.2)
        grf_ax.legend(loc="upper right", fontsize=8)

        for axis_name in ("x", "y"):
            mean = data.get(f"mean__{side_name}_cop_{axis_name}_norm")
            std = data.get(f"std__{side_name}_cop_{axis_name}_norm")
            if mean is None or std is None:
                continue
            cop_ax.plot(
                phase,
                mean,
                linewidth=1.5,
                color=cop_colors[axis_name],
                label=f"CoP {axis_name.upper()}",
            )
            cop_ax.fill_between(
                phase,
                mean - std,
                mean + std,
                color=cop_colors[axis_name],
                alpha=0.18,
            )
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


def _plot_pressure_maps(
    output_path: Path,
    data: dict[str, np.ndarray],
    side_names: list[str],
) -> None:
    max_pressure = 0.0
    for side_name in side_names:
        key = f"mean__{side_name}_pressure_map_pa"
        if key in data:
            max_pressure = max(max_pressure, float(np.max(data[key])))
    vmax = max_pressure if max_pressure > 0.0 else 1.0

    fig, axes = plt.subplots(
        1,
        len(side_names),
        figsize=(5.5 * max(len(side_names), 1), 4.8),
        squeeze=False,
    )

    for col_idx, side_name in enumerate(side_names):
        pressure_map = data.get(f"mean__{side_name}_pressure_map_pa")
        if pressure_map is None:
            raise KeyError(f"Missing pressure map for side {side_name}")
        ax = axes[0][col_idx]
        image = ax.imshow(
            pressure_map,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"{side_name.title()} foot")
        ax.set_xlabel("Normalized fore-aft")
        ax.set_ylabel("Normalized medial-lateral")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Pressure [Pa]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GRF, normalized CoP, and pressure maps from biomechanics exports."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Speed result directory or contact_analysis.npz file.",
    )
    args = parser.parse_args()

    output_dir, data = _load_contact_npz(args.path)
    phase = data["phase"]
    side_names = [str(name) for name in data["side_names"].tolist()]

    _plot_contact_waveforms(output_dir / "contact_waveforms.png", phase, data, side_names)
    _plot_pressure_maps(output_dir / "pressure_maps.png", data, side_names)


if __name__ == "__main__":
    main()
