#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot predicted-vs-reference joint traces and gait cycles from a full-eval artifact."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from protomotions.components.pose_lib import extract_kinematic_info


REPO_ROOT = Path(__file__).resolve().parents[1]
AXIS_ORDER = ("x", "y", "z")
SIDE_ORDER = ("R", "L")
SIDE_LABELS = {"R": "right", "L": "left"}
CONTACT_BODY_NAMES = {
    "R": ("R_Ankle", "R_Toe"),
    "L": ("L_Ankle", "L_Toe"),
}
DOF_NAME_PATTERN = re.compile(r"^(?P<side>[LR])_(?P<joint>[^_]+)_(?P<axis>[xyz])$")


@dataclass
class MotionSequence:
    name: str
    source_path: Path
    dt: float
    dof_pos: np.ndarray
    contacts: np.ndarray


@dataclass
class CycleCollection:
    predicted: list[np.ndarray]
    reference: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate joint-centric predicted-vs-reference timeseries and gait-cycle "
            "plots from a predicted MotionLib saved by full-eval."
        )
    )
    parser.add_argument(
        "--predicted-motion-lib",
        required=True,
        help="Path to predicted_motion_lib_epoch_*.pt saved by full-eval.",
    )
    parser.add_argument(
        "--reference-motion-source",
        default=None,
        help=(
            "Reference motion source (.yaml, .motion, or packaged .pt). "
            "If omitted, infer it from resolved_configs_inference.yaml."
        ),
    )
    parser.add_argument(
        "--model-xml",
        default=None,
        help=(
            "MJCF XML used to recover DOF and body names. "
            "If omitted, infer it from resolved_configs_inference.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for generated plots. Defaults to "
            "<run_dir>/results/joint_angle_plots/<predicted_stem>."
        ),
    )
    parser.add_argument(
        "--units",
        choices=("degrees", "radians"),
        default="degrees",
        help="Angle units for plots and summary statistics.",
    )
    parser.add_argument(
        "--cycle-points",
        type=int,
        default=101,
        help="Number of normalized samples per gait cycle.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=200,
        help="Output DPI for PNG figures.",
    )
    return parser.parse_args()


def resolve_existing_path(path_str: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.exists():
        return candidate

    for anchor in ("HumanRetargeting", "protomotions", "results"):
        if anchor in candidate.parts:
            anchor_idx = candidate.parts.index(anchor)
            repaired = REPO_ROOT.joinpath(*candidate.parts[anchor_idx:])
            if repaired.exists():
                return repaired.resolve()

    raise FileNotFoundError(f"Could not resolve existing path for {path_str}")


def infer_run_dir(predicted_motion_lib: Path) -> Path:
    search_roots = [
        predicted_motion_lib.parent,
        predicted_motion_lib.parent.parent,
        predicted_motion_lib.parent.parent.parent,
    ]
    for root in search_roots:
        if (root / "resolved_configs_inference.yaml").exists():
            return root
        if (root / "resolved_configs.yaml").exists():
            return root
    raise FileNotFoundError(
        f"Could not locate resolved config near {predicted_motion_lib}"
    )


def load_run_config(run_dir: Path) -> dict[str, Any]:
    for file_name in ("resolved_configs_inference.yaml", "resolved_configs.yaml"):
        config_path = run_dir / file_name
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(f"No resolved config found under {run_dir}")


def infer_reference_motion_source(run_config: dict[str, Any]) -> Path:
    return resolve_existing_path(run_config["motion_lib"]["motion_file"], base_dir=REPO_ROOT)


def infer_model_xml(run_config: dict[str, Any]) -> Path:
    asset_root = resolve_existing_path(run_config["robot"]["asset"]["asset_root"])
    return resolve_existing_path(run_config["robot"]["asset"]["asset_file_name"], base_dir=asset_root)


def load_packaged_motion_lib(path: Path) -> list[MotionSequence]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    motion_files = data.get(
        "motion_files",
        tuple(f"motion_{idx}" for idx in range(len(data["motion_num_frames"]))),
    )
    sequences: list[MotionSequence] = []
    for motion_idx, raw_path in enumerate(motion_files):
        start = int(data["length_starts"][motion_idx].item())
        num_frames = int(data["motion_num_frames"][motion_idx].item())
        stop = start + num_frames
        sequences.append(
            MotionSequence(
                name=Path(raw_path).name,
                source_path=Path(raw_path),
                dt=float(data["motion_dt"][motion_idx].item()),
                dof_pos=data["dps"][start:stop].cpu().numpy(),
                contacts=data["contacts"][start:stop].cpu().numpy().astype(bool),
            )
        )
    return sequences


def load_single_motion_file(path: Path) -> MotionSequence:
    data = torch.load(path, map_location="cpu", weights_only=False)
    fps = data.get("fps")
    dof_pos = data.get("dof_pos")
    contacts = data.get("rigid_body_contacts")
    if fps is None or dof_pos is None or contacts is None:
        raise KeyError(
            f"Motion file {path} must contain fps, dof_pos, and rigid_body_contacts"
        )
    return MotionSequence(
        name=path.name,
        source_path=path,
        dt=1.0 / float(fps),
        dof_pos=dof_pos.cpu().numpy(),
        contacts=contacts.cpu().numpy().astype(bool),
    )


def load_yaml_motion_manifest(path: Path) -> list[MotionSequence]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    motions = config.get("motions")
    if not motions:
        raise ValueError(f"No motions found in manifest {path}")
    sequences: list[MotionSequence] = []
    for motion in motions:
        motion_path = resolve_existing_path(motion["file"], base_dir=path.parent)
        sequences.append(load_single_motion_file(motion_path))
    return sequences


def load_motion_source(path: Path) -> list[MotionSequence]:
    if path.is_dir():
        sequences: list[MotionSequence] = []
        for motion_file in sorted(path.glob("*.motion")):
            sequences.append(load_single_motion_file(motion_file))
        if not sequences:
            raise ValueError(f"No .motion files found in directory {path}")
        return sequences
    if path.suffix == ".pt":
        return load_packaged_motion_lib(path)
    if path.suffix == ".yaml":
        return load_yaml_motion_manifest(path)
    if path.suffix == ".motion":
        return [load_single_motion_file(path)]
    raise ValueError(f"Unsupported motion source {path}")


def choose_motion_pairs(
    predicted: list[MotionSequence], reference: list[MotionSequence]
) -> list[tuple[MotionSequence, MotionSequence]]:
    predicted_names = [motion.name for motion in predicted]
    reference_names = [motion.name for motion in reference]

    if predicted_names == reference_names:
        return list(zip(predicted, reference))

    pred_name_counts = Counter(predicted_names)
    ref_name_counts = Counter(reference_names)
    if (
        all(pred_name_counts[name] == 1 for name in predicted_names)
        and all(ref_name_counts[name] == 1 for name in reference_names)
        and set(predicted_names) <= set(reference_names)
    ):
        reference_by_name = {motion.name: motion for motion in reference}
        return [(motion, reference_by_name[motion.name]) for motion in predicted]

    if len(predicted) != len(reference):
        raise ValueError(
            "Could not align predicted and reference motions by name and the counts differ "
            f"({len(predicted)} vs {len(reference)})."
        )

    print("Warning: falling back to index-based motion alignment.")
    return list(zip(predicted, reference))


def sanitize_stem(name: str) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def unit_scale(units: str) -> tuple[float, str]:
    if units == "degrees":
        return 180.0 / math.pi, "deg"
    return 1.0, "rad"


def parse_dof_groups(dof_names: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    groups: dict[str, dict[str, dict[str, int]]] = {}
    for dof_idx, dof_name in enumerate(dof_names):
        match = DOF_NAME_PATTERN.match(dof_name)
        if match is None:
            continue
        side = match.group("side")
        joint = match.group("joint")
        axis = match.group("axis")
        groups.setdefault(joint, {}).setdefault(side, {})[axis] = dof_idx
    return groups


def clip_motion_pair(
    predicted: MotionSequence, reference: MotionSequence
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = min(
        predicted.dof_pos.shape[0],
        reference.dof_pos.shape[0],
        predicted.contacts.shape[0],
        reference.contacts.shape[0],
    )
    predicted_dof = predicted.dof_pos[:num_frames]
    reference_dof = reference.dof_pos[:num_frames]
    reference_contacts = reference.contacts[:num_frames]
    return predicted_dof, reference_dof, reference_contacts


def run_length_encode(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if signal.size == 0:
        return (
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    change_points = np.flatnonzero(signal[1:] != signal[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [signal.size]))
    lengths = ends - starts
    values = signal[starts]
    return values.astype(bool), starts.astype(np.int64), lengths.astype(np.int64)


def smooth_contact_signal(signal: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(signal, dtype=bool).copy()
    if cleaned.size < 5:
        return cleaned

    kernel = np.ones(5, dtype=np.float32)
    majority = np.convolve(cleaned.astype(np.float32), kernel, mode="same")
    cleaned = majority >= 3.0

    for _ in range(4):
        values, starts, lengths = run_length_encode(cleaned)
        if values.size == 0:
            break

        true_lengths = lengths[values]
        false_lengths = lengths[~values]
        min_true = max(
            2,
            int(round((float(np.median(true_lengths)) if true_lengths.size else 4.0) * 0.2)),
        )
        min_false = max(
            2,
            int(round((float(np.median(false_lengths)) if false_lengths.size else 4.0) * 0.2)),
        )

        changed = False
        for run_idx, (value, start, length) in enumerate(zip(values, starts, lengths)):
            threshold = min_true if value else min_false
            if length >= threshold:
                continue

            left_value = values[run_idx - 1] if run_idx > 0 else None
            right_value = values[run_idx + 1] if run_idx + 1 < values.size else None
            if left_value is None and right_value is None:
                continue

            if left_value is not None and right_value is not None and left_value == right_value:
                replacement = left_value
            elif left_value is None:
                replacement = right_value
            elif right_value is None:
                replacement = left_value
            else:
                left_length = lengths[run_idx - 1]
                right_length = lengths[run_idx + 1]
                replacement = left_value if left_length >= right_length else right_value

            if replacement != value:
                cleaned[start : start + length] = replacement
                changed = True

        if not changed:
            break

    return cleaned


def contact_signal_for_side(
    contacts: np.ndarray, body_name_to_index: dict[str, int], side: str
) -> np.ndarray:
    indices = [
        body_name_to_index[name]
        for name in CONTACT_BODY_NAMES[side]
        if name in body_name_to_index
    ]
    if not indices:
        raise KeyError(f"No contact bodies found for side {side}")
    return np.any(contacts[:, indices], axis=1)


def extract_cycle_bounds(contact_signal: np.ndarray) -> list[tuple[int, int]]:
    cleaned = smooth_contact_signal(contact_signal)
    starts = np.flatnonzero(cleaned & ~np.concatenate(([False], cleaned[:-1])))
    if starts.size < 2:
        raw_starts = np.flatnonzero(contact_signal & ~np.concatenate(([False], contact_signal[:-1])))
        starts = raw_starts if raw_starts.size >= 2 else starts
    if starts.size < 2:
        return []

    intervals = np.diff(starts)
    median_interval = float(np.median(intervals))
    min_interval = max(3, int(round(median_interval * 0.5)))
    max_interval = max(min_interval + 1, int(round(median_interval * 1.75)))

    bounds = [
        (int(starts[idx]), int(starts[idx + 1]))
        for idx, interval in enumerate(intervals)
        if min_interval <= interval <= max_interval
    ]
    if bounds:
        return bounds
    return [(int(starts[idx]), int(starts[idx + 1])) for idx in range(starts.size - 1)]


def resample_cycle(trace: np.ndarray, cycle_points: int) -> np.ndarray:
    if trace.size < 2:
        raise ValueError("Cycle trace must have at least 2 samples")
    source_x = np.linspace(0.0, 1.0, num=trace.size, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, num=cycle_points, dtype=np.float64)
    return np.interp(target_x, source_x, trace)


def gather_joint_cycles_for_motion(
    predicted_motion: MotionSequence,
    reference_motion: MotionSequence,
    *,
    joint_sides: dict[str, dict[str, int]],
    body_name_to_index: dict[str, int],
    cycle_points: int,
) -> dict[str, dict[str, CycleCollection]]:
    collections = {
        "right": {axis: CycleCollection(predicted=[], reference=[]) for axis in AXIS_ORDER},
        "left": {axis: CycleCollection(predicted=[], reference=[]) for axis in AXIS_ORDER},
        "both": {axis: CycleCollection(predicted=[], reference=[]) for axis in AXIS_ORDER},
    }

    predicted_dof, reference_dof, reference_contacts = clip_motion_pair(
        predicted_motion, reference_motion
    )
    side_cycles: dict[str, list[tuple[int, int]]] = {}
    for side in SIDE_ORDER:
        signal = contact_signal_for_side(reference_contacts, body_name_to_index, side)
        side_cycles[side] = extract_cycle_bounds(signal)

    for side in SIDE_ORDER:
        side_key = "right" if side == "R" else "left"
        axis_map = joint_sides.get(side, {})
        for cycle_start, cycle_end in side_cycles[side]:
            if cycle_end - cycle_start < 2:
                continue
            for axis, dof_idx in axis_map.items():
                predicted_cycle = resample_cycle(
                    predicted_dof[cycle_start:cycle_end, dof_idx], cycle_points
                )
                reference_cycle = resample_cycle(
                    reference_dof[cycle_start:cycle_end, dof_idx], cycle_points
                )
                collections[side_key][axis].predicted.append(predicted_cycle)
                collections[side_key][axis].reference.append(reference_cycle)
                collections["both"][axis].predicted.append(predicted_cycle)
                collections["both"][axis].reference.append(reference_cycle)

    return collections


def plot_joint_timeseries(
    *,
    motion_name: str,
    joint_name: str,
    joint_sides: dict[str, dict[str, int]],
    predicted_motion: MotionSequence,
    reference_motion: MotionSequence,
    output_path: Path,
    units: str,
    figure_dpi: int,
) -> list[dict[str, Any]]:
    scale, unit_label = unit_scale(units)
    fig, axes = plt.subplots(len(AXIS_ORDER), len(SIDE_ORDER), figsize=(15, 10), sharex=False)
    summary_rows: list[dict[str, Any]] = []
    predicted_dof, reference_dof, _ = clip_motion_pair(predicted_motion, reference_motion)
    time_axis = np.arange(predicted_dof.shape[0], dtype=np.float64) * predicted_motion.dt

    for axis_idx, axis_name in enumerate(AXIS_ORDER):
        for side_idx, side_name in enumerate(SIDE_ORDER):
            axis = axes[axis_idx, side_idx]
            dof_idx = joint_sides.get(side_name, {}).get(axis_name)
            if dof_idx is None:
                axis.axis("off")
                continue

            predicted_trace = predicted_dof[:, dof_idx] * scale
            reference_trace = reference_dof[:, dof_idx] * scale
            axis.plot(time_axis, reference_trace, color="#1f77b4", linewidth=1.1)
            axis.plot(time_axis, predicted_trace, color="#ff7f0e", linewidth=1.0, alpha=0.9)
            rmse_value = float(np.sqrt(np.mean((predicted_trace - reference_trace) ** 2)))
            axis.set_title(
                f"{SIDE_LABELS[side_name]} {axis_name.upper()} | RMSE {rmse_value:.2f} {unit_label}",
                fontsize=10,
            )
            axis.set_xlabel("time (s)", fontsize=9)
            axis.set_ylabel(unit_label, fontsize=9)
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=8)

            summary_rows.append(
                {
                    "motion": motion_name,
                    "joint": joint_name,
                    "plot": "timeseries",
                    "side": SIDE_LABELS[side_name],
                    "axis": axis_name,
                    f"rmse_{unit_label}": rmse_value,
                    "num_segments": 1,
                    "num_cycles": "",
                }
            )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.4, label="reference"),
        Line2D([0], [0], color="#ff7f0e", linewidth=1.4, label="predicted"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.suptitle(
        f"{joint_name} joint angle timeseries | {motion_name}",
        fontsize=13,
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def plot_cycle_panel(
    axis: plt.Axes,
    predicted_cycles: list[np.ndarray],
    reference_cycles: list[np.ndarray],
    *,
    axis_name: str,
    column_title: str,
    unit_label: str,
) -> tuple[float, int]:
    if not predicted_cycles or not reference_cycles:
        axis.text(0.5, 0.5, "no cycles", ha="center", va="center", fontsize=10)
        axis.set_title(column_title, fontsize=10)
        axis.set_xlabel("cycle (%)", fontsize=9)
        axis.set_ylabel(unit_label, fontsize=9)
        axis.grid(alpha=0.2)
        return float("nan"), 0

    predicted = np.asarray(predicted_cycles)
    reference = np.asarray(reference_cycles)
    x_axis = np.linspace(0.0, 100.0, num=predicted.shape[1], dtype=np.float64)

    pred_mean = predicted.mean(axis=0)
    pred_std = predicted.std(axis=0)
    ref_mean = reference.mean(axis=0)
    ref_std = reference.std(axis=0)

    for trace in reference[: min(15, reference.shape[0])]:
        axis.plot(x_axis, trace, color="#1f77b4", alpha=0.08, linewidth=0.8)
    for trace in predicted[: min(15, predicted.shape[0])]:
        axis.plot(x_axis, trace, color="#ff7f0e", alpha=0.08, linewidth=0.8)

    axis.fill_between(x_axis, ref_mean - ref_std, ref_mean + ref_std, color="#1f77b4", alpha=0.18)
    axis.fill_between(x_axis, pred_mean - pred_std, pred_mean + pred_std, color="#ff7f0e", alpha=0.18)
    axis.plot(x_axis, ref_mean, color="#1f77b4", linewidth=2.0)
    axis.plot(x_axis, pred_mean, color="#ff7f0e", linewidth=2.0)

    rmse = float(np.sqrt(np.mean((predicted - reference) ** 2)))
    axis.set_title(f"{column_title} | {axis_name.upper()} | n={predicted.shape[0]} | RMSE {rmse:.2f} {unit_label}", fontsize=10)
    axis.set_xlabel("cycle (%)", fontsize=9)
    axis.set_ylabel(unit_label, fontsize=9)
    axis.grid(alpha=0.25)
    axis.tick_params(labelsize=8)
    return rmse, int(predicted.shape[0])


def plot_joint_cycles(
    *,
    motion_name: str,
    joint_name: str,
    cycle_collections: dict[str, dict[str, CycleCollection]],
    output_path: Path,
    units: str,
    figure_dpi: int,
) -> list[dict[str, Any]]:
    scale, unit_label = unit_scale(units)
    column_keys = ("right", "left", "both")
    column_titles = {
        "right": "right-contact cycles",
        "left": "left-contact cycles",
        "both": "combined left+right cycles",
    }
    fig, axes = plt.subplots(len(AXIS_ORDER), len(column_keys), figsize=(18, 10), sharex=True)
    summary_rows: list[dict[str, Any]] = []

    for axis_idx, axis_name in enumerate(AXIS_ORDER):
        for column_idx, column_key in enumerate(column_keys):
            axis = axes[axis_idx, column_idx]
            collection = cycle_collections[column_key][axis_name]
            predicted_cycles = [cycle * scale for cycle in collection.predicted]
            reference_cycles = [cycle * scale for cycle in collection.reference]
            rmse, num_cycles = plot_cycle_panel(
                axis,
                predicted_cycles,
                reference_cycles,
                axis_name=axis_name,
                column_title=column_titles[column_key],
                unit_label=unit_label,
            )
            summary_rows.append(
                {
                    "motion": motion_name,
                    "joint": joint_name,
                    "plot": "cycles",
                    "side": column_key,
                    "axis": axis_name,
                    f"rmse_{unit_label}": rmse,
                    "num_segments": "",
                    "num_cycles": num_cycles,
                }
            )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.0, label="reference mean"),
        Patch(facecolor="#1f77b4", edgecolor="none", alpha=0.18, label="reference std"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2.0, label="predicted mean"),
        Patch(facecolor="#ff7f0e", edgecolor="none", alpha=0.18, label="predicted std"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.5,
    )
    fig.suptitle(
        f"{joint_name} normalized gait cycles with mean and std | {motion_name}",
        fontsize=13,
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def write_summary_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_joint_angle_plots(
    *,
    predicted_motion_lib: str | Path,
    reference_motion_source: str | Path | None = None,
    model_xml: str | Path | None = None,
    output_dir: str | Path | None = None,
    units: str = "degrees",
    cycle_points: int = 101,
    figure_dpi: int = 200,
) -> Path:
    predicted_motion_lib = resolve_existing_path(predicted_motion_lib, base_dir=REPO_ROOT)
    run_dir = infer_run_dir(predicted_motion_lib)
    run_config = load_run_config(run_dir)

    reference_motion_source = (
        resolve_existing_path(reference_motion_source, base_dir=REPO_ROOT)
        if reference_motion_source is not None
        else infer_reference_motion_source(run_config)
    )
    model_xml = (
        resolve_existing_path(model_xml, base_dir=REPO_ROOT)
        if model_xml is not None
        else infer_model_xml(run_config)
    )

    default_output_dir = (
        run_dir / "results" / "joint_angle_plots" / sanitize_stem(predicted_motion_lib.stem)
    )
    output_dir = Path(output_dir).resolve() if output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kinematic_info = extract_kinematic_info(str(model_xml))
    joint_groups = parse_dof_groups(list(kinematic_info.dof_names))
    body_name_to_index = {name: idx for idx, name in enumerate(kinematic_info.body_names)}

    predicted_sequences = load_packaged_motion_lib(predicted_motion_lib)
    reference_sequences = load_motion_source(reference_motion_source)
    motion_pairs = choose_motion_pairs(predicted_sequences, reference_sequences)

    print(f"Predicted motion lib: {predicted_motion_lib}")
    print(f"Reference motion source: {reference_motion_source}")
    print(f"Model XML: {model_xml}")
    print(f"Output directory: {output_dir}")
    print(f"Motion count: {len(motion_pairs)}")
    print(f"Joint folders: {', '.join(sorted(joint_groups.keys()))}")

    summary_rows: list[dict[str, Any]] = []
    for joint_name in sorted(joint_groups.keys()):
        joint_dir = output_dir / joint_name
        joint_dir.mkdir(parents=True, exist_ok=True)
        joint_sides = joint_groups[joint_name]

        for predicted_motion, reference_motion in motion_pairs:
            motion_dir = joint_dir / sanitize_stem(predicted_motion.name)
            motion_dir.mkdir(parents=True, exist_ok=True)
            cycle_collections = gather_joint_cycles_for_motion(
                predicted_motion,
                reference_motion,
                joint_sides=joint_sides,
                body_name_to_index=body_name_to_index,
                cycle_points=cycle_points,
            )

            summary_rows.extend(
                plot_joint_timeseries(
                    motion_name=predicted_motion.name,
                    joint_name=joint_name,
                    joint_sides=joint_sides,
                    predicted_motion=predicted_motion,
                    reference_motion=reference_motion,
                    output_path=motion_dir / "timeseries.png",
                    units=units,
                    figure_dpi=figure_dpi,
                )
            )
            summary_rows.extend(
                plot_joint_cycles(
                    motion_name=predicted_motion.name,
                    joint_name=joint_name,
                    cycle_collections=cycle_collections,
                    output_path=motion_dir / "cycles.png",
                    units=units,
                    figure_dpi=figure_dpi,
                )
            )
            print(f"Wrote {motion_dir}")

    write_summary_csv(output_dir / "joint_plot_summary.csv", summary_rows)
    print("Done.")
    return output_dir


def main() -> None:
    args = parse_args()
    generate_joint_angle_plots(
        predicted_motion_lib=args.predicted_motion_lib,
        reference_motion_source=args.reference_motion_source,
        model_xml=args.model_xml,
        output_dir=args.output_dir,
        units=args.units,
        cycle_points=args.cycle_points,
        figure_dpi=args.figure_dpi,
    )


if __name__ == "__main__":
    main()
