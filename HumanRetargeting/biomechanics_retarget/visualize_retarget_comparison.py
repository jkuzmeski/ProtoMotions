#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Visualize source lower-body keypoints against retargeted robot motion.

This is a debugging view for the lower-body biomechanics pipeline. It renders the
source 9-point lower-body skeleton next to the retargeted robot body origins so it
is easy to tell whether a bad-looking Newton playback is already present in the
retargeted pose sequence or is being introduced later by asset geometry/rendering.

Usage:
    python visualize_retarget_comparison.py \
        HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/keypoints/S02_15ms_Long.npy \
        HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/retargeted_motions/S02_15ms_Long_retargeted.npz \
        --model-xml protomotions/data/assets/mjcf/smpl_humanoid_lower_body_subject_S_GENERIC.xml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    raise SystemExit(1)


app = typer.Typer(pretty_exceptions_enable=False)

KEYPOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
]

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
]


def _load_source_positions(source_file: Path) -> np.ndarray:
    data = np.load(source_file, allow_pickle=True)
    if getattr(data, "shape", None) == () and getattr(data, "dtype", None) == object:
        data = data.item()

    if isinstance(data, np.ndarray):
        positions = data
    elif isinstance(data, dict) or hasattr(data, "files"):
        if "positions" in data:
            positions = data["positions"]
        elif "keypoints" in data:
            positions = data["keypoints"]
        else:
            raise KeyError(
                f"Could not find 'positions' or 'keypoints' in {source_file}"
            )
    else:
        raise TypeError(f"Unsupported source file format for {source_file}")

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (9, 3):
        raise ValueError(
            f"Expected source positions with shape (T, 9, 3), got {positions.shape}"
        )
    return positions


def _load_retargeted_positions(retargeted_file: Path, model_xml: Path) -> np.ndarray:
    data = np.load(retargeted_file, allow_pickle=True)
    required = ["base_frame_pos", "base_frame_wxyz", "joint_angles"]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing keys in {retargeted_file}: {missing}")

    qpos = np.concatenate(
        [data["base_frame_pos"], data["base_frame_wxyz"], data["joint_angles"]],
        axis=1,
    ).astype(np.float32, copy=False)

    kinematic_info = extract_kinematic_info(str(model_xml))
    robot_state = fk_batch_mjcf_with_velocities(
        kinematic_info,
        torch.from_numpy(qpos),
        fps=30,
        compute_velocities=False,
    )
    body_indices = [kinematic_info.body_names.index(name) for name in KEYPOINT_NAMES]
    return robot_state.rigid_body_pos[:, body_indices].cpu().numpy()


def _compute_travel_directions(pelvis_positions: np.ndarray) -> np.ndarray:
    travel = np.gradient(pelvis_positions, axis=0).astype(np.float32, copy=False)
    travel[:, 2] = 0.0
    norm = np.linalg.norm(travel[:, :2], axis=1, keepdims=True)
    direction = np.zeros_like(travel)
    direction[:, :2] = np.divide(travel[:, :2], np.maximum(norm, 1e-8))
    return direction


def _print_diagnostics(source_positions: np.ndarray, retargeted_positions: np.ndarray) -> None:
    direction = _compute_travel_directions(source_positions[:, 0])

    def _toe_projection_stats(
        positions: np.ndarray, ankle_idx: int, toe_idx: int
    ) -> tuple[float, float]:
        projection = np.sum(
            (positions[:, toe_idx] - positions[:, ankle_idx]) * direction,
            axis=1,
        )
        return float(np.mean(projection)), float(np.mean(projection < 0.0))

    def _ankle_height_stats(positions: np.ndarray) -> tuple[float, float]:
        left = positions[:, 3, 2]
        right = positions[:, 7, 2]
        return float(np.mean((left + right) * 0.5)), float(
            np.mean((left > 0.2) & (right > 0.2))
        )

    for label, positions in [
        ("Source", source_positions),
        ("Retargeted", retargeted_positions),
    ]:
        left_mean, left_backward = _toe_projection_stats(positions, 3, 4)
        right_mean, right_backward = _toe_projection_stats(positions, 7, 8)
        ankle_mean, both_airborne = _ankle_height_stats(positions)
        print(f"{label} diagnostics:")
        print(
            f"  left toe-forward projection mean:  {left_mean:.4f} m, "
            f"backward fraction: {left_backward:.3f}"
        )
        print(
            f"  right toe-forward projection mean: {right_mean:.4f} m, "
            f"backward fraction: {right_backward:.3f}"
        )
        print(
            f"  mean ankle height: {ankle_mean:.4f} m, "
            f"both ankles > 0.2 m fraction: {both_airborne:.3f}"
        )


def _compute_plot_bounds(
    source_positions: np.ndarray, retargeted_positions: np.ndarray
) -> tuple[np.ndarray, float]:
    all_positions = np.concatenate([source_positions, retargeted_positions], axis=0)
    min_vals = np.min(all_positions, axis=(0, 1))
    max_vals = np.max(all_positions, axis=(0, 1))
    mid_vals = (min_vals + max_vals) * 0.5
    max_range = float(np.max(max_vals - min_vals))
    max_range = max(max_range, 0.5)
    return mid_vals, max_range


def _setup_axis(ax, title: str, mid_vals: np.ndarray, max_range: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(mid_vals[0] - max_range * 0.5, mid_vals[0] + max_range * 0.5)
    ax.set_ylim(mid_vals[1] - max_range * 0.5, mid_vals[1] + max_range * 0.5)
    ax.set_zlim(0.0, mid_vals[2] + max_range * 0.5)

    grid_x, grid_y = np.meshgrid(
        np.linspace(mid_vals[0] - max_range * 0.5, mid_vals[0] + max_range * 0.5, 10),
        np.linspace(mid_vals[1] - max_range * 0.5, mid_vals[1] + max_range * 0.5, 10),
    )
    grid_z = np.zeros_like(grid_x)
    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.15, color="gray")


def _render_skeleton(ax, positions: np.ndarray, color: str):
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=color, s=24)
    lines = []
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        line = ax.plot(
            [positions[start_idx, 0], positions[end_idx, 0]],
            [positions[start_idx, 1], positions[end_idx, 1]],
            [positions[start_idx, 2], positions[end_idx, 2]],
            color=color,
            linewidth=2,
        )[0]
        lines.append(line)
    return scatter, lines


@app.command()
def main(
    source_file: Path = typer.Argument(..., exists=True, help="Path to extracted keypoints .npy"),
    retargeted_file: Path = typer.Argument(..., exists=True, help="Path to retargeted .npz"),
    model_xml: Path = typer.Option(..., "--model-xml", exists=True, help="Lower-body MJCF used for FK"),
    fps: int = typer.Option(30, "--fps", min=1, help="Playback FPS"),
    start_frame: int = typer.Option(0, "--start-frame", min=0, help="First frame to display"),
    num_frames: Optional[int] = typer.Option(None, "--num-frames", min=1, help="Limit displayed frames"),
    save_video: bool = typer.Option(False, "--save", help="Save animation to an MP4 next to the retargeted file"),
    snapshot_path: Optional[Path] = typer.Option(None, "--snapshot-path", help="Save a single frame to PNG and exit"),
    snapshot_frame: int = typer.Option(0, "--snapshot-frame", min=0, help="Frame index for --snapshot-path"),
):
    """Visualize source keypoints next to retargeted robot body origins."""
    source_positions = _load_source_positions(source_file)
    retargeted_positions = _load_retargeted_positions(retargeted_file, model_xml)

    frame_count = min(len(source_positions), len(retargeted_positions))
    if frame_count == 0:
        raise ValueError("No frames available to visualize")

    end_frame = frame_count if num_frames is None else min(frame_count, start_frame + num_frames)
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid frame window: start_frame={start_frame}, end_frame={end_frame}"
        )

    source_positions = source_positions[start_frame:end_frame]
    retargeted_positions = retargeted_positions[start_frame:end_frame]
    num_frames_to_draw = source_positions.shape[0]

    print(f"Source frames: {len(source_positions)}")
    print(f"Retargeted frames: {len(retargeted_positions)}")
    _print_diagnostics(source_positions, retargeted_positions)

    mid_vals, max_range = _compute_plot_bounds(source_positions, retargeted_positions)

    fig = plt.figure(figsize=(14, 6))
    source_ax = fig.add_subplot(121, projection="3d")
    retarget_ax = fig.add_subplot(122, projection="3d")
    _setup_axis(source_ax, "Source Keypoints", mid_vals, max_range)
    _setup_axis(retarget_ax, "Retargeted Robot Links", mid_vals, max_range)

    source_scatter, source_lines = _render_skeleton(source_ax, source_positions[0], "tab:blue")
    retarget_scatter, retarget_lines = _render_skeleton(retarget_ax, retargeted_positions[0], "tab:orange")

    frame_text = fig.text(0.5, 0.02, "", ha="center")

    def _update_artist_positions(scatter, lines, positions: np.ndarray):
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        for line, (start_idx, end_idx) in zip(lines, SKELETON_CONNECTIONS):
            line.set_data(
                [positions[start_idx, 0], positions[end_idx, 0]],
                [positions[start_idx, 1], positions[end_idx, 1]],
            )
            line.set_3d_properties(
                [positions[start_idx, 2], positions[end_idx, 2]]
            )

    def update(frame_idx: int):
        _update_artist_positions(source_scatter, source_lines, source_positions[frame_idx])
        _update_artist_positions(retarget_scatter, retarget_lines, retargeted_positions[frame_idx])
        frame_text.set_text(
            f"Frame {start_frame + frame_idx + 1}/{start_frame + num_frames_to_draw}"
        )
        return [source_scatter, retarget_scatter, frame_text, *source_lines, *retarget_lines]

    if snapshot_path is not None:
        frame_idx = min(snapshot_frame, num_frames_to_draw - 1)
        update(frame_idx)
        fig.savefig(snapshot_path, dpi=160, bbox_inches="tight")
        print(f"Saved snapshot to {snapshot_path}")
        return

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames_to_draw,
        interval=1000 / fps,
        blit=False,
    )

    if save_video:
        output_path = retargeted_file.with_suffix(".comparison.mp4")
        animation.save(output_path, writer="ffmpeg", fps=fps)
        print(f"Saved animation to {output_path}")
        return

    plt.show()


if __name__ == "__main__":
    app()
