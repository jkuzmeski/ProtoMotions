#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Utilities for blocking 3D stage comparisons in the biomechanics pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
)

try:
    from .fps_utils import get_resample_indices
except ImportError:
    from fps_utils import get_resample_indices


BODY_NAMES = [
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
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
]
BACKEND_CANDIDATES = ("TkAgg", "QtAgg")
TRACKING_BODY_INDEX = BODY_NAMES.index("Pelvis")
MIN_VIEWPORT_HALF_RANGE = 0.4
VIEWPORT_PADDING = 0.08


def ensure_interactive_matplotlib_backend() -> str:
    """Select an interactive matplotlib backend or raise a clear error."""
    mplconfigdir = os.environ.get("MPLCONFIGDIR")
    if mplconfigdir is None or not os.access(mplconfigdir, os.W_OK):
        cache_dir = tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp")
        os.environ["MPLCONFIGDIR"] = cache_dir

    import matplotlib

    errors: list[str] = []
    for backend in BACKEND_CANDIDATES:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(1, 1))
            plt.close(fig)
            return backend
        except Exception as exc:  # pragma: no cover - environment-specific
            errors.append(f"{backend}: {exc}")
            sys.modules.pop("matplotlib.pyplot", None)

    joined = "\n".join(f"  - {err}" for err in errors)
    raise RuntimeError(
        "No interactive matplotlib backend is available. "
        "Tried TkAgg then QtAgg.\n"
        f"{joined}"
    )


@lru_cache(maxsize=None)
def _body_indices(model_xml: str) -> Tuple[int, ...]:
    kinematic_info = extract_kinematic_info(model_xml)
    return tuple(kinematic_info.body_names.index(name) for name in BODY_NAMES)


def resample_positions(
    positions: np.ndarray,
    source_fps: int,
    target_fps: int,
) -> np.ndarray:
    """Resample position trajectories to a target frame rate."""
    if source_fps == target_fps:
        return np.asarray(positions, dtype=np.float32)

    indices = get_resample_indices(len(positions), source_fps, target_fps)
    return np.asarray(positions[indices], dtype=np.float32)


def load_keypoint_positions(keypoint_file: Path) -> tuple[np.ndarray, int]:
    data = np.load(keypoint_file, allow_pickle=True)
    if data.ndim == 0:
        data = data.item()
    positions = np.asarray(data["positions"], dtype=np.float32)
    fps = int(data.get("fps", 30))
    return positions, fps


def load_retargeted_body_positions(
    retargeted_file: Path,
    model_xml: Path,
) -> tuple[np.ndarray, int]:
    data = np.load(retargeted_file, allow_pickle=True)
    root_pos = torch.from_numpy(np.asarray(data["base_frame_pos"], dtype=np.float32))
    root_rot_wxyz = torch.from_numpy(np.asarray(data["base_frame_wxyz"], dtype=np.float32))
    joint_angles = torch.from_numpy(np.asarray(data["joint_angles"], dtype=np.float32))
    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)

    kinematic_info = extract_kinematic_info(str(model_xml))
    body_indices = list(_body_indices(str(model_xml)))
    root_translation, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    world_positions, _ = compute_forward_kinematics_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_translation,
        joint_rot_mats=joint_rot_mats,
    )
    positions = world_positions[:, body_indices].detach().cpu().numpy().astype(np.float32)
    fps = int(data["fps"]) if "fps" in data else 30
    return positions, fps


def load_motion_body_positions(
    motion_file: Path,
    model_xml: Path,
) -> tuple[np.ndarray, int]:
    data = torch.load(str(motion_file), map_location="cpu", weights_only=False)
    body_indices = list(_body_indices(str(model_xml)))
    positions = data["rigid_body_pos"][:, body_indices].detach().cpu().numpy().astype(np.float32)
    fps = int(data.get("fps", 30))
    return positions, fps


def load_packaged_motion_body_positions(
    packaged_file: Path,
    model_xml: Path,
    motion_index: int,
) -> tuple[np.ndarray, int]:
    data = torch.load(str(packaged_file), map_location="cpu", weights_only=False)
    body_indices = list(_body_indices(str(model_xml)))
    start = int(data["length_starts"][motion_index])
    num_frames = int(data["motion_num_frames"][motion_index])
    end = start + num_frames
    positions = data["gts"][start:end, body_indices].detach().cpu().numpy().astype(np.float32)
    motion_fps = data.get("motion_fps", 30)
    if torch.is_tensor(motion_fps):
        fps = int(motion_fps[motion_index].item())
    else:
        fps = int(motion_fps)
    return positions, fps


def _clip_and_align(
    before_positions: np.ndarray,
    after_positions: np.ndarray,
    fps: int,
    seconds: float,
    start_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_positions = np.asarray(before_positions, dtype=np.float32)
    after_positions = np.asarray(after_positions, dtype=np.float32)

    total_frames = min(len(before_positions), len(after_positions))
    if total_frames == 0:
        raise ValueError("Cannot visualize an empty motion clip.")

    start_frame = max(0, int(round(start_sec * fps)))
    if start_frame >= total_frames:
        raise ValueError(
            f"Visualization start time {start_sec:.2f}s is beyond the available clip "
            f"duration ({total_frames / fps:.2f}s)."
        )

    requested_frames = max(1, int(round(seconds * fps)))
    end_frame = min(total_frames, start_frame + requested_frames)
    frame_indices = np.arange(start_frame, end_frame)
    return before_positions[frame_indices], after_positions[frame_indices], frame_indices


def _compute_tracking_viewport(*position_sets: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute a moving viewport that follows the tracked body through the clip."""
    if not position_sets:
        raise ValueError("At least one position set is required to compute a viewport.")

    relative_positions = []
    for positions in position_sets:
        positions = np.asarray(positions, dtype=np.float32)
        tracking_positions = positions[:, TRACKING_BODY_INDEX : TRACKING_BODY_INDEX + 1, :]
        relative_positions.append(positions - tracking_positions)

    combined_relative = np.concatenate(relative_positions, axis=0)
    min_rel = combined_relative.min(axis=(0, 1))
    max_rel = combined_relative.max(axis=(0, 1))
    center_offset = (min_rel + max_rel) / 2.0
    half_range = max(
        float(np.max(max_rel - min_rel)) / 2.0 + VIEWPORT_PADDING,
        MIN_VIEWPORT_HALF_RANGE,
    )
    return center_offset.astype(np.float32), half_range


def _set_tracking_axes(
    ax,
    tracking_position: np.ndarray,
    center_offset: np.ndarray,
    half_range: float,
) -> None:
    center = np.asarray(tracking_position, dtype=np.float32) + center_offset

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def show_stage_comparison(
    *,
    before_positions: np.ndarray,
    after_positions: np.ndarray,
    before_label: str,
    after_label: str,
    stage_name: str,
    motion_name: str,
    fps: int,
    seconds: float,
    start_sec: float,
) -> None:
    """Show a blocking two-panel 3D animation until the figure is closed."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    before_clip, after_clip, frame_indices = _clip_and_align(
        before_positions=before_positions,
        after_positions=after_positions,
        fps=fps,
        seconds=seconds,
        start_sec=start_sec,
    )

    center_offset, half_range = _compute_tracking_viewport(before_clip, after_clip)
    fig = plt.figure(figsize=(14, 7))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    datasets = [
        (axes[0], before_clip, before_label, "#1f77b4"),
        (axes[1], after_clip, after_label, "#d62728"),
    ]

    artists = []
    for ax, positions, label, color in datasets:
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=18, azim=-65)
        _set_tracking_axes(
            ax,
            tracking_position=positions[0, TRACKING_BODY_INDEX],
            center_offset=center_offset,
            half_range=half_range,
        )

        scat = ax.scatter([], [], [], c=color, s=32)
        lines = [ax.plot([], [], [], color=color, linewidth=2.0)[0] for _ in SKELETON_CONNECTIONS]
        text = ax.text2D(0.03, 0.97, "", transform=ax.transAxes, va="top")
        artists.append((scat, lines, text))

    fig.suptitle(f"{stage_name}: {motion_name}", fontsize=14)
    fig.tight_layout()

    def update(frame_idx: int):
        updated = []
        frame_number = int(frame_indices[frame_idx])
        current_time = frame_number / fps
        for (ax, positions, _, _), (scat, lines, text) in zip(datasets, artists, strict=True):
            current_pos = positions[frame_idx]
            _set_tracking_axes(
                ax,
                tracking_position=current_pos[TRACKING_BODY_INDEX],
                center_offset=center_offset,
                half_range=half_range,
            )
            scat._offsets3d = (current_pos[:, 0], current_pos[:, 1], current_pos[:, 2])
            for line, (start_idx, end_idx) in zip(lines, SKELETON_CONNECTIONS, strict=True):
                line.set_data(
                    [current_pos[start_idx, 0], current_pos[end_idx, 0]],
                    [current_pos[start_idx, 1], current_pos[end_idx, 1]],
                )
                line.set_3d_properties(
                    [current_pos[start_idx, 2], current_pos[end_idx, 2]]
                )
            text.set_text(
                f"Frame {frame_number} | t={current_time:.2f}s | "
                f"pelvis y={current_pos[TRACKING_BODY_INDEX, 1]:.3f}m"
            )
            updated.extend([scat, *lines, text])
        return updated

    ani = FuncAnimation(
        fig,
        update,
        frames=len(before_clip),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    fig._pipeline_animation = ani
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)
