# -*- coding: utf-8 -*-
"""
Treadmill to Overground Motion Transformation Script

This script processes lower-body motion capture data from a treadmill experiment,
transforms it to appear as if performed overground, and saves the results in
multiple formats (.npy, .txt, .csv, .json).

Key Processing Steps:
1.  Loads joint position data from a text file.
2.  Applies an optional coordinate system transformation (e.g., Y-forward to X-forward).
3.  Adjusts motion vertically to ensure the feet contact the ground plane (z=0).
4.  Transforms treadmill motion to overground motion.
5.  Saves the transformed joint positions and associated metadata.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import typer
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation as R

# --- Constants ---
# This order must match the joint order in the input text files.
JOINT_NAMES = [
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
GROUND_CONTACT_JOINT_NAMES = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")

# --- Core Functions ---


def parse_speed_from_filename(filename: str) -> Optional[float]:
    """
    Extracts treadmill speed from filename.
    """
    pattern = r"S\d+_(\d+)ms_.*\.txt"
    match = re.search(pattern, filename)
    if match:
        speed_ms = int(match.group(1))
        return speed_ms / 10.0
    return None


def apply_position_coordinate_transform(
    joint_centers: np.ndarray, transform_type: str
) -> Tuple[np.ndarray, int]:
    """
    Applies a coordinate system transformation to joint position data.
    """
    forward_axis = 1  # Default: Y-forward
    if transform_type == "y_to_x_forward":
        print("   - Applying coordinate transformation to positions.")
        rot = R.from_euler("z", -90, degrees=True)
        n_frames, n_joints, _ = joint_centers.shape
        joint_centers = rot.apply(joint_centers.reshape(-1, 3)).reshape(
            n_frames, n_joints, 3
        )
        forward_axis = 0  # New: X-forward
    return joint_centers, forward_axis


def load_normalized_motion_from_txt(
    motion_filepath: str,
    coordinate_transform: str,
) -> Tuple[np.ndarray, int]:
    """Load motion data, apply coordinate normalization, and ground it without translation."""
    print(f"   - Loading motion data from {motion_filepath}...")
    try:
        motion_df = pd.read_csv(motion_filepath, sep="\t", header=None)
        joint_centers = motion_df.iloc[:, 1:].to_numpy().reshape(motion_df.shape[0], -1, 3)
        print(f"   - Loaded {joint_centers.shape[0]} frames for {joint_centers.shape[1]} joints.")
    except Exception as e:
        print(f"   - ❌ Error loading motion file {motion_filepath}: {e}")
        raise

    joint_centers, forward_axis = apply_position_coordinate_transform(
        joint_centers, coordinate_transform
    )
    print("   - Applying ground plane adjustment...")
    ground_height = estimate_ground_height(joint_centers)
    joint_centers[:, :, 2] -= ground_height
    print(
        "   - Lowered motion by "
        f"{-ground_height:.3f}m using the 1st percentile of ankle/toe heights."
    )
    return joint_centers, forward_axis


def process_motion_file(
    motion_file: Path,
    output_dir: Path,
    fps: int,
    coordinate_transform: str,
    speed_override: Optional[float] = None,
) -> bool:
    """
    Processes a single motion file.
    """
    print(f"\n📁 Processing motion file: {motion_file.name}")
    treadmill_speed = speed_override
    if treadmill_speed is None:
        treadmill_speed = parse_speed_from_filename(motion_file.name)
    
    if treadmill_speed is None:
        print(f"   - ⚠️ Could not parse or find speed for '{motion_file.name}'. Skipping.")
        return False
    
    print(f"   - Using speed: {treadmill_speed:.1f} m/s")
    try:
        joint_centers = create_motion_from_txt(
            str(motion_file), treadmill_speed, fps, coordinate_transform
        )
        output_base = output_dir / motion_file.stem
        save_motion_data(
            joint_data=joint_centers,
            output_path_base=str(output_base),
            fps=fps,
            transform_applied=coordinate_transform,
        )
        print(f"   - ✅ Successfully processed {motion_file.name}")
        return True
    except Exception as e:
        print(f"   - ❌ Error processing motion file: {e}")
        return False


def calculate_kinematics(
    positions: np.ndarray, fps: int
) -> Tuple[np.ndarray, np.ndarray]:
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    time_delta = 1.0 / fps
    velocities = np.gradient(positions, time_delta, axis=0)
    accelerations = np.gradient(velocities, time_delta, axis=0)
    return velocities, accelerations


def detect_stance_phases(
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    foot_accelerations: np.ndarray,
    height_threshold: float = 0.05,
    vertical_velocity_threshold: float = 0.1,
    horizontal_acceleration_threshold: float = 0.5,
) -> np.ndarray:
    """Detect stance using adaptive per-joint thresholds.

    Uses the same adaptive algorithm as the keypoint extractor so that the
    overground transform and the retarget step agree on contact timing.
    The explicit threshold parameters are kept for backward-compatible call
    sites but are used only as fallback caps.
    """
    del foot_accelerations  # not used by adaptive detector

    clearance = foot_positions[:, 2] - np.percentile(foot_positions[:, 2], 1.0)
    horizontal_speed = np.linalg.norm(foot_velocities[:, :2], axis=1)
    vertical_speed = np.abs(foot_velocities[:, 2])

    adaptive_height = max(
        0.012,
        min(float(height_threshold), float(np.percentile(clearance, 20.0) + 0.008)),
    )
    adaptive_hspeed = max(
        0.2,
        min(1.0, float(np.percentile(horizontal_speed, 15.0) + 0.15)),
    )
    adaptive_vspeed = max(
        0.08,
        min(0.5, float(np.percentile(vertical_speed, 20.0) + 0.05)),
    )

    stance_mask = (
        (clearance <= adaptive_height)
        & (horizontal_speed <= adaptive_hspeed)
        & (vertical_speed <= adaptive_vspeed)
    )
    stance_mask = binary_closing(stance_mask, structure=np.ones(7))
    stance_mask = binary_opening(stance_mask, structure=np.ones(3))
    return stance_mask


def estimate_ground_height(joint_centers: np.ndarray) -> float:
    """Estimate floor height from foot-support joints instead of the global minimum."""
    ground_indices = [JOINT_NAMES.index(name) for name in GROUND_CONTACT_JOINT_NAMES if name in JOINT_NAMES]
    if not ground_indices:
        return float(np.min(joint_centers[:, :, 2]))

    support_heights = joint_centers[:, ground_indices, 2].reshape(-1)
    if support_heights.size == 0:
        return float(np.min(joint_centers[:, :, 2]))

    # Use a low percentile so a single bad frame does not define the floor.
    return float(np.percentile(support_heights, 1.0))


def transform_treadmill_to_overground(
    joint_centers: np.ndarray, fps: int, treadmill_speed: float, forward_axis: int
) -> np.ndarray:
    print("   - 🚀 Applying data-driven treadmill-to-overground transformation...")
    try:
        l_foot_idx, r_foot_idx = (
            JOINT_NAMES.index("L_Ankle"),
            JOINT_NAMES.index("R_Ankle"),
        )
    except ValueError as e:
        print(f"   - ⚠️ Warning: A required joint not found ({e}). Skipping transformation.")
        return joint_centers.copy()

    n_frames = joint_centers.shape[0]
    l_foot_pos, r_foot_pos = joint_centers[:, l_foot_idx, :], joint_centers[:, r_foot_idx, :]
    l_foot_vel, l_foot_accel = calculate_kinematics(l_foot_pos, fps)
    r_foot_vel, r_foot_accel = calculate_kinematics(r_foot_pos, fps)
    l_stance = detect_stance_phases(l_foot_pos, l_foot_vel, l_foot_accel)
    r_stance = detect_stance_phases(r_foot_pos, r_foot_vel, r_foot_accel)

    transformed_centers = joint_centers.copy()
    current_offset, last_offset_update = np.zeros(3), np.zeros(3)

    for i in range(1, n_frames):
        offset_update = np.zeros(3)
        # Determine current stance phase and calculate offset update
        # (Logic for single stance, double stance, and flight)
        # ... [omitted for brevity, same as before]
        in_double_stance = l_stance[i] and r_stance[i]
        in_left_stance = l_stance[i] and not r_stance[i]
        in_right_stance = r_stance[i] and not l_stance[i]
        in_flight = not l_stance[i] and not r_stance[i]

        if in_left_stance:
            offset_update = -(l_foot_pos[i] - l_foot_pos[i - 1])
        elif in_right_stance:
            offset_update = -(r_foot_pos[i] - r_foot_pos[i - 1])
        elif in_double_stance:
            avg_disp = ((l_foot_pos[i] + r_foot_pos[i]) / 2.0) - ((l_foot_pos[i - 1] + r_foot_pos[i - 1]) / 2.0)
            offset_update = -avg_disp
        elif in_flight:
            offset_update = last_offset_update
        
        offset_update[2] = 0.0
        current_offset += offset_update
        transformed_centers[i, :, :] += current_offset
        if not in_flight:
            last_offset_update = offset_update

    time_vector = np.linspace(0, (n_frames - 1) / fps, n_frames)
    forward_displacement = treadmill_speed * time_vector
    transformed_centers[:, :, forward_axis] += forward_displacement[:, np.newaxis]
    print("   - ✅ Transformation applied.")
    return transformed_centers


def create_motion_from_txt(
    motion_filepath: str, treadmill_speed: float, mocap_fr: int, coordinate_transform: str
) -> np.ndarray:
    joint_centers, forward_axis = load_normalized_motion_from_txt(
        motion_filepath, coordinate_transform
    )

    if treadmill_speed > 0:
        joint_centers = transform_treadmill_to_overground(joint_centers, mocap_fr, treadmill_speed, forward_axis)
    else:
        print("   - 🔵 Skipping treadmill transformation (speed is 0).")
    return joint_centers


def save_motion_data(
    joint_data: np.ndarray,
    output_path_base: str,
    fps: int,
    transform_applied: str,
):
    print(f"   - 💾 Saving transformed data to '{output_path_base}.*'")
    output_path = Path(output_path_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_path.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path_base = metadata_dir / output_path.name
    n_frames, n_joints, n_dims = joint_data.shape

    np.save(output_path.with_suffix(".npy"), joint_data)

    with open(metadata_path_base.with_suffix(".txt"), "w") as f:
        flat_data = joint_data.reshape(n_frames, -1)
        for i, row_data in enumerate(flat_data):
            f.write(f"{i + 1}\t" + "\t".join(f"{x:.6f}" for x in row_data) + "\n")

    csv_columns = ["Frame"] + [f"{name}_{ax}" for name in JOINT_NAMES for ax in ["X", "Y", "Z"]]
    csv_data = np.hstack((np.arange(1, n_frames + 1)[:, np.newaxis], joint_data.reshape(n_frames, -1)))
    pd.DataFrame(csv_data, columns=csv_columns).to_csv(metadata_path_base.with_suffix(".csv"), index=False, float_format="%.6f")

    coord_system_desc = ("X=forward, Y=left, Z=up" if transform_applied == "y_to_x_forward" else "X=right, Y=forward, Z=up")
    duration = n_frames / fps if fps > 0 else 0.0
    metadata = {
        "generated_timestamp": datetime.now().isoformat(), "fps": fps,
        "duration_seconds": duration, "num_frames": n_frames, "num_joints": n_joints,
        "joint_names": JOINT_NAMES, "units": "meters",
        "coordinate_system": coord_system_desc, "data_type": "positions",
    }
    with open(metadata_path_base.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("   - ✅ Saved .npy file to main output folder.")
    print("   - ✅ Saved .txt, .csv, and .json files to metadata folder.")


def main(
    input_path: Path = typer.Argument(..., exists=True, help="Path to motion file or folder."),
    output_path: str = typer.Argument(..., help="Base output path (e.g., 'output/')."),
    treadmill_speed: Optional[float] = typer.Option(None, "--speed", "-s", help="Override treadmill speed in m/s."),
    fps: int = typer.Option(200, "--fps", "-f", help="Motion capture frame rate (Hz)."),
    coordinate_transform: str = typer.Option("y_to_x_forward", "--transform", "-t", help="Transform: 'none' or 'y_to_x_forward'."),
):
    print("🏃 Treadmill-to-Overground Motion Transformer 🏃")
    print("=" * 50)
    try:
        output_dir = Path(output_path)
        if input_path.is_file() and output_dir.suffix:
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            if input_path.suffix.lower() != ".txt":
                raise typer.BadParameter("Input file must be a .txt file.")
            success = process_motion_file(input_path, output_dir, fps, coordinate_transform, treadmill_speed)
            if not success:
                raise typer.Exit(code=1)
            print(f"\n🎉 Processing completed! Output in: {output_dir}")

        elif input_path.is_dir():
            print(f"\n🗂️ Processing folder: {input_path}")
            files = sorted(list(input_path.glob("*.txt")))
            if not files:
                print("❌ No .txt files found.")
                raise typer.Exit(code=1)
            
            successful, failed = 0, 0
            for f in files:
                if process_motion_file(f, output_dir, fps, coordinate_transform, treadmill_speed):
                    successful += 1
                else:
                    failed += 1

            print("\n🎉 Batch processing completed!")
            print(f"   - Successfully processed: {successful} files")
            print(f"   - Failed to process: {failed} files")
            print(f"   - Output files created in: {output_dir}")
            if failed > 0:
                print("   - ⚠️ Some files failed. Check logs for details.")
        else:
            raise typer.BadParameter("Input path must be a file or directory.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
