#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Convert retargeted motion data to ProtoMotions .motion format.

This script converts retargeted motion output (NPZ files) to the
.motion format used by ProtoMotions for training and inference.

Input format (from the retarget step):
    - base_frame_pos: (T, 3) - root position XYZ
    - base_frame_wxyz: (T, 4) - root orientation quaternion WXYZ
    - joint_angles: (T, num_dofs) - joint angles in radians

Output format (.motion file):
    - Dictionary containing full motion state saved with torch.save()

Usage:
    python convert_retargeted_to_motion.py \\
        input.npz output.motion \\
        --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \\
        --input-fps 200 --output-fps 30 --height-offset 0.09

Author: BioMotions Team
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

# --- Environment Setup ---
# Add project root to path to allow importing protomotions
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add data/scripts to path for motion_filter
sys.path.append(str(PROJECT_ROOT / "data" / "scripts"))

# --- Imports ---
try:
    from protomotions.components.pose_lib import (
        extract_kinematic_info,
        fk_from_transforms_with_velocities,
        compute_cartesian_velocity,
        extract_transforms_from_qpos,
        extract_qpos_from_transforms,
    )
except ImportError as e:
    print(f"Error importing ProtoMotions components: {e}")
    print("Ensure you are running this from the correct directory or have installed the package.")
    sys.exit(1)

try:
    from motion_filter import passes_exclude_motion_filter
except ImportError:
    # print("Warning: Could not import motion_filter. Motion filtering will be disabled.")
    passes_exclude_motion_filter = None

try:
    from .fps_utils import get_resample_indices
except ImportError:
    from fps_utils import get_resample_indices

app = typer.Typer(pretty_exceptions_enable=False)


def _stabilize_joint_angles_for_velocity(
    joint_angles: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    """Select the nearest valid periodic branch before differentiating joint angles."""
    raw_angles = joint_angles.detach().cpu().numpy().astype(np.float32, copy=False)
    lower = lower_limits.detach().cpu().numpy().astype(np.float32, copy=False)
    upper = upper_limits.detach().cpu().numpy().astype(np.float32, copy=False)

    wrapped = ((raw_angles + np.pi) % (2.0 * np.pi)) - np.pi
    stabilized = np.empty_like(wrapped)
    branch_offsets = np.arange(-2, 3, dtype=np.float32) * (2.0 * np.pi)

    for dof_idx in range(wrapped.shape[1]):
        lower_limit = lower[dof_idx]
        upper_limit = upper[dof_idx]
        initial_target = 0.5 * (lower_limit + upper_limit)

        for frame_idx in range(wrapped.shape[0]):
            base_angle = wrapped[frame_idx, dof_idx]
            candidates = base_angle + branch_offsets
            valid_candidates = candidates[
                (candidates >= lower_limit - 1e-5) & (candidates <= upper_limit + 1e-5)
            ]
            target = initial_target if frame_idx == 0 else stabilized[frame_idx - 1, dof_idx]

            if valid_candidates.size == 0:
                chosen = np.clip(base_angle, lower_limit, upper_limit)
            else:
                chosen = valid_candidates[np.argmin(np.abs(valid_candidates - target))]
            stabilized[frame_idx, dof_idx] = chosen

    return torch.from_numpy(stabilized).to(
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    )


def _unwrap_joint_angles_for_motion(joint_angles: torch.Tensor) -> torch.Tensor:
    """Unwrap per-DOF 2pi branch crossings while preserving the exact pose."""
    raw_angles = joint_angles.detach().cpu().numpy().astype(np.float32, copy=False)
    unwrapped = np.unwrap(raw_angles, axis=0)
    return torch.from_numpy(unwrapped).to(
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    )


def load_npz_file(
    npz_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    input_fps: int,
    output_fps: int,
    target_joint_names: Optional[list] = None,
):
    """Load retargeted motion from NPZ file and resample."""
    data = np.load(npz_path, allow_pickle=True)

    base_pos = data["base_frame_pos"]
    num_frames = base_pos.shape[0]
    
    # Get resampling indices
    indices = get_resample_indices(num_frames, input_fps, output_fps)
    
    # Extract and resample
    root_pos = torch.from_numpy(data["base_frame_pos"][indices]).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(data["base_frame_wxyz"][indices]).to(device, dtype)
    
    raw_joint_angles = torch.from_numpy(data["joint_angles"][indices]).to(device, dtype)
    
    # Reorder joints if names are provided
    if "joint_names" in data and target_joint_names is not None:
        source_names = data["joint_names"].tolist()
        
        if len(source_names) == 0:
            print("Warning: Source motion has empty joint names. Skipping reordering.")
            joint_angles = raw_joint_angles
        else:
            # Handle bytes vs string
            if isinstance(source_names[0], bytes):
                source_names = [n.decode("utf-8") for n in source_names]

            if source_names == target_joint_names:
                joint_angles = raw_joint_angles
            else:
                missing_targets = [
                    target_name
                    for target_name in target_joint_names
                    if target_name not in source_names
                ]
                unexpected_sources = [
                    source_name
                    for source_name in source_names
                    if source_name not in target_joint_names
                ]
                if missing_targets or unexpected_sources:
                    raise ValueError(
                        "Joint-name mismatch between retargeted motion and target model. "
                        f"Missing target joints: {missing_targets}. "
                        f"Unexpected source joints: {unexpected_sources}."
                    )

                print(
                    f"Reordering joints from {len(source_names)} source to "
                    f"{len(target_joint_names)} target..."
                )

                reorder_indices = [source_names.index(target_name) for target_name in target_joint_names]
                joint_angles = raw_joint_angles[:, reorder_indices]
    else:
        joint_angles = raw_joint_angles
    
    return root_pos, root_rot_wxyz, joint_angles


def load_contact_labels(
    contact_file: Path,
    motion_length: int,
    left_contact_indices: list[int],
    right_contact_indices: list[int],
    num_bodies: int,
    device: torch.device,
    input_fps: int,
    output_fps: int,
):
    """Load and format contact labels from NPZ file."""
    contact_data = np.load(contact_file, allow_pickle=True)
    # Preferred format preserves ankle/toe channels separately:
    #   left_foot_contacts:  [T, 2]  (ankle, toe)
    #   right_foot_contacts: [T, 2]  (ankle, toe)
    # Older files only have:
    #   foot_contacts: [T, 2] (left/right scalar)
    left_contacts = None
    right_contacts = None
    if "left_foot_contacts" in contact_data and "right_foot_contacts" in contact_data:
        left_contacts = np.asarray(contact_data["left_foot_contacts"])
        right_contacts = np.asarray(contact_data["right_foot_contacts"])
    elif "foot_contacts" in contact_data:
        foot_contacts = np.asarray(contact_data["foot_contacts"])
        left_contacts = foot_contacts[:, [0]]
        right_contacts = foot_contacts[:, [1]]
    else:
        raise KeyError(
            f"Unsupported contact file format for {contact_file}. "
            "Expected left/right foot contact arrays or foot_contacts."
        )

    def _resample_and_align(contacts: np.ndarray) -> np.ndarray:
        contact_frames = contacts.shape[0]
        indices = get_resample_indices(contact_frames, input_fps, output_fps)
        contacts = contacts[indices]

        contact_length = contacts.shape[0]
        if contact_length != motion_length:
            print(
                f"Warning: Contact length ({contact_length}) != motion length ({motion_length}) "
                "after resampling."
            )
            if contact_length > motion_length:
                contacts = contacts[:motion_length]
            else:
                padding = np.repeat(contacts[-1:], motion_length - contact_length, axis=0)
                contacts = np.concatenate([contacts, padding], axis=0)
        return contacts

    left_contacts = _resample_and_align(left_contacts)
    right_contacts = _resample_and_align(right_contacts)

    # The sidecar stores smoothed contact values (for this dataset often 0.2, 0.4, ...).
    # ProtoMotions smooths contacts again on load, so binarize here with a low threshold.
    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)

    def _assign_body_contacts(body_indices: list[int], contact_values: np.ndarray) -> None:
        if contact_values.ndim == 1:
            contact_values = contact_values[:, None]

        if len(body_indices) == 0:
            return

        if contact_values.shape[1] == 1:
            body_contact_columns = [contact_values[:, 0]] * len(body_indices)
        else:
            body_contact_columns = []
            for idx in range(len(body_indices)):
                source_col = min(idx, contact_values.shape[1] - 1)
                body_contact_columns.append(contact_values[:, source_col])

        for body_idx, column in zip(body_indices, body_contact_columns):
            rigid_body_contacts[:, body_idx] = column > 0.1

    _assign_body_contacts(left_contact_indices, left_contacts)
    _assign_body_contacts(right_contact_indices, right_contacts)

    return torch.from_numpy(rigid_body_contacts).to(device)


def convert_npz_to_motion(
    npz_file: Path,
    output_file: Path,
    model_xml: Path,
    input_fps: int = 30,
    output_fps: int = 30,
    contact_file: Optional[Path] = None,
    ignore_first_n_frames: int = 0,
    height_offset: float = 0.0,  # No offset - let motion determine ground height
    apply_motion_filter: bool = False,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.1,
    duration_height_seconds: float = 0.6,
) -> bool:
    """
    Convert a retargeted NPZ file to ProtoMotions .motion format.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Extract kinematic info from model
    kinematic_info = extract_kinematic_info(str(model_xml))
    
    # Get DOF names from kinematic info (excluding root)
    # Note: kinematic_info.dof_names ALREADY excludes root DOFs
    # in current pose_lib implementation
    target_dof_names = kinematic_info.dof_names

    print(f"Loading motion from: {npz_file}")
    root_pos, root_rot_wxyz, joint_angles = load_npz_file(
        npz_file, device, dtype, input_fps, output_fps, target_joint_names=target_dof_names
    )
    # MotionLib linearly interpolates DOF positions during playback. Unwrap each
    # joint's periodic branch crossings here so the exported motion stays
    # kinematically identical but does not generate impossible in-between poses.
    joint_angles = _unwrap_joint_angles_for_motion(joint_angles)
    
    print(f"Loaded motion: {root_pos.shape[0]} frames (resampled from {input_fps} -> {output_fps} fps)")
    
    # Skip initial frames if requested
    if ignore_first_n_frames > 0:
        if ignore_first_n_frames >= root_pos.shape[0]:
            print(f"Error: ignore_first_n_frames ({ignore_first_n_frames}) >= motion length")
            return False
        root_pos = root_pos[ignore_first_n_frames:]
        root_rot_wxyz = root_rot_wxyz[ignore_first_n_frames:]
        joint_angles = joint_angles[ignore_first_n_frames:]
    
    # Extract kinematic info from model
    # kinematic_info = extract_kinematic_info(str(model_xml))
    
    # Build qpos [root_pos, root_rot_wxyz, joint_angles]
    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
    
    # Extract transforms from qpos
    root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    
    # Compute forward kinematics with velocities
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_from_qpos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
    )
    
    # Use the original joint angles directly from the retarget step.
    # The lower-body retargeter outputs Euler XYZ angles for each joint, which is exactly what we need.
    # Re-extracting from transforms can cause angle wrapping issues.
    motion.dof_pos = joint_angles

    # Keep the exported DOF positions on the exact unwrapped branch so
    # ProtoMotions interpolation is stable, but compute velocities from the
    # nearest valid branch so existing velocity-based QC/filter thresholds stay
    # meaningful.
    joint_angles_for_velocity = _stabilize_joint_angles_for_velocity(
        joint_angles,
        lower_limits=kinematic_info.dof_limits_lower.to(device=device, dtype=dtype),
        upper_limits=kinematic_info.dof_limits_upper.to(device=device, dtype=dtype),
    )
    dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles_for_velocity.unsqueeze(1),
        fps=output_fps,
    )
    motion.dof_vel = dof_vel.squeeze(1)
    
    # --- FIX HEIGHT ---
    # User requested to move motion based on minimum value and zero off of that.
    # This implies a global shift (fix_height) rather than per-frame adjustment.
    
    # We skip fix_height_per_frame to preserve flight phases and vertical dynamics.
    # motion.fix_height_per_frame(height_offset=0.02, min_clamp=-10.0)
    
    # Apply global fix
    # Use the provided height_offset directly.
    motion.fix_height(height_offset=height_offset)

    # Handle contact labels
    motion_length = motion.rigid_body_pos.shape[0]
    num_bodies = motion.rigid_body_pos.shape[1]
    body_names = kinematic_info.body_names
    
    # Attempt to automatically find ankle/toe contact bodies. The retarget step exports one
    # left and one right foot-contact signal, so apply each signal to both the ankle
    # and toe bodies when they exist.
    left_contact_indices = [
        body_names.index(name)
        for name in ("L_Ankle", "L_Toe")
        if name in body_names
    ]
    right_contact_indices = [
        body_names.index(name)
        for name in ("R_Ankle", "R_Toe")
        if name in body_names
    ]
    if not left_contact_indices or not right_contact_indices:
        left_contact_indices = [len(body_names) - 2]
        right_contact_indices = [len(body_names) - 1]
        print(
            "Warning: Could not find ankle/toe body names. "
            f"Defaulting contacts to indices {left_contact_indices} and {right_contact_indices}"
        )
    
    if contact_file is not None and contact_file.exists():
        print(f"Loading contact labels from: {contact_file}")
        motion.rigid_body_contacts = load_contact_labels(
            contact_file=contact_file,
            motion_length=motion_length,
            left_contact_indices=left_contact_indices,
            right_contact_indices=right_contact_indices,
            num_bodies=num_bodies,
            device=device,
            input_fps=input_fps,
            output_fps=output_fps
        )
    else:
        # Default: zero contacts (can be recomputed later)
        motion.rigid_body_contacts = torch.zeros(
            motion_length, num_bodies, device=device, dtype=torch.bool
        )
    
    # HACK: prevent motion_lib from interpolating using stored rotations (can cause issues)
    motion.local_rigid_body_rot = None
    
    # Apply motion filter if enabled
    if apply_motion_filter and passes_exclude_motion_filter is not None:
        if not passes_exclude_motion_filter(
            motion,
            min_height_threshold=min_height_threshold,
            max_velocity_threshold=max_velocity_threshold,
            max_dof_vel_threshold=max_dof_vel_threshold,
            duration_height_filter=duration_height_filter,
            duration_height_seconds=duration_height_seconds,
        ):
            print(f"Skipping {npz_file.name} because it does not pass motion filter")
            return False

    # Save motion
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving motion to: {output_file}")
    torch.save(motion.to_dict(), str(output_file))
    return True


@app.command()
def main(
    npz_file: Path = typer.Argument(..., exists=True, help="Input retargeted NPZ file"),
    output_file: Path = typer.Argument(..., help="Output .motion file path"),
    model_xml: Path = typer.Option(
        ..., "--model-xml", "-m", exists=True,
        help="Path to MJCF model file"
    ),
    input_fps: int = typer.Option(30, "--input-fps", help="Input frame rate of the NPZ motion"),
    output_fps: int = typer.Option(30, "--output-fps", help="Target output frame rate"),
    contact_file: Optional[Path] = typer.Option(
        None, "--contact-file", "-c",
        help="Path to contact labels NPZ file"
    ),
    ignore_first_n_frames: int = typer.Option(
        0, "--ignore-first-n", help="Number of frames to skip at start"
    ),
    height_offset: float = typer.Option(
        0.0, "--height-offset", help="Height offset for ground contact (Default: 0.0m)"
    ),
):
    """
    Convert a single retargeted NPZ file to ProtoMotions .motion format.
    """
    with torch.no_grad():
        convert_npz_to_motion(
            npz_file=npz_file,
            output_file=output_file,
            model_xml=model_xml,
            input_fps=input_fps,
            output_fps=output_fps,
            contact_file=contact_file,
            ignore_first_n_frames=ignore_first_n_frames,
            height_offset=height_offset,
        )
    
    print("✅ Conversion complete!")


# --- Batch Processing Utilities ---

def batch_convert(
    retargeted_dir: Path,
    output_dir: Path,
    model_xml: Path,
    contacts_dir: Optional[Path] = None,
    input_fps: int = 200,
    output_fps: int = 30,
    ignore_first_n_frames: int = 0,
    height_offset: float = 0.0,  # No offset - use motion ground height
    force_remake: bool = False,
):
    """Batch convert all NPZ files in a directory."""
    import glob
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(glob.glob(str(retargeted_dir / "*.npz")))
    print(f"Found {len(npz_files)} NPZ files to convert")
    
    for npz_path in tqdm(npz_files, desc="Converting motions"):
        npz_file = Path(npz_path)
        base_name = npz_file.stem.replace("_retargeted", "")
        output_file = output_dir / f"{base_name}.motion"
        
        if output_file.exists() and not force_remake:
            continue
        
        # Find contact file if contacts_dir provided
        contact_file = None
        if contacts_dir is not None:
            contact_file = contacts_dir / f"{base_name}_contacts.npz"
            if not contact_file.exists():
                contact_file = None
        
        try:
            with torch.no_grad():
                convert_npz_to_motion(
                    npz_file=npz_file,
                    output_file=output_file,
                    model_xml=model_xml,
                    input_fps=input_fps,
                    output_fps=output_fps,
                    contact_file=contact_file,
                    ignore_first_n_frames=ignore_first_n_frames,
                    height_offset=height_offset,
                )
        except Exception as e:
            print(f"Error converting {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Batch conversion complete! Output: {output_dir}")


@app.command("batch")
def batch_command(
    retargeted_dir: Path = typer.Argument(..., exists=True, help="Directory with NPZ files"),
    output_dir: Path = typer.Argument(..., help="Output directory for .motion files"),
    model_xml: Path = typer.Option(
        ..., "--model-xml", "-m", exists=True,
        help="Path to MJCF model file"
    ),
    contacts_dir: Optional[Path] = typer.Option(
        None, "--contacts-dir", "-c",
        help="Directory with contact labels"
    ),
    input_fps: int = typer.Option(200, "--input-fps", help="Input frame rate of the motion"),
    output_fps: int = typer.Option(30, "--output-fps", help="Target output frame rate"),
    ignore_first_n_frames: int = typer.Option(
        0, "--ignore-first-n", help="Number of frames to skip at start"
    ),
    height_offset: float = typer.Option(
        0.09, "--height-offset", help="Height offset for ground contact"
    ),
    force_remake: bool = typer.Option(False, "--force", help="Force remake existing files"),
):
    """
    Batch convert all NPZ files in a directory to .motion format.
    """
    batch_convert(
        retargeted_dir=retargeted_dir,
        output_dir=output_dir,
        model_xml=model_xml,
        contacts_dir=contacts_dir,
        input_fps=input_fps,
        output_fps=output_fps,
        ignore_first_n_frames=ignore_first_n_frames,
        height_offset=height_offset,
        force_remake=force_remake,
    )


if __name__ == "__main__":
    app()
