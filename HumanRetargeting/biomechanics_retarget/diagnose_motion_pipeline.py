#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
r"""
Diagnose motion pipeline: compare joint data at every stage.

Compares joint names and values between:
  1. PyRoki .npz  (retargeted output)
  2. .motion file (after convert_retargeted_to_motion.py)
  3. .pt file     (after package_motions.py)

Also extracts MJCF joint ordering and (optionally) Newton simulator ordering
to detect any joint reordering mismatches.

Usage:
    cd D:\Biomotions\newton\ProtoMotions

    # Quick check (compare .npz -> .motion -> .pt joint order + first 3 frames):
    ..\.venv\Scripts\python.exe HumanRetargeting\biomechanics_retarget\diagnose_motion_pipeline.py ^
        --npz HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\retargeted_motions\S02_30ms_Long_retargeted.npz ^
        --motion HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\motion_files\S02_30ms_Long.motion ^
        --pt HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\packaged_data\S_GENERIC.pt ^
        --model-xml HumanRetargeting\rescale\smpl_humanoid_lower_body_adjusted_pd.xml

    # Verbose: print per-joint values at specific frames:
    ..\.venv\Scripts\python.exe HumanRetargeting\biomechanics_retarget\diagnose_motion_pipeline.py ^
        --npz ... --motion ... --pt ... --model-xml ... ^
        --frames 0 10 50 --verbose
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deg(rad):
    """Convert radians to degrees for readability."""
    if isinstance(rad, torch.Tensor):
        return rad.cpu().numpy() * 180.0 / np.pi
    return np.array(rad) * 180.0 / np.pi


def print_header(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    print()
    print(f"--- {title} ---")


def print_joint_table(joint_names, values_deg, label="dof_pos (deg)"):
    """Print a nicely formatted joint table."""
    max_name = max(len(n) for n in joint_names) if joint_names else 10
    print(f"  {'Joint':<{max_name}}  {label:>12}")
    print(f"  {'-'*max_name}  {'-'*12}")
    for name, val in zip(joint_names, values_deg):
        print(f"  {name:<{max_name}}  {val:12.4f}")


def print_joint_comparison(joint_names_a, values_a_deg, label_a,
                           joint_names_b, values_b_deg, label_b):
    """Side-by-side comparison of two sets of joint values."""
    max_name = max(
        max((len(n) for n in joint_names_a), default=10),
        max((len(n) for n in joint_names_b), default=10),
    )
    print(f"  {'Joint':<{max_name}}  {label_a:>12}  {label_b:>12}  {'Diff':>12}")
    print(f"  {'-'*max_name}  {'-'*12}  {'-'*12}  {'-'*12}")

    n = max(len(joint_names_a), len(joint_names_b))
    for i in range(n):
        name_a = joint_names_a[i] if i < len(joint_names_a) else "???"
        name_b = joint_names_b[i] if i < len(joint_names_b) else "???"
        va = values_a_deg[i] if i < len(values_a_deg) else float("nan")
        vb = values_b_deg[i] if i < len(values_b_deg) else float("nan")
        diff = va - vb if not (np.isnan(va) or np.isnan(vb)) else float("nan")
        name_mismatch = " <-- NAME MISMATCH" if name_a != name_b else ""
        val_flag = " ***" if abs(diff) > 0.1 else ""
        print(f"  {name_a:<{max_name}}  {va:12.4f}  {vb:12.4f}  {diff:12.4f}{val_flag}{name_mismatch}")


# ---------------------------------------------------------------------------
# Stage loaders
# ---------------------------------------------------------------------------

def load_npz(npz_path: Path):
    """Load PyRoki retargeted .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    result = {
        "base_frame_pos": data["base_frame_pos"],
        "base_frame_wxyz": data["base_frame_wxyz"],
        "joint_angles": data["joint_angles"],
    }
    if "joint_names" in data:
        names = data["joint_names"].tolist()
        if len(names) > 0 and isinstance(names[0], bytes):
            names = [n.decode("utf-8") for n in names]
        result["joint_names"] = names
    else:
        result["joint_names"] = None
    return result


def load_motion(motion_path: Path):
    """Load .motion file (torch.save dict)."""
    data = torch.load(str(motion_path), map_location="cpu", weights_only=False)
    return data


def load_pt(pt_path: Path):
    """Load packaged MotionLib .pt file."""
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return data


def get_mjcf_joint_order(model_xml: Path):
    """Extract joint ordering from MJCF XML via pose_lib."""
    try:
        from protomotions.components.pose_lib import extract_kinematic_info
        ki = extract_kinematic_info(str(model_xml))
        return ki.dof_names, ki.body_names
    except Exception as e:
        print(f"  Warning: Could not extract kinematic info from MJCF: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose motion pipeline joint ordering and values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--npz", type=Path, default=None, help="PyRoki retargeted .npz file")
    parser.add_argument("--motion", type=Path, default=None, help=".motion file (after convert)")
    parser.add_argument("--pt", type=Path, default=None, help="Packaged MotionLib .pt file")
    parser.add_argument("--model-xml", type=Path, default=None, help="MJCF model XML file")
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[0, 1, 2],
        help="Frame indices to print (default: 0 1 2)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-joint values per frame")
    parser.add_argument(
        "--motion-index", type=int, default=0,
        help="Which motion clip index in the .pt to inspect (default: 0)",
    )
    args = parser.parse_args()

    if not any([args.npz, args.motion, args.pt]):
        parser.error("Provide at least one of --npz, --motion, or --pt")

    # -----------------------------------------------------------------------
    # 1. MJCF Joint Order (ground truth)
    # -----------------------------------------------------------------------
    mjcf_dof_names = None
    mjcf_body_names = None
    if args.model_xml and args.model_xml.exists():
        print_header("MJCF Model Joint Order (ground truth for COMMON format)")
        mjcf_dof_names, mjcf_body_names = get_mjcf_joint_order(args.model_xml)
        if mjcf_dof_names:
            print(f"  DOF count: {len(mjcf_dof_names)}")
            for i, name in enumerate(mjcf_dof_names):
                print(f"    [{i:2d}] {name}")
        if mjcf_body_names:
            print_section("Body names (for rigid_body_pos/rot)")
            for i, name in enumerate(mjcf_body_names):
                print(f"    [{i:2d}] {name}")

    # -----------------------------------------------------------------------
    # 2. PyRoki .npz
    # -----------------------------------------------------------------------
    npz_data = None
    npz_joint_names = None
    if args.npz and args.npz.exists():
        print_header("Stage 1: PyRoki Retargeted .npz")
        npz_data = load_npz(args.npz)
        npz_joint_names = npz_data["joint_names"]

        print(f"  File: {args.npz}")
        print(f"  Frames: {npz_data['joint_angles'].shape[0]}")
        print(f"  DOF count: {npz_data['joint_angles'].shape[1]}")
        print(f"  base_frame_pos shape: {npz_data['base_frame_pos'].shape}")
        print(f"  base_frame_wxyz shape: {npz_data['base_frame_wxyz'].shape}")

        if npz_joint_names:
            print_section("NPZ Joint Names")
            for i, name in enumerate(npz_joint_names):
                print(f"    [{i:2d}] {name}")

            # Compare with MJCF
            if mjcf_dof_names:
                print_section("Joint Name Order Comparison: NPZ vs MJCF")
                match = True
                n = max(len(npz_joint_names), len(mjcf_dof_names))
                for i in range(n):
                    npz_n = npz_joint_names[i] if i < len(npz_joint_names) else "MISSING"
                    mjcf_n = mjcf_dof_names[i] if i < len(mjcf_dof_names) else "MISSING"
                    flag = "" if npz_n == mjcf_n else " <-- MISMATCH!"
                    if flag:
                        match = False
                    print(f"    [{i:2d}] NPZ: {npz_n:<20} MJCF: {mjcf_n:<20}{flag}")
                if match:
                    print("  ✓ Joint order matches between NPZ and MJCF")
                else:
                    print("  ✗ JOINT ORDER MISMATCH between NPZ and MJCF!")
        else:
            print("  Warning: NPZ has no joint_names array!")

        if args.verbose:
            for frame in args.frames:
                if frame < npz_data["joint_angles"].shape[0]:
                    print_section(f"NPZ Frame {frame}")
                    names = npz_joint_names or [f"dof_{i}" for i in range(npz_data["joint_angles"].shape[1])]
                    vals_deg = deg(npz_data["joint_angles"][frame])
                    print_joint_table(names, vals_deg)
                    print(f"  root pos: {npz_data['base_frame_pos'][frame]}")
                    print(f"  root rot (wxyz): {npz_data['base_frame_wxyz'][frame]}")
    elif args.npz:
        print(f"  Warning: NPZ file not found: {args.npz}")

    # -----------------------------------------------------------------------
    # 3. .motion file
    # -----------------------------------------------------------------------
    motion_data = None
    if args.motion and args.motion.exists():
        print_header("Stage 2: .motion File (after convert_retargeted_to_motion)")
        motion_data = load_motion(args.motion)

        print(f"  File: {args.motion}")
        print(f"  Keys: {list(motion_data.keys())}")

        if "dof_pos" in motion_data:
            dof_pos = motion_data["dof_pos"]
            print(f"  dof_pos shape: {dof_pos.shape}")
            print(f"  dof_pos dtype: {dof_pos.dtype}")
        if "rigid_body_pos" in motion_data:
            print(f"  rigid_body_pos shape: {motion_data['rigid_body_pos'].shape}")
        if "rigid_body_rot" in motion_data:
            print(f"  rigid_body_rot shape: {motion_data['rigid_body_rot'].shape}")
        if "dof_vel" in motion_data:
            print(f"  dof_vel shape: {motion_data['dof_vel'].shape}")
        if "fps" in motion_data:
            print(f"  fps: {motion_data['fps']}")

        if args.verbose and "dof_pos" in motion_data:
            dof_pos = motion_data["dof_pos"]
            names = mjcf_dof_names or [f"dof_{i}" for i in range(dof_pos.shape[1])]
            for frame in args.frames:
                if frame < dof_pos.shape[0]:
                    print_section(f".motion Frame {frame}")
                    vals_deg = deg(dof_pos[frame])
                    print_joint_table(names, vals_deg)
                    if "rigid_body_pos" in motion_data:
                        rbp = motion_data["rigid_body_pos"][frame]
                        body_names = mjcf_body_names or [f"body_{i}" for i in range(rbp.shape[0])]
                        print(f"\n  FK Body Positions (world XYZ) — from convert step:")
                        for bi in range(rbp.shape[0]):
                            b_name = body_names[bi] if bi < len(body_names) else f"body_{bi}"
                            print(f"    [{bi:2d}] {b_name:<20} = [{rbp[bi, 0]:8.4f}, {rbp[bi, 1]:8.4f}, {rbp[bi, 2]:8.4f}]")
                    if "rigid_body_rot" in motion_data:
                        rbr = motion_data["rigid_body_rot"][frame]
                        body_names = mjcf_body_names or [f"body_{i}" for i in range(rbr.shape[0])]
                        print(f"\n  FK Body Rotations (xyzw) — from convert step:")
                        for bi in range(rbr.shape[0]):
                            b_name = body_names[bi] if bi < len(body_names) else f"body_{bi}"
                            print(f"    [{bi:2d}] {b_name:<20} = [{rbr[bi, 0]:8.4f}, {rbr[bi, 1]:8.4f}, {rbr[bi, 2]:8.4f}, {rbr[bi, 3]:8.4f}]")

        # Compare with NPZ
        if npz_data is not None and "dof_pos" in motion_data:
            print_section("Comparing NPZ joint_angles vs .motion dof_pos")
            npz_angles = npz_data["joint_angles"]
            mot_dof_pos = motion_data["dof_pos"].numpy()

            # They may have different frame counts due to resampling
            print(f"  NPZ frames: {npz_angles.shape[0]}, .motion frames: {mot_dof_pos.shape[0]}")
            print(f"  NPZ dofs: {npz_angles.shape[1]}, .motion dofs: {mot_dof_pos.shape[1]}")

            # Compare frame 0 (should match if no resampling or first frame survives)
            if npz_angles.shape[1] == mot_dof_pos.shape[1]:
                for frame in args.frames:
                    # Use frame 0 from both since resampling may shift indices
                    npz_f = min(frame, npz_angles.shape[0] - 1)
                    mot_f = min(frame, mot_dof_pos.shape[0] - 1)
                    print_section(f"Frame comparison: NPZ[{npz_f}] vs .motion[{mot_f}]")
                    npz_names = npz_joint_names or [f"dof_{i}" for i in range(npz_angles.shape[1])]
                    mot_names = mjcf_dof_names or [f"dof_{i}" for i in range(mot_dof_pos.shape[1])]
                    print_joint_comparison(
                        npz_names, deg(npz_angles[npz_f]), "NPZ(deg)",
                        mot_names, deg(mot_dof_pos[mot_f]), ".motion(deg)",
                    )
            else:
                print(f"  ✗ DOF count mismatch! Cannot compare directly.")
    elif args.motion:
        print(f"  Warning: .motion file not found: {args.motion}")

    # -----------------------------------------------------------------------
    # 4. .pt MotionLib
    # -----------------------------------------------------------------------
    pt_data = None
    if args.pt and args.pt.exists():
        print_header("Stage 3: Packaged MotionLib .pt")
        pt_data = load_pt(args.pt)

        print(f"  File: {args.pt}")
        print(f"  Top-level keys: {list(pt_data.keys())}")

        # Standard MotionLib field names
        field_map = {
            "gts": "rigid_body_pos",
            "grs": "rigid_body_rot",
            "dps": "dof_pos",
            "dvs": "dof_vel",
            "gavs": "rigid_body_ang_vel",
            "gvs": "rigid_body_vel",
            "contacts": "rigid_body_contacts",
        }

        for key, desc in field_map.items():
            if key in pt_data:
                t = pt_data[key]
                print(f"  {key:>10} ({desc}): shape={t.shape}, dtype={t.dtype}")

        # Motion metadata
        if "motion_num_frames" in pt_data:
            print(f"\n  motion_num_frames: {pt_data['motion_num_frames']}")
        if "length_starts" in pt_data:
            print(f"  length_starts: {pt_data['length_starts']}")
        if "motion_fps" in pt_data:
            print(f"  motion_fps: {pt_data['motion_fps']}")
        if "motion_weights" in pt_data:
            print(f"  motion_weights: {pt_data['motion_weights']}")

        # Show dof_pos for motion_index
        if "dps" in pt_data and "length_starts" in pt_data and "motion_num_frames" in pt_data:
            mi = args.motion_index
            starts = pt_data["length_starts"]
            num_frames = pt_data["motion_num_frames"]

            if mi < len(starts):
                start = int(starts[mi])
                nf = int(num_frames[mi])
                print_section(f"Motion clip {mi}: frames {start}..{start + nf - 1} ({nf} frames)")

                dps = pt_data["dps"]
                names = mjcf_dof_names or [f"dof_{i}" for i in range(dps.shape[1])]

                if args.verbose:
                    for frame in args.frames:
                        abs_frame = start + frame
                        if frame < nf:
                            print_section(f".pt Motion {mi} Frame {frame} (abs {abs_frame})")
                            print_joint_table(names, deg(dps[abs_frame]))

                            if "gts" in pt_data:
                                gts = pt_data["gts"][abs_frame]
                                body_names = mjcf_body_names or [f"body_{i}" for i in range(gts.shape[0])]
                                print(f"\n  Body Positions (world XYZ):")
                                for bi in range(gts.shape[0]):
                                    b_name = body_names[bi] if bi < len(body_names) else f"body_{bi}"
                                    print(f"    [{bi:2d}] {b_name:<20} = [{gts[bi, 0]:8.4f}, {gts[bi, 1]:8.4f}, {gts[bi, 2]:8.4f}]")
                            if "grs" in pt_data:
                                grs = pt_data["grs"][abs_frame]
                                body_names = mjcf_body_names or [f"body_{i}" for i in range(grs.shape[0])]
                                print(f"\n  Body Rotations (xyzw):")
                                for bi in range(grs.shape[0]):
                                    b_name = body_names[bi] if bi < len(body_names) else f"body_{bi}"
                                    print(f"    [{bi:2d}] {b_name:<20} = [{grs[bi, 0]:8.4f}, {grs[bi, 1]:8.4f}, {grs[bi, 2]:8.4f}, {grs[bi, 3]:8.4f}]")

                # Compare .pt frame 0 with .motion frame 0
                if motion_data is not None and "dof_pos" in motion_data:
                    print_section("Comparing .motion dof_pos vs .pt dps (frame 0)")
                    mot_names = mjcf_dof_names or [f"dof_{i}" for i in range(motion_data["dof_pos"].shape[1])]
                    pt_names = mjcf_dof_names or [f"dof_{i}" for i in range(dps.shape[1])]
                    print_joint_comparison(
                        mot_names, deg(motion_data["dof_pos"][0].numpy()), ".motion(deg)",
                        pt_names, deg(dps[start].numpy()), ".pt(deg)",
                    )
    elif args.pt:
        print(f"  Warning: .pt file not found: {args.pt}")

    # -----------------------------------------------------------------------
    # 5. Summary & Recommendations
    # -----------------------------------------------------------------------
    print_header("SUMMARY")
    issues = []

    if npz_joint_names and mjcf_dof_names:
        if npz_joint_names != mjcf_dof_names:
            if set(npz_joint_names) == set(mjcf_dof_names):
                issues.append("NPZ and MJCF have SAME joints but DIFFERENT ORDER → reordering should fix it")
            else:
                issues.append("NPZ and MJCF have DIFFERENT joint sets!")
        else:
            print("  ✓ NPZ joint names match MJCF order")

    if npz_data is not None and motion_data is not None and "dof_pos" in motion_data:
        npz_a = npz_data["joint_angles"]
        mot_d = motion_data["dof_pos"].numpy()
        if npz_a.shape[1] != mot_d.shape[1]:
            issues.append(f"DOF count mismatch: NPZ has {npz_a.shape[1]}, .motion has {mot_d.shape[1]}")

    if motion_data is not None and pt_data is not None and "dof_pos" in motion_data and "dps" in pt_data:
        mot_d = motion_data["dof_pos"]
        pt_d = pt_data["dps"]
        if mot_d.shape[1] != pt_d.shape[1]:
            issues.append(f"DOF count mismatch: .motion has {mot_d.shape[1]}, .pt has {pt_d.shape[1]}")

    if not issues:
        print("  ✓ No joint ordering issues detected in stored data.")
        print()
        print("  If the motion still looks wrong in ProtoMotions env_kinematic_playback,")
        print("  the problem is likely in the COMMON→SIMULATOR dof reordering.")
        print("  Run the Newton diagnostic mode to check:")
        print()
        print("  ...\\.venv\\Scripts\\python.exe examples\\env_kinematic_playback.py \\")
        print("      --experiment-path examples\\experiments\\mimic\\mlp.py \\")
        print("      --motion-file <your.pt> \\")
        print("      --robot-name smpl_lower_body_subject_S_GENERIC \\")
        print("      --simulator newton --num-envs 1")
        print()
        print("  Then check the log for 'dof_convert_to_sim' mapping.")
    else:
        for issue in issues:
            print(f"  ✗ {issue}")

    print()
    print("  TIP: To also see what Newton does with the joint order at runtime,")
    print("  use --diagnose-runtime flag with env_kinematic_playback.py")
    print("  (see diagnose_runtime_joints.py for a patched version).")


if __name__ == "__main__":
    main()
