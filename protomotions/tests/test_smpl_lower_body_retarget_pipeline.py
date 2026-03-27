# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json

import numpy as np
import pytest
import torch
import yaml

from HumanRetargeting.biomechanics_retarget.convert_retargeted_to_motion import (
    convert_npz_to_motion,
    load_npz_file,
)
from HumanRetargeting.biomechanics_retarget.pipeline import PipelineStep, main as run_pipeline
import HumanRetargeting.biomechanics_retarget.stages.package as package_module
from HumanRetargeting.biomechanics_retarget.stages.package import (
    create_motion_manifest,
    generate_experiment_matrix_manifests,
    package_motion_library,
)
from HumanRetargeting.biomechanics_retarget.retarget_qc import (
    evaluate_retargeted_motion,
)
from HumanRetargeting.biomechanics_retarget.validation import validate_retargeted_npz
from HumanRetargeting.biomechanics_retarget.validation import validate_packaged_motion_lib
from HumanRetargeting.biomechanics_retarget.subject_profiles import (
    materialize_height_subject_profile,
)
from protomotions.components.pose_lib import extract_kinematic_info


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = (
    REPO_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "mjcf"
    / "smpl_humanoid_lower_body_subject_S_GENERIC.xml"
)
RETARGET_DIR = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "processed_data"
    / "S_GENERIC"
    / "retargeted_motions"
)
KEYPOINT_DIR = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "processed_data"
    / "S_GENERIC"
    / "keypoints"
)
CONTACT_DIR = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "processed_data"
    / "S_GENERIC"
    / "contacts"
)
MOTION_DIR = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "processed_data"
    / "S_GENERIC"
    / "motion_files"
)
PACKAGED_PATH = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "processed_data"
    / "S_GENERIC"
    / "packaged_data"
    / "S_GENERIC.pt"
)
QC_CONFIG = (
    REPO_ROOT
    / "HumanRetargeting"
    / "biomechanics_retarget"
    / "config"
    / "qc_thresholds.yaml"
)


def _decode_joint_names(joint_names: np.ndarray) -> list[str]:
    names = joint_names.tolist()
    if names and isinstance(names[0], bytes):
        return [name.decode("utf-8") for name in names]
    return [str(name) for name in names]


def _canonical_trial_name(npz_path: Path) -> str:
    return npz_path.stem.removesuffix("_retargeted")


def test_pyrki_retarget_outputs_match_mjcf_contract():
    kinematic_info = extract_kinematic_info(str(MODEL_XML))
    expected_joint_names = kinematic_info.dof_names
    expected_npz_keys = {
        "base_frame_pos",
        "base_frame_wxyz",
        "joint_angles",
        "joint_names",
    }

    npz_paths = sorted(RETARGET_DIR.glob("*_retargeted.npz"))
    assert npz_paths, "expected retargeted PyRoki outputs to exist"

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        assert set(data.files) == expected_npz_keys

        root_pos = np.asarray(data["base_frame_pos"], dtype=np.float32)
        root_rot_wxyz = np.asarray(data["base_frame_wxyz"], dtype=np.float32)
        joint_angles = np.asarray(data["joint_angles"], dtype=np.float32)
        joint_names = _decode_joint_names(np.asarray(data["joint_names"]))

        assert root_pos.shape[1] == 3
        assert root_rot_wxyz.shape[1] == 4
        assert joint_angles.shape[1] == len(expected_joint_names)
        assert joint_names == expected_joint_names
        assert np.isfinite(root_pos).all()
        assert np.isfinite(root_rot_wxyz).all()
        assert np.isfinite(joint_angles).all()

        lower_limits = kinematic_info.dof_limits_lower.numpy()
        upper_limits = kinematic_info.dof_limits_upper.numpy()
        assert np.all(joint_angles >= lower_limits[None, :] - 1e-5)
        assert np.all(joint_angles <= upper_limits[None, :] + 1e-5)

        keypoint_file = KEYPOINT_DIR / f"{_canonical_trial_name(npz_path)}.npy"
        report = evaluate_retargeted_motion(
            keypoint_file=keypoint_file,
            retargeted_file=npz_path,
            model_xml=MODEL_XML,
        )
        assert "passed" in report
        assert isinstance(report["failures"], list)
        assert np.isfinite(report["metrics"]["mean_keypoint_error_m"])
        assert np.isfinite(report["metrics"]["mean_contact_slip_mps"])


def test_sample_pyrki_retarget_outputs_pass_production_validator():
    npz_paths = sorted(RETARGET_DIR.glob("*_retargeted.npz"))
    assert npz_paths, "expected retargeted PyRoki outputs to exist"

    for npz_path in npz_paths:
        keypoint_file = KEYPOINT_DIR / f"{_canonical_trial_name(npz_path)}.npy"
        report = validate_retargeted_npz(
            npz_file=npz_path,
            keypoint_file=keypoint_file,
            model_xml=MODEL_XML,
            qc_config_file=QC_CONFIG,
        )
        assert report["passed"], (npz_path.name, report["failures"], report["quality_report"])


def test_load_npz_file_reorders_named_joints(tmp_path):
    kinematic_info = extract_kinematic_info(str(MODEL_XML))
    target_joint_names = list(kinematic_info.dof_names)
    source_joint_names = target_joint_names[3:] + target_joint_names[:3]

    num_frames = 4
    num_dofs = len(target_joint_names)
    source_joint_angles = np.stack(
        [
            np.arange(num_dofs, dtype=np.float32) + frame * 100.0
            for frame in range(num_frames)
        ],
        axis=0,
    )

    npz_path = tmp_path / "permuted_joint_names.npz"
    np.savez_compressed(
        npz_path,
        base_frame_pos=np.zeros((num_frames, 3), dtype=np.float32),
        base_frame_wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (num_frames, 1),
        ),
        joint_angles=source_joint_angles,
        joint_names=np.asarray(source_joint_names),
    )

    root_pos, root_rot_wxyz, joint_angles = load_npz_file(
        npz_path=npz_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        input_fps=30,
        output_fps=30,
        target_joint_names=target_joint_names,
    )

    assert root_pos.shape == (num_frames, 3)
    assert root_rot_wxyz.shape == (num_frames, 4)
    assert joint_angles.shape == (num_frames, num_dofs)
    expected = source_joint_angles[:, [source_joint_names.index(name) for name in target_joint_names]]
    assert np.allclose(joint_angles.numpy(), expected, atol=1e-6)


def test_load_npz_file_raises_on_missing_named_joint(tmp_path):
    npz_path = tmp_path / "bad_joint_names.npz"
    np.savez_compressed(
        npz_path,
        base_frame_pos=np.zeros((2, 3), dtype=np.float32),
        base_frame_wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (2, 1),
        ),
        joint_angles=np.zeros((2, 2), dtype=np.float32),
        joint_names=np.asarray(["joint_a", "joint_b"]),
    )

    with pytest.raises(ValueError, match="Missing target joints"):
        load_npz_file(
            npz_path=npz_path,
            device=torch.device("cpu"),
            dtype=torch.float32,
            input_fps=30,
            output_fps=30,
            target_joint_names=["joint_a", "joint_c"],
        )


def test_convert_retargeted_motion_preserves_dof_positions(tmp_path):
    npz_path = sorted(RETARGET_DIR.glob("*_retargeted.npz"))[0]
    trial_name = _canonical_trial_name(npz_path)
    contact_file = CONTACT_DIR / f"{trial_name}_contacts.npz"
    output_file = tmp_path / f"{trial_name}.motion"

    success = convert_npz_to_motion(
        npz_file=npz_path,
        output_file=output_file,
        model_xml=MODEL_XML,
        input_fps=30,
        output_fps=30,
        contact_file=contact_file if contact_file.exists() else None,
        ignore_first_n_frames=0,
        height_offset=0.0,
        apply_motion_filter=False,
    )
    assert success

    motion = torch.load(output_file, map_location="cpu", weights_only=False)
    source = np.load(npz_path, allow_pickle=True)
    joint_angles = torch.from_numpy(np.asarray(source["joint_angles"], dtype=np.float32))

    assert int(motion["fps"]) == 30
    assert motion["dof_pos"].shape == joint_angles.shape
    assert torch.allclose(motion["dof_pos"], joint_angles, atol=1e-6)
    assert torch.isfinite(motion["rigid_body_pos"]).all()
    assert torch.isfinite(motion["rigid_body_rot"]).all()
    assert torch.isfinite(motion["dof_vel"]).all()


def test_packaged_motionlib_matches_motion_files_exactly():
    packaged = torch.load(PACKAGED_PATH, map_location="cpu", weights_only=False)
    expected_motion_files = [str(path.resolve()) for path in sorted(MOTION_DIR.glob("*.motion"))]

    assert list(packaged["motion_files"]) == expected_motion_files
    assert len(packaged["motion_num_frames"]) == len(expected_motion_files)
    assert len(packaged["length_starts"]) == len(expected_motion_files)

    for idx, motion_path in enumerate(expected_motion_files):
        motion = torch.load(motion_path, map_location="cpu", weights_only=False)
        start = int(packaged["length_starts"][idx])
        frames = int(packaged["motion_num_frames"][idx])
        end = start + frames

        assert frames == motion["dof_pos"].shape[0]
        assert torch.equal(packaged["dps"][start:end], motion["dof_pos"])
        assert torch.equal(packaged["dvs"][start:end], motion["dof_vel"])
        assert torch.equal(packaged["contacts"][start:end], motion["rigid_body_contacts"])


def test_packaged_validator_accepts_same_files_with_alternate_path_casing():
    expected_motion_files = sorted(MOTION_DIR.glob("*.motion"))
    alternate_motion_files = [
        Path(str(path).replace("/mnt/d/biomotions/protomotions/", "/mnt/d/Biomotions/ProtoMotions/"))
        for path in expected_motion_files
    ]

    assert all(path.exists() for path in alternate_motion_files)

    report = validate_packaged_motion_lib(
        packaged_file=PACKAGED_PATH,
        expected_motion_files=alternate_motion_files,
    )
    assert report["passed"], report


def test_materialize_height_subject_profile_writes_canonical_generated_profile(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile = materialize_height_subject_profile(
        input_dir=REPO_ROOT / "HumanRetargeting" / "biomechanics_retarget" / "treadmill_data" / "S_GENERIC",
        output_path=profile_path,
        height_cm=182,
        subject_id=None,
        model_variant="adjusted_pd",
        fps=200,
        output_fps=30,
        coordinate_transform="y_to_x_forward",
        contact_source="heuristic",
    )

    persisted = profile_path.read_text(encoding="utf-8")
    assert profile.subject_id == "H182"
    assert "profile_mode: generated_from_height" in persisted
    assert "subject_id: H182" in persisted
    assert "height_cm: 182" in persisted


def test_pipeline_assets_step_materializes_profile_and_subject_assets(tmp_path):
    output_dir = tmp_path / "run"
    assets_root = tmp_path / "assets"
    result = run_pipeline(
        input_dir=REPO_ROOT / "HumanRetargeting" / "biomechanics_retarget" / "treadmill_data" / "S_GENERIC",
        output_dir=output_dir,
        height=182,
        step=PipelineStep.ASSETS,
        assets_root=assets_root,
        force=True,
    )

    assert result == output_dir / "profile.yaml"
    summary = json.loads((output_dir / "qc" / "subject_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["generated_profile"] is True
    assert summary["robot_name"] == "smpl_lower_body_subject_H182"
    assert (output_dir / "profile.yaml").exists()
    assert (assets_root / "mjcf" / "smpl_humanoid_lower_body_subject_H182.xml").exists()
    assert (assets_root / "usd" / "smpl_humanoid_lower_body_subject_H182.usda").exists()
    assert (assets_root / "urdf" / "for_retargeting" / "smpl_humanoid_lower_body_subject_H182.urdf").exists()
    assert (assets_root / "subjects" / "smpl_humanoid_lower_body_subject_H182.yaml").exists()


def test_create_motion_manifest_preserves_explicit_metadata_and_fallback_speed(tmp_path):
    motion_dir = tmp_path / "motion_files"
    metadata_dir = motion_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    explicit_motion_path = motion_dir / "custom_stride.motion"
    fallback_motion_path = motion_dir / "S02_30ms_Long.motion"
    torch.save({"rigid_body_pos": torch.zeros((5, 3, 3), dtype=torch.float32)}, explicit_motion_path)
    torch.save({"rigid_body_pos": torch.zeros((4, 3, 3), dtype=torch.float32)}, fallback_motion_path)

    explicit_source_file = tmp_path / "raw" / "subject_trial.csv"
    explicit_source_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir.joinpath("custom_stride.json").write_text(
        json.dumps(
            {
                "subject_id": "S02",
                "trial_name": "custom_stride",
                "speed_mps": 2.75,
                "source_file": str(explicit_source_file),
                "fps": 120,
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "motions_S02.yaml"
    create_motion_manifest(
        motion_files=[fallback_motion_path, explicit_motion_path],
        output_file=manifest_path,
        fps=30,
        subject_id="S02",
        subset_name="speed_subset",
    )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == 1
    assert payload["subject_id"] == "S02"
    assert payload["subset_name"] == "speed_subset"
    assert payload["fps"] == 30
    assert payload["selected_files"] == ["S02_30ms_Long.motion", "custom_stride.motion"]

    fallback_entry, explicit_entry = payload["motions"]
    assert fallback_entry["trial_name"] == "S02_30ms_Long"
    assert fallback_entry["speed_mps"] == pytest.approx(3.0)
    assert fallback_entry["subject_id"] == "S02"
    assert fallback_entry["duration_seconds"] == pytest.approx((4 - 1) / 30.0)
    assert fallback_entry["weight"] == 1.0
    assert fallback_entry["sub_motions"][0]["timings"]["end"] == pytest.approx((4 - 1) / 30.0)

    assert explicit_entry["trial_name"] == "custom_stride"
    assert explicit_entry["speed_mps"] == pytest.approx(2.75)
    assert explicit_entry["subject_id"] == "S02"
    assert explicit_entry["fps"] == 120
    assert explicit_entry["source_file"] == str(explicit_source_file.resolve())
    assert explicit_entry["duration_seconds"] == pytest.approx((5 - 1) / 30.0)
    assert explicit_entry["file"] == str(explicit_motion_path.resolve())


def test_package_motion_library_uses_manifest_path_and_device(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("manifest_version: 1\nmotions: []\n", encoding="utf-8")
    output_path = tmp_path / "packaged" / "motions.pt"

    captured: dict[str, object] = {}

    class FakeMotionLib:
        def __init__(self, config, device):
            captured["motion_file"] = config.motion_file
            captured["device"] = device

        def save_to_file(self, output_file):
            captured["output_file"] = output_file
            Path(output_file).write_text("fake-packaged", encoding="utf-8")

    monkeypatch.setattr(package_module, "MotionLib", FakeMotionLib)

    result = package_motion_library(
        manifest_file=manifest_path,
        output_file=output_path,
        device="cpu",
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "fake-packaged"
    assert captured["motion_file"] == str(manifest_path)
    assert captured["device"] == "cpu"
    assert captured["output_file"] == str(output_path)


def test_generate_experiment_matrix_manifests_uses_explicit_filename_subsets(tmp_path):
    motion_dir = tmp_path / "motion_files"
    metadata_dir = motion_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    trial_names = [
        "S02_15ms_Long",
        "S02_20ms_Long",
        "S02_25ms_Long",
        "S02_30ms_Long",
        "S02_35ms_Long",
        "S02_40ms_Long",
        "S02_45ms_Long",
        "S02_50ms_Long",
    ]

    motion_files = []
    for trial_name in trial_names:
        motion_path = motion_dir / f"{trial_name}.motion"
        torch.save(
            {"rigid_body_pos": torch.zeros((5, 3, 3), dtype=torch.float32)},
            motion_path,
        )
        metadata_dir.joinpath(f"{trial_name}.json").write_text(
            json.dumps(
                {
                    "subject_id": "S02",
                    "trial_name": trial_name,
                    "speed_mps": float(trial_name.split("_")[1].replace("ms", "")) / 10.0,
                    "source_file": str((tmp_path / "raw" / f"{trial_name}.csv").resolve()),
                    "fps": 30,
                }
            ),
            encoding="utf-8",
        )
        motion_files.append(motion_path)

    master_manifest = tmp_path / "motions_S02.yaml"
    create_motion_manifest(
        motion_files=motion_files,
        output_file=master_manifest,
        fps=30,
        subject_id="S02",
        subset_name="all_8",
    )

    output_dir = tmp_path / "experiment_matrix"
    result = generate_experiment_matrix_manifests(
        master_manifest=master_manifest,
        output_dir=output_dir,
    )

    assert set(result.keys()) == {
        "all_8",
        "every_other",
        "anchor_3",
        "speed_2",
        "leave_edge_low",
        "leave_edge_high",
        "loo_15",
        "loo_20",
        "loo_25",
        "loo_30",
        "loo_35",
        "loo_40",
        "loo_45",
        "loo_50",
    }

    every_other_payload = yaml.safe_load(result["every_other"].read_text(encoding="utf-8"))
    speed_2_payload = yaml.safe_load(result["speed_2"].read_text(encoding="utf-8"))
    leave_edge_low_payload = yaml.safe_load(result["leave_edge_low"].read_text(encoding="utf-8"))
    loo_35_payload = yaml.safe_load(result["loo_35"].read_text(encoding="utf-8"))

    assert every_other_payload["selected_files"] == [
        "S02_20ms_Long.motion",
        "S02_30ms_Long.motion",
        "S02_40ms_Long.motion",
        "S02_50ms_Long.motion",
    ]
    assert speed_2_payload["selected_files"] == [
        "S02_15ms_Long.motion",
        "S02_35ms_Long.motion",
    ]
    assert leave_edge_low_payload["selected_files"] == [
        "S02_20ms_Long.motion",
        "S02_25ms_Long.motion",
        "S02_30ms_Long.motion",
        "S02_35ms_Long.motion",
        "S02_40ms_Long.motion",
        "S02_45ms_Long.motion",
        "S02_50ms_Long.motion",
    ]
    assert loo_35_payload["selected_files"] == [
        "S02_15ms_Long.motion",
        "S02_20ms_Long.motion",
        "S02_25ms_Long.motion",
        "S02_30ms_Long.motion",
        "S02_40ms_Long.motion",
        "S02_45ms_Long.motion",
        "S02_50ms_Long.motion",
    ]
