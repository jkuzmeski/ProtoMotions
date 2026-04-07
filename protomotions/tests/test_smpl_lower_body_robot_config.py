# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil

import newton
import pytest

from protomotions.robot_configs.factory import robot_config
import protomotions.robot_configs.smpl_lower_body as smpl_lower_body_config_module
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.newton.simulator import _configure_pressure_field_foot_shapes

from HumanRetargeting.biomechanics_retarget.subject_assets import SubjectAssetBuilder
from HumanRetargeting.biomechanics_retarget.subject_profiles import load_subject_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smpl_lower_body_base_robot_config_loads():
    config = robot_config("smpl_lower_body")

    assert config.kinematic_info.num_dofs == 24
    assert config.kinematic_info.num_bodies == 9
    assert Path(config.asset.asset_root, config.asset.asset_file_name).exists()


def test_smpl_lower_body_ellipsoid_feet_robot_config_loads():
    config = robot_config("smpl_lower_body_ellipsoid_feet")

    assert config.kinematic_info.num_dofs == 24
    assert config.kinematic_info.num_bodies == 9
    assert Path(config.asset.asset_root, config.asset.asset_file_name).exists()
    assert config.contact_bodies == ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]


def test_smpl_lower_body_ellipsoid_feet_asset_imports_all_foot_shapes():
    config = robot_config("smpl_lower_body_ellipsoid_feet")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig()
    builder.add_mjcf(
        str(Path(config.asset.asset_root) / config.asset.asset_file_name),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        floating=True,
        enable_self_collisions=False,
    )

    assert builder.shape_count == 11
    shape_labels = list(builder.shape_label)
    assert any(label.endswith("L_Ankle_geom_0") for label in shape_labels)
    assert any(label.endswith("L_Ankle_geom_1") for label in shape_labels)
    assert any(label.endswith("L_Toe_geom_0") for label in shape_labels)
    assert any(label.endswith("R_Ankle_geom_0") for label in shape_labels)
    assert any(label.endswith("R_Ankle_geom_1") for label in shape_labels)
    assert any(label.endswith("R_Toe_geom_0") for label in shape_labels)


def test_smpl_lower_body_ellipsoid_feet_pressure_field_shapes_configure_cleanly():
    config = robot_config("smpl_lower_body_ellipsoid_feet")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig()
    builder.add_mjcf(
        str(Path(config.asset.asset_root) / config.asset.asset_file_name),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        floating=True,
        enable_self_collisions=False,
    )

    sim_params = NewtonSimParams(
        pressure_field_feet=True,
        pressure_field_foot_sdf_max_resolution=32,
        pressure_field_foot_kh=2.5e7,
    )
    foot_shape_ids = _configure_pressure_field_foot_shapes(
        builder,
        config.contact_bodies,
        sim_params,
    )

    assert len(foot_shape_ids) == 6
    for shape_id in foot_shape_ids:
        assert builder.shape_flags[shape_id] & int(newton.ShapeFlags.HYDROELASTIC)
        assert builder.shape_flags[shape_id] & int(newton.ShapeFlags.HYDROELASTIC_COMPLIANT)
        assert not builder.shape_flags[shape_id] & int(newton.ShapeFlags.HYDROELASTIC_RIGID)
        assert builder.shape_sdf_max_resolution[shape_id] == 32
        assert builder.shape_sdf_target_voxel_size[shape_id] is None
        assert builder.shape_material_kh[shape_id] == pytest.approx(2.5e7)
        assert builder.shape_hydroelastic_contact_workflow[shape_id] == int(
            newton.HydroelasticContactWorkflow.PRESSURE
        )


def test_smpl_lower_body_subject_robot_config_loads_generated_assets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    profile = load_subject_profile(
        REPO_ROOT / "HumanRetargeting" / "biomechanics_retarget" / "profiles" / "S_GENERIC.yaml"
    )
    builder = SubjectAssetBuilder(
        profile=profile,
        rescale_dir=REPO_ROOT / "HumanRetargeting" / "rescale",
        assets_root=tmp_path / "assets",
    )
    assets = builder.build(force=True)

    fake_repo_root = tmp_path / "repo"
    fake_assets_root = fake_repo_root / "protomotions" / "data" / "assets"
    shutil.copytree(assets.asset_root, fake_assets_root, dirs_exist_ok=True)

    monkeypatch.setattr(smpl_lower_body_config_module, "REPO_ROOT", fake_repo_root)
    config = robot_config(f"smpl_lower_body_subject_{profile.subject_id}")

    assert Path(config.asset.asset_root, config.asset.asset_file_name).exists()
    assert config.default_root_height > 0.0
    assert config.kinematic_info.num_dofs == 24
