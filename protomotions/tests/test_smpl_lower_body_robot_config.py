# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from protomotions.robot_configs.factory import robot_config
from protomotions.robot_configs.smpl_lower_body import SmplLowerBodyConfigFactory

from HumanRetargeting.biomechanics_retarget.subject_assets import SubjectAssetBuilder
from HumanRetargeting.biomechanics_retarget.subject_profiles import load_subject_profile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smpl_lower_body_base_robot_config_loads():
    config = robot_config("smpl_lower_body")

    assert config.kinematic_info.num_dofs == 24
    assert config.kinematic_info.num_bodies == 9
    assert Path(config.asset.asset_root, config.asset.asset_file_name).exists()


def test_smpl_lower_body_subject_robot_config_loads_generated_assets(tmp_path):
    profile = load_subject_profile(
        REPO_ROOT / "HumanRetargeting" / "biomechanics_retarget" / "profiles" / "S_GENERIC.yaml"
    )
    builder = SubjectAssetBuilder(
        profile=profile,
        rescale_dir=REPO_ROOT / "HumanRetargeting" / "rescale",
        assets_root=tmp_path / "assets",
    )
    assets = builder.build(force=True)

    config = SmplLowerBodyConfigFactory.create(
        subject_id=profile.subject_id,
        asset_root=assets.asset_root,
    )

    assert Path(config.asset.asset_root, config.asset.asset_file_name).exists()
    assert config.default_root_height > 0.0
    assert config.kinematic_info.num_dofs == 24
