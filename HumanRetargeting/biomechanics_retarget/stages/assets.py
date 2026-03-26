"""Asset and profile materialization helpers for the production pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from HumanRetargeting.biomechanics_retarget.subject_assets import (
    SubjectAssetBuilder,
    SubjectAssets,
)
from HumanRetargeting.biomechanics_retarget.subject_profiles import (
    SubjectProfile,
    load_subject_profile,
    materialize_height_subject_profile,
    subject_profile_to_yaml_data,
)


def resolve_subject_profile(
    *,
    input_dir: Path,
    output_dir: Path,
    subject_profile_path: Path | None,
    height_cm: int | None,
    subject_id: str | None,
    model_variant: str,
    fps: int,
    output_fps: int,
    coordinate_transform: str,
    contact_source: str,
) -> tuple[SubjectProfile, Path, bool]:
    """Load or generate the effective subject profile for one run."""
    output_profile_path = output_dir / "profile.yaml"

    if subject_profile_path is not None:
        profile = load_subject_profile(subject_profile_path)
        profile = replace(
            profile,
            input_dir=input_dir.resolve(),
            profile_path=output_profile_path.resolve(),
        )
        output_profile_path.parent.mkdir(parents=True, exist_ok=True)
        output_profile_path.write_text(
            yaml.safe_dump(subject_profile_to_yaml_data(profile), sort_keys=False),
            encoding="utf-8",
        )
        return profile, output_profile_path, False

    if height_cm is None:
        raise ValueError("Either subject_profile_path or height_cm must be provided")

    profile = materialize_height_subject_profile(
        input_dir=input_dir.resolve(),
        output_path=output_profile_path,
        height_cm=height_cm,
        subject_id=subject_id,
        model_variant=model_variant,
        fps=fps,
        output_fps=output_fps,
        coordinate_transform=coordinate_transform,
        contact_source=contact_source,
    )
    return profile, output_profile_path, True


def build_subject_assets(
    *,
    profile: SubjectProfile,
    rescale_dir: Path,
    assets_root: Path,
    force: bool,
) -> tuple[SubjectAssets, str, dict[str, Any]]:
    """Build or reuse deterministic subject assets and summarize them for reports."""
    builder = SubjectAssetBuilder(
        profile=profile,
        rescale_dir=rescale_dir,
        assets_root=assets_root,
    )
    assets = builder.build(force=force)
    summary = {
        "mjcf": str(assets.mjcf_path),
        "usda": str(assets.usda_path),
        "urdf": str(assets.urdf_path),
        "metadata": str(assets.metadata_path),
        "default_root_height": assets.default_root_height,
    }
    return assets, f"smpl_lower_body_subject_{profile.subject_id}", summary
