#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Subject profiles and study manifests for biomechanics retargeting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_glob(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return value
    return str((base_dir / value).resolve())


@dataclass(slots=True)
class SubjectProfile:
    """Typed subject profile for subject-aware asset generation and processing."""

    subject_id: str
    input_dir: Path
    height_cm: int
    pelvis_width_m: float
    thigh_length_m: float
    shank_length_m: float
    foot_length_m: float
    model_variant: str = "adjusted_pd"
    fps: int = 200
    output_fps: int = 30
    coordinate_transform: str = "y_to_x_forward"
    contact_source: str = "auto"
    trial_glob: str = "*.txt"
    speed_source: str = "filename"
    trial_speed_overrides: dict[str, float] = field(default_factory=dict)
    mass_kg: float | None = None
    foot_width_m: float | None = None
    left_thigh_length_m: float | None = None
    right_thigh_length_m: float | None = None
    left_shank_length_m: float | None = None
    right_shank_length_m: float | None = None
    left_foot_length_m: float | None = None
    right_foot_length_m: float | None = None
    left_foot_width_m: float | None = None
    right_foot_width_m: float | None = None
    grf_glob: str | None = None
    event_glob: str | None = None
    contact_pads: bool = False
    profile_path: Path | None = None

    def __post_init__(self) -> None:
        if self.height_cm <= 0:
            raise ValueError(f"height_cm must be positive, got {self.height_cm}")
        if self.contact_source not in {"auto", "kinetics", "heuristic"}:
            raise ValueError(
                "contact_source must be 'auto', 'kinetics', or 'heuristic', "
                f"got {self.contact_source!r}"
            )

    @property
    def thigh_lengths_m(self) -> tuple[float, float]:
        return (
            self.left_thigh_length_m or self.thigh_length_m,
            self.right_thigh_length_m or self.thigh_length_m,
        )

    @property
    def shank_lengths_m(self) -> tuple[float, float]:
        return (
            self.left_shank_length_m or self.shank_length_m,
            self.right_shank_length_m or self.shank_length_m,
        )

    @property
    def foot_lengths_m(self) -> tuple[float, float]:
        return (
            self.left_foot_length_m or self.foot_length_m,
            self.right_foot_length_m or self.foot_length_m,
        )

    @property
    def foot_widths_m(self) -> tuple[float | None, float | None]:
        return (
            self.left_foot_width_m or self.foot_width_m,
            self.right_foot_width_m or self.foot_width_m,
        )

    def as_metadata(self) -> dict[str, Any]:
        """Serialize profile data to metadata YAML/JSON."""
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        if self.profile_path is not None:
            data["profile_path"] = str(self.profile_path)
        return data

    def trial_speed_override(self, trial_stem: str) -> float | None:
        """Return an optional speed override for one trial stem."""
        return self.trial_speed_overrides.get(trial_stem)


@dataclass(slots=True)
class StudyManifest:
    """Study-level manifest for batch processing."""

    manifest_path: Path
    output_root: Path | None
    defaults: dict[str, Any]
    subjects: list[SubjectProfile]


def _merge_defaults(defaults: dict[str, Any], subject_data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in subject_data.items():
        if key == "trial_speed_overrides":
            combined = dict(merged.get(key, {}))
            combined.update(value or {})
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def subject_profile_from_dict(
    subject_data: dict[str, Any],
    *,
    base_dir: Path,
    defaults: dict[str, Any] | None = None,
    profile_path: Path | None = None,
) -> SubjectProfile:
    """Create a subject profile from inline YAML data."""
    merged = _merge_defaults(defaults or {}, subject_data)
    anthropometry = dict(merged.pop("anthropometry", {}) or {})
    for key, value in anthropometry.items():
        merged.setdefault(key, value)
    input_dir = _resolve_path(base_dir, merged["input_dir"])
    if input_dir is None:
        raise ValueError("subject profile must define input_dir")

    profile = SubjectProfile(
        subject_id=str(merged["subject_id"]),
        input_dir=input_dir,
        height_cm=int(merged["height_cm"]),
        pelvis_width_m=float(merged["pelvis_width_m"]),
        thigh_length_m=float(merged["thigh_length_m"]),
        shank_length_m=float(merged["shank_length_m"]),
        foot_length_m=float(merged["foot_length_m"]),
        model_variant=str(merged.get("model_variant", "adjusted_pd")),
        fps=int(merged.get("fps", 200)),
        output_fps=int(merged.get("output_fps", 30)),
        coordinate_transform=str(merged.get("coordinate_transform", "y_to_x_forward")),
        contact_source=str(merged.get("contact_source", "auto")),
        trial_glob=str(merged.get("trial_glob", "*.txt")),
        speed_source=str(merged.get("speed_source", "filename")),
        trial_speed_overrides={
            str(key): float(value)
            for key, value in (merged.get("trial_speed_overrides") or {}).items()
        },
        mass_kg=(
            float(merged["mass_kg"])
            if merged.get("mass_kg") is not None
            else None
        ),
        foot_width_m=(
            float(merged["foot_width_m"])
            if merged.get("foot_width_m") is not None
            else None
        ),
        left_thigh_length_m=(
            float(merged["left_thigh_length_m"])
            if merged.get("left_thigh_length_m") is not None
            else None
        ),
        right_thigh_length_m=(
            float(merged["right_thigh_length_m"])
            if merged.get("right_thigh_length_m") is not None
            else None
        ),
        left_shank_length_m=(
            float(merged["left_shank_length_m"])
            if merged.get("left_shank_length_m") is not None
            else None
        ),
        right_shank_length_m=(
            float(merged["right_shank_length_m"])
            if merged.get("right_shank_length_m") is not None
            else None
        ),
        left_foot_length_m=(
            float(merged["left_foot_length_m"])
            if merged.get("left_foot_length_m") is not None
            else None
        ),
        right_foot_length_m=(
            float(merged["right_foot_length_m"])
            if merged.get("right_foot_length_m") is not None
            else None
        ),
        left_foot_width_m=(
            float(merged["left_foot_width_m"])
            if merged.get("left_foot_width_m") is not None
            else None
        ),
        right_foot_width_m=(
            float(merged["right_foot_width_m"])
            if merged.get("right_foot_width_m") is not None
            else None
        ),
        grf_glob=_resolve_glob(base_dir, merged.get("grf_glob")),
        event_glob=_resolve_glob(base_dir, merged.get("event_glob")),
        contact_pads=bool(merged.get("contact_pads", False)),
        profile_path=profile_path,
    )
    return profile


def load_subject_profile(profile_path: Path, defaults: dict[str, Any] | None = None) -> SubjectProfile:
    """Load a subject profile YAML file."""
    profile_path = profile_path.resolve()
    with open(profile_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"subject profile at {profile_path} must be a mapping")
    return subject_profile_from_dict(
        data,
        base_dir=profile_path.parent,
        defaults=defaults,
        profile_path=profile_path,
    )


def load_study_manifest(manifest_path: Path) -> StudyManifest:
    """Load a study manifest with inline or referenced subject profiles."""
    manifest_path = manifest_path.resolve()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"study manifest at {manifest_path} must be a mapping")

    defaults = dict(data.get("defaults") or {})
    base_dir = manifest_path.parent
    subjects: list[SubjectProfile] = []
    for entry in data.get("subjects") or []:
        if isinstance(entry, str):
            subjects.append(load_subject_profile(base_dir / entry, defaults))
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                "study manifest subjects entries must be inline mappings or profile paths"
            )
        if "profile" in entry:
            subjects.append(load_subject_profile(base_dir / str(entry["profile"]), defaults))
            continue
        subjects.append(
            subject_profile_from_dict(
                entry,
                base_dir=base_dir,
                defaults=defaults,
            )
        )

    output_root = _resolve_path(base_dir, data.get("output_root"))
    return StudyManifest(
        manifest_path=manifest_path,
        output_root=output_root,
        defaults=defaults,
        subjects=subjects,
    )
