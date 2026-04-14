#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Subject profiles and study manifests for biomechanics retargeting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


_SPEED_FILENAME_RE = re.compile(r"(?P<speed>\d+(?:\.\d+)?)ms(?:[_.-]|$)", re.IGNORECASE)


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


def load_json_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load a JSON sidecar if present, returning an empty mapping when absent."""
    if not metadata_path.exists():
        return {}

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metadata sidecar at {metadata_path} must be a JSON object")
    return data


def parse_speed_mps_from_filename(filename: str) -> float | None:
    """Parse a treadmill speed from a trial filename as a fallback only."""
    match = _SPEED_FILENAME_RE.search(filename)
    if not match:
        return None
    raw_speed = match.group("speed")
    speed = float(raw_speed)
    return speed if "." in raw_speed else speed / 10.0


def resolve_trial_speed_mps(
    trial_name: str,
    *,
    speed_mps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> float | None:
    """Resolve a trial speed from explicit metadata, then filename fallback."""
    if speed_mps is not None:
        return float(speed_mps)

    if metadata is not None:
        metadata_speed = metadata.get("speed_mps")
        if metadata_speed is not None:
            return float(metadata_speed)

    return parse_speed_mps_from_filename(trial_name)


def speed_mps_slug(speed_mps: float | None) -> str:
    """Return a stable filename-safe slug for a speed value."""
    if speed_mps is None:
        return "unknown"
    return f"{speed_mps:g}".replace("-", "neg").replace(".", "p")


def build_trial_metadata_payload(
    *,
    subject_id: str,
    trial_name: str,
    speed_mps: float | None,
    source_file: Path | str,
    fps: int,
    duration_seconds: float,
) -> dict[str, Any]:
    """Build the canonical per-trial metadata payload."""
    return {
        "subject_id": subject_id,
        "trial_name": trial_name,
        "speed_mps": float(speed_mps) if speed_mps is not None else None,
        "source_file": str(Path(source_file).resolve()),
        "fps": int(fps),
        "duration_seconds": float(duration_seconds),
    }


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
    profile_mode: str = "file"

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
        profile_mode=str(merged.get("profile_mode", "file")),
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


def subject_profile_to_yaml_data(profile: SubjectProfile) -> dict[str, Any]:
    """Materialize a profile into the canonical persisted YAML structure."""
    data: dict[str, Any] = {
        "subject_id": profile.subject_id,
        "input_dir": str(profile.input_dir.resolve()),
        "model_variant": profile.model_variant,
        "fps": profile.fps,
        "output_fps": profile.output_fps,
        "coordinate_transform": profile.coordinate_transform,
        "contact_source": profile.contact_source,
        "trial_glob": profile.trial_glob,
        "speed_source": profile.speed_source,
        "profile_mode": profile.profile_mode,
        "anthropometry": {
            "height_cm": profile.height_cm,
            "pelvis_width_m": profile.pelvis_width_m,
            "thigh_length_m": profile.thigh_length_m,
            "shank_length_m": profile.shank_length_m,
            "foot_length_m": profile.foot_length_m,
        },
    }

    if profile.mass_kg is not None:
        data["mass_kg"] = profile.mass_kg
    if profile.foot_width_m is not None:
        data["foot_width_m"] = profile.foot_width_m
    if profile.left_thigh_length_m is not None:
        data["left_thigh_length_m"] = profile.left_thigh_length_m
    if profile.right_thigh_length_m is not None:
        data["right_thigh_length_m"] = profile.right_thigh_length_m
    if profile.left_shank_length_m is not None:
        data["left_shank_length_m"] = profile.left_shank_length_m
    if profile.right_shank_length_m is not None:
        data["right_shank_length_m"] = profile.right_shank_length_m
    if profile.left_foot_length_m is not None:
        data["left_foot_length_m"] = profile.left_foot_length_m
    if profile.right_foot_length_m is not None:
        data["right_foot_length_m"] = profile.right_foot_length_m
    if profile.left_foot_width_m is not None:
        data["left_foot_width_m"] = profile.left_foot_width_m
    if profile.right_foot_width_m is not None:
        data["right_foot_width_m"] = profile.right_foot_width_m
    if profile.grf_glob is not None:
        data["grf_glob"] = profile.grf_glob
    if profile.event_glob is not None:
        data["event_glob"] = profile.event_glob
    if profile.contact_pads:
        data["contact_pads"] = True
    if profile.trial_speed_overrides:
        data["trial_speed_overrides"] = dict(profile.trial_speed_overrides)

    anthropometry = data["anthropometry"]
    if profile.foot_width_m is not None:
        anthropometry["foot_width_m"] = profile.foot_width_m
    if profile.left_thigh_length_m is not None:
        anthropometry["left_thigh_length_m"] = profile.left_thigh_length_m
    if profile.right_thigh_length_m is not None:
        anthropometry["right_thigh_length_m"] = profile.right_thigh_length_m
    if profile.left_shank_length_m is not None:
        anthropometry["left_shank_length_m"] = profile.left_shank_length_m
    if profile.right_shank_length_m is not None:
        anthropometry["right_shank_length_m"] = profile.right_shank_length_m
    if profile.left_foot_length_m is not None:
        anthropometry["left_foot_length_m"] = profile.left_foot_length_m
    if profile.right_foot_length_m is not None:
        anthropometry["right_foot_length_m"] = profile.right_foot_length_m
    if profile.left_foot_width_m is not None:
        anthropometry["left_foot_width_m"] = profile.left_foot_width_m
    if profile.right_foot_width_m is not None:
        anthropometry["right_foot_width_m"] = profile.right_foot_width_m

    return data


def _load_template_profile(template_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"template profile at {template_path} must be a mapping")
    return data


def materialize_height_subject_profile(
    *,
    input_dir: Path,
    output_path: Path,
    height_cm: int,
    subject_id: str | None,
    model_variant: str,
    fps: int,
    output_fps: int,
    coordinate_transform: str,
    contact_source: str,
    template_profile_path: Path | None = None,
) -> SubjectProfile:
    """Generate and persist a canonical height-only subject profile."""
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")

    if template_profile_path is None:
        template_profile_path = (
            Path(__file__).resolve().parent
            / "profiles"
            / "templates"
            / "generic_lower_body.yaml"
        )
    template = _load_template_profile(template_profile_path)
    anthropometry = dict(template.get("anthropometry") or {})
    base_height_cm = int(anthropometry["height_cm"])
    scale = float(height_cm) / float(base_height_cm)

    resolved_subject_id = subject_id or f"H{height_cm}"
    profile = SubjectProfile(
        subject_id=resolved_subject_id,
        input_dir=input_dir.resolve(),
        height_cm=height_cm,
        pelvis_width_m=float(anthropometry["pelvis_width_m"]) * scale,
        thigh_length_m=float(anthropometry["thigh_length_m"]) * scale,
        shank_length_m=float(anthropometry["shank_length_m"]) * scale,
        foot_length_m=float(anthropometry["foot_length_m"]) * scale,
        model_variant=model_variant,
        fps=fps,
        output_fps=output_fps,
        coordinate_transform=coordinate_transform,
        contact_source=contact_source,
        trial_glob=str(template.get("trial_glob", "*.txt")),
        speed_source=str(template.get("speed_source", "filename")),
        mass_kg=(
            float(template["mass_kg"])
            if template.get("mass_kg") is not None
            else None
        ),
        foot_width_m=(
            float(anthropometry["foot_width_m"]) * scale
            if anthropometry.get("foot_width_m") is not None
            else None
        ),
        profile_path=output_path.resolve(),
        profile_mode="generated_from_height",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(subject_profile_to_yaml_data(profile), sort_keys=False),
        encoding="utf-8",
    )
    return profile


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
