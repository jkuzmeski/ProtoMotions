"""Motion packaging stage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from HumanRetargeting.biomechanics_retarget.subject_profiles import (
    build_trial_metadata_payload,
    load_json_metadata,
    resolve_trial_speed_mps,
)


EXPERIMENT_MATRIX_SUBSETS: tuple[str, ...] = (
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
)


def _motion_metadata_path(motion_path: Path) -> Path:
    return motion_path.parent / "metadata" / f"{motion_path.stem}.json"


def _motion_duration_seconds(motion: dict[str, object], fps: int) -> float:
    rigid_body_pos = motion["rigid_body_pos"]
    num_frames = int(getattr(rigid_body_pos, "shape")[0])
    return (num_frames - 1) / float(fps) if num_frames > 0 else 0.0


def _load_manifest(manifest_file: Path) -> dict[str, Any]:
    data = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"motion manifest at {manifest_file} must be a mapping")
    motions = data.get("motions")
    if not isinstance(motions, list):
        raise ValueError(f"motion manifest at {manifest_file} must define a motions list")
    return data


def _manifest_entries_by_selected_file(manifest_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for motion_entry in manifest_data.get("motions", []):
        if not isinstance(motion_entry, dict) or "file" not in motion_entry:
            raise ValueError("motion manifest entries must be mappings containing a file field")
        selected_file = Path(str(motion_entry["file"])).name
        if selected_file in entries:
            raise ValueError(f"duplicate motion file in manifest: {selected_file}")
        entries[selected_file] = motion_entry
    return entries


def _manifest_selected_files(manifest_data: dict[str, Any]) -> list[str]:
    selected_files = manifest_data.get("selected_files")
    if selected_files is not None:
        if not isinstance(selected_files, list):
            raise ValueError("selected_files must be a list when present")
        return [str(item) for item in selected_files]
    return [Path(str(entry["file"])).name for entry in manifest_data["motions"]]


def _subset_payload(
    *,
    manifest_data: dict[str, Any],
    selected_files: Iterable[str],
    subset_name: str,
    source_manifest: Path,
) -> dict[str, Any]:
    entries_by_file = _manifest_entries_by_selected_file(manifest_data)
    ordered_files = list(selected_files)
    motions: list[dict[str, Any]] = []
    for selected_file in ordered_files:
        if selected_file not in entries_by_file:
            raise KeyError(
                f"subset {subset_name!r} references {selected_file!r}, which is not present in "
                f"{source_manifest}"
            )
        motions.append(dict(entries_by_file[selected_file]))

    return {
        "manifest_version": int(manifest_data.get("manifest_version", 1)),
        "subject_id": manifest_data.get("subject_id"),
        "subset_name": subset_name,
        "source_manifest": str(source_manifest.resolve()),
        "fps": manifest_data.get("fps"),
        "selected_files": ordered_files,
        "motions": motions,
    }


def create_motion_manifest_from_selected_files(
    *,
    source_manifest: Path,
    output_file: Path,
    selected_files: Iterable[str],
    subset_name: str,
) -> Path:
    """Derive a YAML subset manifest from explicit motion filenames."""
    manifest_data = _load_manifest(source_manifest)
    payload = _subset_payload(
        manifest_data=manifest_data,
        selected_files=selected_files,
        subset_name=subset_name,
        source_manifest=source_manifest,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_file


def build_experiment_matrix_subset_specs(selected_files: list[str]) -> dict[str, list[str]]:
    """Build the frozen experiment matrix from the canonical eight trial filenames."""
    if len(selected_files) != 8:
        raise ValueError(
            "experiment matrix generation expects exactly 8 canonical trial files; "
            f"got {len(selected_files)}"
        )

    return {
        "all_8": list(selected_files),
        "every_other": [selected_files[i] for i in (1, 3, 5, 7)],
        "anchor_3": [selected_files[i] for i in (0, 3, 7)],
        "speed_2": [selected_files[i] for i in (0, 4)],
        "leave_edge_low": selected_files[1:],
        "leave_edge_high": selected_files[:-1],
        "loo_15": [selected_files[i] for i in range(1, 8)],
        "loo_20": [selected_files[i] for i in (0, 2, 3, 4, 5, 6, 7)],
        "loo_25": [selected_files[i] for i in (0, 1, 3, 4, 5, 6, 7)],
        "loo_30": [selected_files[i] for i in (0, 1, 2, 4, 5, 6, 7)],
        "loo_35": [selected_files[i] for i in (0, 1, 2, 3, 5, 6, 7)],
        "loo_40": [selected_files[i] for i in (0, 1, 2, 3, 4, 6, 7)],
        "loo_45": [selected_files[i] for i in (0, 1, 2, 3, 4, 5, 7)],
        "loo_50": [selected_files[i] for i in range(7)],
    }


def generate_experiment_matrix_manifests(
    *,
    master_manifest: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate the frozen experiment-matrix subset manifests from a master manifest."""
    manifest_data = _load_manifest(master_manifest)
    selected_files = _manifest_selected_files(manifest_data)
    specs = build_experiment_matrix_subset_specs(selected_files)

    output_paths: dict[str, Path] = {}
    for subset_name, subset_files in specs.items():
        subset_file = output_dir / f"{subset_name}.yaml"
        create_motion_manifest_from_selected_files(
            source_manifest=master_manifest,
            output_file=subset_file,
            selected_files=subset_files,
            subset_name=subset_name,
        )
        output_paths[subset_name] = subset_file
    return output_paths


def _build_motion_entry(
    *,
    motion_path: Path,
    fps: int,
    subject_id: str | None,
) -> tuple[dict[str, object], str]:
    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    metadata = load_json_metadata(_motion_metadata_path(motion_path))
    speed_mps = resolve_trial_speed_mps(
        motion_path.stem,
        speed_mps=metadata.get("speed_mps"),
        metadata=metadata,
    )

    duration_seconds = _motion_duration_seconds(motion, fps)
    source_file = metadata.get("source_file") or str(motion_path.resolve())
    entry_metadata = build_trial_metadata_payload(
        subject_id=str(subject_id or metadata.get("subject_id") or ""),
        trial_name=str(metadata.get("trial_name") or motion_path.stem),
        speed_mps=speed_mps,
        source_file=source_file,
        fps=int(metadata.get("fps", fps)),
        duration_seconds=duration_seconds,
    )
    entry_metadata["file"] = str(motion_path.resolve())
    entry_metadata["weight"] = 1.0
    entry_metadata["sub_motions"] = [
        {
            "idx": 0,
            "timings": {"start": 0.0, "end": duration_seconds},
            "weight": 1.0,
        }
    ]
    return entry_metadata, motion_path.name


def create_motion_manifest(
    *,
    motion_files: list[Path],
    output_file: Path,
    fps: int,
    subject_id: str | None = None,
    subset_name: str | None = None,
) -> Path:
    """Create a reproducible YAML manifest for a packaged motion library."""
    motions = []
    selected_files: list[str] = []
    for motion_path in sorted(motion_files):
        entry, selected_name = _build_motion_entry(
            motion_path=motion_path,
            fps=fps,
            subject_id=subject_id,
        )
        motions.append(entry)
        selected_files.append(selected_name)

    payload: dict[str, object] = {
        "manifest_version": 1,
        "subject_id": subject_id,
        "subset_name": subset_name or "all",
        "fps": fps,
        "selected_files": selected_files,
        "motions": motions,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output_file


def package_motion_library(
    *,
    manifest_file: Path,
    output_file: Path,
    device: str = "cpu",
) -> Path:
    """Package motion files into a MotionLib .pt file."""
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(manifest_file)), device=device)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    motion_lib.save_to_file(str(output_file))
    return output_file
