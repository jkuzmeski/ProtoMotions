#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Production single-subject biomechanics retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import typer
from rich.console import Console

from HumanRetargeting.biomechanics_retarget.contact_sources import load_trial_contacts
from HumanRetargeting.biomechanics_retarget.stages.assets import (
    build_subject_assets,
    resolve_subject_profile,
)
from HumanRetargeting.biomechanics_retarget.stages.convert import (
    run_motion_conversion,
)
from HumanRetargeting.biomechanics_retarget.stages.keypoints import (
    run_keypoint_extraction,
)
from HumanRetargeting.biomechanics_retarget.extract_keypoints_from_overground import (
    extract_anthropometry_from_keypoints,
)
from HumanRetargeting.biomechanics_retarget.stages.overground import (
    run_overground_trial,
)
from HumanRetargeting.biomechanics_retarget.stages.package import (
    create_motion_manifest,
    generate_experiment_matrix_manifests,
    package_motion_library,
)
from HumanRetargeting.biomechanics_retarget.stages.retarget import (
    resolve_pyroki_runtime,
    run_pyroki_retarget_trial,
    verify_pyroki_runtime,
)
from HumanRetargeting.biomechanics_retarget.subject_assets import SubjectAssets
from HumanRetargeting.biomechanics_retarget.subject_profiles import SubjectProfile
from HumanRetargeting.biomechanics_retarget.subject_profiles import (
    build_trial_metadata_payload,
    load_json_metadata,
    resolve_trial_speed_mps,
    speed_mps_slug,
)
from HumanRetargeting.biomechanics_retarget.validation import (
    ensure_validation_passed,
    validate_motion_file,
    validate_packaged_motion_lib,
    validate_retargeted_npz,
    write_validation_report,
)


console = Console()
app = typer.Typer(pretty_exceptions_enable=False)


class PipelineStep(str, Enum):
    """Supported production pipeline stages."""

    ASSETS = "assets"
    OVERGROUND = "overground"
    KEYPOINTS = "keypoints"
    RETARGET = "retarget"
    CONVERT = "convert"
    PACKAGE = "package"
    ALL = "all"


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for one single-subject pipeline run."""

    input_dir: Path
    output_dir: Path
    subject_profile_path: Path | None
    height_cm: int | None
    subject_id: str | None
    model_variant: str
    fps: int
    output_fps: int
    coordinate_transform: str
    contact_source: str
    step: PipelineStep
    force: bool
    pyroki_python: Path | None
    pyroki_script: Path | None
    assets_root: Path
    rescale_dir: Path
    qc_config_file: Path
    skip_qc: bool = False
    export_profile: Path | None = None

    @property
    def profile_output_path(self) -> Path:
        return self.output_dir / "profile.yaml"

    @property
    def overground_dir(self) -> Path:
        return self.output_dir / "overground_data"

    @property
    def keypoints_dir(self) -> Path:
        return self.output_dir / "keypoints"

    @property
    def contacts_dir(self) -> Path:
        return self.output_dir / "contacts"

    @property
    def retargeted_dir(self) -> Path:
        return self.output_dir / "retargeted_motions"

    @property
    def motion_dir(self) -> Path:
        return self.output_dir / "motion_files"

    @property
    def yaml_dir(self) -> Path:
        return self.output_dir / "yaml_data"

    @property
    def packaged_dir(self) -> Path:
        return self.output_dir / "packaged_data"

    @property
    def qc_dir(self) -> Path:
        return self.output_dir / "qc"

    @property
    def qc_keypoints_dir(self) -> Path:
        return self.qc_dir / "keypoints"

    @property
    def qc_retarget_dir(self) -> Path:
        return self.qc_dir / "retarget"

    @property
    def qc_motion_dir(self) -> Path:
        return self.qc_dir / "motion"

    @property
    def qc_package_dir(self) -> Path:
        return self.qc_dir / "package"

    @property
    def subject_summary_path(self) -> Path:
        return self.qc_dir / "subject_summary.json"

    def create_directories(self) -> None:
        for path in (
            self.output_dir,
            self.overground_dir,
            self.keypoints_dir,
            self.contacts_dir,
            self.retargeted_dir,
            self.motion_dir,
            self.yaml_dir,
            self.packaged_dir,
            self.qc_keypoints_dir,
            self.qc_retarget_dir,
            self.qc_motion_dir,
            self.qc_package_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class SubjectContext:
    """Resolved subject-specific production context."""

    profile: SubjectProfile
    profile_path: Path
    generated_profile: bool
    assets: SubjectAssets | None = None
    robot_name: str | None = None
    assets_summary: dict[str, Any] | None = None

    @property
    def model_xml(self) -> Path:
        if self.assets is None:
            raise RuntimeError("Subject assets have not been built yet")
        return self.assets.mjcf_path

    @property
    def retarget_urdf(self) -> Path:
        if self.assets is None:
            raise RuntimeError("Subject assets have not been built yet")
        return self.assets.urdf_path


class ProductionPipeline:
    """Single-subject production pipeline with explicit contract validation."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.config.create_directories()
        self.context: SubjectContext | None = None
        self._trial_metadata: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {
            "status": "running",
            "step": self.config.step.value,
            "input_dir": str(self.config.input_dir.resolve()),
            "output_dir": str(self.config.output_dir.resolve()),
            "qc_config_file": str(self.config.qc_config_file.resolve()),
        }
        self._summary_written = False

    def _write_json(self, output_file: Path, payload: dict[str, Any]) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _trial_metadata_path(self, base_dir: Path, trial_name: str) -> Path:
        return base_dir / "metadata" / f"{trial_name}.json"

    def _load_trial_metadata(self, trial_name: str) -> dict[str, Any]:
        cached = self._trial_metadata.get(trial_name)
        if cached is not None:
            return cached

        for base_dir in (self.config.motion_dir, self.config.overground_dir):
            metadata = load_json_metadata(self._trial_metadata_path(base_dir, trial_name))
            if metadata:
                return metadata
        return {}

    def _write_trial_metadata(self, base_dir: Path, trial_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        metadata_path = self._trial_metadata_path(base_dir, trial_name)
        existing = load_json_metadata(metadata_path)
        merged = {**existing, **payload}
        self._write_json(metadata_path, merged)
        self._trial_metadata[trial_name] = merged
        return merged

    def _trial_speed_mps(self, trial_name: str) -> float | None:
        metadata = self._load_trial_metadata(trial_name)
        return resolve_trial_speed_mps(trial_name, metadata=metadata)

    def _write_subject_summary(self) -> None:
        self._write_json(self.config.subject_summary_path, self.summary)
        self._summary_written = True

    def _resolve_subject_context(self) -> SubjectContext:
        if self.context is not None:
            return self.context

        profile, profile_path, generated = resolve_subject_profile(
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            subject_profile_path=self.config.subject_profile_path,
            height_cm=self.config.height_cm,
            subject_id=self.config.subject_id,
            model_variant=self.config.model_variant,
            fps=self.config.fps,
            output_fps=self.config.output_fps,
            coordinate_transform=self.config.coordinate_transform,
            contact_source=self.config.contact_source,
        )
        if self.config.export_profile is not None:
            self.config.export_profile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(profile_path, self.config.export_profile)

        self.context = SubjectContext(
            profile=profile,
            profile_path=profile_path,
            generated_profile=generated,
        )
        self.summary.update(
            {
                "subject_id": profile.subject_id,
                "profile_path": str(profile_path),
                "generated_profile": generated,
                "robot_name": f"smpl_lower_body_subject_{profile.subject_id}",
            }
        )
        self._write_subject_summary()
        return self.context

    def _ensure_assets(self) -> SubjectContext:
        context = self._resolve_subject_context()
        if context.assets is not None:
            return context

        assets, robot_name, assets_summary = build_subject_assets(
            profile=context.profile,
            rescale_dir=self.config.rescale_dir,
            assets_root=self.config.assets_root,
            force=self.config.force,
        )
        context.assets = assets
        context.robot_name = robot_name
        context.assets_summary = assets_summary
        self.summary.update(
            {
                "robot_name": robot_name,
                "assets": assets_summary,
                "model_xml": str(assets.mjcf_path),
                "retarget_urdf": str(assets.urdf_path),
            }
        )
        self._write_subject_summary()
        return context

    def _find_input_trials(self) -> list[Path]:
        context = self._resolve_subject_context()
        trials = sorted(context.profile.input_dir.glob(context.profile.trial_glob))
        if not trials:
            raise FileNotFoundError(
                f"No treadmill motion files matched {context.profile.trial_glob!r} in "
                f"{context.profile.input_dir}"
            )
        return trials

    def _write_keypoint_report(
        self,
        *,
        trial_stem: str,
        keypoint_file: Path,
        payload: dict[str, Any],
    ) -> None:
        report = {
            "passed": True,
            "failures": [],
            "trial": trial_stem,
            "keypoint_file": str(keypoint_file),
            **payload,
        }
        self._write_json(self.config.qc_keypoints_dir / f"{trial_stem}.json", report)

    def _apply_contact_source_to_keypoints(self, keypoint_file: Path) -> None:
        context = self._resolve_subject_context()
        keypoint_data = np.load(keypoint_file, allow_pickle=True).item()
        positions = np.asarray(keypoint_data["positions"], dtype=np.float32)
        payload: dict[str, Any] = {
            "metrics": {
                "num_frames": int(positions.shape[0]),
                "num_keypoints": int(positions.shape[1]),
            }
        }

        resolved = load_trial_contacts(
            profile=context.profile,
            trial_stem=keypoint_file.stem,
            expected_frames=len(positions),
        )
        if resolved is None:
            keypoint_data["contact_source"] = "heuristic"
            keypoint_data["external_contact_path"] = None
            np.save(keypoint_file, keypoint_data)
            payload["contact_source"] = "heuristic"
            payload["external_contact_path"] = None
            self._write_keypoint_report(
                trial_stem=keypoint_file.stem,
                keypoint_file=keypoint_file,
                payload=payload,
            )
            return

        left_contacts, right_contacts, source_name = resolved
        min_len = min(
            len(keypoint_data["positions"]),
            len(left_contacts),
            len(right_contacts),
        )
        if min_len <= 0:
            raise RuntimeError(f"Resolved empty external contacts for {keypoint_file.name}")

        keypoint_data["positions"] = keypoint_data["positions"][:min_len]
        keypoint_data["orientations"] = keypoint_data["orientations"][:min_len]
        keypoint_data["left_foot_contacts"] = left_contacts[:min_len]
        keypoint_data["right_foot_contacts"] = right_contacts[:min_len]
        keypoint_data["contact_source"] = "kinetics"
        keypoint_data["external_contact_path"] = source_name
        np.save(keypoint_file, keypoint_data)

        payload["contact_source"] = "kinetics"
        payload["external_contact_path"] = source_name
        payload["metrics"]["num_frames"] = min_len
        payload["metrics"]["external_contact_shape"] = [int(min_len), 2]
        self._write_keypoint_report(
            trial_stem=keypoint_file.stem,
            keypoint_file=keypoint_file,
            payload=payload,
        )

    def _resolve_stage_inputs(self, directory: Path, suffix: str) -> list[Path]:
        input_trials = self._find_input_trials()
        files: list[Path] = []
        missing: list[str] = []
        for trial in input_trials:
            stage_file = directory / f"{trial.stem}{suffix}"
            if not stage_file.exists():
                missing.append(stage_file.name)
            else:
                files.append(stage_file)
        if missing:
            raise FileNotFoundError(
                f"Missing required stage inputs in {directory}: {', '.join(missing)}"
            )
        return files

    def run_assets(self) -> SubjectContext:
        context = self._ensure_assets()
        self.summary["completed_step"] = PipelineStep.ASSETS.value
        self.summary["status"] = "ok"
        self._write_subject_summary()
        return context

    def run_overground(self) -> list[Path]:
        context = self._resolve_subject_context()
        output_files: list[Path] = []
        for motion_file in self._find_input_trials():
            output_file = self.config.overground_dir / f"{motion_file.stem}.npy"
            if output_file.exists() and not self.config.force:
                overground_positions = np.load(output_file, mmap_mode="r")
                speed_mps = resolve_trial_speed_mps(
                    motion_file.stem,
                    speed_mps=context.profile.trial_speed_override(motion_file.stem),
                )
                overground_metadata = build_trial_metadata_payload(
                    subject_id=context.profile.subject_id,
                    trial_name=motion_file.stem,
                    speed_mps=speed_mps,
                    source_file=motion_file.resolve(),
                    fps=context.profile.fps,
                    duration_seconds=(
                        float(overground_positions.shape[0]) / float(context.profile.fps)
                        if overground_positions.shape[0] > 0
                        else 0.0
                    ),
                )
                self._write_trial_metadata(self.config.overground_dir, motion_file.stem, overground_metadata)
                output_files.append(output_file)
                continue
            converted = run_overground_trial(
                motion_file=motion_file,
                output_dir=self.config.overground_dir,
                fps=context.profile.fps,
                coordinate_transform=context.profile.coordinate_transform,
                speed_override=context.profile.trial_speed_override(motion_file.stem),
            )
            if converted is None:
                raise RuntimeError(f"Overground conversion failed for {motion_file.name}")
            overground_positions = np.load(converted, mmap_mode="r")
            speed_mps = resolve_trial_speed_mps(
                motion_file.stem,
                speed_mps=context.profile.trial_speed_override(motion_file.stem),
            )
            overground_metadata = build_trial_metadata_payload(
                subject_id=context.profile.subject_id,
                trial_name=motion_file.stem,
                speed_mps=speed_mps,
                source_file=motion_file.resolve(),
                fps=context.profile.fps,
                duration_seconds=(
                    float(overground_positions.shape[0]) / float(context.profile.fps)
                    if overground_positions.shape[0] > 0
                    else 0.0
                ),
            )
            self._write_trial_metadata(self.config.overground_dir, motion_file.stem, overground_metadata)
            output_files.append(converted)

        self.summary["completed_step"] = PipelineStep.OVERGROUND.value
        self.summary["num_overground_files"] = len(output_files)
        self._write_subject_summary()
        return output_files

    def run_keypoints(self) -> list[Path]:
        context = self._resolve_subject_context()
        overground_files = self._resolve_stage_inputs(self.config.overground_dir, ".npy")
        output_files: list[Path] = []
        for overground_file in overground_files:
            output_file = self.config.keypoints_dir / overground_file.name
            if not output_file.exists() or self.config.force:
                run_keypoint_extraction(
                    input_file=overground_file,
                    output_file=output_file,
                    fps=context.profile.fps,
                    output_fps=context.profile.output_fps,
                )
            self._apply_contact_source_to_keypoints(output_file)
            output_files.append(output_file)

        # Auto-calibrate: compare profile anthropometry to data and warn on mismatch.
        measured = extract_anthropometry_from_keypoints(output_files)
        mismatches: list[str] = []
        for key, threshold in (
            ("thigh_length_m", 0.01),
            ("shank_length_m", 0.01),
            ("foot_length_m", 0.008),
            ("pelvis_width_m", 0.01),
        ):
            profile_val = getattr(context.profile, key)
            data_val = measured[key]
            if abs(profile_val - data_val) > threshold:
                mismatches.append(
                    f"  {key}: profile={profile_val:.4f}  data={data_val:.4f}  "
                    f"delta={data_val - profile_val:+.4f} ({100*(data_val - profile_val)/profile_val:+.1f}%)"
                )
        if mismatches:
            console.print(
                "[bold yellow]⚠ Anthropometry mismatch between profile and data:[/bold yellow]"
            )
            for line in mismatches:
                console.print(line)
            console.print(
                "[yellow]Consider updating the subject profile with data-derived "
                "measurements to improve retarget quality.[/yellow]"
            )
        self.summary["measured_anthropometry"] = measured

        self.summary["completed_step"] = PipelineStep.KEYPOINTS.value
        self.summary["num_keypoint_files"] = len(output_files)
        self._write_subject_summary()
        return output_files

    def run_retarget(self) -> list[Path]:
        context = self._ensure_assets()
        keypoint_files = self._resolve_stage_inputs(self.config.keypoints_dir, ".npy")
        python_path, script_path = resolve_pyroki_runtime(
            repo_root=REPO_ROOT,
            retarget_python=self.config.pyroki_python,
            retarget_script=self.config.pyroki_script,
        )
        console.print(f"PyRoki interpreter: {python_path}")
        console.print(f"PyRoki retarget script: {script_path}")
        verify_pyroki_runtime(python_path)

        output_files: list[Path] = []
        for keypoint_file in keypoint_files:
            npz_file, _contact_file = run_pyroki_retarget_trial(
                python_path=python_path,
                script_path=script_path,
                keypoint_file=keypoint_file,
                retargeted_dir=self.config.retargeted_dir,
                contacts_dir=self.config.contacts_dir,
                retarget_fps=context.profile.output_fps,
                retarget_urdf_path=context.retarget_urdf,
                force=self.config.force,
            )
            report = validate_retargeted_npz(
                npz_file=npz_file,
                keypoint_file=keypoint_file,
                model_xml=context.model_xml,
                qc_config_file=self.config.qc_config_file,
            )
            report["trial"] = keypoint_file.stem
            write_validation_report(
                report,
                self.config.qc_retarget_dir / f"{keypoint_file.stem}.json",
            )
            if not self.config.skip_qc:
                ensure_validation_passed(
                    report,
                    f"Retarget validation failed for {keypoint_file.stem}",
                )
            output_files.append(npz_file)

        self.summary["completed_step"] = PipelineStep.RETARGET.value
        self.summary["num_retargeted_files"] = len(output_files)
        self.summary["pyroki_python"] = str(python_path)
        self.summary["pyroki_script"] = str(script_path)
        self._write_subject_summary()
        return output_files

    def run_convert(self) -> list[Path]:
        context = self._ensure_assets()
        npz_files = self._resolve_stage_inputs(self.config.retargeted_dir, "_retargeted.npz")
        output_files: list[Path] = []
        for npz_file in npz_files:
            trial_stem = npz_file.stem.removesuffix("_retargeted")
            output_file = self.config.motion_dir / f"{trial_stem}.motion"
            contact_file = self.config.contacts_dir / f"{trial_stem}_contacts.npz"
            if not output_file.exists() or self.config.force:
                run_motion_conversion(
                    npz_file=npz_file,
                    output_file=output_file,
                    model_xml=context.model_xml,
                    input_fps=context.profile.output_fps,
                    output_fps=context.profile.output_fps,
                    contact_file=contact_file if contact_file.exists() else None,
                    apply_motion_filter=False,
                )
            trial_metadata = self._load_trial_metadata(trial_stem)
            motion_data = torch.load(output_file, map_location="cpu", weights_only=False)
            speed_mps = resolve_trial_speed_mps(trial_stem, metadata=trial_metadata)
            motion_metadata = build_trial_metadata_payload(
                subject_id=context.profile.subject_id,
                trial_name=trial_stem,
                speed_mps=speed_mps,
                source_file=trial_metadata.get("source_file", npz_file.resolve()),
                fps=context.profile.output_fps,
                duration_seconds=(
                    (int(motion_data["rigid_body_pos"].shape[0]) - 1)
                    / float(context.profile.output_fps)
                    if int(motion_data["rigid_body_pos"].shape[0]) > 0
                    else 0.0
                ),
            )
            self._write_trial_metadata(self.config.motion_dir, trial_stem, motion_metadata)
            report = validate_motion_file(
                motion_file=output_file,
                model_xml=context.model_xml,
                qc_config_file=self.config.qc_config_file,
            )
            report["trial"] = trial_stem
            write_validation_report(
                report,
                self.config.qc_motion_dir / f"{trial_stem}.json",
            )
            if not self.config.skip_qc:
                ensure_validation_passed(
                    report,
                    f"Motion validation failed for {trial_stem}",
                )
            output_files.append(output_file)

        self.summary["completed_step"] = PipelineStep.CONVERT.value
        self.summary["num_motion_files"] = len(output_files)
        self._write_subject_summary()
        return output_files

    def run_package(self) -> Path:
        context = self._ensure_assets()
        motion_files = self._resolve_stage_inputs(self.config.motion_dir, ".motion")
        manifest_file = self.config.yaml_dir / f"motions_{context.profile.subject_id}.yaml"
        create_motion_manifest(
            motion_files=motion_files,
            output_file=manifest_file,
            fps=context.profile.output_fps,
            subject_id=context.profile.subject_id,
            subset_name="all_8",
        )

        speed_subset_dir = self.config.yaml_dir / "subsets"
        grouped_motion_files: dict[str, list[Path]] = defaultdict(list)
        for motion_file in motion_files:
            speed_mps = self._trial_speed_mps(motion_file.stem)
            grouped_motion_files[speed_mps_slug(speed_mps)].append(motion_file)

        subset_manifests: dict[str, str] = {}
        for speed_label, selected_motion_files in sorted(grouped_motion_files.items()):
            subset_manifest = speed_subset_dir / f"motions_{context.profile.subject_id}_speed_{speed_label}.yaml"
            create_motion_manifest(
                motion_files=selected_motion_files,
                output_file=subset_manifest,
                fps=context.profile.output_fps,
                subject_id=context.profile.subject_id,
                subset_name=f"speed_{speed_label}",
            )
            subset_manifests[speed_label] = str(subset_manifest)

        matrix_subset_dir = self.config.yaml_dir / "experiment_matrix"
        matrix_manifests = generate_experiment_matrix_manifests(
            master_manifest=manifest_file,
            output_dir=matrix_subset_dir,
        )

        packaged_file = self.config.packaged_dir / f"{context.profile.subject_id}.pt"
        if self.config.force or not packaged_file.exists():
            package_motion_library(
                manifest_file=manifest_file,
                output_file=packaged_file,
                device="cpu",
            )

        report = validate_packaged_motion_lib(
            packaged_file=packaged_file,
            expected_motion_files=motion_files,
        )
        report["subject_id"] = context.profile.subject_id
        write_validation_report(report, self.config.qc_package_dir / "package.json")
        if not self.config.skip_qc:
            ensure_validation_passed(
                report,
                f"Packaged MotionLib validation failed for {context.profile.subject_id}",
            )

        self.summary["completed_step"] = PipelineStep.PACKAGE.value
        self.summary["packaged_file"] = str(packaged_file)
        self.summary["motion_manifest"] = str(manifest_file)
        self.summary["motion_subset_manifests"] = {
            "speed_grouped": subset_manifests,
            "experiment_matrix": {name: str(path) for name, path in matrix_manifests.items()},
        }
        self._write_subject_summary()
        return packaged_file

    def run(self) -> Path | None:
        try:
            if self.config.step == PipelineStep.ASSETS:
                self.run_assets()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return self.config.profile_output_path
            if self.config.step == PipelineStep.OVERGROUND:
                self.run_overground()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return self.config.overground_dir
            if self.config.step == PipelineStep.KEYPOINTS:
                self.run_keypoints()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return self.config.keypoints_dir
            if self.config.step == PipelineStep.RETARGET:
                self.run_retarget()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return self.config.retargeted_dir
            if self.config.step == PipelineStep.CONVERT:
                self.run_convert()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return self.config.motion_dir
            if self.config.step == PipelineStep.PACKAGE:
                packaged_file = self.run_package()
                self.summary["status"] = "ok"
                self._write_subject_summary()
                return packaged_file

            self.run_assets()
            self.run_overground()
            self.run_keypoints()
            self.run_retarget()
            self.run_convert()
            packaged_file = self.run_package()
            self.summary["status"] = "ok"
            self._write_subject_summary()
            return packaged_file
        except Exception as exc:
            self.summary["status"] = "failed"
            self.summary["error"] = str(exc)
            self._write_subject_summary()
            raise


def main(
    input_dir: Path,
    output_dir: Path,
    *,
    subject_profile_path: Path | None = None,
    height: int | None = None,
    subject_id: str | None = None,
    model_variant: str = "adjusted_pd",
    fps: int = 200,
    output_fps: int = 30,
    coordinate_transform: str = "y_to_x_forward",
    contact_source: str = "heuristic",
    step: PipelineStep = PipelineStep.ALL,
    force: bool = False,
    skip_qc: bool = False,
    pyroki_python: Path | None = None,
    pyroki_script: Path | None = None,
    export_profile: Path | None = None,
    qc_config_file: Path | None = None,
    assets_root: Path | None = None,
    subject_height: int | None = None,
    model_xml: Path | None = None,
    speed_override: float | None = None,
    auto_scale: bool = True,
    scale_override: float | None = None,
    force_remake: bool | None = None,
    clean_intermediate: bool = False,
    retarget_python: Path | None = None,
    retarget_script: Path | None = None,
    retarget_urdf_path: Path | None = None,
    contact_pads: bool = False,
) -> Path | None:
    """Run the production single-subject pipeline.

    Deprecated keyword arguments are accepted so legacy wrappers can still call
    into the production path, but they are not part of the supported CLI.
    """
    del model_xml, speed_override, auto_scale, scale_override, clean_intermediate, retarget_urdf_path
    del contact_pads

    if force_remake is not None:
        force = force_remake
    if retarget_python is not None and pyroki_python is None:
        pyroki_python = retarget_python
    if retarget_script is not None and pyroki_script is None:
        pyroki_script = retarget_script
    if height is None and subject_height is not None:
        height = subject_height

    if subject_profile_path is None and height is None:
        raise ValueError("Provide either --subject-profile or --height")
    if subject_profile_path is not None and height is not None:
        raise ValueError("Use either --subject-profile or --height, not both")

    qc_config = qc_config_file or (
        REPO_ROOT
        / "HumanRetargeting"
        / "biomechanics_retarget"
        / "config"
        / "qc_thresholds.yaml"
    )
    assets_root = assets_root or (
        REPO_ROOT
        / "protomotions"
        / "data"
        / "assets"
    )

    pipeline = ProductionPipeline(
        PipelineConfig(
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            subject_profile_path=subject_profile_path.resolve() if subject_profile_path else None,
            height_cm=height,
            subject_id=subject_id,
            model_variant=model_variant,
            fps=fps,
            output_fps=output_fps,
            coordinate_transform=coordinate_transform,
            contact_source=contact_source,
            step=step,
            force=force,
            skip_qc=skip_qc,
            pyroki_python=Path(pyroki_python).absolute() if pyroki_python else None,
            pyroki_script=Path(pyroki_script).absolute() if pyroki_script else None,
            assets_root=assets_root.resolve(),
            rescale_dir=(REPO_ROOT / "HumanRetargeting" / "rescale").resolve(),
            qc_config_file=qc_config.resolve(),
            export_profile=export_profile.resolve() if export_profile else None,
        )
    )
    return pipeline.run()


@app.command()
def cli(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    subject_profile_path: Path | None = typer.Option(
        None,
        "--subject-profile",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Checked-in subject profile YAML.",
    ),
    height: int | None = typer.Option(
        None,
        "--height",
        help="Generate a canonical height-only profile such as H182.",
    ),
    subject_id: str | None = typer.Option(
        None,
        "--subject-id",
        help="Explicit subject id. Height-only runs default to H<height>.",
    ),
    model_variant: str = typer.Option(
        "adjusted_pd",
        "--model-variant",
        help="Lower-body template variant used when materializing subject assets.",
    ),
    fps: int = typer.Option(200, "--fps", help="Input treadmill frame rate."),
    output_fps: int = typer.Option(30, "--output-fps", help="Output frame rate."),
    coordinate_transform: str = typer.Option(
        "y_to_x_forward",
        "--coordinate-transform",
        help="Coordinate transform applied during treadmill-to-overground conversion.",
    ),
    contact_source: str = typer.Option(
        "heuristic",
        "--contact-source",
        help="Default contact source for height-only generated profiles.",
    ),
    step: PipelineStep = typer.Option(
        PipelineStep.ALL,
        "--step",
        help="Single pipeline step to run, or all for the full production flow.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild existing outputs instead of reusing them.",
    ),
    skip_qc: bool = typer.Option(
        False,
        "--skip-qc",
        help="Skip QC validation so any motion data passes through.",
    ),
    pyroki_python: Path | None = typer.Option(
        None,
        "--pyroki-python",
        file_okay=True,
        dir_okay=False,
        help="Override the default production PyRoki interpreter.",
    ),
    pyroki_script: Path | None = typer.Option(
        None,
        "--pyroki-script",
        file_okay=True,
        dir_okay=False,
        help="Override the default production PyRoki wrapper script.",
    ),
    export_profile: Path | None = typer.Option(
        None,
        "--export-profile",
        file_okay=True,
        dir_okay=False,
        help="Optional extra copy of the resolved run profile.",
    ),
    qc_config_file: Path | None = typer.Option(
        None,
        "--qc-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Checked-in QC threshold config.",
    ),
) -> None:
    """CLI entrypoint for the production single-subject pipeline."""
    try:
        result = main(
            input_dir=input_dir,
            output_dir=output_dir,
            subject_profile_path=subject_profile_path,
            height=height,
            subject_id=subject_id,
            model_variant=model_variant,
            fps=fps,
            output_fps=output_fps,
            coordinate_transform=coordinate_transform,
            contact_source=contact_source,
            step=step,
            force=force,
            skip_qc=skip_qc,
            pyroki_python=pyroki_python,
            pyroki_script=pyroki_script,
            export_profile=export_profile,
            qc_config_file=qc_config_file,
        )
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if result is not None:
        console.print(f"[green]Pipeline output:[/green] {result}")


if __name__ == "__main__":
    app()
