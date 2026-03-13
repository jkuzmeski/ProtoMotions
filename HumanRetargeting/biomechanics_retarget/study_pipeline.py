#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Batch study runner for subject-aware biomechanics retargeting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
import yaml

try:
    from .pipeline import PipelineStep, main as run_subject_pipeline
    from .subject_profiles import SubjectProfile, load_study_manifest
except ImportError:
    from pipeline import PipelineStep, main as run_subject_pipeline
    from subject_profiles import SubjectProfile, load_study_manifest


app = typer.Typer(pretty_exceptions_enable=False)


def _materialize_profile(profile: SubjectProfile, output_root: Path) -> Path:
    profile_dir = output_root / "_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{profile.subject_id}.yaml"
    profile_path.write_text(
        yaml.safe_dump(profile.as_metadata(), sort_keys=False),
        encoding="utf-8",
    )
    return profile_path


@app.command()
def main(
    manifest: Path = typer.Option(..., "--manifest", exists=True, help="Study manifest YAML."),
    output_root: Path = typer.Option(None, "--output-root", help="Root directory for processed study outputs."),
    step: PipelineStep = typer.Option(PipelineStep.ALL, "--step", help="Pipeline step to run for each subject."),
    force: bool = typer.Option(False, "--force", help="Force subject reprocessing."),
    pyroki_python: Path | None = typer.Option(
        None,
        "--pyroki-python",
        help="Optional explicit PyRoki interpreter for every subject.",
    ),
    jax_platform: str = typer.Option("cuda", "--jax-platform", help="JAX backend."),
) -> None:
    """Run the single-subject pipeline for every subject in a study manifest."""
    study = load_study_manifest(manifest)
    effective_output_root = output_root or study.output_root or (manifest.parent / "processed_data")
    effective_output_root = effective_output_root.resolve()
    effective_output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for profile in study.subjects:
        subject_output_dir = effective_output_root / profile.subject_id
        subject_profile_path = profile.profile_path or _materialize_profile(profile, effective_output_root)
        run_subject_pipeline(
            input_dir=profile.input_dir,
            output_dir=subject_output_dir,
            model_xml=None,
            subject_height=None,
            model_variant=profile.model_variant,
            fps=profile.fps,
            output_fps=profile.output_fps,
            speed_override=None,
            coordinate_transform=profile.coordinate_transform,
            auto_scale=True,
            scale_override=None,
            force_remake=force,
            step=step,
            clean_intermediate=False,
            pyroki_python=pyroki_python,
            pyroki_urdf_path=None,
            jax_platform=jax_platform,
            subject_profile_path=subject_profile_path,
            contact_source=profile.contact_source,
            contact_pads=profile.contact_pads,
        )
        summary_path = subject_output_dir / "qc" / "subject_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            summaries.append(
                {
                    "subject_id": profile.subject_id,
                    "output_dir": str(subject_output_dir),
                    "error": "missing subject summary",
                }
            )

    summary_json = effective_output_root / "study_summary.json"
    summary_csv = effective_output_root / "study_summary.csv"
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    app()
