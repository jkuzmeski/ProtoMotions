#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate the frozen speed-conditioned experiment-matrix manifests."""

from __future__ import annotations

import sys
from pathlib import Path

import typer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HumanRetargeting.biomechanics_retarget.stages.package import (  # noqa: E402
    generate_experiment_matrix_manifests,
)


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    master_manifest: Path = typer.Argument(..., exists=True, dir_okay=False, help="Master YAML manifest."),
    output_dir: Path = typer.Argument(..., file_okay=False, help="Directory for derived subset manifests."),
) -> None:
    """Write all frozen experiment-matrix subset manifests."""
    subset_manifests = generate_experiment_matrix_manifests(
        master_manifest=master_manifest,
        output_dir=output_dir,
    )
    for subset_name, manifest_path in sorted(subset_manifests.items()):
        print(f"{subset_name}: {manifest_path}")


if __name__ == "__main__":
    app()
