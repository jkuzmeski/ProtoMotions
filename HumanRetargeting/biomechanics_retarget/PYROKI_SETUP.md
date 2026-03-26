# PyRoki Integration Guide

This pipeline now treats PyRoki as an external dependency again.

Two things are required for the retarget step:

1. A dedicated Python interpreter with upstream `pyroki` installed.
2. The lower-body PyRoki wrapper script used by the production pipeline.

The biomechanics pipeline no longer falls back to the local in-repo lower-body
solver. It defaults to the production paths below, prints the exact paths it is
using, and stops with a clear error instead of silently using a different
implementation.

## Upstream Install

Upstream `pyroki` still installs as an editable package from the GitHub repo:

```bash
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

At the time of writing, upstream declares `requires-python >=3.10` and depends on
`jax`, `jaxlib`, `jaxls`, `yourdfpy`, `viser`, and related packages via its
`pyproject.toml`.

## Recommended Local Setup

This repo now includes a helper script that creates a dedicated virtualenv,
clones upstream PyRoki, and installs it:

```bash
./scripts/setup_pyroki_env.sh
```

Defaults:

- Python executable: `python3`
- virtualenv: `.venvs/pyroki`
- upstream checkout: `third_party/pyroki`

Custom locations are also supported:

```bash
./scripts/setup_pyroki_env.sh python3.12 .venvs/pyroki third_party/pyroki
```

The final interpreter path will be:

```bash
.venvs/pyroki/bin/python
```

## Running The Biomechanics Pipeline

For the lower-body biomechanics flow, the production defaults are:

- interpreter: `./.venvs/pyroki/bin/python`
- wrapper script: `./pyroki/batch_retarget_to_smpl_lower_body.py`

Example:

```bash
python HumanRetargeting/biomechanics_retarget/pipeline.py \
    ./HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    ./HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    --subject-profile HumanRetargeting/biomechanics_retarget/profiles/S_GENERIC.yaml
```

Override flags are still available if you need them:

```bash
--pyroki-python
--pyroki-script
```

## What The Pipeline Validates

Before launching the retarget step, the pipeline now verifies that the selected
interpreter can import:

- `pyroki`
- `jax`
- `jaxls`
- `yourdfpy`

If that import check fails, the pipeline aborts before any retargeting work runs.

## Legacy Study Runner

The old batch study helper now lives under `tools/legacy/` for auditability.
It is not the supported production entrypoint.

If you still need it for a migration or comparison run, invoke:

```bash
python HumanRetargeting/biomechanics_retarget/tools/legacy/study_pipeline.py \
    --manifest /path/to/study.yaml \
    --pyroki-python ./.venvs/pyroki/bin/python \
    --pyroki-script ./pyroki/batch_retarget_to_smpl_lower_body.py
```

## Robot Retargeting

The G1 and H1_2 retargeting scripts under `pyroki/` still use upstream `pyroki`
as a library and should also be run with the dedicated PyRoki interpreter.

See:

- `scripts/retarget_single_motion_to_robot.sh`
- `scripts/retarget_amass_to_robot.sh`
