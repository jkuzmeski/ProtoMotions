# Biomechanics Retargeting

This package contains the treadmill-to-MotionLib pipeline used for
ProtoMotions subject retargeting. The supported production path is the single
subject pipeline in `pipeline.py`, with PyRoki used as the external retargeter.

## Supported Flow

```text
Treadmill motion (.txt)
    -> treadmill2overground.py
Overground motion (.npy)
    -> extract_keypoints_from_overground.py
Keypoints + contacts
    -> PyRoki retargeter
Retargeted motion (.npz)
    -> convert_retargeted_to_motion.py
ProtoMotions motion (.motion)
    -> pipeline packaging
MotionLib package (.pt)
```

## What Lives Where

Production and pipeline-adjacent helpers stay at the package root:

- `pipeline.py` - main orchestrator
- `treadmill2overground.py` - treadmill to overground conversion
- `extract_keypoints_from_overground.py` - keypoint extraction
- `convert_retargeted_to_motion.py` - PyRoki output to ProtoMotions motion
- `pipeline_visualization.py` - visualization helper used by the standalone tooling
- `retarget_qc.py` - QC thresholds and reporting helpers
- `subject_assets.py` - subject asset generation
- `subject_profiles.py` - profile loading and materialization

Non-production tooling is quarantined under `tools/`:

- `tools/debug/` - diagnostics and inspection utilities
- `tools/visualization/` - standalone visualization helpers
- `tools/legacy/` - old packaging or study helpers kept for auditability

## Quick Start

Run the production pipeline from the repo root:

```bash
.venv/bin/python HumanRetargeting/biomechanics_retarget/pipeline.py \
    HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    --subject-profile HumanRetargeting/biomechanics_retarget/profiles/S_GENERIC.yaml \
    --step all \
    --force
```

Run the full production pipeline with a generated height-only subject:

```bash
python HumanRetargeting/biomechanics_retarget/pipeline.py \
    HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    HumanRetargeting/biomechanics_retarget/processed_data/H182 \
    --height 182
```

If you only know the subject height, the pipeline can materialize a generated
profile into `processed_data/<subject>/profile.yaml` and use that as the run
input. The checked-in template profile is
`profiles/templates/generic_lower_body.yaml`.

The production retarget defaults are:

- interpreter: `./.venvs/pyroki/bin/python`
- wrapper script: `./pyroki/batch_retarget_to_smpl_lower_body.py`

The pipeline prints the exact paths it uses and hard-fails if either one is
missing or the interpreter cannot import the PyRoki runtime.

Useful flags:

- rerun everything: `--force`
- run one stage: `--step assets|overground|keypoints|retarget|convert|package|all`


## Debugging

Use the quarantined tools when you need to inspect pipeline artifacts:

- `tools/debug/check_pipeline_data.py` - quick keypoint or retargeted artifact checks
- `tools/debug/diagnose_motion_pipeline.py` - compare `.npz`, `.motion`, and `.pt`
- `tools/debug/diagnose_runtime_joints.py` - inspect simulator joint ordering
- `tools/debug/inspect_npy.py` - quick inspection of a keypoint `.npy`
- `tools/visualization/visualize_keypoints.py` - standalone 3D keypoint viewer
- `tools/visualization/visualize_pipeline_stage.py` - pairwise or full pipeline matplotlib comparison

Examples:

```bash
python HumanRetargeting/biomechanics_retarget/tools/debug/diagnose_motion_pipeline.py \
    --npz HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/retargeted_motions/S02_30ms_Long_retargeted.npz \
    --motion HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/motion_files/S02_30ms_Long.motion \
    --pt HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/packaged_data/S_GENERIC.pt
```

```bash
python HumanRetargeting/biomechanics_retarget/tools/visualization/visualize_keypoints.py \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/keypoints/S02_15ms_Long.npy
```

```bash
python HumanRetargeting/biomechanics_retarget/tools/visualization/visualize_pipeline_stage.py \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    S02_15ms_Long \
    --stage full \
    --seconds 2.0 \
    --start-sec 0.0
```

## Legacy Helpers

The following scripts are kept under `tools/legacy/` for reference and
one-off migrations, but they are not the supported production entrypoints:

- `tools/legacy/batch_retarget_lower_body.py`
- `tools/legacy/create_motion_yaml.py`
- `tools/legacy/package_motions.py`
- `tools/legacy/quick_rescale.py`
- `tools/legacy/study_pipeline.py`

## Output Layout

Typical pipeline outputs are written under:

```text
processed_data/<subject>/
├── overground_data/
├── keypoints/
├── contacts/
├── retargeted_motions/
├── motion_files/
├── yaml_data/
├── packaged_data/
└── qc/
```

The pipeline keeps intermediate artifacts by default so each stage can be
debugged independently. QC reports are written alongside the run output.
