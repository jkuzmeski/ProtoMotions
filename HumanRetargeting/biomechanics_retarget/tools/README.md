# Tools

This directory quarantines non-production helpers that are useful for debugging,
visualization, or legacy migrations.

Production orchestration stays in `HumanRetargeting/biomechanics_retarget/pipeline.py`.

## Debugging

Use these when you need to inspect pipeline data or simulator ordering:

- `tools/debug/check_pipeline_data.py`
- `tools/debug/diagnose_motion_pipeline.py`
- `tools/debug/diagnose_runtime_joints.py`
- `tools/debug/inspect_npy.py`

Example:

```bash
python HumanRetargeting/biomechanics_retarget/tools/debug/diagnose_motion_pipeline.py \
    --npz HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/retargeted_motions/S02_30ms_Long_retargeted.npz \
    --motion HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/motion_files/S02_30ms_Long.motion \
    --pt HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/packaged_data/S_GENERIC.pt
```

## Visualization

- `tools/visualization/visualize_keypoints.py`
- `tools/visualization/visualize_pipeline_stage.py`

Example:

```bash
python HumanRetargeting/biomechanics_retarget/tools/visualization/visualize_keypoints.py \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/keypoints/S02_15ms_Long.npy
```

```bash
python HumanRetargeting/biomechanics_retarget/tools/visualization/visualize_pipeline_stage.py \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    S02_15ms_Long \
    --stage full
```

## Legacy helpers

These files are kept for auditability and migration work, but they are not the
production entrypoints:

- `tools/legacy/batch_retarget_lower_body.py`
- `tools/legacy/create_motion_yaml.py`
- `tools/legacy/package_motions.py`
- `tools/legacy/quick_rescale.py`
- `tools/legacy/study_pipeline.py`
