# Quarantined: `transformer_bm_speed`

This branch keeps the low-level speed-conditioning infrastructure, but the
full teacher-student experiment at
`examples/experiments/masked_mimic/transformer_bm_speed.py` is intentionally
quarantined.

## Why it is quarantined

The experiment file was started against an older observation/control API and
now mixes that older shape with the current `MdpComponent` and `EnvContext`
system. The result is not a single bug. It is a partial port.

The immediate import failure from `speed_obs_functions.py` has been fixed, but
the experiment file itself still references incompatible APIs and missing
modules.

## What is safe to keep

- `SpeedControl`
- `speed_obs_factory()`
- motion-metadata speed lookup in `SpeedControl`
- the passing tests in `protomotions/tests/test_speed_control_and_obs.py`

These pieces are general infrastructure and are useful even without the full
teacher-student experiment.

## What must be ported before re-enabling

1. Replace old observation factory assumptions with current `MdpComponent`
   factories and `EnvContext` bindings.

2. Remove or rewrite imports that do not exist on this branch:
   - `protomotions.envs.obs.general`
   - `protomotions.envs.obs.masked_mimic_obs_functions`
   - old `observation_component`-style codepaths

3. Rewrite factory calls that use stale signatures. Known examples:
   - `reduced_coords_obs_factory(observation_noise=True)`
   - `historical_reduced_coords_obs_factory(observation_noise=True)`
   - `mimic_target_poses_reduced_coords_factory(num_future_steps=1, observation_noise=True)`

4. Rebuild the masked-mimic target observation components using the same style
   as `examples/experiments/masked_mimic/transformer.py`, not the older helper
   layout.

5. Verify the student/prior token shapes against the current masked-mimic
   model config before training.

6. Add a focused smoke test that imports the experiment module and builds:
   - `env_config(...)`
   - `agent_config(...)`

7. Only after that, run a small training sanity check with a tiny env count and
   no viewer.

## Recommended restart point

When this work resumes, do not start from the quarantined file and patch
forward line by line. Start from
`examples/experiments/masked_mimic/transformer.py` and re-apply only the
speed-conditioned prior changes on top of the current branch API.
