# Biomechanics Retarget Refactor

- [x] Lock the production architecture and validation policy.
- [x] Make `HumanRetargeting.biomechanics_retarget` a real package.
- [x] Add explicit stage modules under `stages/`.
- [x] Hard-cut production retargeting to PyRoki only.
- [x] Add explicit `assets` pipeline step.
- [x] Replace `--model` production entry with profile-driven flow.
- [x] Implement `--height` as generated profile sugar.
- [x] Add checked-in generic profile template.
- [x] Persist the effective profile to `processed_data/<subject>/profile.yaml`.
- [x] Add structured QC output tree under `qc/`.
- [x] Add retarget NPZ contract validation and hard-fail reports.
- [x] Promote retarget quality checks to production hard-fail.
- [x] Add `.motion` contract validation and hard-fail reports.
- [x] Add packaged `.pt` contract validation and hard-fail reports.
- [x] Keep `yaml_data/` as a reproducible artifact.
- [x] Move diagnostics and visualization scripts into `tools/`.
- [x] Remove visualization from the production pipeline.
- [x] Delete legacy production entrypoints and fallback retarget path.
- [x] Update docs to match the new production flow and debugging tools.
- [x] Replace legacy tests with contract-focused coverage.
- [x] Run the validation suite and an end-to-end smoke check.

## This Pass

- [x] Quarantine non-production helpers under `tools/`.
- [x] Update README and setup docs to point at the quarantined tools.
- [x] Keep the production pipeline files in place.
