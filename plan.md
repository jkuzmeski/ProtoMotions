# Speed-Conditioned Masked Mimic Plan

## Objective

Build a new single-subject experiment suite to answer:

> What is the minimum number of discrete treadmill speed conditions needed for a speed-only policy to recover subject-specific biomechanics?

The final shipped policy must run at inference with:

- no motion library
- no reference motion data
- fixed forward heading
- only a scalar speed command

## Fixed Decisions

These decisions are closed unless a later implementation blocker forces reconsideration.

- Teacher architecture: `examples/experiments/mimic/mlp_bm.py`
- Student architecture: transformer MaskedMimic variant
- Teacher retraining: one teacher per subset condition
- Student training env: motion-backed
- Student encoder at training time: privileged full target pose access
- Student prior at training time: speed-conditioned only
- Prior inputs: `target_speed + fixed forward heading + current proprioception + existing history`
- Deployment env: separate speed-control env with no motion library
- Deployment reset: default standing humanoid pose
- Deployment behavior: fixed speed only
- Inference failure mode: fail hard and loud if motion-library-backed control is present
- Metadata persistence: YAML / sidecar only for now, not packaged `.pt`
- Subset generation: master manifest plus derived subset manifests
- Subset selection: explicit trial filenames, not runtime filename parsing
- Evaluation: compare both held-out speeds and training speeds
- Event source: simulator contacts
- Left cycle anchor: left foot strike transition from no left-foot contact to any left-foot contact
- Burn-in gate: start metric collection only after cycle-mean root forward speed is within `+-10%` of target for 2 consecutive left-anchored gait cycles
- Trial success: only successful if 10 left-anchored gait cycles are collected after burn-in
- Metrics window: 10 full left-anchored gait cycles
- Root speed metric: root forward speed
- Stride length: left strike to next left strike forward displacement
- Cadence: steps per minute
- Waveforms: pelvis, hip, knee, ankle flex/add/rot mean waveforms plus std envelopes
- Evaluation episodes: 20 per target speed

## Non-Goals

- No runtime speed-change curriculum in v1
- No walk/run class labels in the model
- No metadata schema extension in packaged `MotionLib` `.pt` files
- No true continuous in-between-speed ground-truth comparator in v1
- No learned reset-state distribution in v1

## Scientific Scope

Primary claim:

- minimum number of discrete speed conditions needed for recovery of subject-specific biomechanics

Secondary claims:

- held-out real-speed reconstruction from speed command alone
- deployment feasibility of speed-only control without a motion library
- sensitivity of biomechanical fidelity to reduced speed-condition coverage

## Canonical Trial Set

Subject example:

- `S02_15ms_Long`
- `S02_20ms_Long`
- `S02_25ms_Long`
- `S02_30ms_Long`
- `S02_35ms_Long`
- `S02_40ms_Long`
- `S02_45ms_Long`
- `S02_50ms_Long`

Interpretation:

- `15ms -> 1.5 m/s`
- `20ms -> 2.0 m/s`
- `25ms -> 2.5 m/s`
- `30ms -> 3.0 m/s`
- `35ms -> 3.5 m/s`
- `40ms -> 4.0 m/s`
- `45ms -> 4.5 m/s`
- `50ms -> 5.0 m/s`

## Experiment Matrix

### Master Manifest

- `all_8`
  - train: all 8 speeds
  - eval: all 8 speeds

### Reduced-Coverage Subsets

- `every_other`
  - train: `S02_20ms_Long`, `S02_30ms_Long`, `S02_40ms_Long`, `S02_50ms_Long`
  - eval controls: training speeds above
  - eval held-out speeds: `S02_15ms_Long`, `S02_25ms_Long`, `S02_35ms_Long`, `S02_45ms_Long`

- `anchor_3`
  - train: `S02_15ms_Long`, `S02_30ms_Long`, `S02_50ms_Long`
  - eval controls: training speeds above
  - eval held-out speeds: all remaining real speeds

- `speed_2`
  - train: `S02_15ms_Long`, `S02_35ms_Long`
  - eval controls: training speeds above
  - eval held-out speeds: all remaining real speeds

### Edge Holdout

- `leave_edge_low`
  - train: all except `S02_15ms_Long`
  - eval held-out: `S02_15ms_Long`
  - eval controls: all train speeds

- `leave_edge_high`
  - train: all except `S02_50ms_Long`
  - eval held-out: `S02_50ms_Long`
  - eval controls: all train speeds

### Leave-One-Out

- `loo_15`
- `loo_20`
- `loo_25`
- `loo_30`
- `loo_35`
- `loo_40`
- `loo_45`
- `loo_50`

For each `loo_*` run:

- train: all except the named file
- eval held-out: the named file
- eval controls: all train speeds

## High-Level Architecture

### Training-Time Teacher

- Full-body motion tracking teacher
- BeyondMimic reward stack
- Subset-specific motion manifest
- No access to held-out speed trials for that subset condition

### Training-Time Student

- MaskedMimic-style teacher-student distillation
- Encoder remains privileged and full-pose-conditioned during training
- Prior is changed from masked pose conditioning to speed conditioning
- Student trains against teacher actions

### Deployment-Time Policy

- Separate speed-control environment
- No motion library
- No motion manager
- No masked-mimic control component
- Only speed command and fixed forward heading
- Standing reset only

### Evaluation

- Motion-backed evaluation for teacher/student scientific comparison
- Speed-only deployment evaluation for final-product feasibility
- Cycle-based biomechanical summaries

## Proposed File/Module Targets

These are planning targets, not an implementation promise that every file must change.

### Metadata and manifests

- `HumanRetargeting/biomechanics_retarget/stages/package.py`
- `HumanRetargeting/biomechanics_retarget/pipeline.py`
- `HumanRetargeting/biomechanics_retarget/subject_profiles.py`
- `HumanRetargeting/biomechanics_retarget/treadmill2overground.py`
- `scripts/` new subset-manifest generator helper

### New control / obs path

- `protomotions/envs/control/speed_control.py` new
- `protomotions/envs/obs/` new speed observation helper or constrained reuse of steering-style helpers
- `protomotions/envs/obs/__init__.py`
- `protomotions/envs/control/__init__.py` if needed

### Student experiment

- `examples/experiments/masked_mimic/` new speed-conditioned transformer experiment
- likely fork from `transformer_bm.py` rather than plain `transformer.py`

### Deployment / inference

- `protomotions/inference_agent.py`
- experiment-specific `apply_inference_overrides`
- deployment env config path with motion library explicitly disabled

### Evaluation

- `protomotions/agents/evaluators/mimic_evaluator.py` extend only if clean
- or `protomotions/agents/evaluators/` new dedicated biomechanics evaluator
- plotting / raw-cycle export helper under `protomotions/agents/evaluators/` or `scripts/`

### Tests

- `protomotions/tests/` new targeted tests for:
  - metadata generation
  - subset manifest generation
  - speed control semantics
  - deployment env hard-failure behavior
  - gait-cycle event extraction

## Dependency Order

The work should be done in this order to avoid rework:

1. metadata plumbing
2. subset manifest generation
3. subset-specific teacher training wiring
4. speed control component and speed observation path
5. speed-conditioned student experiment
6. deployment env and fail-loud inference path
7. biomechanics evaluator
8. automation scripts for experiment matrix
9. validation and docs

## Orchestration Strategy

Use disjoint write sets so multiple workers can operate in parallel with minimal merge risk.

### Main orchestrator responsibilities

- keep design constraints frozen
- sequence dependencies
- review worker patches
- reconcile config naming and experiment conventions
- run integration validation
- keep `plan.md` current during execution

### Suggested worker lanes

- Worker A: metadata and manifest generation
- Worker B: new speed control component and observations
- Worker C: student experiment config and inference path
- Worker D: evaluator and biomechanics outputs
- Worker E: tests and validation harness

## Workstream Checklists

### W0. Guardrails And Baseline Reproduction

Owner: Main orchestrator

- [ ] Record exact subject trial inventory and speed mapping for the first subject under test
- [ ] Confirm baseline teacher training command for `mimic/mlp_bm.py`
- [ ] Confirm current masked mimic training path and where prior inputs are defined
- [ ] Confirm current inference path loads frozen configs and would incorrectly preserve motion-backed control without intervention
- [ ] Write down the exact commands that will later be used for teacher training, student training, and inference
- [ ] Freeze naming convention for subset manifests and output directories

### W1. Metadata And Manifest Plumbing

Owner: Worker A

Write set:

- `HumanRetargeting/biomechanics_retarget/stages/package.py`
- `HumanRetargeting/biomechanics_retarget/pipeline.py`
- `HumanRetargeting/biomechanics_retarget/subject_profiles.py`
- `scripts/` new helper files only

Tasks:

- [ ] Define per-trial metadata fields in YAML / sidecar:
  - [ ] `subject_id`
  - [ ] `trial_name`
  - [ ] `speed_mps`
  - [ ] `source_file`
  - [ ] `fps`
  - [ ] `duration_seconds`
- [ ] Preserve current `speed_source: filename` behavior as fallback only
- [ ] Ensure explicit trial-level metadata is emitted during package stage
- [ ] Create one master manifest per subject
- [ ] Create derived subset manifests using explicit filename lists only
- [ ] Prevent subset contamination by making manifests self-contained and explicit
- [ ] Define deterministic output locations for subset manifests
- [ ] Add tests for metadata emission and subset-manifest exact file membership

Acceptance checks:

- [ ] Master manifest includes all 8 trials with correct speed metadata
- [ ] Derived subset manifests include exactly the intended filenames
- [ ] No experiment code needs to regex parse filenames at runtime

### W2. Teacher Subset Training Plumbing

Owner: Main orchestrator or Worker A if disjoint helper files are used

Write set:

- likely new helper under `scripts/` or experiment launch docs only

Tasks:

- [ ] Standardize teacher training input to consume subset manifests directly
- [ ] Define output directory naming convention per subset:
  - [ ] subject
  - [ ] subset id
  - [ ] teacher tag
- [ ] Ensure held-out trials are excluded from teacher training for every subset condition
- [ ] Define teacher checkpoint discovery convention for later student training

Acceptance checks:

- [ ] A teacher can be launched from a derived subset manifest without manual file editing
- [ ] Held-out teacher contamination is impossible if the manifest is correct

### W3. New Speed Control Component

Owner: Worker B

Write set:

- `protomotions/envs/control/speed_control.py` new
- `protomotions/envs/control/__init__.py` only if needed

Design constraints:

- reuse steering-control design philosophy where useful
- fixed forward heading only
- fixed speed for entire episode
- no runtime speed changes in v1
- fail loud if misconfigured

Tasks:

- [ ] Implement `SpeedControlConfig`
- [ ] Implement `SpeedControl`
- [ ] Expose `tar_speed`
- [ ] Expose fixed forward heading / direction representation compatible with the rest of the codebase
- [ ] Define reset semantics for standing start
- [ ] Ensure no hidden random heading logic remains
- [ ] Add optional visualization markers only if cheap and useful
- [ ] Add unit tests for:
  - [ ] fixed heading
  - [ ] fixed speed persistence
  - [ ] reset behavior
  - [ ] misconfiguration failure cases

Acceptance checks:

- [ ] Control context is deterministic under a fixed command
- [ ] No motion manager access is required
- [ ] Component can exist in an env with empty `MotionLib`

### W4. Speed Observation Path

Owner: Worker B

Write set:

- `protomotions/envs/obs/` new helper file or minimal additions
- `protomotions/envs/obs/__init__.py`

Tasks:

- [ ] Decide whether to fork steering-style observation code or add a small speed-specific helper
- [ ] Encode `target_speed + fixed forward heading` in local frame
- [ ] Keep interface simple and explicit
- [ ] Add tests for observation shape and semantics

Acceptance checks:

- [ ] Observation component can be used in both training and deployment envs
- [ ] Observation does not depend on motion references

### W5. Speed-Conditioned Student Experiment

Owner: Worker C

Write set:

- `examples/experiments/masked_mimic/` new experiment file
- possibly small helper additions if strictly required

Tasks:

- [ ] Fork from the BeyondMimic-flavored masked mimic transformer baseline
- [ ] Keep privileged encoder path intact
- [ ] Remove masked-pose conditioning from the prior path
- [ ] Replace prior conditioning with:
  - [ ] speed observation
  - [ ] current state token
  - [ ] historical pose/state token
- [ ] Keep teacher-action distillation logic unchanged unless a clean extension is required
- [ ] Ensure training env remains motion-backed
- [ ] Ensure teacher observations still line up with subset-specific teacher config
- [ ] Define naming and CLI arguments for:
  - [ ] teacher checkpoint path
  - [ ] subject/subset manifest path
  - [ ] deployment-mode inference override

Acceptance checks:

- [ ] Student training can run with a teacher checkpoint trained on the same subset manifest
- [ ] Prior inputs contain no motion-reference target poses
- [ ] Encoder still has privileged full-target information during training

### W6. Deployment Env And Inference Overrides

Owner: Worker C

Write set:

- deployment experiment file or inference override path
- `protomotions/inference_agent.py` only if required

Tasks:

- [ ] Create a speed-only deployment config path
- [ ] Disable motion library for deployment mode
- [ ] Disable motion-backed control components for deployment mode
- [ ] Ensure speed control is the only task-control component
- [ ] Add hard failure if a motion-backed masked-mimic or mimic control path survives deployment override
- [ ] Add hard failure if motion library is present in deployment mode
- [ ] Confirm standing reset path works without motion references

Acceptance checks:

- [ ] Deployment config starts with empty `MotionLib`
- [ ] Deployment env initializes successfully with speed control only
- [ ] Deployment env fails loudly on accidental motion-backed config leakage

### W7. Biomechanics Evaluator

Owner: Worker D

Write set:

- `protomotions/agents/evaluators/` new evaluator or plugin modules
- plotting/export helpers if needed

Tasks:

- [ ] Decide whether to extend `MimicEvaluator` or add a dedicated evaluator
- [ ] Implement left-foot-strike event detection from simulator contacts:
  - [ ] `no left foot contact -> any left foot contact`
  - [ ] add debounce / minimum interval to avoid chatter-induced fake cycles
- [ ] Compute cycle-mean root forward speed
- [ ] Implement burn-in gate:
  - [ ] within `+-10%` of target
  - [ ] for 2 consecutive left-anchored cycles
- [ ] Begin metrics only after burn-in gate is satisfied
- [ ] Require 10 full post-burn-in left-anchored cycles for success
- [ ] Run 20 episodes per target speed
- [ ] Compute summary metrics:
  - [ ] stride length
  - [ ] cadence in steps/min
  - [ ] success/failure rate
  - [ ] root-speed tracking summary
- [ ] Extract cycle-normalized waveforms for:
  - [ ] pelvis flex/add/rot
  - [ ] hip flex/add/rot
  - [ ] knee flex/add/rot if available in the target representation
  - [ ] ankle flex/add/rot if available in the target representation
- [ ] Save outputs:
  - [ ] per-run JSON summary
  - [ ] CSV table for aggregate metrics
  - [ ] raw normalized cycle arrays
  - [ ] mean +- std plots
- [ ] Support evaluation on:
  - [ ] training-speed controls
  - [ ] held-out real speeds

Acceptance checks:

- [ ] Evaluator cleanly reports unsuccessful episodes that never pass burn-in
- [ ] Evaluator exports reusable raw arrays for later plotting
- [ ] Waveform outputs are left-cycle aligned and deterministic from the saved arrays

### W8. Experiment Automation

Owner: Worker E or Main orchestrator

Write set:

- `scripts/` new launch or orchestration helpers
- docs command snippets if needed

Tasks:

- [ ] Create helper to generate all explicit subset manifests from a master manifest
- [ ] Create helper to enumerate teacher/student/eval runs for the full experiment matrix
- [ ] Standardize output directory layout:
  - [ ] subject
  - [ ] subset
  - [ ] teacher
  - [ ] student
  - [ ] eval
- [ ] Ensure teacher and student for a subset cannot accidentally point at different manifests
- [ ] Add dry-run mode to print the full matrix without launching

Acceptance checks:

- [ ] Full matrix can be enumerated reproducibly
- [ ] Manifest and checkpoint paths are explicit in every generated command

### W9. Validation And Tests

Owner: Worker E

Write set:

- `protomotions/tests/`
- possibly targeted script-level tests under `scripts/` if the repo already supports them

Tasks:

- [ ] Metadata/manifest tests
- [ ] Speed control tests
- [ ] Speed observation tests
- [ ] Deployment fail-loud tests
- [ ] Gait-cycle event detection tests
- [ ] Burn-in gate tests
- [ ] Evaluator export-shape tests
- [ ] At least one end-to-end smoke test that builds a subset manifest and verifies the config wiring

Acceptance checks:

- [ ] Unit tests cover the new control and evaluation semantics
- [ ] One integration-style test confirms deployment mode does not require motion references

### W10. Documentation

Owner: Main orchestrator

Tasks:

- [ ] Add command examples for:
  - [ ] generating master metadata / subset manifests
  - [ ] subset-specific teacher training
  - [ ] subset-specific student training
  - [ ] deployment-mode inference
  - [ ] biomechanics evaluation
- [ ] Document the exact experiment matrix
- [ ] Document what counts as a successful evaluation episode
- [ ] Document that v1 uses standing reset and not learned reset-state initialization
- [ ] Document that metadata is YAML / sidecar only in v1

## Validation Plan

### Minimum pre-merge validation

- [ ] Metadata generation test passes
- [ ] Subset manifest generation test passes
- [ ] Speed control unit tests pass
- [ ] Student experiment config materializes with subset teacher checkpoint path
- [ ] Deployment config starts with empty `MotionLib`
- [ ] Deployment config fails loudly if motion-backed control remains
- [ ] Evaluator unit tests pass

### Recommended end-to-end validation

- [ ] Train one teacher on `anchor_3`
- [ ] Train one student on `anchor_3`
- [ ] Run deployment-mode inference with speed-only command
- [ ] Run biomechanics evaluator on both:
  - [ ] a training speed
  - [ ] a held-out speed
- [ ] Verify at least one successful episode reaches 10 post-burn-in cycles
- [ ] Inspect waveform plots and summary tables manually

## Risks And Failure Modes

- Risk: student prior silently still receives motion-reference data
  - Mitigation: explicit prior input audit and tests

- Risk: deployment override accidentally preserves motion-backed control
  - Mitigation: hard-failure assertions in deployment path

- Risk: contact chatter corrupts cycle detection
  - Mitigation: debounce / minimum interval logic and event tests

- Risk: burn-in gate is too strict and yields low successful-episode counts
  - Mitigation: export unsuccessful episode reasons; tune only after baseline evidence

- Risk: subset contamination through incorrect manifest generation
  - Mitigation: explicit filename lists plus manifest-membership tests

- Risk: teacher/student mismatch on subset definition
  - Mitigation: standardized manifest and output naming convention

- Risk: standing reset causes long acceleration transients
  - Mitigation: burn-in gate plus many-reset evaluation

## Open Questions To Defer

These are intentionally deferred until after v1 works.

- [ ] learned reset-state distribution
- [ ] runtime speed changes within an episode
- [ ] true in-between-speed commands like `2.25 m/s`
- [ ] comparator design for non-dataset interpolation speeds
- [ ] metadata persistence inside packaged `.pt`

## Definition Of Done

This plan is complete when all of the following are true:

- [ ] subset-specific teachers can be trained from explicit manifests
- [ ] a speed-conditioned transformer student can be trained against those teachers
- [ ] deployment-mode inference runs with only speed command and no motion library
- [ ] deployment mode fails loudly on motion-backed misconfiguration
- [ ] evaluator reports stride length, cadence, root-speed gate success, and waveform plots
- [ ] outputs exist for both training-speed controls and held-out real speeds
- [ ] the full experiment matrix can be enumerated reproducibly from explicit manifests

