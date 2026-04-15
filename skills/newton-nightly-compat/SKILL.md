---
name: newton-nightly-compat
description: Maintain ProtoMotions compatibility with a fast-moving local Newton fork after nightly rebases or upstream syncs. Use when Newton changed and Codex needs to run the standard Newton smoke training command, trace API drift in `protomotions/simulator/newton/simulator.py`, patch adapter breakages, and rerun until the smoke threshold clears.
---

# Newton Nightly Compat

## Overview

Keep ProtoMotions working against the local Newton fork by running one stable smoke command, fixing drift in the Newton adapter layer, and rerunning the same command until it clears the requested epoch threshold.

Prefer this skill immediately after a nightly Newton rebase, after manual Newton sync work, or whenever a previously working Newton training command starts failing during simulator initialization, rollout, contact sensing, viewer updates, or early training.

## Quick Start

1. Run the repo-local smoke wrapper first.
   `bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10`
2. Keep the command stable while debugging.
   Reuse the exact failing command until the error is gone.
3. Patch the Newton adapter before touching broader training code.
   Start in `protomotions/simulator/newton/simulator.py`.
4. Rerun until the same smoke command reaches the target epoch.
   Minimum nightly success is epoch 10 without a runtime error.

## Workflow

### 1. Rebuild context from source

Read the traceback first. Use the Newton codebase itself as the primary API reference:

- Read ProtoMotions call sites in `protomotions/simulator/newton/simulator.py`.
- Read the corresponding implementation in `/mnt/d/Biomotions/newton/newton/_src/...`.
- Avoid patching from memory when Newton APIs are moving quickly.

If the traceback lands in a known hotspot, read `references/drift-hotspots.md`.

### 2. Run the canonical smoke

Prefer the wrapper script for the nightly loop:

```bash
bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10
```

Use viewer mode only when the failure is render- or input-related:

```bash
bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10 --viewer viser
```

Use `--dry-run` when you only want the exact command string:

```bash
bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10 --dry-run
```

### 3. Patch the narrowest compatibility surface

Prefer narrow adapter fixes over broad rewrites.

Typical order:

1. Label or selection drift
2. Builder-to-model field drift
3. MuJoCo attribute drift
4. Contact sensor API drift
5. Viewer API drift
6. Only then inspect higher-level env or agent code

Do not spread fixes across unrelated files unless the traceback proves the adapter is not the right layer.

### 4. Validate with the same command

After each patch:

1. Rerun the exact smoke command that failed.
2. Wait through slow startup before assuming a hang.
3. Stop only after the run reaches the requested epoch threshold or exposes a new concrete failure.

Keep the seed and smoke parameters stable during the debugging loop so each rerun is comparable.

## Nightly Rebase Loop

Use this skill alongside the existing nightly Newton rebase job.

Recommended sequence:

1. Complete the Newton rebase or sync.
2. Run the 10-epoch headless smoke immediately.
3. If it fails, patch the adapter and rerun the same smoke.
4. If the headless smoke passes, optionally run one viewer smoke when recent changes touched rendering or input handling.
5. Treat the smoke as the gate for whether the rebase is safe for normal ProtoMotions work.

Read `references/nightly-workflow.md` when wiring this into a recurring maintenance routine.

## Success Criteria

Treat the nightly check as healthy when all of the following are true:

- The smoke command reaches epoch 10 without a runtime exception.
- The same command no longer regresses on rerun.
- Any warnings are understood as non-fatal or separately tracked.

Warnings alone are not a failure unless they predict broken training behavior.

## Resources

### scripts/

- `scripts/run_mimic_smoke.sh`: Standard Newton smoke wrapper with stable defaults and optional viewer mode.

### references/

- `references/drift-hotspots.md`: Fast map of known Newton adapter drift surfaces.
- `references/nightly-workflow.md`: Recommended operating pattern for nightly rebases and smoke validation.
