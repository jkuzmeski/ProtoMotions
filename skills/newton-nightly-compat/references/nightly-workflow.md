# Nightly Workflow

Use this sequence after the Newton nightly rebase finishes.

## Standard loop

1. Record the Newton commit or branch tip that landed.
2. Run the headless 10-epoch smoke:
   `bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10`
3. If it fails, keep the exact command fixed while debugging.
4. Patch the narrowest compatibility layer that explains the traceback.
5. Rerun until the same smoke reaches epoch 10.

## When to run a viewer smoke

Run one viewer pass after the headless smoke only when recent Newton changes touched:

- viewer backends
- input handling
- camera APIs
- render-time state access

Example:

`bash skills/newton-nightly-compat/scripts/run_mimic_smoke.sh --epochs 10 --viewer viser`

## What counts as a regression

Treat these as nightly failures:

- simulator initialization crash
- rollout crash before epoch 10
- training crash before epoch 10
- deterministic repro of a new traceback on rerun

Treat these as warnings to review but not automatic failures:

- known Python syntax warnings from third-party parsing
- MuJoCo CCD warnings that do not stop training
- PyTorch stream mismatch warnings when training still proceeds

## Escalation rule

Escalate beyond the adapter only when:

- the traceback does not touch the Newton integration layer, or
- the adapter can no longer express the required behavior without deeper config or env changes.
