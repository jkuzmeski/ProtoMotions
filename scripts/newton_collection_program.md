# Newton Collection Autoresearch

You are optimizing Newton collection throughput in ProtoMotions.

Objective:
- Maximize `samples_per_s` from the fixed benchmark command provided by the runner.

Allowed edit scope:
- `protomotions/simulator/newton/simulator.py`
- `protomotions/simulator/newton/config.py`

Do not edit:
- Benchmark harnesses
- Runner scripts
- Tests
- Unrelated simulator backends

Constraints:
- Preserve correctness and API compatibility.
- Keep the Newton backend usable for normal training.
- Prefer changes that improve repeated collection-time paths, not only startup.
- Avoid one-off benchmark hacks keyed to the benchmark script.
- The benchmark uses the real Newton rollout collection stack, not a synthetic simulator-only loop.

What matters most:
- Reducing repeated `wp.to_torch` / `wp.from_torch` bridge overhead
- Avoiding repeated full-state materialization in a single collection step
- Reducing Python-side allocation and list/dict work on hot paths
- Improving the default built-in PD path

Good changes:
- Add step-local caches that are invalidated when simulation state changes
- Reuse persistent buffers instead of reallocating every step
- Replace Python loops in hot paths with precomputed indices or tensorized code
- Keep benchmark-independent design quality

Bad changes:
- Removing functionality that collection relies on
- Special-casing behavior only for the benchmark mode
- Editing unrelated files to suppress work instead of making Newton faster

Workflow:
1. Inspect the benchmark target and hot path.
2. Make one coherent optimization pass.
3. Leave a concise final note explaining what changed and why it should improve throughput.
