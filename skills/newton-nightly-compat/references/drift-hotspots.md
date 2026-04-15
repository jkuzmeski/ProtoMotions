# Drift Hotspots

Check these surfaces first when Newton rebases break ProtoMotions.

## 1. Label and selection drift

Newton now commonly preserves hierarchical labels in finalized models.
Leaf names like `robot`, `L_Ankle`, or `R_Toe` may need wildcard selectors such as `*/robot` or `*/L_Ankle`.

Start here when tracebacks mention:

- `ArticulationView`
- `SensorContact`
- `No articulations matching pattern`
- `No bodies matched`

## 2. Builder identifier drift

Older code may expect builder fields like `joint_key` or `articulation_key`.
Newer Newton builds may expose `joint_label` or `articulation_label` instead.

Prefer deriving leaf names from labels instead of assuming the older field exists.

## 3. MuJoCo attribute drift

`model.mujoco` is not stable across Newton revisions.
Some builds no longer expose attributes such as `geom_gap` or `geom_margin` even though `solver.mjw_model` still does.

Guard attribute writes with `hasattr(...)` and update whichever surface exists in the current fork.

## 4. Contact sensor API drift

Older code may call `sensor.eval(contacts)` and read `sensor.net_force`.
Newer builds may require `sensor.update(state, contacts)` and expose `sensor.total_force`.

Handle both forms when possible.

## 5. Viewer drift

`ViewerGL` and `ViewerViser` do not expose camera state the same way.
`ViewerViser` may cache camera data in `_camera_request` instead of a `.camera.pos` object.

Treat render and camera code as backend-specific.

## 6. Patch target

Start in `protomotions/simulator/newton/simulator.py`.
Only move into env, agent, or terrain code when the traceback proves the drift is not confined to the Newton adapter.
