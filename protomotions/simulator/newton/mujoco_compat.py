from __future__ import annotations

import os
import numpy as np
import warp as wp


_JOINT_DOF_MUJOCO_ATTRS = (
    "limit_margin",
    "solimplimit",
    "solreffriction",
    "solimpfriction",
    "dof_passive_stiffness",
    "dof_passive_damping",
    "dof_springref",
    "dof_ref",
    "jnt_actgravcomp",
)

_ADD_JOINT_COERCION_INSTALLED = False


def canonicalize_joint_dof_attribute(
    values: np.ndarray,
    joint_dof_count: int,
) -> np.ndarray:
    """Normalize a MuJoCo joint-DOF attribute to a solver-friendly shape.

    MuJoCo 3.7 rejects length-1 ndarrays for scalar joint fields such as
    ``stiffness``. Newton normally stores these attributes as flat arrays, but
    defensive reshaping here avoids failures if a singleton dimension is
    introduced upstream.
    """

    array = np.asarray(values)
    if array.ndim <= 1 or joint_dof_count <= 0 or array.size == 0:
        return array
    if array.size % joint_dof_count != 0:
        return array

    canonical = array.reshape(joint_dof_count, -1)
    if canonical.shape[1] == 1:
        return canonical.reshape(joint_dof_count)
    return canonical


def normalize_joint_dof_mujoco_attributes(model) -> None:
    """Collapse singleton dimensions on MuJoCo joint-DOF custom attributes."""

    mujoco_attrs = getattr(model, "mujoco", None)
    if mujoco_attrs is None:
        return

    joint_dof_count = int(getattr(model, "joint_dof_count", 0))
    if joint_dof_count <= 0:
        return

    for attr_name in _JOINT_DOF_MUJOCO_ATTRS:
        attr = getattr(mujoco_attrs, attr_name, None)
        if attr is None:
            continue

        values = attr.numpy()
        canonical = canonicalize_joint_dof_attribute(values, joint_dof_count)
        if canonical.shape != values.shape:
            setattr(
                mujoco_attrs,
                attr_name,
                wp.array(
                    canonical,
                    dtype=attr.dtype,
                    device=attr.device,
                    requires_grad=getattr(attr, "requires_grad", False),
                ),
            )


def _coerce_mujoco_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.reshape(-1)[0].item()
        return [_coerce_mujoco_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_coerce_mujoco_value(item) for item in value]
    if isinstance(value, list):
        return [_coerce_mujoco_value(item) for item in value]
    return value


def install_mujoco_joint_arg_coercion() -> None:
    """Patch MuJoCo's joint builder to accept NumPy-derived kwargs robustly."""

    global _ADD_JOINT_COERCION_INSTALLED
    if _ADD_JOINT_COERCION_INSTALLED:
        return

    import mujoco

    body_cls = type(mujoco.MjSpec().worldbody)
    original_add_joint = body_cls.add_joint

    def coerced_add_joint(self, *args, **kwargs):
        try:
            coerced_kwargs = {
                key: _coerce_mujoco_value(value) for key, value in kwargs.items()
            }
            return original_add_joint(self, *args, **coerced_kwargs)
        except TypeError:
            if os.environ.get("PROTO_DEBUG_MUJOCO_JOINTS") == "1":
                debug_items = ", ".join(
                    f"{key}={type(value).__name__}:{repr(value)[:120]}"
                    for key, value in kwargs.items()
                )
                print(f"[DEBUG] MuJoCo add_joint kwargs: {debug_items}")
            raise

    body_cls.add_joint = coerced_add_joint
    _ADD_JOINT_COERCION_INSTALLED = True
