from __future__ import annotations

import numpy as np
import warp as wp

from protomotions.simulator.newton.mujoco_compat import (
    canonicalize_joint_dof_attribute,
    install_mujoco_joint_arg_coercion,
    normalize_joint_dof_mujoco_attributes,
)


def test_canonicalize_joint_dof_attribute_flattens_scalar_row_vector() -> None:
    values = np.array([[250.0, 200.0, 150.0]], dtype=np.float32)

    canonical = canonicalize_joint_dof_attribute(values, joint_dof_count=3)

    np.testing.assert_array_equal(canonical, np.array([250.0, 200.0, 150.0], dtype=np.float32))


def test_canonicalize_joint_dof_attribute_flattens_scalar_column_vector() -> None:
    values = np.array([[250.0], [200.0], [150.0]], dtype=np.float32)

    canonical = canonicalize_joint_dof_attribute(values, joint_dof_count=3)

    np.testing.assert_array_equal(canonical, np.array([250.0, 200.0, 150.0], dtype=np.float32))


def test_canonicalize_joint_dof_attribute_preserves_vector_attributes() -> None:
    values = np.array([[[0.9, 0.95], [0.8, 0.85], [0.7, 0.75]]], dtype=np.float32)

    canonical = canonicalize_joint_dof_attribute(values, joint_dof_count=3)

    np.testing.assert_array_equal(
        canonical,
        np.array(
            [
                [0.9, 0.95],
                [0.8, 0.85],
                [0.7, 0.75],
            ],
            dtype=np.float32,
        ),
    )


def test_canonicalize_joint_dof_attribute_leaves_unmatched_shapes_unchanged() -> None:
    values = np.ones((2, 2), dtype=np.float32)

    canonical = canonicalize_joint_dof_attribute(values, joint_dof_count=3)

    assert canonical is values


def test_normalize_joint_dof_mujoco_attributes_replaces_bad_scalar_shape() -> None:
    class DummyNamespace:
        pass

    class DummyModel:
        joint_dof_count = 3
        mujoco = DummyNamespace()

    model = DummyModel()
    model.mujoco.dof_passive_stiffness = wp.array(
        np.array([[250.0, 200.0, 150.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )

    normalize_joint_dof_mujoco_attributes(model)

    assert model.mujoco.dof_passive_stiffness.shape == (3,)
    np.testing.assert_array_equal(
        model.mujoco.dof_passive_stiffness.numpy(),
        np.array([250.0, 200.0, 150.0], dtype=np.float32),
    )


def test_install_mujoco_joint_arg_coercion_accepts_numpy_array_scalar() -> None:
    import mujoco

    install_mujoco_joint_arg_coercion()

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="body")
    joint = body.add_joint(
        name="joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        stiffness=np.array([250.0], dtype=np.float32),
    )

    assert joint is not None
