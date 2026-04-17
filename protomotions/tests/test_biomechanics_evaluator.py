# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _register_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_biomechanics_evaluator_module():
    _register_package("protomotions")
    _register_package("protomotions.agents")
    _register_package("protomotions.agents.evaluators")
    _register_package("protomotions.envs")
    _register_package("protomotions.envs.control")
    utils_module = _register_package("protomotions.utils")

    base_evaluator_module = types.ModuleType(
        "protomotions.agents.evaluators.base_evaluator"
    )

    class BaseEvaluator:
        def __init__(self, agent=None, fabric=None, config=None):
            self.agent = agent
            self.fabric = fabric
            self.config = config

    base_evaluator_module.BaseEvaluator = BaseEvaluator
    sys.modules[base_evaluator_module.__name__] = base_evaluator_module

    config_module = types.ModuleType("protomotions.agents.evaluators.config")

    class BiomechanicsEvaluatorConfig:
        pass

    config_module.BiomechanicsEvaluatorConfig = BiomechanicsEvaluatorConfig
    sys.modules[config_module.__name__] = config_module

    speed_control_module = types.ModuleType("protomotions.envs.control.speed_control")

    class SpeedControl:
        pass

    speed_control_module.SpeedControl = SpeedControl
    sys.modules[speed_control_module.__name__] = speed_control_module

    steering_control_module = types.ModuleType(
        "protomotions.envs.control.steering_control"
    )

    class SteeringControl:
        pass

    steering_control_module.SteeringControl = SteeringControl
    sys.modules[steering_control_module.__name__] = steering_control_module

    rotations_module = types.ModuleType("protomotions.utils.rotations")
    utils_module.rotations = rotations_module
    sys.modules[rotations_module.__name__] = rotations_module

    module_path = (
        Path(__file__).resolve().parents[1]
        / "agents"
        / "evaluators"
        / "biomechanics_evaluator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "biomechanics_evaluator_test_module",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load_biomechanics_evaluator_module()
BiomechanicsEvaluator = _MODULE.BiomechanicsEvaluator
SteeringControl = sys.modules[
    "protomotions.envs.control.steering_control"
].SteeringControl


def test_cycle_normalized_joint_plot_specs_prefers_sagittal_lower_body_joints():
    evaluator = BiomechanicsEvaluator.__new__(BiomechanicsEvaluator)
    evaluator._feature_names = [
        "pelvis_flex",
        "left_ankle_flex",
        "right_ankle_flex",
        "left_hip_flex",
        "right_hip_flex",
        "left_knee_flex",
        "right_knee_flex",
        "left_hip_add",
    ]

    specs = evaluator._cycle_normalized_joint_plot_specs()

    assert specs == [
        ("Hip", {"Left": "left_hip_flex", "Right": "right_hip_flex"}),
        ("Knee", {"Left": "left_knee_flex", "Right": "right_knee_flex"}),
        ("Ankle", {"Left": "left_ankle_flex", "Right": "right_ankle_flex"}),
    ]


def test_log_cycle_normalized_joint_figure_uses_speed_specific_tensorboard_tag():
    calls = []

    evaluator = BiomechanicsEvaluator.__new__(BiomechanicsEvaluator)
    evaluator.fabric = SimpleNamespace(loggers=[object()])
    evaluator._create_cycle_normalized_joint_figure = lambda **_: "joint-figure"
    evaluator._log_tensorboard_figure = lambda tag, figure: calls.append(
        {"tag": tag, "figure": figure}
    )

    evaluator._log_cycle_normalized_joint_figure(
        speed_tag="1p25",
        target_speed=1.25,
        phase=[],
        waveform_exports={},
        post_burn_in_cycle_count=3,
        feature_names=["left_hip_flex", "right_hip_flex"],
    )

    assert calls == [
        {
            "tag": "eval/biomechanics/cycle_normalized_joints/1p25",
            "figure": "joint-figure",
        }
    ]


def test_cache_and_restore_eval_state_uses_env_snapshot_and_restores_control_state():
    saved_snapshot = {"snapshot": "env"}
    restore_calls = []

    control = SteeringControl()
    control._heading_change_steps = torch.tensor([4, 9], dtype=torch.int64)
    control._tar_dir_theta = torch.tensor([0.1, 0.2], dtype=torch.float32)
    control._tar_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    control._tar_face_dir = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
    )
    control._tar_speed = torch.tensor([1.25, 1.75], dtype=torch.float32)
    control._prev_root_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
    )

    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([3, 7], dtype=torch.long),
        motion_times=torch.tensor([0.25, 0.75], dtype=torch.float32),
    )
    env = SimpleNamespace(
        save_state=lambda: saved_snapshot,
        restore_state=lambda snapshot: restore_calls.append(snapshot),
        motion_manager=motion_manager,
        control_manager=SimpleNamespace(components={"steering": control}),
    )

    evaluator = BiomechanicsEvaluator.__new__(BiomechanicsEvaluator)
    evaluator.agent = SimpleNamespace(env=env)
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))
    evaluator.env = env

    evaluator._cache_eval_state()

    motion_manager.motion_ids.fill_(0)
    motion_manager.motion_times.fill_(0.0)
    control._heading_change_steps.fill_(0)
    control._tar_dir_theta.fill_(0.0)
    control._tar_dir.fill_(0.0)
    control._tar_face_dir.fill_(0.0)
    control._tar_speed.fill_(0.0)
    control._prev_root_pos.fill_(0.0)

    evaluator._restore_eval_state()

    assert restore_calls == [saved_snapshot]
    assert torch.equal(motion_manager.motion_ids, torch.tensor([3, 7], dtype=torch.long))
    assert torch.equal(
        motion_manager.motion_times, torch.tensor([0.25, 0.75], dtype=torch.float32)
    )
    assert torch.equal(control._heading_change_steps, torch.tensor([4, 9], dtype=torch.int64))
    assert torch.equal(
        control._tar_dir_theta, torch.tensor([0.1, 0.2], dtype=torch.float32)
    )
    assert torch.equal(
        control._tar_dir, torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    )
    assert torch.equal(
        control._tar_face_dir,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
    )
    assert torch.equal(control._tar_speed, torch.tensor([1.25, 1.75], dtype=torch.float32))
    assert torch.equal(
        control._prev_root_pos,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
    )
    assert evaluator._env_snapshot is None
