# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import math
import json

import pytest
import torch

from protomotions.envs.control.speed_control import SpeedControl, SpeedControlConfig
from protomotions.envs.obs.speed_obs_functions import compute_speed_obs, speed_obs_factory


class _FakeSimulator:
    def __init__(self, root_pos: torch.Tensor):
        self._root_state = SimpleNamespace(root_pos=root_pos)

    def get_root_state(self):
        return self._root_state


class _FakeEnv:
    def __init__(self, num_envs: int, root_pos: torch.Tensor, motion_lib=None, motion_manager=None):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.simulator = _FakeSimulator(root_pos=root_pos)
        self.progress_buf = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.motion_lib = motion_lib
        self.motion_manager = motion_manager


def test_speed_control_holds_standing_then_activates_fixed_speed():
    env = _FakeEnv(num_envs=2, root_pos=torch.zeros((2, 3), dtype=torch.float32))
    config = SpeedControlConfig(
        target_speed=2.5,
        heading_theta=math.pi / 2.0,
        standing_reset_steps=2,
    )

    control = SpeedControl(config=config, env=env)
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    control.reset(env_ids)

    context = control.get_context()
    assert torch.allclose(context["tar_dir"], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert torch.allclose(context["tar_speed"], torch.zeros(2))
    assert torch.equal(context["standing_reset_steps_remaining"], torch.tensor([2, 2]))
    assert torch.equal(context["is_standing"], torch.tensor([True, True]))

    control.step()
    assert torch.allclose(control.get_context()["tar_speed"], torch.zeros(2))
    assert torch.equal(control.get_context()["standing_reset_steps_remaining"], torch.tensor([1, 1]))

    control.step()
    context = control.get_context()
    assert torch.allclose(context["tar_speed"], torch.full((2,), 2.5))
    assert torch.equal(context["standing_reset_steps_remaining"], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(context["is_standing"], torch.tensor([False, False]))

    reset_buf, terminate_buf = control.check_resets_and_terminations()
    assert not reset_buf.any()
    assert not terminate_buf.any()


def test_speed_control_rejects_invalid_configuration():
    env = _FakeEnv(num_envs=1, root_pos=torch.zeros((1, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="target_speed must be non-negative"):
        SpeedControl(
            config=SpeedControlConfig(target_speed=-0.1),
            env=env,
        )


def test_compute_speed_obs_returns_speed_and_local_heading():
    root_rot = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    tar_dir = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tar_speed = torch.tensor([1.5, 3.5], dtype=torch.float32)

    obs = compute_speed_obs(root_rot=root_rot, tar_dir=tar_dir, tar_speed=tar_speed)

    assert obs.shape == (2, 3)
    assert torch.allclose(obs, torch.tensor([[1.5, 1.0, 0.0], [3.5, 0.0, 1.0]]))


def test_speed_obs_factory_exposes_expected_context_keys():
    config = speed_obs_factory()

    assert config.function is compute_speed_obs
    assert config.variables == {
        "root_rot": "current_state_root_rot",
        "tar_dir": "tar_dir",
        "tar_speed": "tar_speed",
    }


def test_speed_control_can_source_command_from_motion_metadata(tmp_path):
    motion_dir = tmp_path / "motion_files"
    metadata_dir = motion_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    motion_paths = [
        motion_dir / "S02_15ms_Long.motion",
        motion_dir / "custom_stride.motion",
    ]
    metadata_dir.joinpath("S02_15ms_Long.json").write_text(
        json.dumps({"speed_mps": 1.5}),
        encoding="utf-8",
    )
    metadata_dir.joinpath("custom_stride.json").write_text(
        json.dumps({"speed_mps": 2.75}),
        encoding="utf-8",
    )

    motion_lib = SimpleNamespace(motion_files=tuple(str(path) for path in motion_paths))
    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    env = _FakeEnv(
        num_envs=2,
        root_pos=torch.zeros((2, 3), dtype=torch.float32),
        motion_lib=motion_lib,
        motion_manager=motion_manager,
    )

    control = SpeedControl(
        config=SpeedControlConfig(
            target_speed=9.9,
            speed_source="motion_file",
            standing_reset_steps=1,
        ),
        env=env,
    )
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    control.reset(env_ids)
    assert torch.allclose(control.get_context()["commanded_tar_speed"], torch.tensor([1.5, 2.75]))
    assert torch.allclose(control.get_context()["tar_speed"], torch.zeros(2))

    control.step()
    assert torch.allclose(control.get_context()["tar_speed"], torch.tensor([1.5, 2.75]))
