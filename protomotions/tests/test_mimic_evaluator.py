# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "agents" / "evaluators" / "eval_pruning.py"
)
_SPEC = importlib.util.spec_from_file_location("eval_pruning_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

get_eval_termination_mask = _MODULE.get_eval_termination_mask
prune_completed_eval_envs = _MODULE.prune_completed_eval_envs


def test_eval_termination_mask_uses_max_joint_error_threshold():
    mask = get_eval_termination_mask(
        max_joint_err=torch.tensor([0.25, 1.0, 1.25], dtype=torch.float32),
        threshold=1.0,
    )

    assert torch.equal(mask, torch.tensor([False, False, True]))


def test_eval_termination_mask_can_be_disabled():
    mask = get_eval_termination_mask(
        max_joint_err=torch.tensor([0.25, 1.25], dtype=torch.float32),
        threshold=None,
    )

    assert torch.equal(mask, torch.tensor([False, False]))


def test_prune_completed_eval_envs_removes_done_and_failed_envs():
    active_env_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    active_motion_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    dones = torch.tensor([False, False, False, False, True])
    terminated = torch.tensor([False, False, False, False, False])
    max_joint_err = torch.tensor([1.2, 0.3, 0.8], dtype=torch.float32)

    next_env_ids, next_motion_ids = prune_completed_eval_envs(
        active_env_ids=active_env_ids,
        active_motion_ids=active_motion_ids,
        dones=dones,
        terminated=terminated,
        max_joint_err=max_joint_err,
        early_terminate_max_joint_err=1.0,
    )

    assert torch.equal(next_env_ids, torch.tensor([2], dtype=torch.long))
    assert torch.equal(next_motion_ids, torch.tensor([11], dtype=torch.long))
