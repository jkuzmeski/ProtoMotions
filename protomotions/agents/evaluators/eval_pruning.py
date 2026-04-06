# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from torch import Tensor


def get_eval_termination_mask(
    max_joint_err: Tensor, threshold: Optional[float]
) -> Tensor:
    """Return which active evaluation envs should stop collecting frames."""
    if threshold is None:
        return torch.zeros_like(max_joint_err, dtype=torch.bool)
    return max_joint_err > threshold


def prune_completed_eval_envs(
    active_env_ids: Tensor,
    active_motion_ids: Tensor,
    dones: Tensor,
    terminated: Tensor,
    max_joint_err: Tensor,
    early_terminate_max_joint_err: Optional[float],
) -> Tuple[Tensor, Tensor]:
    """Drop active evaluation envs that are done or have clearly failed."""
    finished_mask = dones[active_env_ids] | terminated[active_env_ids]
    eval_terminated_mask = get_eval_termination_mask(
        max_joint_err=max_joint_err,
        threshold=early_terminate_max_joint_err,
    )
    keep_mask = ~(finished_mask | eval_terminated_mask)
    return active_env_ids[keep_mask], active_motion_ids[keep_mask]
