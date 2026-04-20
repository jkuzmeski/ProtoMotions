import torch

from protomotions.envs.rewards import (
    compute_foot_crossover_rew,
    compute_foot_slip_rew,
)


def test_compute_foot_slip_rew_only_penalizes_planted_feet():
    current_rigid_body_vel = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    sim_contacts = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )
    ref_contacts = torch.tensor(
        [[False, False, False], [False, True, False]],
        dtype=torch.bool,
    )
    foot_body_ids = torch.tensor([1], dtype=torch.long)

    rew = compute_foot_slip_rew(
        current_rigid_body_vel=current_rigid_body_vel,
        sim_contacts=sim_contacts,
        ref_contacts=ref_contacts,
        foot_body_ids=foot_body_ids,
    )

    assert torch.allclose(rew, torch.tensor([0.5, 1.0]))


def test_compute_foot_crossover_rew_penalizes_feet_that_swap_sides():
    current_rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.10, 0.0], [0.0, -0.10, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -0.02, 0.0], [0.0, 0.01, 0.0]],
        ],
        dtype=torch.float32,
    )
    root_pos = torch.zeros((2, 3), dtype=torch.float32)
    root_rot = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    left_foot_body_ids = torch.tensor([1], dtype=torch.long)
    right_foot_body_ids = torch.tensor([2], dtype=torch.long)

    rew = compute_foot_crossover_rew(
        current_rigid_body_pos=current_rigid_body_pos,
        root_pos=root_pos,
        root_rot=root_rot,
        left_foot_body_ids=left_foot_body_ids,
        right_foot_body_ids=right_foot_body_ids,
        min_lateral_separation=0.06,
    )

    assert torch.allclose(rew, torch.tensor([0.0, 0.09]))