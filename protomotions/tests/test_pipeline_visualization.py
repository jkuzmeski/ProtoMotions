# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from HumanRetargeting.biomechanics_retarget.pipeline_visualization import (
    TRACKING_BODY_INDEX,
    _compute_tracking_viewport,
)


def test_compute_tracking_viewport_tracks_pose_not_total_forward_distance():
    base_pose = np.array(
        [
            [0.0, 0.0, 0.95],
            [-0.10, -0.02, 0.88],
            [-0.11, -0.03, 0.48],
            [-0.12, -0.05, 0.09],
            [-0.05, 0.12, 0.02],
            [0.10, -0.02, 0.88],
            [0.11, -0.03, 0.48],
            [0.12, -0.05, 0.09],
            [0.05, 0.12, 0.02],
        ],
        dtype=np.float32,
    )

    num_frames = 40
    forward_offsets = np.linspace(0.0, 12.0, num_frames, dtype=np.float32)
    positions = np.repeat(base_pose[None, :, :], num_frames, axis=0)
    positions[:, :, 1] += forward_offsets[:, None]

    center_offset, half_range = _compute_tracking_viewport(positions)

    assert half_range < 1.0

    start_center_y = positions[0, TRACKING_BODY_INDEX, 1] + center_offset[1]
    end_center_y = positions[-1, TRACKING_BODY_INDEX, 1] + center_offset[1]
    assert np.isclose(end_center_y - start_center_y, 12.0, atol=1e-6)
