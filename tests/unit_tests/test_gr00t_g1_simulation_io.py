# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GR00T G1 DreamDojo I/O bridge."""

import numpy as np
import torch

from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_g1_dex3_obs_to_gr00t_format,
    convert_to_g1_dex3_action,
)


def test_convert_g1_dex3_observation():
    observation = {
        "main_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
        "states": torch.arange(56, dtype=torch.float32).reshape(2, 28),
        "task_descriptions": ["task a", "task b"],
    }

    converted = convert_g1_dex3_obs_to_gr00t_format(observation)

    assert converted["video.head_view"].shape == (2, 1, 480, 640, 3)
    np.testing.assert_array_equal(
        converted["state.left_arm"], observation["states"][:, None, 0:7].numpy()
    )
    np.testing.assert_array_equal(
        converted["state.right_hand"], observation["states"][:, None, 21:28].numpy()
    )
    assert converted["annotation.human.task_description"] == ["task a", "task b"]


def test_convert_g1_dex3_action_order():
    for prefix in ("", "action."):
        action_chunk = {
            f"{prefix}left_arm": np.full((2, 25, 7), 1.0),
            f"{prefix}right_arm": np.full((2, 25, 7), 2.0),
            f"{prefix}left_hand": np.full((2, 25, 7), 3.0),
            f"{prefix}right_hand": np.full((2, 25, 7), 4.0),
        }

        converted = convert_to_g1_dex3_action(action_chunk, chunk_size=25)

        assert converted.shape == (2, 25, 28)
        np.testing.assert_array_equal(converted[..., 0:7], 1.0)
        np.testing.assert_array_equal(converted[..., 7:14], 2.0)
        np.testing.assert_array_equal(converted[..., 14:21], 3.0)
        np.testing.assert_array_equal(converted[..., 21:28], 4.0)
