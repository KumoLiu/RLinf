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

"""Tests for G1-to-DreamDojo action conversion."""

import json

import pytest
import torch

from rlinf.envs.world_model.dreamdojo_action import (
    DREAMDOJO_G1_SLOT,
    G1DreamDojoActionBridge,
)


@pytest.fixture
def bridge(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {
                    "min": [0.0] * 43,
                    "max": [2.0] * 43,
                }
            }
        )
    )
    return G1DreamDojoActionBridge(stats_path)


def test_expand_policy_actions_reorders_g1_joints(bridge):
    actions = torch.arange(28, dtype=torch.float32).reshape(1, 1, 28)

    expanded = bridge.expand_policy_actions(actions)

    torch.testing.assert_close(expanded[..., :15], torch.zeros(1, 1, 15))
    torch.testing.assert_close(expanded[..., 15:22], actions[..., 0:7])
    torch.testing.assert_close(expanded[..., 22:29], actions[..., 14:21])
    torch.testing.assert_close(expanded[..., 29:36], actions[..., 7:14])
    torch.testing.assert_close(expanded[..., 36:43], actions[..., 21:28])


def test_encode_sampled_actions_matches_block_relative_deltas(bridge):
    sampled = torch.zeros(1, 13, 28)
    sampled[:, :, 0] = torch.arange(13)

    encoded = bridge.encode_sampled_actions(sampled)

    assert encoded.shape == (1, 12, 384)
    expected = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.float32)
    # min=0, max=2 maps x to x-1, so subtraction preserves each delta.
    torch.testing.assert_close(encoded[0, :, 73], expected)
    assert torch.count_nonzero(encoded[..., : DREAMDOJO_G1_SLOT.start]) == 0
    assert torch.count_nonzero(encoded[..., DREAMDOJO_G1_SLOT.stop :]) == 0


def test_encode_30hz_actions_selects_even_offsets(bridge):
    actions = torch.zeros(2, 25, 28)
    actions[:, :, 0] = torch.arange(25)

    encoded = bridge.encode_30hz_actions(actions)

    expected = torch.tensor([2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8], dtype=torch.float32)
    torch.testing.assert_close(encoded[0, :, 73], expected)


def test_encode_30hz_actions_rejects_short_horizon(bridge):
    with pytest.raises(ValueError, match=r"\(25, 28\)"):
        bridge.encode_30hz_actions(torch.zeros(1, 24, 28))
