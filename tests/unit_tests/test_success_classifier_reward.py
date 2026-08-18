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

"""Tests for conservative temporal success aggregation."""

from unittest.mock import patch

import torch
import torch.nn as nn
import torchvision.models as models

from rlinf.envs.world_model.success_classifier_reward import (
    SuccessClassifierReward,
)


def _write_classifier_checkpoint(path):
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Module()
    model.backbone = backbone
    model.head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )
    torch.save({"model": model.state_dict()}, path)


def test_classifier_checkpoint_key_remap(tmp_path):
    checkpoint = tmp_path / "classifier.pt"
    _write_classifier_checkpoint(checkpoint)

    reward = SuccessClassifierReward(checkpoint, num_envs=2)

    assert reward.backbone.fc[0].weight.shape == (256, 512)


def test_temporal_aggregation_rejects_single_frame_spike(tmp_path):
    checkpoint = tmp_path / "classifier.pt"
    _write_classifier_checkpoint(checkpoint)
    reward = SuccessClassifierReward(
        checkpoint,
        num_envs=1,
        temporal_window=4,
        min_stable_frames=4,
        success_threshold=0.8,
        majority_fraction=0.75,
    )
    probabilities = torch.tensor([[0.1, 0.1, 0.99, 0.1]])

    with patch.object(reward, "predict_probabilities", return_value=probabilities):
        rewards, success = reward.evaluate_chunk(
            torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8)
        )

    assert not success.item()
    assert rewards.sum().item() == 0


def test_temporal_aggregation_accepts_persistent_success(tmp_path):
    checkpoint = tmp_path / "classifier.pt"
    _write_classifier_checkpoint(checkpoint)
    reward = SuccessClassifierReward(
        checkpoint,
        num_envs=1,
        temporal_window=4,
        min_stable_frames=4,
        success_threshold=0.8,
        majority_fraction=0.75,
    )
    probabilities = torch.tensor([[0.82, 0.85, 0.9, 0.88]])

    with patch.object(reward, "predict_probabilities", return_value=probabilities):
        rewards, success = reward.evaluate_chunk(
            torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8)
        )

    assert success.item()
    assert rewards[0, -1].item() == 1
