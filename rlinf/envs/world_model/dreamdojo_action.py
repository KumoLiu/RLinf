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

"""Action conversion utilities for the G1 DreamDojo world model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

G1_POLICY_ACTION_DIM = 28
G1_DREAMDOJO_ACTION_DIM = 43
DREAMDOJO_CONDITION_DIM = 384
DREAMDOJO_G1_SLOT = slice(58, 101)
DREAMDOJO_CHUNK_SIZE = 12
DREAMDOJO_POLICY_HORIZON = 25


class G1DreamDojoActionBridge:
    """Convert decoded GR00T G1 actions to DreamDojo action conditioning.

    GR00T emits 28 physical joint targets ordered as left arm, right arm,
    left hand, and right hand. DreamDojo was trained with a 43-dimensional G1
    layout that additionally contains fixed legs and waist, and embeds
    normalized action deltas into dimensions ``[58:101]`` of a 384-dimensional
    conditioning vector.
    """

    def __init__(self, statistics_path: str | Path):
        statistics_path = Path(statistics_path).expanduser()
        with statistics_path.open() as f:
            statistics = json.load(f)
        action_statistics = statistics["action"]
        self._minimum = torch.tensor(action_statistics["min"], dtype=torch.float32)
        self._maximum = torch.tensor(action_statistics["max"], dtype=torch.float32)
        if self._minimum.shape != (G1_DREAMDOJO_ACTION_DIM,):
            raise ValueError(
                "DreamDojo action statistics must contain 43 values, got "
                f"{tuple(self._minimum.shape)}"
            )

    @staticmethod
    def expand_policy_actions(actions: torch.Tensor) -> torch.Tensor:
        """Expand and reorder 28-D G1 actions into DreamDojo's 43-D layout."""
        if actions.shape[-1] != G1_POLICY_ACTION_DIM:
            raise ValueError(
                f"Expected {G1_POLICY_ACTION_DIM}-D G1 actions, got {actions.shape[-1]}"
            )
        expanded = actions.new_zeros(*actions.shape[:-1], G1_DREAMDOJO_ACTION_DIM)
        expanded[..., 15:22] = actions[..., 0:7]  # left arm
        expanded[..., 22:29] = actions[..., 14:21]  # left hand
        expanded[..., 29:36] = actions[..., 7:14]  # right arm
        expanded[..., 36:43] = actions[..., 21:28]  # right hand
        return expanded

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply DreamDojo's min-max normalization to physical G1 actions."""
        minimum = self._minimum.to(device=actions.device, dtype=actions.dtype)
        maximum = self._maximum.to(device=actions.device, dtype=actions.dtype)
        varying = minimum != maximum
        normalized = torch.zeros_like(actions)
        normalized[..., varying] = (
            2
            * (actions[..., varying] - minimum[varying])
            / (maximum[varying] - minimum[varying])
            - 1
        )
        return normalized

    @staticmethod
    def compute_block_deltas(sampled_actions: torch.Tensor) -> torch.Tensor:
        """Reproduce DreamDojo's 4-action block-relative delta encoding.

        Args:
            sampled_actions: Normalized actions shaped ``[..., 13, 43]``. The
                first item is the baseline and the remaining 12 items condition
                the generated frame chunk.

        Returns:
            Tensor shaped ``[..., 12, 43]``.
        """
        if sampled_actions.shape[-2:] != (
            DREAMDOJO_CHUNK_SIZE + 1,
            G1_DREAMDOJO_ACTION_DIM,
        ):
            raise ValueError(
                "Expected sampled actions ending in (13, 43), got "
                f"{tuple(sampled_actions.shape)}"
            )
        deltas = []
        for start in (1, 5, 9):
            deltas.append(
                sampled_actions[..., start : start + 4, :]
                - sampled_actions[..., start - 1 : start, :]
            )
        return torch.cat(deltas, dim=-2)

    def encode_sampled_actions(
        self, sampled_policy_actions: torch.Tensor
    ) -> torch.Tensor:
        """Encode 13 policy-space samples into 12 DreamDojo conditions."""
        expanded = self.expand_policy_actions(sampled_policy_actions)
        normalized = self.normalize(expanded)
        deltas = self.compute_block_deltas(normalized)
        conditioning = deltas.new_zeros(*deltas.shape[:-1], DREAMDOJO_CONDITION_DIM)
        conditioning[..., DREAMDOJO_G1_SLOT] = deltas
        return conditioning

    def encode_30hz_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        """Encode a 25-step 30 Hz GR00T horizon for the 15 Hz DreamDojo model.

        DreamDojo training sampled source actions at offsets
        ``0, 2, ..., 24``. Consequently, exact conditioning requires 25 policy
        targets; a 24-step horizon cannot represent the final sample.
        """
        if policy_actions.shape[-2:] != (
            DREAMDOJO_POLICY_HORIZON,
            G1_POLICY_ACTION_DIM,
        ):
            raise ValueError(
                "Expected policy actions ending in (25, 28), got "
                f"{tuple(policy_actions.shape)}"
            )
        sampled_actions = policy_actions[..., 0:25:2, :]
        return self.encode_sampled_actions(sampled_actions)
