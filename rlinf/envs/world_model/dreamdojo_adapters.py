# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor adapters between GR00T G1-Dex3 and DreamDojo conditioning."""

from __future__ import annotations

import json
from pathlib import Path

import torch

G1_DIM = 43
DREAMDOJO_ACTION_DIM = 384
DREAMDOJO_G1_SLICE = slice(58, 101)

_LEFT_ARM = slice(15, 22)
_LEFT_HAND = slice(22, 29)
_RIGHT_ARM = slice(29, 36)
_RIGHT_HAND = slice(36, 43)

_SRC_LEFT_ARM = slice(0, 7)
_SRC_RIGHT_ARM = slice(7, 14)
_SRC_LEFT_HAND = slice(14, 21)
_SRC_RIGHT_HAND = slice(21, 28)


def pad_g1_dex3_28_to_43(values: torch.Tensor) -> torch.Tensor:
    """Pad the upper-body Dex3 layout into DreamDojo's official G1 layout."""
    if values.shape[-1] != 28:
        raise ValueError(f"Expected 28-D G1-Dex3 values, got {tuple(values.shape)}")
    out = values.new_zeros((*values.shape[:-1], G1_DIM))
    out[..., _LEFT_ARM] = values[..., _SRC_LEFT_ARM]
    out[..., _LEFT_HAND] = values[..., _SRC_LEFT_HAND]
    out[..., _RIGHT_ARM] = values[..., _SRC_RIGHT_ARM]
    out[..., _RIGHT_HAND] = values[..., _SRC_RIGHT_HAND]
    return out


def update_g1_dex3_state(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Advance proprio from GR00T's decoded absolute joint targets.

    GR00T represents arm actions relatively inside its processor, but
    ``unapply_action`` converts them back to absolute targets before the
    environment action converter is called.
    """
    if state.shape[-1] != 28 or actions.shape[-1] != 28:
        raise ValueError("G1-Dex3 state and actions must both be 28-D")
    return actions.clone()


class G1DreamDojoActionBridge:
    """Reproduce DreamDojo's 30-Hz G1 preprocessing for online policy chunks."""

    def __init__(self, statistics_path: str | Path, device: torch.device | str):
        all_statistics = json.loads(Path(statistics_path).read_text())
        statistics = all_statistics["action"]
        self.device = torch.device(device)
        self.minimum = torch.tensor(
            statistics["min"], dtype=torch.float32, device=self.device
        )
        self.maximum = torch.tensor(
            statistics["max"], dtype=torch.float32, device=self.device
        )
        self.state_minimum = torch.tensor(
            all_statistics["observation.state"]["min"],
            dtype=torch.float32,
            device=self.device,
        )
        self.state_maximum = torch.tensor(
            all_statistics["observation.state"]["max"],
            dtype=torch.float32,
            device=self.device,
        )

    def _normalise(self, actions_43: torch.Tensor) -> torch.Tensor:
        return self._min_max(actions_43, self.minimum, self.maximum)

    @staticmethod
    def _min_max(values, minimum, maximum):
        """Match training, including values outside the reference statistics."""
        span = maximum - minimum
        mask = span != 0
        normalised = torch.zeros_like(values)
        normalised[..., mask] = 2 * (values[..., mask] - minimum[mask]) / span[mask] - 1
        return normalised

    def encode_30hz_actions(
        self, actions_30hz: torch.Tensor, baseline_state: torch.Tensor
    ) -> torch.Tensor:
        """Encode raw a[t-2:t+23] and state s[t-2] for condition image I[t]."""
        squeeze = actions_30hz.ndim == 2
        if squeeze:
            actions_30hz = actions_30hz.unsqueeze(0)
        if actions_30hz.ndim != 3 or actions_30hz.shape[1] != 25:
            raise ValueError(
                "Expected 25 raw actions shaped [B,25,28] or [B,25,43], "
                f"got {tuple(actions_30hz.shape)}"
            )
        actions_30hz = actions_30hz.to(self.device).float()
        if actions_30hz.shape[-1] == 28:
            actions_43 = pad_g1_dex3_28_to_43(actions_30hz)
        elif actions_30hz.shape[-1] == G1_DIM:
            actions_43 = actions_30hz
        else:
            raise ValueError(
                f"Expected 28-D or 43-D actions, got {actions_30hz.shape[-1]}"
            )

        sampled = self._normalise(actions_43)[:, ::2]
        blocks = []
        for start in range(0, 12, 4):
            blocks.append(
                sampled[:, start + 1 : start + 5] - sampled[:, start : start + 1]
            )
        deltas = torch.cat(blocks, dim=1)
        encoded = deltas.new_zeros(
            deltas.shape[0], deltas.shape[1], DREAMDOJO_ACTION_DIM
        )
        encoded[..., DREAMDOJO_G1_SLICE] = deltas
        baseline_state = baseline_state.to(self.device, dtype=torch.float32)
        if baseline_state.ndim == 1:
            baseline_state = baseline_state.unsqueeze(0)
        if baseline_state.shape[-1] == 28:
            baseline_state = pad_g1_dex3_28_to_43(baseline_state)
        if baseline_state.shape != (encoded.shape[0], G1_DIM):
            raise ValueError("baseline_state must have shape [B,28] or [B,43]")
        normalized_state = self._min_max(
            baseline_state, self.state_minimum, self.state_maximum
        )
        # Training writes through its __key__ view into this action slice.
        encoded[:, 0, :29] = normalized_state[:, :29]
        return encoded.squeeze(0) if squeeze else encoded

    def encode_policy_chunk(
        self,
        previous_action_28: torch.Tensor,
        actions_28: torch.Tensor,
        baseline_state: torch.Tensor,
    ) -> torch.Tensor:
        """Keep commands at 30 Hz; hold only the unused future padding.

        Consume only the first 6 new 15-Hz frames from the generated video.
        """
        if actions_28.ndim != 3 or actions_28.shape[1:] != (12, 28):
            raise ValueError(
                f"Expected policy actions [B,12,28], got {tuple(actions_28.shape)}"
            )
        if previous_action_28.shape != (actions_28.shape[0], 28):
            raise ValueError(
                f"Expected previous actions [B,28], got {tuple(previous_action_28.shape)}"
            )
        # The t-1 slot is ignored by ::2; t-2 is the historical baseline.
        raw_window = torch.cat(
            [
                previous_action_28[:, None].repeat(1, 2, 1),
                actions_28,
                actions_28[:, -1:].repeat(1, 11, 1),
            ],
            dim=1,
        )
        return self.encode_30hz_actions(raw_window, baseline_state)
