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

"""Batched deployment wrapper for DreamDojo's v2 milestone reward."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

FRAME_OFFSETS = (0, 4, 8, 16)
CONFIRM_WINDOW = (15, 15, 2)
CONFIRM_COUNT = (13, 13, 2)
THRESHOLDS = (0.8, 0.8, 0.8)
NUM_HEADS = 3
STORE_SIZE = (270, 360)
CROP_SIZE = (240, 320)


class _MilestoneNet(nn.Module):
    """Checkpoint-compatible copy of DreamDojo's small reward network."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            old.in_channels * len(FRAME_OFFSETS),
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        backbone.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(backbone.fc.in_features, NUM_HEADS)
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BatchedMilestoneReward:
    """Shared model with independent temporal ratchets for each environment."""

    def __init__(
        self,
        checkpoint: str | Path,
        num_envs: int,
        device: torch.device | str,
        *,
        duplicate_for_30fps: bool = True,
    ) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = _MilestoneNet()
        self.model.load_state_dict(state["model"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.num_envs = int(num_envs)
        self.duplicate_for_30fps = duplicate_for_30fps
        self.mean = (
            torch.tensor([0.485, 0.456, 0.406], device=self.device)
            .repeat(len(FRAME_OFFSETS))
            .view(1, -1, 1, 1)
        )
        self.std = (
            torch.tensor([0.229, 0.224, 0.225], device=self.device)
            .repeat(len(FRAME_OFFSETS))
            .view(1, -1, 1, 1)
        )
        self.reset()

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if not hasattr(self, "frame_history"):
            self.frame_history = [
                deque(maxlen=max(FRAME_OFFSETS) + 1) for _ in range(self.num_envs)
            ]
            self.recent = [
                [deque(maxlen=w) for w in CONFIRM_WINDOW] for _ in range(self.num_envs)
            ]
            self.stage = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        indices = (
            range(self.num_envs)
            if env_mask is None
            else env_mask.nonzero(as_tuple=False).flatten().tolist()
        )
        for index in indices:
            self.frame_history[index].clear()
            for votes in self.recent[index]:
                votes.clear()
            self.stage[index] = 0

    @torch.no_grad()
    def prime_handover(self, frames: torch.Tensor, env_indices: list[int]) -> None:
        """Seed raw 30-Hz visual history and known picked state, without payout.

        Only real frames at/before reset may be passed. Future-head confirmation
        votes deliberately start empty: no pre-reset transition earns credit.
        """
        if frames.ndim != 5 or frames.shape[:3] != (len(env_indices), 17, 3):
            raise ValueError("Expected 17 raw history frames per selected environment")
        if len(set(env_indices)) != len(env_indices) or any(
            index < 0 or index >= self.num_envs for index in env_indices
        ):
            raise ValueError("Invalid reward warmup environment indices")
        for index in env_indices:
            self.frame_history[index].clear()
            for votes in self.recent[index]:
                votes.clear()
        for time_index in range(frames.shape[1]):
            prepared = self._prepare(frames[:, time_index])
            for local_index, env_index in enumerate(env_indices):
                self.frame_history[env_index].append(prepared[local_index])
        self.stage[env_indices] = 1

    def _prepare(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            frames_uint8 = frames
        elif frames.min() < 0:
            frames_uint8 = ((frames.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        else:
            frames_uint8 = (frames.float() * 255).clamp(0, 255).to(torch.uint8)
        arrays = frames_uint8.permute(0, 2, 3, 1).cpu().numpy()
        y0 = (STORE_SIZE[0] - CROP_SIZE[0]) // 2
        x0 = (STORE_SIZE[1] - CROP_SIZE[1]) // 2
        crops = [
            cv2.resize(
                frame, (STORE_SIZE[1], STORE_SIZE[0]), interpolation=cv2.INTER_AREA
            )[y0 : y0 + CROP_SIZE[0], x0 : x0 + CROP_SIZE[1]]
            for frame in arrays
        ]
        return (
            torch.from_numpy(np.stack(crops))
            .permute(0, 3, 1, 2)
            .to(self.device, dtype=torch.float32)
            .div(255)
        )

    def _predict(self, frame_rgb: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare(frame_rgb.to(self.device))
        stacks = []
        for env_index in range(self.num_envs):
            history = self.frame_history[env_index]
            history.append(prepared[env_index])
            newest_first = list(history)[::-1]
            stacks.append(
                torch.cat(
                    [
                        newest_first[min(offset, len(newest_first) - 1)]
                        for offset in FRAME_OFFSETS
                    ],
                    dim=0,
                )
            )
        batch = (torch.stack(stacks) - self.mean) / self.std
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(batch)
        return torch.sigmoid(logits.float())

    def advance(self, probs: torch.Tensor) -> torch.Tensor:
        """Advance each ratchet from a `[B,3]` probability tensor."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        for env_index in range(self.num_envs):
            before = int(self.stage[env_index])
            for head in range(NUM_HEADS):
                self.recent[env_index][head].append(
                    bool(probs[env_index, head] >= THRESHOLDS[head])
                )
            stage = before
            while (
                stage < NUM_HEADS
                and sum(self.recent[env_index][stage]) >= CONFIRM_COUNT[stage]
            ):
                stage += 1
            self.stage[env_index] = stage
            rewards[env_index] = stage - before
        return rewards

    @torch.no_grad()
    def score_chunk(
        self, frames_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score `[B,T,3,H,W]` 15-fps frames and return rewards/probabilities."""
        if frames_rgb.ndim != 5 or frames_rgb.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected [B,T,3,H,W] for B={self.num_envs}, "
                f"got {tuple(frames_rgb.shape)}"
            )
        chunk_rewards = torch.zeros(
            self.num_envs, frames_rgb.shape[1], device=self.device
        )
        chunk_probs = torch.zeros(
            self.num_envs, frames_rgb.shape[1], NUM_HEADS, device=self.device
        )
        repeats = 2 if self.duplicate_for_30fps else 1
        for time_index in range(frames_rgb.shape[1]):
            for _ in range(repeats):
                probs = self._predict(frames_rgb[:, time_index])
                chunk_rewards[:, time_index] += self.advance(probs)
            chunk_probs[:, time_index] = probs
        return chunk_rewards, chunk_probs

    def to(self, device: torch.device | str) -> "BatchedMilestoneReward":
        device = torch.device(device)
        self.device = device
        self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.stage = self.stage.to(device)
        self.frame_history = [
            deque((frame.to(device) for frame in history), maxlen=history.maxlen)
            for history in self.frame_history
        ]
        return self
