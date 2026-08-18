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

"""Temporally aggregated ResNet success reward for world-model environments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SuccessClassifierReward(nn.Module):
    """Load a frame classifier and produce conservative episode rewards."""

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_envs: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: int = 224,
        temporal_window: int = 8,
        success_threshold: float = 0.8,
        majority_fraction: float = 0.75,
        min_stable_frames: int = 4,
    ):
        super().__init__()
        if temporal_window <= 0:
            raise ValueError("temporal_window must be positive")
        if not 0 <= success_threshold <= 1:
            raise ValueError("success_threshold must be in [0, 1]")
        if not 0 <= majority_fraction <= 1:
            raise ValueError("majority_fraction must be in [0, 1]")

        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.backbone = backbone
        self.image_size = image_size
        self.temporal_window = temporal_window
        self.success_threshold = success_threshold
        self.majority_fraction = majority_fraction
        self.min_stable_frames = min(min_stable_frames, temporal_window)

        self.register_buffer(
            "_mean",
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_probability_history",
            torch.zeros(num_envs, temporal_window),
            persistent=False,
        )
        self.register_buffer(
            "_history_lengths",
            torch.zeros(num_envs, dtype=torch.long),
            persistent=False,
        )
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        remapped = {}
        for key, value in state.items():
            if key.startswith("head."):
                key = key.replace("head.", "backbone.fc.", 1)
            remapped[key] = value
        self.load_state_dict(remapped, strict=True)

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        """Clear temporal reward state for all or selected environments."""
        if env_indices is None:
            self._probability_history.zero_()
            self._history_lengths.zero_()
            return
        self._probability_history[env_indices] = 0
        self._history_lengths[env_indices] = 0

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected four-dimensional images, got {images.shape}")
        if images.shape[-1] in (1, 3, 4) and images.shape[1] not in (1, 3, 4):
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.is_floating_point():
            minimum = float(images.min().detach())
            maximum = float(images.max().detach())
            if minimum < 0 or maximum > 1:
                raise ValueError(
                    "Floating-point classifier images must be in [0, 1], got "
                    f"[{minimum}, {maximum}]"
                )
            images = images.float()
        else:
            raise TypeError(f"Unsupported classifier image dtype: {images.dtype}")
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (images - self._mean) / self._std

    @torch.no_grad()
    def predict_probabilities(self, frames: torch.Tensor) -> torch.Tensor:
        """Predict per-frame success probabilities.

        Args:
            frames: RGB frames shaped ``[B, T, H, W, C]`` in uint8 [0, 255]
                or floating point [0, 1].
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected frames shaped [B,T,H,W,C], got {frames.shape}")
        batch_size, chunk_size = frames.shape[:2]
        flat_frames = frames.reshape(batch_size * chunk_size, *frames.shape[2:])
        model_parameter = next(self.parameters())
        images = self._preprocess(flat_frames).to(
            device=model_parameter.device,
            dtype=model_parameter.dtype,
        )
        logits = self.backbone(images).squeeze(-1)
        return torch.sigmoid(logits).reshape(batch_size, chunk_size)

    @torch.no_grad()
    def evaluate_chunk(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse chunk rewards and stable success decisions."""
        probabilities = self.predict_probabilities(frames)
        batch_size, chunk_size = probabilities.shape
        if batch_size != self._probability_history.shape[0]:
            raise ValueError(
                f"Expected {self._probability_history.shape[0]} envs, got {batch_size}"
            )

        stable_success = torch.zeros(
            batch_size, dtype=torch.bool, device=probabilities.device
        )
        for step in range(chunk_size):
            self._probability_history = torch.roll(
                self._probability_history, shifts=-1, dims=1
            )
            self._probability_history[:, -1] = probabilities[:, step]
            self._history_lengths.add_(1).clamp_(max=self.temporal_window)

            positions = torch.arange(
                self.temporal_window, device=probabilities.device
            ).unsqueeze(0)
            valid = positions >= (
                self.temporal_window - self._history_lengths.unsqueeze(1)
            )
            count = valid.sum(dim=1).clamp_min(1)
            mean_probability = (
                self._probability_history.masked_fill(~valid, 0).sum(dim=1) / count
            )
            majority = (
                (self._probability_history >= self.success_threshold) & valid
            ).sum(dim=1) / count
            stable_success |= (
                (self._history_lengths >= self.min_stable_frames)
                & (mean_probability >= self.success_threshold)
                & (majority >= self.majority_fraction)
            )

        rewards = probabilities.new_zeros(probabilities.shape)
        rewards[:, -1] = stable_success.to(rewards.dtype)
        return rewards, stable_success
