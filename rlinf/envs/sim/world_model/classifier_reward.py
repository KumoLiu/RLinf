# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batched ordered-class image rewards for medical pads."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn


class ClassifierReward(nn.Module):
    """Score each frame independently; progress can decrease after a refold."""

    def __init__(self, checkpoint, num_envs=0, device="cpu", config_file=None):
        super().__init__()
        from torchvision.models import resnet18

        checkpoint = Path(checkpoint).expanduser()
        if checkpoint.suffix == ".json":
            checkpoint = Path(json.loads(checkpoint.read_text())["checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = resnet18(weights=None)
        self.legacy_pil = "architecture" in state
        if self.legacy_pil:
            if state["architecture"] != "resnet18" or state["classes"] != list(
                range(5)
            ):
                raise ValueError(
                    "Expected the legacy ResNet18 medical-pads 0–4 checkpoint"
                )
            self.num_classes = 5
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            weights = state["model"]
            self.image_size = (int(state["image_size"]),) * 2
            mean, std = state["normalization_mean"], state["normalization_std"]
        else:
            config_path = (
                Path(config_file)
                if config_file
                else checkpoint.with_name("inference_config.yaml")
            )
            config = yaml.safe_load(config_path.read_text())
            if (config["arch"], config["precision"]) != ("resnet18", "fp32"):
                raise ValueError("Expected an FP32 ResNet18 inference configuration")
            if config["num_classes"] not in (4, 5):
                raise ValueError("Expected four or five classifier classes")
            self.num_classes = config["num_classes"]
            self.pre_resize = config.get(
                "pre_resize", (240, 320) if self.num_classes == 5 else None
            )
            self.channels_last = bool(config.get("channels_last", False))
            self.image_size = tuple(config["image_size"][1:])
            if len(self.image_size) != 2 or config["image_size"][0] != 3:
                raise ValueError("Classifier image_size must be [3, height, width]")
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, config["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["hidden_dim"], self.num_classes),
            )
            weights = {k.removeprefix("backbone."): v for k, v in state.items()}
            mean = [0.485, 0.456, 0.406] if config["normalize"] else [0.0] * 3
            std = [0.229, 0.224, 0.225] if config["normalize"] else [1.0] * 3
        values = (
            list(range(self.num_classes))
            if self.legacy_pil
            else config.get("reward_values", list(range(self.num_classes)))
        )
        if (
            not self.legacy_pil
            and self.num_classes == 5
            and values not in ([-1, 0, 1, 2, 3], [-2, -1, 0, 1, 2])
        ):
            raise ValueError(
                "Expected five-class reward_values [-1,0,1,2,3] or [-2,-1,0,1,2]"
            )
        self.register_buffer("reward_values", torch.tensor(values, dtype=torch.float32))
        self.model.load_state_dict(weights, strict=True)
        self.num_envs = num_envs
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("scores", torch.zeros(num_envs))
        self.register_buffer("labels", torch.zeros(num_envs, dtype=torch.long))
        self.reset()
        self.eval().to(device)

    @property
    def device(self):
        return self.mean.device

    def reset(self, num_envs=None):
        if num_envs is not None:
            self.num_envs = int(num_envs)
        self.scores = torch.zeros(self.num_envs, device=self.device)
        self.labels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def score_chunk(self, frames):
        """Return expected scores [B,T] and softmax probabilities [B,T,C].

        Frames are RGB [B,T,3,H,W], uint8 or floats in [-1,1]. Preprocessing
        uses tensor-bilinear FP32 inference for exported stage classifiers, and
        PIL-bilinear preprocessing for the legacy five-class checkpoint.
        """
        if frames.ndim != 5 or frames.shape[0] != self.num_envs or frames.shape[2] != 3:
            raise ValueError("Expected RGB frames [num_envs,T,3,H,W]")
        if frames.dtype != torch.uint8:
            frames = ((frames.float() + 1) * 127.5).round().clamp(0, 255).byte()
        if self.legacy_pil:
            arrays = frames.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            resized = np.stack(
                [
                    np.asarray(
                        Image.fromarray(frame).resize(
                            self.image_size[::-1], Image.Resampling.BILINEAR
                        )
                    )
                    for frame in arrays
                ]
            )
            inputs = (
                torch.from_numpy(resized).permute(0, 3, 1, 2).to(self.device).float()
                / 255
            )
        else:
            inputs = frames.flatten(0, 1).contiguous().to(self.device).float() / 255
            if self.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            if self.pre_resize and inputs.shape[-2:] != self.pre_resize:
                inputs = F.interpolate(
                    inputs, size=self.pre_resize, mode="bilinear", align_corners=False
                )
            if inputs.shape[-2:] != self.image_size:
                inputs = F.interpolate(
                    inputs, size=self.image_size, mode="bilinear", align_corners=False
                )
        inputs = (inputs - self.mean) / self.std
        probabilities = []
        with torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.legacy_pil and self.device.type == "cuda",
        ):
            for batch in inputs.split(64):
                probabilities.append(self.model(batch).float().softmax(-1))
        probabilities = torch.cat(probabilities).reshape(
            *frames.shape[:2], self.num_classes
        )
        values = self.reward_values
        scores = (
            probabilities @ values
            if self.legacy_pil
            else (probabilities * values).sum(-1)
        )
        self.scores = scores[:, -1]
        self.labels = probabilities[:, -1].argmax(-1)
        return scores, probabilities
