# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Action-conditioned H2 RGB generation through Cosmos forward dynamics."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Sequence

import torch
import yaml

from rlinf.envs.sim.world_model.registry import register_backend

from . import FrameQueue


@register_backend("cosmos3")
class Cosmos3Backend:
    """Keep an RGB condition, absolute-action anchor and noise seed per slot."""

    supports_kir = False

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.image_size = tuple(cfg.image_size)
        if (self.chunk, self.condition_frame_length, cfg.action_dim) != (16, 1, 58):
            raise ValueError(
                "Cosmos H2 requires chunk=16, condition_frame_length=1, action_dim=58"
            )
        if cfg.num_frames != 17 or int(cfg.get("wm_batch_size", 1)) < 1:
            raise ValueError("Cosmos H2 requires num_frames=17 and wm_batch_size >= 1")
        self._sessions = {}
        self.pipe = self._build_pipeline()

    def _build_pipeline(self):
        root = str(Path(self.cfg.cosmos3_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.utils.h2_action_codec import (
            RELATIVE_REPRESENTATION,
            H2ActionCodec,
        )

        training = yaml.safe_load(Path(self.cfg.config_file).read_text())
        data = training["dataloader_train"]["dataloader"]["datasets"]["h2_sharpa"][
            "dataset"
        ]
        if (
            data["chunk_length"] != self.chunk
            or data["action_representation"] != RELATIVE_REPRESENTATION
            or data.get("action_channels", "all") != "all"
            or data.get("use_state", False)
            or data["camera_key"] != self.cfg.video_key
        ):
            raise ValueError(
                "Expected the H2 16-step, 58-joint, left-head relative-action checkpoint"
            )
        self.action_codec = H2ActionCodec(
            self.cfg.action_statistics, data["action_stats_sha256"]
        )
        setup = OmniSetupOverrides(
            checkpoint_path=self.cfg.model_path,
            config_file=self.cfg.config_file,
            output_dir=Path(self.cfg.pipeline_log_dir)
            / f"env_{socket.gethostname()}_{os.getpid()}",
            use_torch_compile=False,
            diffusion_cache=False,
            guardrails=False,
            dp_shard_size=1,
            dp_replicate_size=1,
            cp_size=1,
            cfgp_size=1,
        ).build_setup()
        return OmniInference.create(setup)

    @staticmethod
    def load_reward_model(cfg):
        from rlinf.envs.sim.world_model.classifier_reward import ClassifierReward

        return ClassifierReward(
            cfg.reward_model.from_pretrained,
            config_file=cfg.reward_model.get("config_file"),
        )

    def reward_instructions(self, env):
        return None

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
        task_descriptions: Sequence[str] | None = None,
    ) -> None:
        if not (len(env_ids) == len(init_frames) == len(init_actions) == len(seeds)):
            raise ValueError(
                "Session slots, conditions, action anchors and seeds must have equal lengths"
            )
        tasks = (
            list(task_descriptions)
            if task_descriptions is not None
            else [""] * len(env_ids)
        )
        if len(tasks) != len(env_ids):
            raise ValueError("Each Cosmos session requires one task description")
        for row, (env_id, seed) in enumerate(zip(env_ids, seeds, strict=True)):
            self._sessions[int(env_id)] = {
                "frame": init_frames[row][-1].detach().clone().to(self.device),
                "anchor": init_actions[row, -1].detach().clone().to(self.device),
                "seed": int(seed),
                "steps": 0,
                "prompt": tasks[row],
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    @torch.no_grad()
    def generate(self, env_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.inference.inference import (
            _finalize_data_batch,
            _merge_data_batches,
        )

        env_ids = [int(i) for i in env_ids]
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        expected = (len(env_ids), self.chunk, 58)
        if tuple(actions.shape) != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        missing = [i for i in env_ids if i not in self._sessions]
        if missing:
            raise RuntimeError(f"Cosmos slots require open sessions: {missing}")
        batches = []
        for row, env_id in enumerate(env_ids):
            session = self._sessions[env_id]
            encoded = self.action_codec.encode(
                actions[row].cpu().numpy(), session["anchor"].cpu().numpy()
            )
            condition = ((session["frame"] + 1) * 127.5).round().clamp(0, 255).byte()
            batches.append(
                build_action_batch(
                    video=condition,
                    action=torch.from_numpy(encoded).to(self.device),
                    raw_action_dim=58,
                    prompt=session["prompt"],
                    view_point="third_person_view",
                    domain_name="h2_sharpa",
                    model_mode=ModelMode.FORWARD_DYNAMICS,
                    action_chunk_size=self.chunk,
                    fps=30,
                    resolution="480",
                    input_video_key=self.pipe.model.input_video_key,
                    device=self.device,
                )
            )
        decoded = []
        batch_size = int(self.cfg.get("wm_batch_size", 1))
        for start in range(0, len(env_ids), batch_size):
            selected = batches[start : start + batch_size]
            batch = _finalize_data_batch(
                _merge_data_batches(selected), len(selected), self.pipe.model
            )
            seeds = [
                self._sessions[i]["seed"] + self._sessions[i]["steps"]
                for i in env_ids[start : start + batch_size]
            ]
            outputs = self.pipe.model.generate_samples_from_batch(
                batch,
                guidance=float(self.cfg.get("guidance", 1.0)),
                seed=seeds,
                num_steps=int(self.cfg.num_inference_steps),
                shift=float(self.cfg.get("shift", 10.0)),
            )
            decoded.extend(
                self.pipe.model.decode(vision) for vision in outputs["vision"]
            )
        height, width = self.image_size
        frames = torch.cat(decoded)[:, :, 1 : self.chunk + 1, :height, :width].clamp(
            -1, 1
        )
        expected = (len(env_ids), 3, self.chunk, height, width)
        if tuple(frames.shape) != expected:
            raise ValueError(
                f"Expected generated RGB {expected}, got {tuple(frames.shape)}"
            )
        for row, env_id in enumerate(env_ids):
            session = self._sessions[env_id]
            session["frame"] = frames[row, :, -1:].detach().clone()
            session["anchor"] = actions[row, -1].detach().clone()
            session["steps"] += self.chunk
        return frames

    def _move(self, device):
        self.pipe.model.to(device)
        tokenizer = self.pipe.model.tokenizer_vision_gen
        tokenizer.model.model.to(device)
        tokenizer.model.clear_encoder_cache()
        for session in self._sessions.values():
            for key in ("frame", "anchor"):
                session[key] = session[key].to(device)

    def offload(self) -> None:
        self._move("cpu")

    def onload(self) -> None:
        self._move(self.device)
