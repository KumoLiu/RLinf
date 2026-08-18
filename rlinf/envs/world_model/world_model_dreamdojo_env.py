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

"""DreamDojo teacher world-model environment for G1 reinforcement learning."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.dreamdojo_action import (
    DREAMDOJO_CHUNK_SIZE,
    DREAMDOJO_POLICY_HORIZON,
    G1DreamDojoActionBridge,
)
from rlinf.envs.world_model.success_classifier_reward import (
    SuccessClassifierReward,
)

__all__ = ["DreamDojoEnv"]


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _DreamDojoResetDataset:
    """Expose episode-start RGB and physical G1 state from DreamDojo data."""

    _STATE_KEYS = (
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    )

    def __init__(self, dreamdojo_path: Path, dataset_path: Path):
        with _working_directory(dreamdojo_path):
            from groot_dreams.dataloader import MultiVideoActionDataset

            dataset = MultiVideoActionDataset(
                num_frames=DREAMDOJO_CHUNK_SIZE + 1,
                dataset_path=str(dataset_path),
                data_split="full",
                single_base_index=True,
                deterministic_uniform_sampling=False,
            )
        if len(dataset.datasets) != 1:
            raise ValueError("DreamDojoEnv currently requires exactly one dataset")
        self._dataset = dataset.datasets[0].lerobot_dataset
        self._episode_starts = [
            (trajectory_id, step_index)
            for trajectory_id, step_index in self._dataset.all_steps
            if step_index == 0
        ]
        if not self._episode_starts:
            raise ValueError("DreamDojo dataset contains no episode-start frames")

    def __len__(self) -> int:
        return len(self._episode_starts)

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self._episode_starts[index]
        raw = self._dataset.get_step_data(trajectory_id, base_index)
        image = np.asarray(raw["video.ego_view"][0])
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        state_components = [
            np.asarray(raw[key][0], dtype=np.float32).reshape(-1)
            for key in self._STATE_KEYS
        ]
        return {
            "image": image,
            "state": np.concatenate(state_components),
            "trajectory_id": int(trajectory_id),
        }


class DreamDojoEnv(BaseWorldEnv):
    """Use a post-trained DreamDojo teacher as a vectorized G1 environment."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
        worker_info=None,
    ):
        self.dreamdojo_path = Path(cfg.dreamdojo_path).expanduser().resolve()
        if str(self.dreamdojo_path) not in sys.path:
            sys.path.insert(0, str(self.dreamdojo_path))
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
            record_metrics,
        )

        self.chunk = DREAMDOJO_POLICY_HORIZON
        self.world_model_chunk = DREAMDOJO_CHUNK_SIZE
        if cfg.chunk != self.chunk:
            raise ValueError(
                f"DreamDojoEnv requires chunk={self.chunk}, got {cfg.chunk}"
            )
        self.group_size = cfg.group_size
        if num_envs % self.group_size:
            raise ValueError("num_envs must be divisible by group_size")
        self.num_groups = num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", True)
        self.task_description = cfg.task_description
        self.image_size = tuple(cfg.image_size)
        self._generator = torch.Generator().manual_seed(self.seed)
        self.update_reset_state_ids()

        self.action_bridge = G1DreamDojoActionBridge(cfg.action_statistics_path)
        self.pipeline = self._build_pipeline()
        reward_cfg = cfg.reward_model
        self.reward_model = (
            SuccessClassifierReward(
                reward_cfg.checkpoint_path,
                num_envs,
                hidden_dim=reward_cfg.get("hidden_dim", 256),
                dropout=reward_cfg.get("dropout", 0.1),
                temporal_window=reward_cfg.get("temporal_window", 8),
                success_threshold=reward_cfg.get("success_threshold", 0.8),
                majority_fraction=reward_cfg.get("majority_fraction", 0.75),
                min_stable_frames=reward_cfg.get("min_stable_frames", 4),
            )
            .eval()
            .to(self.device)
        )
        self.current_images = torch.zeros(
            num_envs, *self.image_size, 3, dtype=torch.uint8, device=self.device
        )
        self.current_states = torch.zeros(
            num_envs, 28, dtype=torch.float32, device=self.device
        )
        self.task_descriptions = [self.task_description] * num_envs
        self._is_offloaded = False

    def _build_dataset(self, cfg):
        dataset_path = Path(cfg.initial_dataset_path).expanduser().resolve()
        return _DreamDojoResetDataset(self.dreamdojo_path, dataset_path)

    def _build_pipeline(self):
        with _working_directory(self.dreamdojo_path):
            from omegaconf import OmegaConf

            # DreamDojo registers these common resolvers at import time without
            # ``replace=True``. RLinf has already registered equivalent ones.
            for resolver_name in ("add", "subtract"):
                if OmegaConf.has_resolver(resolver_name):
                    OmegaConf.clear_resolver(resolver_name)
            from cosmos_predict2._src.predict2.inference.video2world import (
                Video2WorldInference,
            )

            return Video2WorldInference(
                experiment_name=self.cfg.experiment_name,
                ckpt_path=str(Path(self.cfg.model_path).expanduser().resolve()),
                s3_credential_path="",
                context_parallel_size=self.cfg.get("context_parallel_size", 1),
                    config_file=self.cfg.config_file,
            )

    def update_reset_state_ids(self) -> None:
        episode_ids = torch.randint(
            len(self.dataset),
            (self.num_groups,),
            generator=self._generator,
        )
        self.reset_state_ids = episode_ids.repeat_interleave(self.group_size)

    def _wrap_obs(self) -> dict:
        return {
            "main_images": self.current_images,
            "wrist_images": None,
            "states": self.current_states,
            "task_descriptions": self.task_descriptions,
        }

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        del options
        self.onload()
        self.elapsed_steps = 0
        if seed is not None:
            seed_value = seed[0] if isinstance(seed, list) else seed
            self._generator.manual_seed(seed_value)
        if episode_indices is None:
            if self.is_start and self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            else:
                episode_indices = torch.randperm(
                    len(self.dataset), generator=self._generator
                )[: self.num_envs]
        self.is_start = False
        if isinstance(episode_indices, np.ndarray):
            episode_indices = torch.from_numpy(episode_indices)

        images = []
        states = []
        for episode_index in episode_indices.tolist():
            item = self.dataset[int(episode_index)]
            images.append(torch.from_numpy(item["image"].copy()))
            states.append(torch.from_numpy(item["state"].copy()))
        self.current_images = torch.stack(images).to(self.device)
        self.current_states = torch.stack(states).to(self.device)
        self.task_descriptions = [self.task_description] * self.num_envs
        self.reward_model.reset()
        self._reset_metrics()
        return self._wrap_obs(), {}

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        del actions, auto_reset
        raise NotImplementedError("DreamDojoEnv only supports chunk_step")

    def _generate_one(
        self,
        condition_image: torch.Tensor,
        action_condition: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        image = condition_image.permute(2, 0, 1).unsqueeze(0).cpu()
        input_video = torch.cat(
            [
                image,
                torch.zeros_like(image).repeat(self.world_model_chunk, 1, 1, 1),
            ],
            dim=0,
        )
        input_video = input_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
        with _working_directory(self.dreamdojo_path):
            generated = self.pipeline.generate_vid2world(
                prompt="",
                input_path=input_video,
                action=action_condition.cpu(),
                guidance=self.cfg.get("guidance", 0),
                num_video_frames=self.world_model_chunk + 1,
                num_latent_conditional_frames=1,
                resolution=f"{self.image_size[0]},{self.image_size[1]}",
                seed=seed,
                num_steps=self.cfg.get("num_inference_steps", 35),
                lam_video=None,
            )
        frames = (
            (torch.clamp((generated[0] + 1.0) / 2.0, 0, 1) * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
        )
        return frames[-self.world_model_chunk :]

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        actions = torch.as_tensor(
            policy_output_action, dtype=torch.float32, device=self.device
        )
        action_condition = self.action_bridge.encode_30hz_actions(actions)
        generated_chunks = [
            self._generate_one(
                self.current_images[env_index],
                action_condition[env_index],
                self.seed + self.elapsed_steps + env_index,
            )
            for env_index in range(self.num_envs)
        ]
        frames = torch.stack(generated_chunks).to(self.device)
        self.current_images = frames[:, -1]
        self.current_states = actions[:, -1]
        self.elapsed_steps += self.chunk

        frame_rewards, success = self.reward_model.evaluate_chunk(frames)
        rewards = torch.zeros(
            self.num_envs, self.chunk, device=self.device, dtype=torch.float32
        )
        rewards[:, -1] = frame_rewards[:, -1]
        terminations = torch.zeros_like(rewards, dtype=torch.bool)
        terminations[:, -1] = success
        truncations = torch.zeros_like(terminations)
        truncations[:, -1] = self.elapsed_steps >= self.cfg.max_episode_steps
        dones = terminations[:, -1] | truncations[:, -1]

        observation = self._wrap_obs()
        infos = self._record_metrics(rewards.sum(dim=1), terminations[:, -1], {})
        if dones.any() and self.auto_reset:
            final_observation = observation
            final_info = infos
            observation, infos = self.reset()
            infos.update(
                {
                    "final_observation": final_observation,
                    "final_info": final_info,
                    "_final_observation": dones,
                    "_final_info": dones,
                }
            )
        return [observation], rewards, terminations, truncations, [infos]

    def offload(self) -> None:
        """Offload the reward model; teacher manages its own staged offloading."""
        if self._is_offloaded:
            return
        self.reward_model.to("cpu")
        self.current_images = self.current_images.cpu()
        self.current_states = self.current_states.cpu()
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self) -> None:
        """Restore the reward model to the environment execution device."""
        if not self._is_offloaded:
            return
        self.reward_model.to(self.device)
        self.current_images = self.current_images.to(self.device)
        self.current_states = self.current_states.to(self.device)
        self._is_offloaded = False

    def close(self) -> None:
        """Release DreamDojo inference resources."""
        self.pipeline.cleanup()
