# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
IsaacLabEnv is a wrapper for Isaac Lab environments to be used with the RLinf framework.

This class adapts the Isaac Lab environment to the interface expected by RLinf's
workers and runners. It handles:
- Initialization and configuration of the underlying Isaac Lab simulation.
- Conversion of raw Isaac Lab observations into a standardized format for VLA models.
- Implementation of the core gym.Env interface (`step`, `reset`).
- Action chunking (`chunk_step`) for efficient rollout collection.
- Metrics tracking for logging episode statistics (return, success, etc.).
- Automatic reset of finished sub-environments in a vectorized setup.
"""

import gymnasium as gym
import torch
from omegaconf import OmegaConf

# --- Isaac Lab specific imports (placeholders) ---
# You will need to replace these with the actual Isaac Lab imports.
# For example, from omni.isaac.lab.app import AppLauncher
# For example, from omni.isaac.lab.envs import ManagerBasedRLEnv
# For example, from omni.isaac.lab_tasks.utils import load_cfg_from_registry
# --- End Isaac Lab specific imports ---

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

__all__ = ["IsaacLabEnv"]


class IsaacLabEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes):
        """
        Initializes the IsaacLabEnv wrapper.

        Args:
            cfg: The Hydra configuration object for the environment.
            seed_offset: An offset to add to the base seed, for multi-process training.
            total_num_processes: The total number of parallel processes.
        """
        self.cfg = cfg
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self._is_start = True

        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.record_metrics = cfg.get("record_metrics", True)

        # --- 1. Your Isaac Lab Initialization Code Here ---
        # This is where you will instantiate and configure your Isaac Lab environment.
        # The exact code will depend on your specific Isaac Lab task.
        #
        # Example using AppLauncher and a registered task:
        #
        # from omni.isaac.lab.app import AppLauncher
        # self.lab_app = AppLauncher(headless=True).launch()
        #
        # from omni.isaac.lab_tasks.utils import load_cfg_from_registry
        # env_cfg_path = f"isaaclab.envs.{self.cfg.task_name}" # e.g., "isaaclab.envs.ANYmal-v0"
        # env_cfg = load_cfg_from_registry(env_cfg_path)
        #
        # # Set number of environments and other params from your main config
        # env_cfg.scene.num_envs = self.cfg.num_envs
        #
        # from omni.isaac.lab.envs import ManagerBasedRLEnv
        # self.env = ManagerBasedRLEnv(cfg=env_cfg)
        #
        # print("Isaac Lab Environment Initialized.")
        # --- End Initialization Code ---

        if self.record_metrics:
            self._init_metrics()

    def _wrap_obs(self, raw_obs: dict) -> dict:
        """
        Converts raw Isaac Lab observations to the standardized format for VLA models.

        This is the most critical method to implement. You need to map the sensor
        data from `raw_obs` to the keys expected by the policy network.

        Args:
            raw_obs: The raw observation dictionary from the underlying Isaac Lab env.

        Returns:
            A dictionary containing processed observations, including 'images',
            'wrist_images', and 'task_descriptions'.
        """
        # --- 2. Your Observation Conversion Logic Here ---
        # The keys inside `raw_obs` will depend on your Isaac Lab task configuration.
        # You need to inspect `raw_obs` and find the keys for your camera sensors.
        #
        # Example assuming the raw_obs has a 'policy' key for observations:
        #
        # head_cam_obs = raw_obs["policy"]["camera_head_rgb"] # Shape: [num_envs, H, W, C]
        # wrist_cam_obs = raw_obs["policy"]["camera_wrist_rgb"] # Shape: [num_envs, H, W, C]
        #
        # # Convert from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        # # and normalize if needed (e.g., divide by 255.0).
        # head_images = head_cam_obs.permute(0, 3, 1, 2).float() / 255.0
        # wrist_images = wrist_cam_obs.permute(0, 3, 1, 2).float() / 255.0
        #
        # wrapped_obs = {
        #     "images": head_images,  # [num_envs, C, H, W]
        #     "wrist_images": wrist_images, # [num_envs, C, H, W]
        #     "task_descriptions": [self.cfg.task_description] * self.num_envs,
        # }
        #
        # if self.cfg.get("use_proprio", False):
        #     # Example for proprioceptive state
        #     proprio_state = raw_obs["policy"]["proprio_state"]
        #     wrapped_obs["proprio"] = proprio_state
        #
        # return wrapped_obs
        # --- End Conversion Logic ---
        raise NotImplementedError("Please implement the `_wrap_obs` method.")

    def reset(self):
        """Resets the environment and returns the initial observation."""
        raw_obs, _ = self.env.reset()
        obs = self._wrap_obs(raw_obs)
        infos = {}
        if self.record_metrics:
            self._reset_metrics()
        return obs, infos

    def step(self, actions: torch.Tensor):
        """Executes a step in the environment."""
        if self._is_start:
            self._is_start = False
            obs, infos = self.reset()
            rewards = torch.zeros(self.num_envs, device=self.device)
            terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, rewards, terminations, truncations, infos

        raw_next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs = self._wrap_obs(raw_next_obs)

        if self.record_metrics:
            infos = self._record_metrics(rewards, terminations, truncations, infos)

        if self.ignore_terminations:
            terminations[:] = False

        dones = torch.logical_or(terminations, truncations)
        if dones.any() and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, rewards, terminations, truncations, infos

    def chunk_step(self, chunk_actions: torch.Tensor):
        """Executes a chunk of actions."""
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            # Use auto_reset=False within the chunk
            obs, step_reward, terminations, truncations, infos = self.step(actions)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        # The 'done' for the chunk is determined by the last step's termination/truncation
        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    def _handle_auto_reset(self, dones: torch.Tensor, obs: dict, infos: dict):
        """Resets environments that are done and updates observations."""
        # This is a simplified auto-reset. Isaac Lab's `ManagerBasedRLEnv` might
        # handle this automatically. If so, you may not need this method and can
        # remove the call from `step`.
        # If Isaac Lab does not auto-reset, you would call `self.env.reset(dones)`
        # and merge the new observations into the `obs` dictionary.
        return obs, infos

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    # --- Metrics and Logging (Boilerplate) ---

    def _init_metrics(self):
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True

        self.returns[mask] = 0.0
        self.success_once[mask] = False
        self.episode_lengths[mask] = 0

    def _record_metrics(self, rewards, terminations, truncations, infos):
        self.returns += rewards
        self.episode_lengths += 1
        self.success_once |= terminations  # Assuming termination means success

        dones = torch.logical_or(terminations, truncations)
        if dones.any():
            if "episode" not in infos:
                infos["episode"] = {}
            
            done_indices = dones.nonzero(as_tuple=True)[0]

            infos["episode"]["return"] = self.returns[done_indices].clone()
            infos["episode"]["len"] = self.episode_lengths[done_indices].clone()
            infos["episode"]["success_once"] = self.success_once[done_indices].clone()

            # Reset metrics for envs that are done
            self._reset_metrics(done_indices)

        return infos
