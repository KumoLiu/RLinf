# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
import cv2
import numpy as np
from av.container import Container
from av.stream import Stream
import av
import os
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

try:
    # isaaclab-related imports
    from isaaclab.app import AppLauncher
    # from isaaclab.envs import ManagerBasedRLEnv
except ImportError as e:
    raise ImportError(
        f"Failed to import Isaac Lab modules. Please ensure 'isaaclab' and "
        f"'isaaclab_tasks' are installed and in your PYTHONPATH. Details: {e}"
    )


__all__ = ["IsaacLabEnv"]


class IsaacLabEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes):
        """
        Initializes the IsaacLabEnv wrapper.

        The `cfg` object is expected to have an `isaaclab` sub-configuration with
        the following keys:
        - `task_name` (str): The name of the Isaac Lab task to load.
        - `num_envs` (int): The number of parallel environments to run.
        - `headless` (bool): Whether to run Isaac Sim in headless mode.
        - `task_description` (str): The natural language instruction for the task.
        - `use_proprio` (bool, optional): Whether to include proprioceptive state.

        Args:
            cfg: The Hydra configuration object for the environment.
            seed_offset: An offset to add to the base seed for multi-process training.
            total_num_processes: The total number of parallel processes.
        """
        self.cfg = cfg
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = cfg.get("record_metrics", True)
        self._is_start = True

        self.auto_reset = cfg.auto_reset

        # --- Isaac Lab Initialization ---
        print("Launching Isaac Lab App...")
        # self.lab_app = AppLauncher(headless=self.cfg.isaaclab.headless).launch()
        app_launcher = AppLauncher(headless=self.cfg.isaaclab.headless, enable_cameras=True)
        self.simulation_app = app_launcher.app
        print("Isaac Lab App Launched.")

        # Parse the environment configuration file
        print(f"Parsing Isaac Lab environment config for task: {self.cfg.isaaclab.task_name}")
        # The 'import tasks' is necessary for gym.make to find and register the custom env.
        # We import it locally from its new location.
        from rlinf.envs.isaaclab import tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        env_cfg = parse_env_cfg(
            self.cfg.isaaclab.task_name,
            device=self.cfg.isaaclab.device,
            num_envs=self.cfg.isaaclab.num_envs,
        )

        env_cfg.seed = self.seed_offset
        env_cfg.env_name = self.cfg.isaaclab.task_name
        self.env = gym.make(self.cfg.isaaclab.task_name, cfg=env_cfg).unwrapped
        self.env.seed(self.seed_offset)

        # Create the Isaac Lab environment
        # self.env = ManagerBasedRLEnv(cfg=env_cfg)
        print("Isaac Lab Environment Initialized.")
        # --- End Initialization ---

        if self.record_metrics:
            self._init_metrics()

        # --- Video Recording Initialization ---
        self._video_writer = None
        if self.cfg.isaaclab.video_cfg.save_video:
            os.makedirs(str(self.cfg.isaaclab.video_cfg.video_base_dir), exist_ok=True)
            video_name = str(self.cfg.isaaclab.video_cfg.video_base_dir) + "/isaaclab_video.mp4"
            # Define video resolution and format
            resolution = (480, 640 + 240) # Combined frame size
            container = av.open(video_name, mode="w")
            stream = container.add_stream("libx264", rate=30)
            stream.width = resolution[1]
            stream.height = resolution[0]
            stream.pix_fmt = "yuv420p"
            self._video_writer = (container, stream)
        # --- End Video Initialization ---

    def _wrap_obs(self, raw_obs: dict) -> dict:
        """
        Converts raw Isaac Lab observations to the standardized format for VLA models.

        This implementation is based on the provided configuration files:
        - It expects observations to be nested under the "policy" key.
        - It extracts RGB images from "front_camera", "left_wrist_camera", and
          "right_wrist_camera".
        - It combines joint and gripper states for the proprioceptive input.

        Args:
            raw_obs: The raw observation dictionary from the underlying Isaac Lab env.

        Returns:
            A dictionary containing processed observations for the RLinf framework.
        """
        # --- Observation Conversion Logic ---
        policy_obs = raw_obs["policy"]
        cam_data = raw_obs["camera_images"]

        # Extract images. Isaac Lab returns [N, H, W, C] tensors.
        # Permute to [N, C, H, W] for the VLA model.
        # The images are typically float tensors in [0, 1], so no normalization needed.
        front_image_raw = cam_data["front_camera"]
        if front_image_raw.ndim == 3:
            front_image_raw = front_image_raw.unsqueeze(0)  # Add batch dimension
        front_image = front_image_raw.permute(0, 3, 1, 2)

        left_wrist_image_raw = cam_data["left_wrist_camera"]
        if left_wrist_image_raw.ndim == 3:
            left_wrist_image_raw = left_wrist_image_raw.unsqueeze(0)
        left_wrist_image = left_wrist_image_raw.permute(0, 3, 1, 2)

        right_wrist_image_raw = cam_data["right_wrist_camera"]
        if right_wrist_image_raw.ndim == 3:
            right_wrist_image_raw = right_wrist_image_raw.unsqueeze(0)
        right_wrist_image = right_wrist_image_raw.permute(0, 3, 1, 2)

        # Construct the observation dict for RLinf.
        # 'images' is typically the head/third-person camera.
        # 'wrist_images' is for the wrist camera.
        wrapped_obs = {
            "images": front_image, # [N_ENV, C, H, W]
            "wrist_images": torch.stack([left_wrist_image, right_wrist_image], dim=1), # [N_ENV, N_IMG, C, H, W]
            "task_descriptions": [self.cfg.isaaclab.task_description] * self.num_envs,
        }

        if self.cfg.isaaclab.get("use_proprio", False):
            # Extract and concatenate joint and gripper states for proprioception.
            
            # The robot_joint_state is a tensor of shape (N, num_joints * 3) with pos, vel, torque concatenated.
            # We only need the positions, which are the first `num_joints` columns (29 for the body).
            body_joint_state_tensor = policy_obs["robot_joint_state"]
            body_joint_pos = body_joint_state_tensor[:, :29]

            # The robot_gipper_state is a tensor of just the gripper joint positions.
            gripper_joint_pos = policy_obs["robot_gipper_state"]

            proprio_state = torch.cat([body_joint_pos, gripper_joint_pos], dim=-1)
            wrapped_obs["proprio"] = proprio_state

        return wrapped_obs
        # --- End Conversion Logic ---

    def reset(self):
        """Resets the environment and returns the initial observation."""
        raw_obs, infos = self.env.reset()
        obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        print('****** infos', infos, self.env.num_envs, raw_obs["camera_images"]["front_camera"].shape)
        infos = self._record_metrics(rewards, infos)
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

        obs, _reward, terminations, truncations, infos = self.env.step(actions)
        print('*********reward*********', _reward)
        # For Debug, Only record table image with the first batch
        img = ((obs["camera_images"]["front_camera"][0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        self.images.append(img.cpu().numpy())

        obs = self._wrap_obs(obs)


        raw_next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        print('****** raw_next_obs', type(raw_next_obs))
        print('****** raw_next_obs[0]', type(raw_next_obs[0]))
        print('****** raw_next_obs[0].keys()', raw_next_obs[0].keys())
        obs = self._wrap_obs(raw_next_obs)

        if self.cfg.isaaclab.video_cfg.save_video:
            self._write_video(raw_next_obs)

        if self.record_metrics:
            infos = self._record_metrics(rewards, terminations, truncations, infos)

        if self.ignore_terminations:
            terminations[:] = False

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
            extracted_obs, step_reward, terminations, truncations, infos = self.step(actions)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos


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

    @property
    def video_writer(self) -> tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def flush_video(self) -> None:
        """
        Flush the video writer.
        """
        if self.cfg.isaaclab.video_cfg.save_video:
            self.video_writer = None

    def _write_video(self, raw_obs) -> None:
        """
        Write the current robot observations to video.
        This function processes the first environment's observations.
        """
        if self._video_writer is None:
            return
        
        container, stream = self._video_writer
        cam_data = raw_obs["camera_images"]
        
        # Process tensors for the first environment (index 0)
        # Convert from GPU tensor [H, W, C] to CPU numpy array
        # Isaac Lab images are float [0, 1], so convert to uint8 [0, 255]
        head_rgb = (cam_data["front_camera"][0].cpu().numpy() * 255).astype(np.uint8)
        left_wrist_rgb = (cam_data["left_wrist_camera"][0].cpu().numpy() * 255).astype(np.uint8)
        right_wrist_rgb = (cam_data["right_wrist_camera"][0].cpu().numpy() * 255).astype(np.uint8)

        # Resize for consistent video layout
        head_rgb_resized = cv2.resize(head_rgb, (640, 480))
        left_wrist_resized = cv2.resize(left_wrist_rgb, (240, 240))
        right_wrist_resized = cv2.resize(right_wrist_rgb, (240, 240))

        # Create a blank canvas to arrange the wrist images
        wrist_canvas = np.zeros((480, 240, 3), dtype=np.uint8)
        wrist_canvas[:240, :, :] = left_wrist_resized
        wrist_canvas[240:, :, :] = right_wrist_resized
        
        # Combine head and wrist views side-by-side
        final_frame = np.hstack([head_rgb_resized, wrist_canvas])

        # Write the frame to the video
        frame = av.VideoFrame.from_ndarray(final_frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # --- Metrics and Logging (Boilerplate) ---

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        if self.record_metrics:
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            episode_info = {
                "success": info.get("done", {}).get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            if "success" in info:
                self.success_once[env_idx] = (
                    self.success_once[env_idx] | info["success"]
                )
                episode_info["success_once"] = self.success_once[env_idx].clone()
            if "fail" in info:
                self.fail_once[env_idx] = self.fail_once[env_idx] | info["fail"]
                episode_info["fail_once"] = self.fail_once[env_idx].clone()
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    def _handle_auto_reset(self, dones: torch.Tensor, extracted_obs: dict, infos: dict):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos
