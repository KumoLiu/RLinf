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

"""DreamDojo world-model environment (scaffold).

This module wires the NVIDIA DreamDojo generalist robot world model
(https://github.com/NVIDIA/DreamDojo, based on Cosmos-Predict2.5) into RLinf
as a `world-model env`, mirroring the design of :class:`WanEnv`
(see ``rlinf/envs/world_model/world_model_wan_env.py``).

Task setting this scaffold is written for:
    - Unitree G1 + Dex3 hand, bimanual (action_dim = 28, state_dim = 28)
    - Single head camera (``head_view``, no wrist views)
    - GR00T N1.7 as the policy (chunk length = 16 timesteps)
    - LeRobot v2.1 init dataset (parquet + mp4) via
      :class:`LeRobotV21InitDataset` — no offline ``.npy`` conversion needed
    - 4-class progress reward model producing a scalar reward in ``[0, 1]``
      (already reduced inside the reward model itself)

Everything DreamDojo-specific is behind ``TODO(dreamdojo)`` markers; fill them
in once you drop the DreamDojo package under ``$VENV_DIR/dreamdojo`` (or wire
its imports below).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image  # noqa: F401  (kept for parity with WanEnv; delete if unused)

from rlinf.data.datasets.lerobot_world_model import LeRobotV21InitDataset
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

# TODO(dreamdojo): replace these placeholder imports once the DreamDojo package
# is installed in the venv (or vendored under rlinf/models/embodiment/dreamdojo/).
# Example (subject to change based on the real DreamDojo API):
#
#     from dreamdojo.pipelines import DreamDojoVideoPipeline, ModelConfig
#     from dreamdojo.models.reward_model import ProgressResnetRewModel
DreamDojoVideoPipeline = None  # type: ignore[assignment]
ProgressResnetRewModel = None  # type: ignore[assignment]


__all__ = ["DreamDojoEnv"]


class DreamDojoEnv(BaseWorldEnv):
    """World-model environment backed by DreamDojo.

    Behavioural contract (must match :class:`WanEnv` so ``EnvWorker`` and the
    GR00T actor can consume either interchangeably):

    * ``reset()`` -> ``(obs_dict, info_dict)`` where ``obs_dict`` has keys
      ``main_images``, ``wrist_images``, ``states``, ``task_descriptions``.
    * ``chunk_step(policy_output_action)`` ->
      ``(obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list)``.
    * ``offload()`` / ``onload()`` for VRAM management when
      ``env.*.enable_offload: True``.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
        worker_info=None,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
            record_metrics,
        )

        # ---- Reset-state / grouping bookkeeping (mirrors WanEnv) ----
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ---- WM hyperparameters (defaults chosen to mirror WanEnv semantics) ----
        # These names are intentionally identical to WanEnv so YAML overrides
        # transfer 1:1 between the two envs.
        self.num_inference_steps: int = cfg.num_inference_steps
        self.chunk: int = cfg.chunk  # action-chunk length, e.g. 8
        self.condition_frame_length: int = cfg.condition_frame_length  # e.g. 5
        self.num_frames: int = cfg.num_frames  # condition + chunk
        assert self.num_frames == self.condition_frame_length + self.chunk, (
            "num_frames must equal condition_frame_length + chunk"
        )

        self.image_size = tuple(cfg.image_size)  # e.g. (256, 256)

        self.retain_action: bool = cfg.get("retain_action", True)
        self.enable_kir: bool = cfg.get("enable_kir", True)

        # G1 + Dex3 has 28-D actions and 28-D state. Keep both configurable so
        # the same env can serve other robots without code edits.
        self.action_dim: int = cfg.get("action_dim", 28)
        self.state_dim: int = cfg.get("state_dim", 28)

        # Optional scalar reward threshold for marking a chunk successful.
        # For a 4-class progress reward mapped to [0, 1] (phase index / 3),
        # phase-3 corresponds to reward ~= 1.0 -> a threshold around 0.9 works.
        self.success_reward_threshold: float = cfg.get(
            "success_reward_threshold", 0.9
        )

        # ---- Build heavy components ----
        self.pipe = self._build_pipeline()
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # ---- Runtime state ----
        # ``current_obs`` layout matches WanEnv: [num_envs, 3, 1, T, H, W] in [-1, 1]
        self.current_obs: Optional[torch.Tensor] = None
        self.task_descriptions: list[str] = [""] * self.num_envs
        self.init_ee_poses: list = [None] * self.num_envs

        # Per-env queue of the most recent ``condition_frame_length`` frames.
        # Each element is a [3, 1, H, W] float tensor in [-1, 1].
        self.image_queue: list[list[Optional[torch.Tensor]]] = [
            [None] * self.condition_frame_length for _ in range(self.num_envs)
        ]

        # Condition-action buffer for AR generation: last ``T_c`` executed
        # actions per env. Dtype follows the actor's precision at runtime.
        self.condition_action = torch.zeros(
            self.num_envs, self.condition_frame_length, self.action_dim
        )

        # Trocar task does not have a "gripper open on reset" concept the way
        # LIBERO does; we keep the flag but default it to False.
        self.reset_gripper_open: bool = cfg.get("reset_gripper_open", False)

        # Normalisation to [-1, 1] for the DreamDojo pipeline input frames.
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                )
            ]
        )

        self._is_offloaded = False

    # ------------------------------------------------------------------ #
    #                        Component construction                       #
    # ------------------------------------------------------------------ #

    def _build_dataset(self, cfg):
        """Load per-episode init trajectories from a LeRobot v2.1 tree.

        Expected on-disk layout (matches ``pick_trocar_teleop_success_*``)::

            cfg.initial_image_path/
                meta/{info.json, modality.json, tasks.jsonl, episodes.jsonl}
                data/chunk-000/episode_XXXXXX.parquet
                videos/chunk-000/<original_key>/episode_XXXXXX.mp4

        The dataset returns per-episode dicts shaped identically to
        :class:`NpyTrajectoryDatasetWrapper` (``start_items``, ``target_items``,
        ``task``, ``episode_index``, ``dataset_meta``) so the reset path stays
        generic across world-model envs.
        """
        return LeRobotV21InitDataset(
            data_path=cfg.initial_image_path,
            video_key=cfg.get("video_key", "head_view"),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            enable_kir=self.enable_kir,
            kir_context_len=self.condition_frame_length - 1,
            image_size=self.image_size,
            episodes=cfg.get("episodes"),
            random_start_frame=cfg.get("random_start_frame", False),
            video_backend=cfg.get("video_backend", "decord"),
        )

    def _build_pipeline(self):
        """Instantiate the DreamDojo video-generation pipeline.

        Contract this method must satisfy:

        * The returned object exposes a callable (either ``pipe(**kwargs)`` or
          ``pipe.generate(**kwargs)``) that, given ``condition_frames`` (List[PIL]
          or float tensor in [-1, 1]) and ``action`` (float tensor of shape
          ``[B, T, action_dim]``), returns ``num_frames`` predicted future
          frames per env in a form we can convert to
          ``[3, T_out, H, W]`` float tensors in [-1, 1].
        * All heavy sub-modules (DiT / UNet, VAE, text encoder, action
          tokenizer, ...) live on ``self.device`` after this returns.
        """
        # TODO(dreamdojo): replace with the real DreamDojo pipeline builder.
        # Example scaffold (rename to whatever DreamDojo exports):
        #
        #     pipe = DreamDojoVideoPipeline.from_pretrained(
        #         torch_dtype=torch.bfloat16,
        #         device=self._get_runtime_device_str(),
        #         model_configs=[
        #             ModelConfig(path=self.cfg.model_path,   offload_device="cpu"),
        #             ModelConfig(path=self.cfg.VAE_path,     offload_device="cpu"),
        #             # DreamDojo may also want the text encoder / action tokenizer
        #             # ModelConfig(path=self.cfg.text_encoder_path, offload_device="cpu"),
        #         ],
        #     )
        #     pipe.dit.to(self.device)
        #     pipe.vae.to(self.device)
        #     return pipe
        raise NotImplementedError(
            "TODO(dreamdojo): wire NVIDIA/DreamDojo pipeline construction here."
        )

    def _load_reward_model(self):
        """Load the 4-class progress reward model.

        The model must expose ``predict_rew(images) -> Tensor`` returning a
        scalar reward per frame in ``[0, 1]`` (typically ``phase_index / 3``).
        For a raw 4-class classifier, do the reduction inside the model's
        ``predict_rew`` so the env can stay generic.

        Input shape: ``[B*chunk, 3, H, W]``  float in [-1, 1] or [0, 1]
                     (match what your reward model was trained on).
        Output:      ``[B*chunk]``           float in [0, 1]
        """
        rew_type = self.cfg.reward_model.type
        # TODO(dreamdojo): register your reward model class(es) here.
        # if rew_type == "ProgressResnetRewModel":
        #     return ProgressResnetRewModel(self.cfg.reward_model.from_pretrained)
        raise NotImplementedError(
            f"TODO(dreamdojo): load reward model type={rew_type!r} "
            f"from {self.cfg.reward_model.from_pretrained!r}."
        )

    # ------------------------------------------------------------------ #
    #                              Metrics                                #
    # ------------------------------------------------------------------ #

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0.0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self.elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        if not self.record_metrics:
            return infos
        self.returns += step_reward
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor
        episode_info = {
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": torch.full(
                (self.num_envs,),
                self.elapsed_steps,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"].clamp_min(1)
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        """Convert per-frame chunk rewards to per-chunk scalar step rewards.

        Mirrors WanEnv: takes the max reward inside the chunk as the step
        signal so a single high-progress frame propagates back to the actor.
        Adjust if you want mean, last-frame, or delta-based rewards instead.
        """
        return chunk_rewards.max(dim=1).values

    # ------------------------------------------------------------------ #
    #                       Reset / rollout / step                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def reset(self, *, seed=None, options=None, episode_indices=None):
        """Sample init trajectories, seed condition frames + actions, and
        return the initial observation dict."""
        self.onload()

        # 1. Pick episode indices for each env (respecting group_size so that
        #    all members of a GRPO group start from the same init state).
        if episode_indices is None:
            episode_indices = self._sample_reset_episode_indices()

        img_tensors: list[torch.Tensor] = []
        condition_actions: list[torch.Tensor] = []
        task_descriptions: list[str] = []
        init_ee_poses: list = []

        # 2. For each env, load its init episode and build condition frames.
        for env_idx, episode_idx in enumerate(episode_indices):
            episode_data = self.dataset[int(episode_idx)]
            if not episode_data.get("start_items"):
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]
            task_desc = str(episode_data.get("task", ""))
            task_descriptions.append(task_desc)

            if "image" not in first_frame:
                raise ValueError(
                    f"No 'image' key in first frame of episode {episode_idx}"
                )

            img_tensor = first_frame["image"]  # [3, H, W] float in [0, 1]

            if "observation.state" in first_frame:
                init_ee_poses.append(first_frame["observation.state"].numpy())
            else:
                init_ee_poses.append(None)

            if img_tensor.shape[1:] != self.image_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            img_tensor = self.trans_norm(img_tensor)  # [-1, 1]

            # Repeat first frame across condition-frame slots.
            env_img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )  # [3, T_c, H, W]

            env_condition_action = np.zeros(
                (self.condition_frame_length, self.action_dim), dtype=np.float32
            )

            # KIR: fill last (T_c - 1) condition frames with real pre-keyframe
            # context and their executed actions, if available.
            target_items = episode_data.get("target_items", [])
            if len(target_items) == self.condition_frame_length - 1:
                for target_idx, target_frame in enumerate(target_items):
                    if "image" not in target_frame or "action" not in target_frame:
                        raise ValueError(
                            "Missing image/action in KIR target frame for "
                            f"episode {episode_idx}"
                        )
                    target_img = target_frame["image"]
                    if target_img.shape[1:] != self.image_size:
                        target_img = F.interpolate(
                            target_img.unsqueeze(0),
                            size=self.image_size,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    target_img = self.trans_norm(target_img)
                    env_img_tensor[:, target_idx + 1] = target_img
                    env_condition_action[target_idx + 1] = (
                        target_frame["action"].numpy()
                        if isinstance(target_frame["action"], torch.Tensor)
                        else target_frame["action"]
                    )

            img_tensors.append(env_img_tensor)
            condition_actions.append(torch.from_numpy(env_condition_action))

        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)
        # [num_envs, 3, 1, T_c, H, W] to leave room for the "view" axis Wan uses.
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        self.condition_action = torch.stack(condition_actions, dim=0).to(self.device)

        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = [
                self.current_obs[env_idx, :, 0, t : t + 1, :, :]
                for t in range(self.condition_frame_length)
            ]

        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        self._reset_metrics()
        return self._wrap_obs(), {}

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "step() is not implemented for DreamDojoEnv; use chunk_step() instead."
        )

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Run one action-chunk through DreamDojo and score the generated frames.

        Args:
            policy_output_action: ``[num_envs, chunk, action_dim]`` float tensor
                (already denormalised into the env's action space).

        Returns:
            tuple[list[dict], Tensor, Tensor, Tensor, list[dict]] shaped like
            :meth:`WanEnv.chunk_step` so ``EnvWorker.env_interact_step`` can
            consume either env without branching.
        """
        self.onload()
        self._infer_next_chunk_frames(policy_output_action)
        chunk_rewards = self._infer_next_chunk_rewards()  # [num_envs, chunk]

        step_reward = self._calc_step_reward(chunk_rewards)  # [num_envs]
        chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, device=self.device, dtype=torch.bool
        )
        chunk_truncations = torch.zeros_like(chunk_terminations)

        # Mark termination on the last frame of the chunk when the progress
        # reward crosses the success threshold.
        success_mask = step_reward >= self.success_reward_threshold
        chunk_terminations[:, -1] = success_mask

        self.elapsed_steps += self.chunk
        infos: dict = {}
        infos = self._record_metrics(step_reward, chunk_terminations[:, -1], infos)

        extracted_obs = self._wrap_obs()

        if self.auto_reset and success_mask.any():
            # Reset only the envs that just terminated so the runner sees fresh
            # episodes at the next chunk boundary. WanEnv does the equivalent
            # via ``_handle_auto_reset``; keep the same semantics.
            extracted_obs, infos = self._handle_auto_reset(
                success_mask, extracted_obs, infos
            )

        return [extracted_obs], chunk_rewards, chunk_terminations, chunk_truncations, [infos]

    # ------------------------------------------------------------------ #
    #                    DreamDojo-specific hot loops                     #
    # ------------------------------------------------------------------ #

    def _infer_next_chunk_frames(self, actions):
        """Roll DreamDojo forward one chunk and update ``current_obs`` / ``image_queue``.

        Contract:
            * ``actions`` shape ``[num_envs, chunk, action_dim]`` (float32 or bf16).
            * On exit, ``self.current_obs`` is
              ``[num_envs, 3, 1, condition_frame_length + chunk, H, W]``
              (older frames evicted via a sliding window) and each
              ``self.image_queue[i]`` holds the last ``condition_frame_length``
              frames as ``[3, 1, H, W]`` tensors in [-1, 1].
        """
        actions_tensor = (
            torch.from_numpy(actions) if isinstance(actions, np.ndarray) else actions
        ).to(device=self.device)

        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )
        if self.retain_action:
            actions_tensor = torch.cat(
                [self.condition_action, actions_tensor], dim=1
            )

        # Slide condition-action buffer: keep the most recent (T_c - 1) actions.
        self.condition_action[:, 1 : self.condition_frame_length, :] = actions_tensor[
            :, -(self.condition_frame_length - 1) :, :
        ]

        # TODO(dreamdojo): call the real DreamDojo generation API. A rough
        # template mirroring ``WanEnv._infer_next_chunk_frames``:
        #
        #     kwargs = {
        #         "condition_frames": [
        #             self._decode_env_condition_frames(env_idx)  # e.g. List[PIL]
        #             for env_idx in range(self.num_envs)
        #         ],
        #         "action": actions_tensor,                       # [B, T, action_dim]
        #         "prompt": self.task_descriptions,               # List[str]
        #         "height": self.image_size[0],
        #         "width":  self.image_size[1],
        #         "num_frames": self.num_frames,
        #         "num_inference_steps": self.num_inference_steps,
        #         "cfg_scale": 1.0,
        #     }
        #     output = self.pipe(**kwargs)                        # e.g. List[List[PIL]]
        #
        # Then convert each per-env output to a [3, T_out, H, W] tensor in
        # [-1, 1], update ``self.image_queue[env_idx]`` with the last 4 frames,
        # and concatenate the new chunk onto ``self.current_obs`` along dim=3.
        raise NotImplementedError(
            "TODO(dreamdojo): implement DreamDojo AR generation call here."
        )

    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        """Score the newest ``chunk`` frames with the progress reward model.

        Returns:
            ``[num_envs, chunk]`` float tensor in ``[0, 1]``.
        """
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # current_obs: [num_envs, 3, 1, T_c + chunk, H, W]  in [-1, 1]
        b, c, v, t, h, w = self.current_obs.shape
        # Reorder to [num_envs, T_c+chunk, 3, 1, H, W] and take the last ``chunk`` frames.
        chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)[
            :, -self.chunk :
        ]
        # [num_envs*chunk, 3, H, W]
        chunk_obs = chunk_obs.reshape(b * self.chunk, c, v, h, w).squeeze(2)
        chunk_obs = chunk_obs.to(self.device)

        # ``predict_rew`` MUST return already-reduced scalars in [0, 1].
        # For a 4-class progress classifier, reduce with
        #     reward = (softmax(logits) * torch.arange(4, device=logits.device)).sum(-1) / 3.0
        # inside your model's ``predict_rew``.
        rewards = self.reward_model.predict_rew(chunk_obs)
        return rewards.reshape(b, self.chunk)

    # ------------------------------------------------------------------ #
    #                       Observation wrapping                          #
    # ------------------------------------------------------------------ #

    def _wrap_obs(self) -> dict:
        """Return the observation dict expected by GR00T / LIBERO-style actors.

        Keys and shapes are chosen to match :meth:`LiberoEnv._wrap_obs` so no
        actor-side branching is needed.

        Since DreamDojo (in this setup) generates only the main camera and
        does not expose proprioception, ``wrist_images`` is ``None`` and
        ``states`` is a zero placeholder. Adjust ``state_dim`` via config if
        your GR00T converter wants a specific shape (LIBERO converter reads 8
        columns, dual-arm Dex3 might want more).
        """
        # Last generated frame per env.
        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [B, 3, H, W]
        full_image = ((last_frame + 1.0) / 2.0 * 255.0).clamp(0, 255)

        if tuple(full_image.shape[-2:]) != self.image_size:
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
        full_image = full_image.permute(0, 2, 3, 1).to(torch.uint8)  # [B, H, W, 3]

        # G1 + Dex3 layout: [left_arm(0:7), right_arm(7:14), left_hand(14:21),
        # right_hand(21:28)]. DreamDojo does not expose proprio, so we fill
        # zeros here. If you later want to feed the last-known state (from
        # reset or a state estimator), populate ``self.current_state`` in
        # ``reset()`` / ``chunk_step()`` and swap this out.
        states = torch.zeros(
            (self.num_envs, self.state_dim), device=self.device, dtype=torch.float32
        )

        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": list(self.task_descriptions),
        }

    def _handle_auto_reset(self, done_mask: torch.Tensor, extracted_obs, infos):
        """Reset only the envs that finished this chunk.

        WanEnv resets everything unconditionally; for GRPO groups you often
        want per-env resets so surviving group members can keep rolling. Keep
        this stub simple for the first version — full episode boundary logic
        (final-obs vs. reset-obs) can be added when needed.
        """
        # TODO(dreamdojo): implement selective reset. As a placeholder, reset
        # all envs whenever any one is done, matching the WanEnv default.
        if done_mask.any():
            extracted_obs, infos_reset = self.reset()
            infos.update(infos_reset)
        return extracted_obs, infos

    def _sample_reset_episode_indices(self) -> list[int]:
        """Sample dataset indices for the next reset, grouping by GRPO group.

        Members of the same GRPO group get the same init trajectory, so the
        relative advantage isolates policy behaviour rather than init noise.
        """
        n_episodes = len(self.dataset)
        if self.use_fixed_reset_state_ids:
            base = torch.arange(self.num_group) % n_episodes
        else:
            base = torch.randint(
                0, n_episodes, (self.num_group,), generator=self._generator
            )
        return base.repeat_interleave(self.group_size).tolist()

    # ------------------------------------------------------------------ #
    #                              Offload                                #
    # ------------------------------------------------------------------ #

    def offload(self):
        if self._is_offloaded:
            return
        # TODO(dreamdojo): move heavy DreamDojo modules to CPU (e.g.
        # ``self.pipe.dit.to("cpu")``, ``self.pipe.vae.to("cpu")``).
        self.reward_model.to("cpu")
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to("cpu")
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        # TODO(dreamdojo): move DreamDojo modules back to ``self.device``.
        self.reward_model.to(self.device)
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to(self.device)
        self._is_offloaded = False
