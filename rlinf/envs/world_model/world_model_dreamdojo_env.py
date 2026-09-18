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

"""DreamDojo world-model environment.

This module wires the NVIDIA DreamDojo generalist robot world model
(https://github.com/NVIDIA/DreamDojo, based on Cosmos-Predict2.5) into RLinf
as a `world-model env`, mirroring the design of :class:`WanEnv`
(see ``rlinf/envs/world_model/world_model_wan_env.py``).

Task setting:
    - Unitree G1 + Dex3 hand, bimanual (action_dim = 28, state_dim = 28)
    - Single head camera (``head_view``, no wrist views)
    - GR00T N1.7 as the policy (chunk length = 12 timesteps)
    - LeRobot v2.1 init dataset (parquet + mp4) via
      :class:`LeRobotV21InitDataset` — no offline ``.npy`` conversion needed
    - v2 three-head milestone classifier with independent per-env ratchets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from rlinf.data.datasets.lerobot_world_model import (
    LeRobotV21InitDataset,
    MixedLeRobotV21InitDataset,
)
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.dreamdojo_adapters import (
    G1DreamDojoActionBridge,
    update_g1_dex3_state,
)
from rlinf.envs.world_model.dreamdojo_reward import BatchedMilestoneReward

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
        # BaseWorldEnv builds the reset dataset, so dimensions must exist first.
        self.action_dim = int(cfg.get("action_dim", 28))
        self.state_dim = int(cfg.get("state_dim", 28))
        self.chunk = int(cfg.chunk)
        # RLinf counts 30-Hz commands; DreamDojo produces 15-Hz frames.
        self.wm_chunk = 12
        self.wm_steps_per_chunk = self.chunk // 2
        self.condition_frame_length = int(cfg.get("condition_frame_length", 1))
        self.image_size = tuple(cfg.get("image_size", (480, 640)))
        self.enable_kir = bool(cfg.get("enable_kir", False))
        self.kir_probability = float(cfg.get("kir_probability", 0.5))
        self.kir_max_offset_frames = int(cfg.get("kir_max_offset_frames", 30))
        if not 0 <= self.kir_probability <= 1 or self.kir_max_offset_frames < 1:
            raise ValueError("Invalid KIR probability or offset")
        if self.enable_kir and (
            cfg.get("is_eval", False) or cfg.get("random_start_frame", False)
        ):
            raise ValueError(
                "KIR is train-only and incompatible with random_start_frame"
            )
        # Separate RNG: enabling KIR does not perturb the episode/source sampler.
        self._kir_generator = torch.Generator().manual_seed(
            int(cfg.seed) + int(seed_offset) + 100000
        )
        self.seed_offset = int(seed_offset)
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
        self._init_eval_mask()

        self.video_audit = None
        if cfg.get("video_audit_dir"):
            if (
                not cfg.get("is_eval", False)
                or not self.eval_unique_episodes
                or cfg.max_episode_steps != 240
                or not cfg.reward_model.get("duplicate_for_30fps", True)
            ):
                raise ValueError(
                    "Video audit requires unique native eval, 240 actions and 30Hz reward scoring"
                )
            from rlinf.envs.world_model.dreamdojo_reward import THRESHOLDS
            from rlinf.envs.world_model.dreamdojo_video_audit import MilestoneVideoAudit

            self.video_audit = MilestoneVideoAudit(
                Path(cfg.video_audit_dir) / f"seed_{self.seed_offset}",
                cfg.get("video_audit_label") or "GR00T policy / DreamDojo WM",
                THRESHOLDS,
            )

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ---- WM hyperparameters (defaults chosen to mirror WanEnv semantics) ----
        # These names are intentionally identical to WanEnv so YAML overrides
        # transfer 1:1 between the two envs.
        self.num_inference_steps = int(cfg.num_inference_steps)
        self.num_frames = int(cfg.num_frames)
        assert self.num_frames == self.condition_frame_length + self.wm_chunk, (
            "num_frames must equal condition_frame_length + wm_chunk"
        )

        if self.chunk != 12 or self.action_dim != 28:
            raise ValueError("DreamDojo G1 currently requires chunk=12, action_dim=28")
        if self.condition_frame_length != 1 or self.auto_reset:
            raise ValueError(
                "DreamDojo requires one condition frame and auto_reset off"
            )

        # ---- Build heavy components ----
        self.pipe = self._build_pipeline()
        self.reward_model = self._load_reward_model()
        self.action_bridge = G1DreamDojoActionBridge(cfg.action_statistics, self.device)

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
        self.previous_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.current_state = torch.zeros(
            self.num_envs, self.state_dim, device=self.device
        )
        self.previous_state = torch.zeros_like(self.current_state)
        self.last_chunk_probs = torch.zeros(
            self.num_envs, self.wm_steps_per_chunk, 3, device=self.device
        )
        self._render_frames = None

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
        dataset_kwargs = {
            "video_key": cfg.get("video_key", "head_view"),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            # DreamDojo KIR changes the reset state, not Wan's multi-frame context.
            "enable_kir": False,
            "kir_context_len": self.condition_frame_length - 1,
            "image_size": self.image_size,
            "episodes": cfg.get("episodes"),
            "random_start_frame": cfg.get("random_start_frame", False),
            "video_backend": cfg.get("video_backend", "decord"),
            "history_frame_offset": 2,
        }
        paths = cfg.initial_image_path
        if isinstance(paths, str):
            return LeRobotV21InitDataset(paths, **dataset_kwargs)
        return MixedLeRobotV21InitDataset(
            list(paths),
            mixing_weights=cfg.get("initial_image_mixing_weights"),
            **dataset_kwargs,
        )

    def _build_pipeline(self):
        """Instantiate batched Video2WorldInference with explicit offload flags."""
        root = Path(self.cfg.dreamdojo_root).expanduser().resolve()
        checkpoint = Path(self.cfg.model_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DreamDojo checkpoint not found: {checkpoint}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from cosmos_predict2._src.predict2.inference.video2world import (
            Video2WorldInference,
        )

        config_file = (
            "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
        )
        previous_cwd = Path.cwd()
        offload = bool(self.cfg.get("enable_offload", True))

        def offload_flag(name):
            override = self.cfg.get(name)
            return offload if override is None else bool(override)

        try:
            os.chdir(root)
            return Video2WorldInference(
                experiment_name=self.cfg.experiment_name,
                ckpt_path=str(checkpoint),
                s3_credential_path="",
                context_parallel_size=1,
                config_file=str(config_file),
                offload_diffusion_model=offload_flag("wm_offload_diffusion_model"),
                offload_text_encoder=offload_flag("wm_offload_text_encoder"),
                offload_tokenizer=offload_flag("wm_offload_tokenizer"),
                cache_text_embeddings=bool(
                    self.cfg.get("cache_text_embeddings", False)
                ),
                skip_zero_guidance=bool(self.cfg.get("skip_zero_guidance", False)),
            )
        finally:
            os.chdir(previous_cwd)

    def _load_reward_model(self):
        """Load the v2 cumulative-head milestone reward and ratchet."""
        rew_type = self.cfg.reward_model.type
        if rew_type != "MilestoneRewardV2":
            raise ValueError(f"Unsupported DreamDojo reward model: {rew_type}")
        return BatchedMilestoneReward(
            self.cfg.reward_model.from_pretrained,
            self.num_envs,
            self.device,
            duplicate_for_30fps=bool(
                self.cfg.reward_model.get("duplicate_for_30fps", True)
            ),
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
        episode_info["reward"] = episode_info["return"] / episode_info[
            "episode_len"
        ].clamp_min(1)
        episode_info["episode_seconds"] = episode_info["episode_len"] / 30.0
        for head, name in enumerate(("picked", "handed", "placed")):
            episode_info[name] = (self.reward_model.stage > head).clone()
            episode_info[f"{name}_prob_max"] = (
                self.last_chunk_probs[:, :, head].amax(dim=1).clone()
            )
        if getattr(self, "enable_kir", False):
            episode_info["kir_fraction"] = self.reset_is_kir.float().clone()
            episode_info["kir_eligible_fraction"] = (
                self.reset_kir_eligible.float().clone()
            )
            episode_info["reset_frame"] = self.reset_start_frames.float().clone()
            episode_info["initial_stage"] = self.reset_initial_stage.float().clone()
            # Conditional means are recovered as numerator / fraction in reports.
            episode_info["kir_success_numerator"] = (
                self.success_once.float() * self.reset_is_kir
            )
            episode_info["original_success_numerator"] = (
                self.success_once.float() * ~self.reset_is_kir
            )
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        """Convert per-frame chunk rewards to per-chunk scalar step rewards.

        Sum the one-time milestone payouts over the executed interval.
        """
        return chunk_rewards.sum(dim=1)

    # ------------------------------------------------------------------ #
    #                       Reset / rollout / step                        #
    # ------------------------------------------------------------------ #

    def _load_reset_items(self, episode_indices):
        """Choose once per GRPO group; never change the sampled episode/source."""
        if not getattr(self, "enable_kir", False):
            return [self.dataset[int(index)] for index in episode_indices]
        if len(episode_indices) != self.num_envs or self.num_envs % self.group_size:
            raise ValueError("KIR reset requires complete GRPO groups")
        items, flags, eligible_flags, starts = [], [], [], []
        self.reset_manifest = []
        for offset in range(0, self.num_envs, self.group_size):
            indices = [
                int(index)
                for index in episode_indices[offset : offset + self.group_size]
            ]
            if len(set(indices)) != 1:
                raise ValueError("KIR group members must share the same episode")
            index = indices[0]
            start = self.dataset.handover_start_frame(index, self.kir_max_offset_frames)
            eligible = start is not None
            use_kir = (
                eligible
                and torch.rand((), generator=self._kir_generator).item()
                < self.kir_probability
            )
            item = self.dataset.get_at_frame(
                index,
                start if use_kir else None,
                reward_history_frames=17 if use_kir else 0,
            )
            actual_start = item["dataset_meta"]["start_frame"]
            items.extend([item] * self.group_size)
            flags.extend([use_kir] * self.group_size)
            eligible_flags.extend([eligible] * self.group_size)
            starts.extend([actual_start] * self.group_size)
            self.reset_manifest.append(
                {
                    "flat_episode_index": index,
                    "episode_index": item["episode_index"],
                    "source_path": item["dataset_meta"].get("source_path"),
                    "start_frame": actual_start,
                    "is_kir": use_kir,
                    "eligible": eligible,
                    "initial_stage": int(use_kir),
                }
            )
        self.reset_is_kir = torch.tensor(flags, device=self.device, dtype=torch.bool)
        self.reset_kir_eligible = torch.tensor(
            eligible_flags, device=self.device, dtype=torch.bool
        )
        self.reset_start_frames = torch.tensor(starts, device=self.device)
        self.reset_initial_stage = self.reset_is_kir.long()
        return items

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
        initial_states: list[torch.Tensor] = []
        initial_actions: list[torch.Tensor] = []
        baseline_states: list[torch.Tensor] = []
        task_descriptions: list[str] = []
        init_ee_poses: list = []

        # 2. For each env, load its init episode and build condition frames.
        reset_items = self._load_reset_items(episode_indices)
        for env_idx, (episode_idx, episode_data) in enumerate(
            zip(episode_indices, reset_items)
        ):
            if not episode_data.get("start_items"):
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]
            task_desc = str(episode_data.get("task", ""))
            task_descriptions.append(task_desc)

            if "image" not in first_frame:
                raise ValueError(
                    f"No 'image' key in first frame of episode {episode_idx}"
                )

            # KIR group members share the same decoded item. Normalization below
            # is in-place, so each environment must own its input image.
            img_tensor = first_frame["image"].clone()  # [3, H, W] in [0, 1]

            if "observation.state" in first_frame:
                state = torch.as_tensor(first_frame["observation.state"]).float()
                init_ee_poses.append(state.numpy())
            else:
                state = torch.zeros(self.state_dim)
                init_ee_poses.append(None)
            initial_states.append(state)
            initial_actions.append(
                torch.as_tensor(first_frame["wm_previous_action"]).float()
            )
            baseline_states.append(
                torch.as_tensor(first_frame["wm_previous_state"]).float()
            )

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

            img_tensors.append(env_img_tensor)

        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)
        # [num_envs, 3, 1, T_c, H, W] to leave room for the "view" axis Wan uses.
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        self.current_state = torch.stack(initial_states).to(self.device)
        self.previous_action = torch.stack(initial_actions).to(self.device)
        self.previous_state = torch.stack(baseline_states).to(self.device)

        for env_idx in range(self.num_envs):
            self.image_queue[env_idx] = [
                self.current_obs[env_idx, :, 0, t : t + 1, :, :]
                for t in range(self.condition_frame_length)
            ]

        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses
        self.reward_model.reset()
        if getattr(self, "enable_kir", False) and self.reset_is_kir.any():
            selected = self.reset_is_kir.nonzero(as_tuple=False).flatten().tolist()
            histories = torch.stack(
                [reset_items[index]["reward_history"] for index in selected]
            )
            self.reward_model.prime_handover(histories, selected)
        self._render_frames = None

        self._reset_metrics()
        if getattr(self, "video_audit", None) is not None:
            cases = [
                {
                    "flat_episode_index": int(episode_index),
                    "episode_index": int(item["episode_index"]),
                    "source_path": item["dataset_meta"]["source_path"],
                    "start_frame": int(item["dataset_meta"]["start_frame"]),
                    "valid": bool(self._eval_valid_mask[index]),
                }
                for index, (episode_index, item) in enumerate(
                    zip(episode_indices, reset_items, strict=True)
                )
            ]
            self.video_audit.start(
                self._wrap_obs()["main_images"].detach().cpu().numpy(), cases
            )
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

        # Success means all three ordered milestones have been confirmed.
        success_mask = self.reward_model.stage == 3
        if not self.ignore_terminations:
            chunk_terminations[:, -1] = success_mask

        self.elapsed_steps += self.chunk
        if self.elapsed_steps >= self.cfg.max_episode_steps:
            chunk_truncations[:, -1] = True
        if self.eval_unique_episodes:
            # Native EnvWorker counts only newly-done slots. Padding still
            # participates in batched inference but never emits an episode.
            chunk_terminations &= self._eval_valid_mask[:, None]
            chunk_truncations &= self._eval_valid_mask[:, None]
        infos: dict = {}
        infos = self._record_metrics(step_reward, success_mask, infos)

        if (
            getattr(self, "video_audit", None) is not None
            and self.elapsed_steps >= self.cfg.max_episode_steps
        ):
            self.video_audit.finish()

        extracted_obs = self._wrap_obs()

        return (
            [extracted_obs],
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    # ------------------------------------------------------------------ #
    #                    DreamDojo-specific hot loops                     #
    # ------------------------------------------------------------------ #

    def _infer_next_chunk_frames(self, actions):
        """Roll DreamDojo forward one chunk and update ``current_obs`` / ``image_queue``.

        Contract:
            * ``actions`` shape ``[num_envs, chunk, action_dim]`` (float32 or bf16).
            * On exit, ``self.current_obs`` is
              ``[num_envs, 3, 1, condition_frame_length + wm_steps_per_chunk, H, W]``
              (older frames evicted via a sliding window) and each
              ``self.image_queue[i]`` holds the last ``condition_frame_length``
              frames as ``[3, 1, H, W]`` tensors in [-1, 1].
        """
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        if actions_tensor.shape != (self.num_envs, self.chunk, self.action_dim):
            raise ValueError(
                "Expected actions shaped "
                f"{(self.num_envs, self.chunk, self.action_dim)}, "
                f"got {tuple(actions_tensor.shape)}"
            )
        encoded = self.action_bridge.encode_policy_chunk(
            self.previous_action, actions_tensor, self.previous_state
        )

        condition = self.current_obs[:, :, 0, -1]
        condition_uint8 = ((condition + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        input_video = torch.cat(
            [
                condition_uint8.unsqueeze(2),
                torch.zeros_like(condition_uint8)
                .unsqueeze(2)
                .repeat(1, 1, self.wm_chunk, 1, 1),
            ],
            dim=2,
        )
        generated = self.pipe.generate_vid2world(
            prompt=[""] * self.num_envs,
            input_path=input_video,
            action=encoded,
            guidance=0,
            num_video_frames=self.wm_chunk + 1,
            num_latent_conditional_frames=1,
            resolution=f"{self.image_size[0]},{self.image_size[1]}",
            seed=self.seed + self.elapsed_steps,
            lam_video=None,
            num_steps=self.num_inference_steps,
        )
        # Ignore the padded unexecuted future: 12 policy commands = 6 frames.
        new_frames = generated[:, :, 1 : 1 + self.wm_steps_per_chunk].to(self.device)
        condition = self.current_obs[:, :, :, -1:]
        self.current_obs = torch.cat([condition, new_frames.unsqueeze(2)], dim=3)
        if self.video_cfg.save_video:
            self._render_frames = (
                ((new_frames + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 4, 1)
                .cpu()
            )
        # For next image I[t+12], baseline is a[t+10], s[t+10].
        # Proprio remains an explicit perfect-target-following proxy.
        self.previous_action = actions_tensor[:, -2].detach()
        self.previous_state = actions_tensor[:, -3].detach().clone()
        self.current_state = update_g1_dex3_state(
            self.current_state, actions_tensor[:, -1]
        )
        for env_index in range(self.num_envs):
            self.image_queue[env_index] = [self.current_obs[env_index, :, 0, -1:]]

    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        """Score the newest WM frames and align payouts to control commands.

        Returns:
            ``[num_envs, chunk]`` rewards, nonzero only on image boundaries.
            Multiple confirmed milestones can pay out at one boundary.
        """
        frames = self.current_obs[:, :, 0, -self.wm_steps_per_chunk :].permute(
            0, 2, 1, 3, 4
        )
        audit = getattr(self, "video_audit", None)
        stage_before = self.reward_model.stage.clone() if audit is not None else None
        frame_rewards, self.last_chunk_probs = self.reward_model.score_chunk(frames)
        if audit is not None:
            stages = stage_before[:, None] + frame_rewards.cumsum(dim=1)
            audit.append(
                ((frames + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 1, 3, 4, 2)
                .cpu()
                .numpy(),
                self.last_chunk_probs.detach().cpu().numpy(),
                stages.detach().cpu().numpy(),
            )
        # Images arrive after two control commands; pay at that boundary.
        rewards = frame_rewards.new_zeros(self.num_envs, self.chunk)
        rewards[:, 1::2] = frame_rewards
        return rewards

    # ------------------------------------------------------------------ #
    #                       Observation wrapping                          #
    # ------------------------------------------------------------------ #

    def _wrap_obs(self) -> dict:
        """Return the observation dict expected by GR00T / LIBERO-style actors.

        Keys and shapes are chosen to match :meth:`LiberoEnv._wrap_obs` so no
        actor-side branching is needed.

        DreamDojo generates the main camera only. ``states`` is the last-known
        proprio proxy assuming perfect tracking of decoded absolute targets.
        """
        # Last generated frame per env.
        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [B, 3, H, W]
        full_image = ((last_frame + 1.0) / 2.0 * 255.0).clamp(0, 255)

        if tuple(full_image.shape[-2:]) != self.image_size:
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
        full_image = full_image.permute(0, 2, 3, 1).to(torch.uint8)  # [B, H, W, 3]

        # G1 + Dex3 layout: left_arm, right_arm, left_hand, right_hand.
        states = self.current_state.to(self.device, dtype=torch.float32)

        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": list(self.task_descriptions),
        }

    def capture_image(self) -> torch.Tensor:
        """Return all newly generated frames for RLinf's video recorder."""
        if self._render_frames is not None:
            return self._render_frames
        return self._wrap_obs()["main_images"][:, None]

    def _init_eval_mask(self) -> None:
        """Exclude compute-only padding slots from native eval episode counts."""
        self.eval_unique_episodes = bool(self.cfg.get("eval_unique_episodes", False))
        self._eval_valid_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if not self.eval_unique_episodes:
            return
        if (
            not self.cfg.get("is_eval", False)
            or not self.use_fixed_reset_state_ids
            or self.group_size != 1
            or not self.ignore_terminations
        ):
            raise ValueError(
                "Unique eval requires is_eval, fixed resets, group_size=1, "
                "and ignore_terminations=True"
            )
        count = len(self.dataset)
        slots = self.num_envs * self.total_num_processes
        if count <= 0 or slots < count or slots - count >= self.total_num_processes:
            raise ValueError(
                "Unique eval needs the smallest evenly sharded batch covering the dataset"
            )
        indices = self.seed_offset * self.num_envs + torch.arange(
            self.num_envs, device=self.device
        )
        self._eval_valid_mask = indices < count

    def _sample_reset_episode_indices(self) -> list[int]:
        """Sample dataset indices for the next reset, grouping by GRPO group.

        Members of the same GRPO group get the same init trajectory, so the
        relative advantage isolates policy behaviour rather than init noise.
        """
        n_episodes = len(self.dataset)
        if self.use_fixed_reset_state_ids:
            start = self.seed_offset * self.num_group
            base = (torch.arange(self.num_group) + start) % n_episodes
        elif hasattr(self.dataset, "sample_indices"):
            base = torch.tensor(
                self.dataset.sample_indices(self.num_group, self._generator)
            )
        else:
            base = torch.randint(
                0, n_episodes, (self.num_group,), generator=self._generator
            )
        return base.repeat_interleave(self.group_size).tolist()

    # ------------------------------------------------------------------ #
    #                              Offload                                #
    # ------------------------------------------------------------------ #

    def _move_resident_wm(self, device):
        """Move chunk-resident modules only at the outer rollout boundary."""
        if not self.pipe.offload_diffusion_model:
            self.pipe.model.net.to(device)
            self.pipe.model.conditioner.to(device)
        if not self.pipe.offload_tokenizer:
            tokenizer = self.pipe.model.tokenizer
            if hasattr(tokenizer, "encoder") and hasattr(tokenizer, "decoder"):
                tokenizer.encoder.to(device)
                tokenizer.decoder.to(device)
            else:
                # Wan2pt1VAEInterface has one shared nn.Module under model.model,
                # not the separate encoder/decoder interface used by Cosmos.
                tokenizer.clear_cache()
                tokenizer.model.model.to(device)

    def offload(self):
        if self._is_offloaded:
            return
        # Per-chunk offloaded modules are already on CPU. Resident modules
        # must also release memory before the actor training phase.
        self._move_resident_wm("cpu")
        self.reward_model.to("cpu")
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to("cpu")
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self._move_resident_wm(self.device)
        self.reward_model.to(self.device)
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to(self.device)
        self._is_offloaded = False
