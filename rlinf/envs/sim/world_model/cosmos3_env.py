# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot H2 episodes with a Cosmos backend and framewise progress rewards."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from rlinf.data.datasets.lerobot_world_model import (
    LeRobotV21InitDataset,
)
from rlinf.envs.sim.world_model.env import WorldModelEnv


class Cosmos3Env(WorldModelEnv):
    """Keep the H2 reset, proprioception and evaluation contracts across backends."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        self.seed_offset = int(seed_offset)
        self.action_dim, self.state_dim = int(cfg.action_dim), int(cfg.state_dim)
        self.image_size = tuple(cfg.image_size)
        if cfg.auto_reset or self.state_dim != 58 or num_envs % cfg.group_size:
            raise ValueError(
                "Cosmos H2 requires auto_reset=false, state_dim=58 and complete GRPO groups"
            )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            record_metrics=record_metrics,
            worker_info=worker_info,
        )
        self.reward_model.reset(self.num_envs)
        self.current_state = torch.zeros(num_envs, 58, device=self.device)
        self.success_label = self.reward_model.num_classes - 1
        self.last_chunk_probs = torch.zeros(
            num_envs, self.chunk, self.reward_model.num_classes, device=self.device
        )
        self._render_frames = None
        self._init_eval_mask()

    def _build_dataset(self, cfg):
        if cfg.get("enable_kir", False) or cfg.get("video_audit_dir"):
            raise ValueError("Cosmos H2 uses ordinary resets and video_cfg recording")
        split = json.loads(Path(cfg.split_path).read_text())
        episodes = [
            row["episode_index"]
            for row in split["episodes"]
            if row["split"] == cfg.split
        ]
        if not episodes:
            raise ValueError(f"No episodes in split {cfg.split!r}")
        if cfg.get("episodes") is not None:
            if not set(cfg.episodes).issubset(episodes):
                raise ValueError(
                    f"Requested episodes must belong to split {cfg.split!r}"
                )
            episodes = list(cfg.episodes)
        video_key = cfg.video_key
        modality_path = Path(cfg.initial_image_path) / "meta/modality.json"
        if modality_path.is_file():
            videos = json.loads(modality_path.read_text()).get("video", {})
            if video_key not in videos:
                aliases = [
                    key
                    for key, value in videos.items()
                    if value.get("original_key") == video_key
                ]
                if len(aliases) != 1:
                    raise ValueError(
                        f"Expected one alias for camera {video_key!r}, found {aliases}"
                    )
                video_key = aliases[0]
        dataset = LeRobotV21InitDataset(
            cfg.initial_image_path,
            video_key=video_key,
            state_dim=58,
            action_dim=58,
            image_size=self.image_size,
            episodes=episodes,
            random_start_frame=cfg.get("random_start_frame", False),
            video_backend=cfg.get("video_backend", "pyav"),
        )
        if dataset.fps != 30 or dataset.video_original_key != cfg.video_key:
            raise ValueError("Cosmos H2 requires 30-Hz data and a canonical video_key")
        joint_names = json.loads(Path(cfg.action_statistics).read_text())["joint_names"]
        for key in ("action", "observation.state"):
            names = dataset.info["features"][key]["names"]
            if len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            if names != joint_names:
                raise ValueError(
                    f"{key} joint order differs from Cosmos action statistics"
                )
        self._quality_pools = None
        if cfg.get("balanced_quality_sampling", False):
            if cfg.get("is_eval", False):
                raise ValueError("Balanced sampling is training-only")
            lookup = {
                r["episode_index"]: r.get("quality", r.get("review_status"))
                for r in split["episodes"]
            }
            qualities = cfg.get("balanced_qualities")
            if qualities is not None:
                if (
                    isinstance(qualities, str)
                    or len(qualities) < 2
                    or len(set(qualities)) != len(qualities)
                ):
                    raise ValueError(
                        "Balanced qualities must contain at least two distinct quality names"
                    )
                self._quality_pools = [
                    [
                        i
                        for i, ep in enumerate(dataset.episodes)
                        if lookup[ep] == quality
                    ]
                    for quality in qualities
                ]
            else:
                failure_qualities = set(
                    cfg.get("balanced_failure_qualities", ["imperfect2"])
                )
                if not failure_qualities or "success" in failure_qualities:
                    raise ValueError(
                        "Failure reset qualities must be nonempty and exclude success"
                    )
                self._quality_pools = [
                    [
                        i
                        for i, ep in enumerate(dataset.episodes)
                        if lookup[ep] == "success"
                    ],
                    [
                        i
                        for i, ep in enumerate(dataset.episodes)
                        if lookup[ep] in failure_qualities
                    ],
                ]
            if sum(map(len, self._quality_pools)) != len(dataset) or not all(
                self._quality_pools
            ):
                raise ValueError(
                    "Expected nonempty quality pools covering all selected training data"
                )
        self._quality_draw = 0
        return dataset

    def update_reset_state_ids(self):
        """Draw episode groups lazily in reset, keeping the seeded sampler order."""
        if not hasattr(self, "reset_state_ids"):
            self.reset_state_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _sample_reset_episode_indices(self):
        count = len(self.dataset)
        if self.use_fixed_reset_state_ids:
            base = (
                torch.arange(self.num_group) + self.seed_offset * self.num_group
            ) % count
        elif self.cfg.get("balanced_quality_sampling", False):
            base = []
            for group in range(self.num_group):
                quality = (
                    self.seed_offset * self.num_group + self._quality_draw + group
                ) % len(self._quality_pools)
                pool = self._quality_pools[quality]
                base.append(
                    pool[int(torch.randint(len(pool), (1,), generator=self._generator))]
                )
            self._quality_draw += self.num_group
            base = torch.tensor(base)
        else:
            base = torch.randint(0, count, (self.num_group,), generator=self._generator)
        return base.repeat_interleave(self.group_size).tolist()

    def _init_eval_mask(self):
        self.eval_unique_episodes = bool(self.cfg.get("eval_unique_episodes", False))
        self._eval_valid_mask = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if not self.eval_unique_episodes:
            return
        if (
            not self.cfg.get("is_eval", False)
            or not self.use_fixed_reset_state_ids
            or self.group_size != 1
            or (
                not self.ignore_terminations
                and not self.cfg.get("terminal_streak_frames", 0)
            )
        ):
            raise ValueError(
                "Unique eval requires is_eval, fixed resets, group_size=1 and either ignored or streak-based terminations"
            )
        count, slots = len(self.dataset), self.num_envs * self.total_num_processes
        if count <= 0 or slots < count or slots - count >= self.total_num_processes:
            raise ValueError(
                "Unique eval requires the smallest evenly sharded batch covering the dataset"
            )
        indices = self.seed_offset * self.num_envs + torch.arange(
            self.num_envs, device=self.device
        )
        self._eval_valid_mask = indices < count

    @torch.no_grad()
    def reset(self, *, seed=None, options=None, episode_indices=None, env_idx=None):
        if env_idx is not None:
            raise ValueError(
                "Cosmos H2 resets complete GRPO batches; env_idx must be None"
            )
        self.onload()
        indices = (
            self._sample_reset_episode_indices()
            if episode_indices is None
            else list(episode_indices)
        )
        if len(indices) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} reset episodes, got {len(indices)}"
            )
        self.reset_state_ids = torch.tensor(indices, device=self.device)
        items = [self.dataset[int(i)] for i in indices]
        starts = [item["start_items"][0] for item in items]
        images = [
            self._to_condition_frame(f["image"].clone()).unsqueeze(1) for f in starts
        ]
        self.current_obs = torch.stack(images).to(self.device).unsqueeze(2)
        self.current_state = torch.stack(
            [torch.as_tensor(f["observation.state"]).float() for f in starts]
        ).to(self.device)
        anchors = torch.stack(
            [torch.as_tensor(f["wm_previous_action"]).float() for f in starts]
        ).to(self.device)
        self.task_descriptions = [str(item.get("task", "")) for item in items]
        slots = list(range(self.num_envs))
        self.backend.close_session(slots)
        self.backend.open_session(
            env_ids=slots,
            init_frames=[[image] for image in images],
            init_actions=anchors[:, None],
            seeds=[self.seed + i for i in slots],
            task_descriptions=self.task_descriptions,
        )
        self.reward_model.reset(self.num_envs)
        self._reset_metrics()
        scores, _ = self.reward_model.score_chunk(
            self.current_obs[:, :, 0, -1:].permute(0, 2, 1, 3, 4)
        )
        self.prev_step_reward = scores[:, 0].clone()
        if self.cfg.get("terminal_streak_frames", 0):
            if (
                self.ignore_terminations
                or self.reward_model.reward_values.tolist()
                not in ([-1, 0, 1, 2, 3], [-2, -1, 0, 1, 2])
            ):
                raise ValueError(
                    "Streak termination requires supported signed five-class rewards and ignore_terminations=false"
                )
            self._terminal = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._terminal_success = self._terminal.clone()
            self._failure_streak = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._success_streak = self._failure_streak.clone()
            self._success_by_streak = self._terminal.clone()
            self._success_at_timeout = self._terminal.clone()
            self._time_limit_reached = self._terminal.clone()
        self._render_frames = None
        self._is_start = False
        return self._wrap_obs(), {}

    def _infer_next_chunk_frames(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        super()._infer_next_chunk_frames(actions)
        self.current_state = actions[:, -1].detach().clone()
        if self.video_cfg.save_video:
            self._render_frames = (
                ((self.current_obs[:, :, 0, -self.chunk :] + 1) * 127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 4, 1)
                .cpu()
            )

    def _infer_next_chunk_rewards(self):
        frames = self.current_obs[:, :, 0, -self.chunk :].permute(0, 2, 1, 3, 4)
        scores, self.last_chunk_probs = self.reward_model.score_chunk(frames)
        return scores

    def _calc_step_reward(self, scores):
        rewards = scores
        if self.use_rel_reward:
            previous = torch.cat(
                [self.prev_step_reward[:, None], scores[:, :-1]], dim=1
            )
            rewards = scores - previous
        self.prev_step_reward = scores[:, -1].clone()
        return rewards * self.cfg.reward_coef

    def _estimate_success_from_rewards(self, chunk_rewards):
        success = self.reward_model.labels == self.success_label
        return torch.zeros_like(success) if self.ignore_terminations else success

    def _timeout_metric_key(self):
        """Keep legacy names only for the legacy three-point success scale."""
        return (
            "success_at_timeout_final3"
            if float(self.reward_model.reward_values[self.success_label]) == 3
            else "success_at_timeout_final_success"
        )

    def _record_metrics(self, step_reward, terminations, infos):
        if not self.record_metrics:
            return infos
        if self.cfg.get("terminal_streak_frames", 0):
            infos = super()._record_metrics(step_reward, self._terminal_success, infos)
            infos["episode"].update(
                {
                    "success_final": self._terminal_success.clone(),
                    "failure": self._terminal
                    & ~self._terminal_success
                    & (self._failure_streak >= self.cfg.terminal_streak_frames),
                    "timeout": self._terminal
                    & ~(
                        self._terminal_success
                        | (self._failure_streak >= self.cfg.terminal_streak_frames)
                    ),
                    "success_by_streak": self._success_by_streak.clone(),
                    self._timeout_metric_key(): self._success_at_timeout.clone(),
                    "time_limit_reached": self._time_limit_reached.clone(),
                    "score": self.reward_model.scores.clone(),
                    "episode_seconds": self._elapsed_steps.float() / 30,
                }
            )
            return infos
        infos = super()._record_metrics(
            step_reward, self.reward_model.labels == self.success_label, infos
        )
        infos["episode"].update(
            {
                "episode_seconds": self._elapsed_steps.float() / 30,
                "score": self.reward_model.scores.clone(),
                "success_final": self.reward_model.labels == self.success_label,
                "success_prob_max": self.last_chunk_probs[
                    :, :, self.success_label
                ].amax(dim=1),
            }
        )
        return infos

    def _wrap_obs(self):
        obs = super()._wrap_obs()
        obs["states"] = self.current_state.to(self.device, dtype=torch.float32)
        return obs

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        if self.cfg.get("terminal_streak_frames", 0):
            obs, rewards, terminations, truncations, infos = self._streak_chunk_step(
                policy_output_action
            )
        else:
            obs, rewards, terminations, truncations, infos = super().chunk_step(
                policy_output_action
            )
        if self.eval_unique_episodes:
            terminations &= self._eval_valid_mask[:, None]
            truncations &= self._eval_valid_mask[:, None]
        result_dir = self.cfg.get("evaluation_result_dir")
        if result_dir and self.cfg.get("is_eval", False):
            completed = (
                (terminations | truncations).any(dim=1).nonzero().flatten().tolist()
            )
            if completed:
                directory = Path(result_dir)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / f"episodes-{os.getpid()}.jsonl").open("a") as stream:
                    for slot in completed:
                        index = int(self.reset_state_ids[slot])
                        row = {
                            "episode_index": int(self.dataset.episodes[index]),
                            "success_once": bool(self.success_once[slot]),
                            "success_final": bool(self._terminal_success[slot])
                            if self.cfg.get("terminal_streak_frames", 0)
                            else bool(
                                self.reward_model.labels[slot] == self.success_label
                            ),
                            "final_score": float(self.reward_model.scores[slot]),
                            "final_probabilities": self.last_chunk_probs[
                                slot, -1
                            ].tolist(),
                            "elapsed_steps": int(self._elapsed_steps[slot]),
                            "seed": self.seed + slot,
                        }
                        if self.cfg.get("terminal_streak_frames", 0):
                            row.update(
                                success_by_streak=bool(self._success_by_streak[slot]),
                                time_limit_reached=bool(self._time_limit_reached[slot]),
                                failure=bool(
                                    self._failure_streak[slot]
                                    >= self.cfg.terminal_streak_frames
                                    and not self._terminal_success[slot]
                                ),
                                timeout=bool(
                                    self._time_limit_reached[slot]
                                    and not self._terminal_success[slot]
                                ),
                                final_predicted_reward=int(
                                    self.reward_model.reward_values[
                                        self.reward_model.labels[slot]
                                    ]
                                ),
                            )
                        if self.cfg.get("terminal_streak_frames", 0):
                            row[self._timeout_metric_key()] = bool(
                                self._success_at_timeout[slot]
                            )
                            row["success_reward"] = float(
                                self.reward_model.reward_values[self.success_label]
                            )
                            row["reward_values"] = (
                                self.reward_model.reward_values.tolist()
                            )
                        stream.write(json.dumps(row) + "\n")
        return obs, rewards, terminations, truncations, infos

    @torch.no_grad()
    def _streak_chunk_step(self, policy_output_action):
        self.onload()
        actions = torch.as_tensor(
            policy_output_action, device=self.device, dtype=torch.float32
        )
        active = ~self._terminal
        old_frame = self.current_obs[:, :, :, -1:].clone()
        videos = old_frame[:, :, 0].expand(-1, -1, self.chunk, -1, -1).clone()
        slots = active.nonzero().flatten().tolist()
        if slots:
            videos[slots] = self.backend.generate(
                env_ids=slots, actions=actions[slots]
            ).to(videos)
        self.current_obs = torch.cat([old_frame, videos.unsqueeze(2)], dim=3)
        scores = self._infer_next_chunk_rewards()
        labels = self.last_chunk_probs.argmax(-1)
        terms = torch.zeros_like(scores, dtype=torch.bool)
        truncs = torch.zeros_like(terms)
        valid = torch.zeros_like(terms)
        stop = torch.full(
            (self.num_envs,), self.chunk - 1, device=self.device, dtype=torch.long
        )
        for i in range(self.chunk):
            alive = ~self._terminal
            valid[:, i] = alive
            self._elapsed_steps += alive.long()
            self._failure_streak = torch.where(
                alive & (self.reward_model.reward_values[labels[:, i]] < 0),
                self._failure_streak + 1,
                torch.where(
                    alive, torch.zeros_like(self._failure_streak), self._failure_streak
                ),
            )
            self._success_streak = torch.where(
                alive & (labels[:, i] == self.success_label),
                self._success_streak + 1,
                torch.where(
                    alive, torch.zeros_like(self._success_streak), self._success_streak
                ),
            )
            failure = alive & (self._failure_streak >= self.cfg.terminal_streak_frames)
            success = alive & (self._success_streak >= self.cfg.terminal_streak_frames)
            terms[:, i] = failure | success
            truncs[:, i] = (
                alive
                & ~terms[:, i]
                & (self._elapsed_steps >= self.cfg.max_episode_steps)
            )
            ended = terms[:, i] | truncs[:, i]
            stop[ended] = i
            # Time-limit success uses the last valid frame, never a padded future frame.
            timeout_success = (
                truncs[:, i] & (labels[:, i] == self.success_label)
                if self.cfg.get(
                    "timeout_success_if_final_success",
                    self.cfg.get("timeout_success_if_final_label3", False),
                )
                else torch.zeros_like(alive)
            )
            self._success_by_streak |= success
            self._success_at_timeout |= timeout_success
            self._time_limit_reached |= truncs[:, i]
            self._terminal_success |= success | timeout_success
            self._terminal |= ended
        previous = self.prev_step_reward.clone()
        rewards = torch.zeros_like(scores)
        for i in range(self.chunk):
            rewards[:, i] = torch.where(
                valid[:, i],
                scores[:, i] - previous if self.use_rel_reward else scores[:, i],
                0,
            )
            previous = torch.where(valid[:, i], scores[:, i], previous)
        self.prev_step_reward = previous
        # Tail frames and commands are padding, excluded by dones from the actor loss.
        for slot in range(self.num_envs):
            if not active[slot]:
                self.current_obs[slot] = old_frame[slot].expand_as(
                    self.current_obs[slot]
                )
                continue
            end = int(stop[slot])
            self.current_state[slot] = actions[slot, end]
            self.current_obs[slot, :, :, end + 2 :] = self.current_obs[
                slot, :, :, end + 1 : end + 2
            ]
            self.last_chunk_probs[slot, end + 1 :] = self.last_chunk_probs[
                slot, end : end + 1
            ].clone()
        self.reward_model.scores = previous.clone()
        final_labels = self.last_chunk_probs[:, -1].argmax(-1)
        self.reward_model.labels = torch.where(
            active, final_labels, self.reward_model.labels
        )
        if self.video_cfg.save_video:
            self._render_frames = (
                ((self.current_obs[:, :, 0, -self.chunk :] + 1) * 127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 4, 1)
                .cpu()
            )
        rewards *= self.cfg.reward_coef
        infos = self._record_metrics(rewards.sum(1), self._terminal_success, {})
        return [self._wrap_obs()], rewards, terms, truncs, [infos]

    def capture_image(self):
        if self._render_frames is not None:
            return self._render_frames
        return self._wrap_obs()["main_images"][:, None]
