# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native seven-worker evaluation covers 55 examples without padding bias."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
from rlinf.utils.metric_utils import compute_evaluate_metrics
from toolkits.world_model.dreamdojo_validation import config


def fake_env(rank, count=55, workers=7, per_worker=8):
    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.cfg = OmegaConf.create(
        {"eval_unique_episodes": True, "is_eval": True, "max_episode_steps": 240}
    )
    env.device = torch.device("cpu")
    env.num_envs = env.num_group = per_worker
    env.total_num_processes = workers
    env.seed_offset = rank
    env.group_size = 1
    env.use_fixed_reset_state_ids = True
    env.ignore_terminations = True
    env.dataset = range(count)
    env._init_eval_mask()
    return env


def test_main_yaml_uses_native_train_eval_on_seven_healthy_gpus():
    cfg = config()
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-3,5-7"
    assert cfg.runner.max_epochs == 1000 and cfg.runner.max_steps == -1
    assert cfg.runner.val_check_interval == cfg.runner.save_interval == 5
    assert cfg.env.train.total_num_envs == cfg.actor.global_batch_size == 56
    assert cfg.actor.micro_batch_size == 2
    assert cfg.env.eval.total_num_envs == 56
    assert cfg.env.eval.group_size == 1
    assert cfg.env.eval.eval_unique_episodes
    assert cfg.env.eval.use_fixed_reset_state_ids
    assert cfg.env.eval.ignore_terminations and not cfg.env.eval.auto_reset
    assert cfg.env.eval.initial_image_mixing_weights is None
    assert not cfg.env.train.eval_unique_episodes
    for part in (cfg.env.train, cfg.env.eval):
        assert part.num_inference_steps == 35
        assert part.video_cfg.fps == 15
        assert part.enable_offload
        assert part.cache_text_embeddings and part.skip_zero_guidance
        assert part.wm_offload_diffusion_model is False
        assert part.wm_offload_tokenizer is False
        assert part.wm_offload_text_encoder
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None


def test_seven_shards_cover_each_validation_episode_once():
    valid, padding = [], []
    for rank in range(7):
        env = fake_env(rank)
        ids = env._sample_reset_episode_indices()
        assert ids == env._sample_reset_episode_indices()
        for index, keep in zip(ids, env._eval_valid_mask.tolist(), strict=True):
            (valid if keep else padding).append(index)
    assert valid == list(range(55))
    assert padding == [0]


def test_low_noise_recipe_records_native_train_video_without_other_policy_changes():
    cfg = config()
    head = cfg.actor.model.rl_head_config
    assert head.noise_method == "flow_sde"
    assert head.noise_level == pytest.approx(0.1)
    assert not head.noise_anneal
    assert head.action_noise_scale == 0
    assert cfg.actor.model.denoising_steps == 4
    assert cfg.env.train.video_cfg.save_video
    assert cfg.env.eval.video_cfg.save_video
    assert cfg.env.train.initial_image_mixing_weights == [0.34, 0.33, 0.33]
    assert not cfg.env.train.random_start_frame
    assert cfg.actor.optim.lr == pytest.approx(5e-6)


@pytest.mark.parametrize("count", [0, 49, 57])
def test_unique_eval_rejects_incomplete_or_excessive_slots(count):
    with pytest.raises(ValueError, match="smallest evenly sharded"):
        fake_env(0, count)


def test_padding_never_finishes_or_biases_native_metrics():
    shards = []
    for rank in range(7):
        env = fake_env(rank)
        env.chunk = 12
        env.elapsed_steps = 228
        env.auto_reset = False
        env.onload = lambda: None
        env._infer_next_chunk_frames = lambda actions: None
        env._infer_next_chunk_rewards = lambda: torch.zeros(8, 12)
        # One true success (episode 0), and a deliberately successful padding.
        successes = torch.tensor(env._sample_reset_episode_indices()) == 0
        env.reward_model = SimpleNamespace(stage=successes.long() * 3)
        env._record_metrics = lambda reward, done, info: {
            "episode": {
                "success_once": successes.clone(),
                "return": successes.float() * 3,
            }
        }
        env._wrap_obs = lambda: {}
        _, _, terms, truncs, infos = env.chunk_step(torch.zeros(8, 12, 28))
        done = (terms | truncs).any(dim=1)
        assert torch.equal(done, env._eval_valid_mask)
        # Same newly_done selection used by the native EnvWorker.
        shards.append({k: v[done] for k, v in infos[0]["episode"].items()})
    result = compute_evaluate_metrics(shards)
    assert result["num_trajectories"] == 55
    assert result["success_once"] == pytest.approx(1 / 55)
    assert result["return"] == pytest.approx(3 / 55)


def test_training_mask_is_inactive():
    env = fake_env(6)
    env.cfg.eval_unique_episodes = False
    env.cfg.is_eval = False
    env._init_eval_mask()
    assert not env.eval_unique_episodes
    assert env._eval_valid_mask.all()


def test_eight_worker_cluster_eval_counts_55_not_56():
    valid = []
    for rank in range(8):
        env = fake_env(rank, workers=8, per_worker=7)
        valid.extend(
            index
            for index, keep in zip(
                env._sample_reset_episode_indices(),
                env._eval_valid_mask.tolist(),
                strict=True,
            )
            if keep
        )
    assert valid == list(range(55))
