# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unique native evaluation, video audit and portable diagnostics."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.world_model import dreamdojo_video_audit as audit_module
from rlinf.envs.world_model.dreamdojo_video_audit import (
    MilestoneVideoAudit,
    annotate_frame,
    prediction_status,
)
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    compute_group_reward_metrics,
)
from toolkits.world_model import dreamdojo_validation as validation
from toolkits.world_model.dreamdojo_validation import config

# Native evaluation and padding


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


def test_local_recipe_preserves_native_train_eval_defaults():
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


def test_local_recipe_preserves_flow_sde_and_video_settings():
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


# Video audit and original reward traces


def case(index=0, valid=True):
    return {
        "flat_episode_index": index,
        "episode_index": index,
        "source_path": "/assets/data/pick_trocar_teleop_success_validation",
        "start_frame": 1,
        "valid": valid,
    }


def test_audit_is_disabled_for_normal_training_and_eval():
    cfg = config()
    assert cfg.env.train.video_audit_dir is None
    assert cfg.env.eval.video_audit_dir is None


def test_success_is_confirmed_state_not_instantaneous_probability():
    assert prediction_status(0, 0, False) == "NOT YET"
    assert prediction_status(0, 0, True) == "FAIL"
    assert prediction_status(2, 1, False) == "SUCCESS"
    assert prediction_status(2, 2, True) == "FAIL"
    assert prediction_status(3, 2, True) == "SUCCESS"


def test_overlay_preserves_every_source_pixel():
    frame = np.full((480, 640, 3), (100, 140, 180), np.uint8)
    original = frame.copy()
    rendered = annotate_frame(
        frame, np.array([0.1, 0.2, 0.3]), 3, 120, True, case(), "policy", (0.8,) * 3
    )
    assert rendered.shape == (480, 1040, 3)
    np.testing.assert_array_equal(rendered[:, :640], original)
    np.testing.assert_array_equal(frame, original)


def test_full_trace_is_aligned_padding_excluded_and_not_rescored(tmp_path, monkeypatch):
    writers = {}

    class Writer:
        def __init__(self, path, **kwargs):
            self.count = 0
            writers[str(path)] = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def append_data(self, frame):
            self.count += 1
            self.last = frame.copy()

    monkeypatch.setattr(audit_module.imageio, "get_writer", Writer)
    audit = MilestoneVideoAudit(tmp_path, "policy", (0.8,) * 3)
    initial = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    frames = np.full((2, 120, 16, 32, 3), 120, dtype=np.uint8)
    probs = np.full((2, 120, 3), 0.2, dtype=np.float32)
    stages = np.tile(np.repeat([0, 1, 2, 3], 30), (2, 1))
    audit.start(initial, [case(0), case(1, False)])
    audit.append(frames, probs, stages)
    frames[:] = 0
    probs[:] = 0
    stages[:] = 0
    summaries = audit.finish()
    assert len(summaries) == 1 and summaries[0]["classifier_success"]
    assert summaries[0]["final_stage"] == 3
    directory = tmp_path / "rollout_000/case_000"
    assert not (tmp_path / "rollout_000/case_001").exists()
    assert len(writers) == 2 and all(writer.count == 121 for writer in writers.values())
    np.testing.assert_array_equal(
        writers[str(directory / "raw.mp4")].last, np.full((16, 32, 3), 120, np.uint8)
    )
    with (directory / "trace.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == 121
    assert trace[0]["pick_probability"] == ""
    assert int(trace[-1]["raw_action_count"]) == 240
    assert float(trace[-1]["relative_seconds"]) == 8
    assert float(trace[-1]["place_probability"]) == pytest.approx(0.2)
    assert trace[-1]["place_confirmed"] == "True"
    assert json.loads((directory / "summary.json").read_text())["classifier_success"]


def test_incomplete_audit_fails_instead_of_exporting_misleading_video(tmp_path):
    audit = MilestoneVideoAudit(tmp_path, "policy", (0.8,) * 3)
    audit.start(np.zeros((1, 16, 32, 3), np.uint8), [case()])
    with pytest.raises(RuntimeError, match="previous rollout"):
        audit.start(np.zeros((1, 16, 32, 3), np.uint8), [case()])
    audit.append(
        np.zeros((1, 6, 16, 32, 3), np.uint8), np.zeros((1, 6, 3)), np.zeros((1, 6))
    )
    with pytest.raises(ValueError, match="full 240-action"):
        audit.finish()


def test_native_audit_uses_original_payouts_and_does_not_change_reward():
    env = DreamDojoEnv.__new__(DreamDojoEnv)
    env.current_obs = torch.zeros(2, 3, 1, 7, 16, 32)
    env.wm_steps_per_chunk, env.num_envs, env.chunk = 6, 2, 12
    payouts = torch.tensor(
        [[0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.float32
    )
    probs = torch.full((2, 6, 3), 0.3)
    seen = []
    env.reward_model = SimpleNamespace(stage=torch.tensor([0, 1]))

    def score(frames):
        env.reward_model.stage = torch.tensor([3, 2])
        return payouts, probs

    env.reward_model.score_chunk = score
    env.video_audit = SimpleNamespace(append=lambda *args: seen.append(args))
    before = env.current_obs.clone()
    reward = env._infer_next_chunk_rewards()
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][2], [[0, 1, 1, 2, 2, 3], [1, 1, 1, 1, 1, 2]])
    torch.testing.assert_close(reward[:, 1::2], payouts)
    assert torch.count_nonzero(reward[:, ::2]) == 0
    torch.testing.assert_close(env.current_obs, before)
    env.video_audit = None
    env.reward_model.stage = torch.tensor([0, 1])
    torch.testing.assert_close(env._infer_next_chunk_rewards(), reward)


# Portable diagnostics and reward metrics


def test_diagnostic_keeps_model_when_eval_returns_none():
    calls = []

    class Policy:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            # Match the actual GR00T override, which returns None.

    model = Policy()
    assert validation.prepare_diagnostic_policy(model, "cpu") is model
    assert calls == [("to", "cpu"), ("eval",)]


@pytest.mark.parametrize("step", [1, 15, 220, 1000])
def test_checkpoint_labels_follow_global_step(step):
    assert validation.checkpoint_case(
        f"/trial/global_step_{step}/actor/weights.pt"
    ) == (f"global_step_{step}")
    assert validation.checkpoint_case("base") == "base"


def test_group_metrics_preserve_scores():
    scores = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
    original = scores.clone()
    metrics = compute_group_reward_metrics(scores)
    assert metrics["group_flat_fraction"] == 0.5
    assert metrics["group_zero_fraction"] == 0.5
    assert metrics["group_return_std_mean"] == 0.5
    torch.testing.assert_close(scores, original)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_group_metrics_reject_non_finite(bad):
    with pytest.raises(ValueError, match="Non-finite"):
        compute_group_reward_metrics(torch.tensor([[0.0, bad]]))


def test_cluster_paths_override_all_local_defaults(monkeypatch, tmp_path):
    paths = {
        "DREAMDOJO_ROOT": tmp_path / "wm_repo",
        "DREAMDOJO_DATA_ROOT": tmp_path / "datasets",
        "DREAMDOJO_WM_CHECKPOINT": tmp_path / "weights/wm.pt",
        "DREAMDOJO_REWARD_CHECKPOINT": tmp_path / "weights/reward.pt",
        "DREAMDOJO_ACTION_STATISTICS": tmp_path / "stats.json",
        "GR00T_MODEL_PATH": tmp_path / "weights/policy",
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("DREAMDOJO_GPUS", "0-7")
    monkeypatch.chdir(tmp_path)
    cfg = validation.config()
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-7"
    for part in (cfg.env.train, cfg.env.eval):
        assert part.dreamdojo_root == str(paths["DREAMDOJO_ROOT"])
        assert part.model_path == str(paths["DREAMDOJO_WM_CHECKPOINT"])
        assert part.reward_model.from_pretrained == str(
            paths["DREAMDOJO_REWARD_CHECKPOINT"]
        )
        assert part.action_statistics == str(paths["DREAMDOJO_ACTION_STATISTICS"])
        assert all(
            Path(path).parent == paths["DREAMDOJO_DATA_ROOT"]
            for path in part.initial_image_path
        )
    assert (
        cfg.actor.model.model_path
        == cfg.rollout.model.model_path
        == str(paths["GR00T_MODEL_PATH"])
    )


def test_diagnostics_write_results_not_source_snapshots(monkeypatch, tmp_path):
    output = tmp_path / "diagnostic"
    input_path = tmp_path / "previous_run"
    monkeypatch.setattr(
        "sys.argv",
        ["diagnostic", "report", "--output", str(output), "--inputs", str(input_path)],
    )
    monkeypatch.setattr(
        validation,
        "report",
        lambda args: validation.save_json(args.output / "results.json", {}),
    )
    validation.main()
    assert {p.name for p in output.iterdir()} == {
        "arguments.json",
        "results.json",
        "timing.json",
    }
    assert json.loads((output / "arguments.json").read_text())["inputs"] == [
        str(input_path)
    ]
