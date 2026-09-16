# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Causal keyframe selection, grouped reset and no-credit reward warmup."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from rlinf.data.datasets.lerobot_world_model import (
    LeRobotV21InitDataset,
    MixedLeRobotV21InitDataset,
)
from rlinf.envs.world_model.dreamdojo_reward import BatchedMilestoneReward
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv


def dataset(picked=30, handed=90, placed=110):
    ds = LeRobotV21InitDataset.__new__(LeRobotV21InitDataset)
    ds.episodes = [0]
    ds.root = Path("/fixture")
    ds.fps = 30.0
    ds.history_frame_offset = 2
    ds.kir_context_len = 0
    ds.enable_kir = ds.random_start_frame = False
    ds.state_dim = ds.action_dim = 28
    ds.image_size = (2, 3)
    ds._episode_meta = {
        0: {
            "episode_index": 0,
            "length": 120,
            "success": True,
            "tasks": ["trocar"],
            "source_annotation": {
                "cleaned_phase_frames": {
                    "left_hand_pickup": picked,
                    "handover_to_right_hand": handed,
                    "placed_on_plate": placed,
                }
            },
        }
    }
    values = np.arange(120, dtype=np.float32)[:, None].repeat(28, axis=1)
    table = pa.table(
        {"observation.state": values.tolist(), "action": (2 * values).tolist()}
    )
    ds._parquet_path = lambda _: Path("/fixture/parquet")
    ds._read_parquet = lambda _: table
    ds._video_path = lambda _: Path("/fixture/video")
    ds._decode_video_frames = lambda _, indices: (
        np.array(indices, dtype=np.uint8)[:, None, None, None]
        * np.ones((len(indices), 2, 3, 3), dtype=np.uint8)
    )
    return ds


def reward(count=8):
    obj = BatchedMilestoneReward.__new__(BatchedMilestoneReward)
    obj.device = torch.device("cpu")
    obj.num_envs = count
    obj.reset()
    obj._prepare = lambda frames: frames.float() / 255
    return obj


def env(ds, count=8, group_size=8, probability=1.0):
    obj = DreamDojoEnv.__new__(DreamDojoEnv)
    obj.enable_kir = True
    obj.kir_probability = probability
    obj.kir_max_offset_frames = 30
    obj._kir_generator = torch.Generator().manual_seed(100000)
    obj.dataset = ds
    obj.num_envs = count
    obj.group_size = group_size
    obj.device = torch.device("cpu")
    obj.state_dim = 28
    obj.image_size = (2, 3)
    obj.condition_frame_length = 1
    obj.onload = lambda: None
    obj.trans_norm = lambda frames: frames * 2 - 1
    obj.reward_model = reward(count)
    obj.image_queue = [[] for _ in range(count)]
    obj.record_metrics = True
    obj.prev_step_reward = torch.zeros(count)
    obj.success_once = torch.zeros(count, dtype=torch.bool)
    obj.returns = torch.zeros(count)
    return obj


@pytest.mark.parametrize(
    "picked,handed,expected", [(30, 90, 60), (30, 70, 50), (30, 110, 80)]
)
def test_adaptive_handover_start(picked, handed, expected):
    ds = dataset(picked, handed, placed=119)
    assert ds.handover_start_frame(0) == expected
    assert ds.handover_start_frame(0, 10) == handed - 10
    assert ds.handover_start_frame(0, 15) == handed - 15


@pytest.mark.parametrize(
    "change", ["missing", "failure", "unordered", "boundary", "ambiguous"]
)
def test_unreliable_labels_fall_back_without_changing_episode(change):
    ds = dataset()
    meta = ds._episode_meta[0]
    phases = meta["source_annotation"]["cleaned_phase_frames"]
    if change == "missing":
        meta.pop("source_annotation")
    if change == "failure":
        meta["success"] = False
    if change == "unordered":
        phases["left_hand_pickup"] = 100
    if change == "boundary":
        phases["placed_on_plate"] = 120
    if change == "ambiguous":
        phases["handover_to_right_hand"] = 38
    assert ds.handover_start_frame(0) is None
    obj = env(ds)
    obj.reset(episode_indices=[0] * 8)
    assert not obj.reset_is_kir.any()
    assert (obj.reset_start_frames == 2).all()
    assert (obj.reward_model.stage == 0).all()


def test_explicit_frame_aligns_proprio_history_and_only_past_images():
    ds = dataset()
    item = ds.get_at_frame(0, 60, reward_history_frames=17)
    first = item["start_items"][0]
    assert item["dataset_meta"]["start_frame"] == 60
    torch.testing.assert_close(first["observation.state"], torch.full((28,), 60.0))
    torch.testing.assert_close(first["wm_previous_state"], torch.full((28,), 58.0))
    torch.testing.assert_close(first["wm_previous_action"], torch.full((28,), 116.0))
    torch.testing.assert_close(
        item["reward_history"][:, 0, 0, 0], torch.arange(44, 61, dtype=torch.uint8)
    )
    assert not item["target_items"]
    assert ds[0]["dataset_meta"]["start_frame"] == 2


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_shared_group_images_survive_native_inplace_normalization(probability):
    from torchvision import transforms

    ds = dataset()
    start = 60 if probability else 2
    original = ds.get_at_frame(
        start_frame=start, idx=0, reward_history_frames=17 if probability else 0
    )
    before = original["start_items"][0]["image"].clone()
    ds.get_at_frame = lambda *args, **kwargs: original
    obj = env(ds, probability=probability)
    obj.trans_norm = transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True)
    obs, _ = obj.reset(episode_indices=[0] * 8)
    torch.testing.assert_close(original["start_items"][0]["image"], before)
    torch.testing.assert_close(
        obj.current_obs[:, :, 0, 0], (before * 2 - 1)[None].repeat(8, 1, 1, 1)
    )
    torch.testing.assert_close(
        obs["main_images"], obs["main_images"][:1].expand_as(obs["main_images"])
    )
    assert obj.current_obs.min() >= -1 and obj.current_obs.max() <= 1


@pytest.mark.parametrize(
    "start,history", [(1, 0), (120, 0), (60.5, 0), (2, 17), (60, -1)]
)
def test_invalid_explicit_frames_fail(start, history):
    with pytest.raises(ValueError):
        dataset().get_at_frame(0, start, reward_history_frames=history)


def test_group_reset_shares_frame_state_and_does_not_pay_for_picked():
    obj = env(dataset())
    obs, _ = obj.reset(episode_indices=[0] * 8)
    assert obj.reset_is_kir.all()
    assert (obj.reset_start_frames == 60).all()
    torch.testing.assert_close(obs["states"], torch.full((8, 28), 60.0))
    assert (obs["main_images"] == obs["main_images"][0]).all()
    assert (obj.reward_model.stage == 1).all()
    assert not obj.returns.any() and not obj.success_once.any()
    assert all(len(h) == 17 for h in obj.reward_model.frame_history)
    assert all(not votes for heads in obj.reward_model.recent for votes in heads)
    assert (
        obj.reward_model.advance(torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)).sum()
        == 0
    )
    payouts = sum(
        obj.reward_model.advance(torch.tensor([[1.0, 1.0, 0.0]]).repeat(8, 1))
        for _ in range(13)
    )
    torch.testing.assert_close(payouts, torch.ones(8))


def test_zero_probability_is_same_as_original_reset():
    ds = dataset()
    original = env(ds)
    original.enable_kir = False
    enabled = env(ds, probability=0)
    obs0, _ = original.reset(episode_indices=[0] * 8)
    obs1, _ = enabled.reset(episode_indices=[0] * 8)
    for key in ("main_images", "states"):
        torch.testing.assert_close(obs0[key], obs1[key])
    for key in ("previous_state", "previous_action", "current_obs"):
        torch.testing.assert_close(getattr(original, key), getattr(enabled, key))
    assert not enabled.reset_is_kir.any()
    assert all(not history for history in enabled.reward_model.frame_history)


def test_group_mismatch_rejected():
    with pytest.raises(ValueError, match="share"):
        env(dataset())._load_reset_items([0] * 7 + [1])


def test_reproducible_group_selection_and_separate_episode_rng():
    a, b = (
        env(dataset(), count=64, probability=0.5),
        env(dataset(), count=64, probability=0.5),
    )
    a._generator = torch.Generator().manual_seed(123)
    before = a._generator.get_state().clone()
    a._load_reset_items([0] * 64)
    b._load_reset_items([0] * 64)
    assert torch.equal(a.reset_is_kir, b.reset_is_kir)
    assert torch.equal(before, a._generator.get_state())
    assert all(len(set(group)) == 1 for group in a.reset_is_kir.reshape(-1, 8).tolist())
    assert 0 < a.reset_is_kir.sum() < 64


def test_mixed_dataset_keeps_flat_index_and_source_metadata():
    mixed = MixedLeRobotV21InitDataset.__new__(MixedLeRobotV21InitDataset)
    mixed.datasets = [dataset(), dataset(20, 60, 110)]
    mixed.offsets = [0, 1, 2]
    assert mixed.handover_start_frame(1) == 40
    item = mixed.get_at_frame(1, 40, reward_history_frames=17)
    assert item["dataset_meta"]["source_index"] == 1
    assert item["dataset_meta"]["start_frame"] == 40
    with pytest.raises(IndexError):
        mixed.get_at_frame(2)


def test_warmup_only_changes_selected_environments_and_no_future_votes():
    obj = reward(2)
    frames = torch.zeros(1, 17, 3, 2, 3, dtype=torch.uint8)
    obj.prime_handover(frames, [1])
    assert obj.stage.tolist() == [0, 1]
    assert len(obj.frame_history[0]) == 0 and len(obj.frame_history[1]) == 17
    assert not any(obj.recent[1])
    with pytest.raises(ValueError):
        obj.prime_handover(frames, [2])
    with pytest.raises(ValueError):
        obj.prime_handover(frames[:, :16], [1])


def test_kir_settings_reject_eval_and_uncontrolled_random_start():
    from omegaconf import OmegaConf

    from toolkits.world_model.dreamdojo_validation import config

    ec = config().env.train
    ec.enable_kir = True
    for key in ("is_eval", "random_start_frame"):
        candidate = OmegaConf.create(OmegaConf.to_container(ec))
        candidate[key] = True
        with pytest.raises(ValueError, match="train-only"):
            DreamDojoEnv(candidate, 8, 0, 1)


def test_stage_initialization_does_not_count_as_episode_success_or_return():
    obj = env(dataset())
    obj.reset(episode_indices=[0] * 8)
    obj.elapsed_steps = 12
    obj.last_chunk_probs = torch.zeros(8, 6, 3)
    info = obj._record_metrics(torch.zeros(8), torch.zeros(8, dtype=torch.bool), {})[
        "episode"
    ]
    assert info["picked"].all()
    assert not info["handed"].any() and not info["success_once"].any()
    assert not info["return"].any()
    assert info["kir_fraction"].sum() == 8


def test_runtime_mounts_and_chain_guard_cover_same_explicit_kir_files():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "docker/dreamdojo/run_cluster.slurm").read_text()
    chain = (root / "docker/dreamdojo/chain.py").read_text()
    for path in (
        "rlinf/data/datasets/lerobot_world_model.py",
        "rlinf/envs/world_model/world_model_dreamdojo_env.py",
        "rlinf/envs/world_model/dreamdojo_reward.py",
        "examples/embodiment/config/env/dreamdojo_trocar.yaml",
        "toolkits/world_model/dreamdojo_validation.py",
    ):
        assert path in launcher and path in chain
    assert (
        "${DREAMDOJO_KIR_SOURCE_ROOT}/${relative}:/opt/src/RLinf/${relative}:ro"
        in launcher
    )
